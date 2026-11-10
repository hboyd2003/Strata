# `serve/dynamic-yarn`

Adds `--rope-scaling dynamic`, which extends the context past the model's trained 262,144 positions with YaRN only when a conversation actually needs it. Stacked on `main`, after `perf/gfx906-kv-grow`.

This is a port of upstream PR [#2961](https://github.com/Niko1221/Strata/pull/2961) by sergqwer. The PR was closed when the upstream tracker was reset for 0.2.0, not rejected. The fork adds two changes of its own (below).

## How it works

- A conversation that fits the trained context runs with no rope scaling, exactly as without the flag. Qwen notes that static YaRN, which is always on, makes short texts worse.
- Past 262,144 positions, the conversation runs YaRN with factor = length / 262,144, rounded up to a multiple of 0.125 (`--rope-dynamic-step`) and capped at `--rope-scale`. Left unset, the cap comes from `--max-context`, so `--max-context 524288` gives at most 2.
- The K cache holds keys after rope. When a conversation's factor changes, kernels rotate the cached keys to the new angles instead of reading the prompt again. V, deeper layers and the linear-attention state are not recomputed. This is the same trick llama.cpp uses for its K-shift.
- The rope kernels read the float64 angle table (as with `STRATA_ROPE_TABLE=1`), and a factor change rebuilds the table in place.
- The factor is saved in session files, so a restored conversation comes back with its own factor.
- Limits: one GPU, `--serve`, single session (no `--batch`), int8 or fp16 K/V (q4_0 and k8v4 are refused), no layer split.

Most implementations ship static YaRN only (vLLM, SGLang, llama.cpp). Hugging Face's "dynamic" mode is dynamic NTK, and it leaves cached post-rope keys at their old angles.

## What the fork adds

- **Delta session saves after a rotation write a full file.** With `STRATA_SESSION_DELTA=1` (`serve/session-delta`), a save normally appends only new K/V to the conversation's log. After a rotation, the log still holds keys at the old angles, so the next save now writes the whole file again. Later saves at the same factor append as usual.
- **Two table copies are kept in pinned host memory.** In PR #2961, every switch between conversations at different factors rebuilt the table on the CPU. That cost about 300 ms at 512K positions on this box, even when switching back to a factor it had just used. The branch keeps the two most recently used tables (each remembers its scaling) and rebuilds only the least recently used one. Switching back to a held factor is then just an upload: 9-10 ms at 1M positions. The log line for a switch says whether the table was held or built, and where the time went.

## Measured (MI50, Qwen3.8-Flash-Next IQ3_S, int8 K/V, `--kv-grow`)

- **Identity.** Without the flag, the output is bitwise the same as `main`. With `--rope-scaling dynamic --max-context 524288`, a short conversation is bitwise the same as `main` with `STRATA_ROPE_TABLE=1` at the same max context. Compared with `main` at 262,144, first-token KL is 0.0003 (2K prompt) and 0.0004 (32K prompt), with the same top token. That difference includes the float64 table and the ~250 expert slots that the larger context reserves.
- **Parity on gfx906.** Rotated keys versus keys written at the new factor: RMS error 1.31-1.45x that of a fresh write, worst 1.02 fp16 ulps / 1.21 int8 steps.
- **One conversation grown to 459,846 tokens** in 40K steps, with a short conversation asked between steps:
  - 0 of 31 codewords wrong, and 0 of 4 after a restart restored it from disk.
  - Factors went 1 → 1.125 → 1.375 → 1.5 → 1.625 → 1.875, and each rotation of 255K-420K cells took 28-45 ms.
  - The elastic K/V grew to 466,944 cells. The fewest slots still holding experts was 9,830 (13,130 at short context).
  - Decode of ~200-token answers: 73 tok/s at 174K, 70 at 294K, 62-66 at 379K-420K, 60 at 460K.
- **Near 1M** (`--max-context 1048576`, factor up to 3.125): 0 of 17 codewords wrong up to 802,122 tokens. Past that, this box ran out of VRAM: the K/V could not grow further. A restore of that conversation after a restart was refused, and the engine exited on the next request. That is not looked into, because the box runs a maximum factor of 2.

Not measured: how much YaRN improves quality past 262K over no scaling. PR #2961 reports about 1% lower perplexity past 262K on one corpus. A factor change during decoding (replies here never crossed a step) is also untested.

Output: rounding only. Without the flag nothing changes. With it, short conversations change only by the float64 table and the larger context's expert reservation.
