# gfx906 fork: start here

This fork tunes [Strata](https://github.com/Niko1221/Strata) for one machine: a single AMD Instinct MI50
running one Qwen model for long agent conversations. Everything in it was measured on that machine, and
nothing was kept unless it made that machine faster (see the warning in the [README](../../README.md)).

- [maintenance.md](maintenance.md): how `main` is rebuilt, why it is force-pushed, and how to follow it
- [Releases](https://github.com/hboyd2003/Strata/releases): tagged `v<upstream>+<fork>-gfx906`, with a Docker image each
  ([how they work](maintenance.md#releases))
- [branches/gfx906-ab.md](branches/gfx906-ab.md): the benchmark harness every number here was measured with
  (`bench/gfx906/ab.py`). Run its standard suite on your machine and compare with ours.
- [tested.md](tested.md): what was tried and **not** kept (forks, upstream PRs, switches, tuning), with numbers
- [branches/](branches/): one page per custom branch, written on that branch (what it does, its switches, its A/B);
  the table at the bottom links each one

The rest of `docs/` is upstream's and describes upstream Strata. It is left untouched so catch-ups stay clean.

Numbers below are as of 2026-11-03, unless a line gives its own date.

## The box

| Part | What |
| --- | --- |
| GPU | AMD Instinct MI50 (gfx906, wave64), 32 GiB HBM2 VRAM, 225 W cap |
| GPU link | PCIe 4.0 x16, 16 GT/s (through a riser, zero AER errors) |
| CPU | AMD Ryzen 9 7945HX, 16 cores / 32 threads |
| RAM | 64 GiB DDR5 (2 GiB of it is carved out for the CPU's integrated GPU) |
| Software | built and run in the `mixa3607/rocm-gfx906` ROCm image (the fork's `docker/gfx906/Dockerfile`) |

Things about this card that shaped the work:

- **Decode is latency bound, not bandwidth bound.** About 70% of a decode window is the host waiting for
  the GPU to get through many small kernels (gemv, router, hc, experts). Power sits around 150 W during decode.
  Wins come from fewer, faster small kernels, not from the CPU side.
- **The SDMA engine caps host-to-device copies at ~13 GB/s** on this card, while a kernel reading mapped host
  memory gets 25–27 GB/s. Expert swaps therefore use a kernel copy.
- **gfx906 is wave64.** CUDA "warps" in shared code are half-waves here, and the fork's own wave64 MMVQ
  kernel replaces the kernels many upstream fusions are built on, so those fusions often do nothing.
- **`STRATA_USE_HIP` is not defined in the gfx906 build** (only the generic HIP backend defines it). Code under
  `#if defined(STRATA_USE_HIP)` is compiled out here and its CUDA-side default applies. Several HIP tweaks
  were silently missing for that reason; check a switch's guard before concluding it does nothing.

## What runs in production

**Model:** Qwen3.8-Flash-Next at IQ3_S (experts IQ2_S / IQ3_XXS / IQ3_S gate/up, IQ4_NL down, Q2_0 down in 9 layers), int8 KV cache,
up to 262,144 tokens of context, MTP speculative decoding.

**Engine flags:** `--spec 4 --spec-min-p 0.6`, `--expert-cache auto`, `--pcie-frac 0`, `--prefill auto:32768`,
`--adapt-every 1 --adapt-swaps 80 --adapt-decay 0.97`, `--conversation-cache-min-free-mib 1536`.

**Server config:** `"auto_sessions": true` (conversations saved to disk on a switch), `"qwen_code_tail": true`,
`"gpu_temp_sensor": "junction"`.

**Environment switches** the box sets. Most turn on something that is off by default; the `=0` rows turn off a
default that is on in this build:

| Switch(es) | Branch | Output |
| --- | --- | --- |
| `SIDE_MIXER=2`, `EXP_MODE=8`, `MMVF_ROWS=1`, `ATTN_MERGE_V2=1`, `GFX906_ATTN_QUERY_SWIZZLE=1`, `GFX906_ATTN_REDUCE12=1`, `GR_NORM_FAST=1`, `VERIFY_QDEDUP=1` | upstream (off by default on gfx906) | exact |
| `GFX906_TOPK_SPLIT=1`; `GFX906_ROUTER_T1_ROWS=1`, `GFX906_ROUTER_T2_EXACT=1` (set, but `ROUTER_GEMV=auto` now takes those calls) | `perf/topk-router` | exact |
| `MMVQ_WAVE_ROWS=2` | `gfx906/mmvq-wave-rows` | exact |
| `MMVQ_MUL24=1`, `POST_COPY_WIDE=1` | `perf/gfx906-decode-kernels` | exact |
| `DOWN_LPR=4`, `MMVQ_IACC=1` | `perf/gfx906-decode-kernels` | rounding only |
| `GFX906_ROUTER_W64=1`, `GFX906_ROUTER_GEMV=auto` | `perf/gfx906-router` | exact |
| `GU16=1` / `GR_UP_EXACT_T=1` | `perf/gfx906-gu16` | rounding only / exact |
| `HIP_ADAPT_KERNEL_COPY=1` | upstream switch, built for gfx906 by `perf/gfx906-adapt-copy` | exact |
| `ADAPT_ASYNC=0` (the async adapt tier is armed by default here) | upstream | exact |
| `ROUTE_TAIL_SKIP=0` (upstream's approximate tail skip is on by default here) | upstream | keeps the output exact |
| `ADAPT_COPY_BLOCKS=16` | `perf/gfx906-draft-copy` | exact |
| `PREFILL_CPU_SHARE=auto`, `PREFILL_CPU_SHARE_MAX=8192` / `PREFILL_CPU_TOKENS=16` | upstream / `perf/prefill-cpu-tokens` | rounding only |
| `SELECT_SIMT=1` (prefill only) | upstream | rounding only |
| `GDN_HEAD=1`, `GDN_PP=2`, `GDN_CONVL2=1`, `GDN_NOY=1` (prefill only) | upstream | exact |
| `QSA_ROWS=1` (long-context decode: the ≤4-query QSA score calls) | upstream | rounding only |
| `SESSION_DELTA=1` | `serve/session-delta` | exact |
| `GR_RO=1` | `perf/gfx906-hc-ro` | rounding only |

`DECODE_TIMING=1` is also set: it only logs per-window timings and does not change speed or output.

Built in, with no switch to set: [`perf/gfx906-dpp-shfl`](branches/perf-gfx906-dpp-shfl.md) (exact DPP / `ds_swizzle`
xor shuffles instead of `ds_bpermute`, ~+1.9% decode; bitwise).

Upstream defaults that matter here (since the 2026-11-03 catch-up): `--lookup-max-t 12` (12-row prompt-lookup
windows: +8.5% on replies that copy their context, neutral otherwise, 38 MiB of VRAM), and penalties count only the
reply's tokens (upstream #4590). The model card's presence penalty 1.5 is sent for thinking-off requests, so their
greedy outputs differ from builds before 2026-11-03.

All names take the `STRATA_` prefix. "Exact" means the greedy output hashes did not change; "rounding only"
means the order of floating-point sums changed and nothing else. Nothing that lowers precision below
what the quant already has (for example a 4-bit KV cache) is used.

**Speed** (measured with [`bench/gfx906/ab.py`](branches/gfx906-ab.md)): about 85 tok/s decode on short prompts
(85.9 on 2026-11-03 after the catch-up to upstream of 2026-11-02; 85.8 on 2026-11-02 after `perf/gfx906-dpp-shfl`). On real agent traffic (the last 100 requests on 2026-10-30: context median
144K tokens, p90 185K, ~730 new tokens a turn) decode is 82 tok/s median with 0.74 draft acceptance, and prompt
reading takes ~45% of engine time. Prefill ~780 tok/s at 16K and ~736 tok/s on a 32K document (2026-11-03), ~720
tok/s at 100K (measured 2026-10-30, before the GDN prompt switches). The fork started at ~59 tok/s decode on the same box and model.

## Branches merged into `main`

In merge order. "Source" says where the code came from.

| Branch | Source | What it does |
| --- | --- | --- |
| `rebased/cassettesgoboom` | [cassettesgoboom](https://github.com/cassettesgoboom/Strata) fork | gfx906 build, compat layer, parity tests |
| `rebased/janus-reith` | [janus-reith](https://github.com/janus-reith/Strata) fork | gfx906 prefill port |
| `rebased/pr-3565` | upstream PR #3565 | auto sessions (save and restore conversations to disk) |
| `rebased/pr-3251` | upstream PR #3251 | restore a session without holding its K/V in RAM |
| `rebased/pr-4156` | upstream PR #4156 | HIP stream-priority mapping |
| [`gfx906/local`](branches/gfx906-local.md) | own | gfx906 image fixes (AVX-512 host CPU code, ROCm libomp) and notes on the ported PRs |
| [`gfx906/mmvq-wave-rows`](branches/gfx906-mmvq-wave-rows.md) | own | two rows per wave in the wave64 MMVQ kernel |
| [`serve/gpu-temp-sensor`](branches/serve-gpu-temp-sensor.md) | own | picks the AMD card's temperature probe for the monitor |
| [`perf/topk-router`](branches/perf-topk-router.md) | own + 0FL01 + MarcoSon (PR #3987) | gfx906 1- and 2-token router kernels, two-stage decode top-k |
| [`serve/auto-sessions-monitor`](branches/serve-auto-sessions-monitor.md) | own | an Auto sessions card in the monitor |
| [`serve/request-transcripts`](branches/serve-request-transcripts.md) | own | read a recent request as a chat in the monitor |
| [`perf/pcie-off-graph`](branches/perf-pcie-off-graph.md) | own (idea from ramanujbhattacharjee93) | `--pcie-frac 0` drops the PCIe branch from the verify graph |
| [`fix/session-restore-real-peak`](branches/fix-session-restore-real-peak.md) | own | session restores price the metadata the file declares, not the worst case |
| [`serve/qwen-code-tail`](branches/serve-qwen-code-tail.md) | own | keeps Qwen Code's per-turn reminder out of the prompt cache checkpoint |
| [`perf/prefill-cpu-tokens`](branches/perf-prefill-cpu-tokens.md) | own (from baptisterajaut) | CPU share for short prefill chunks |
| `rebased/pr-4411` | upstream PR #4411 | pipelined session-file saves |
| [`perf/gfx906-decode-kernels`](branches/perf-gfx906-decode-kernels.md) | own | gfx906 decode kernel switches |
| [`perf/gfx906-router`](branches/perf-gfx906-router.md) | own (partly from VlDubr) | 64-lane router top-10 and router GEMV |
| [`perf/gfx906-gu16`](branches/perf-gfx906-gu16.md) | own (partly from VlDubr) | 16-lane expert gate/up |
| [`perf/gfx906-adapt-copy`](branches/perf-gfx906-adapt-copy.md) | own | kernel copies for adaptive expert swaps |
| [`perf/gfx906-draft-copy`](branches/perf-gfx906-draft-copy.md) | own (partly from VlDubr) | smaller swap-copy grid so copies stop crowding the MTP draft |
| [`serve/session-delta`](branches/serve-session-delta.md) | own | delta session files: append a turn's new K/V instead of rewriting the file |
| [`fix/rebased-review`](branches/fix-rebased-review.md) | own | review fixes for the forks' and PRs' code (auto sessions, session restore, correctness trace) |
| [`perf/gfx906-hc-ro`](branches/perf-gfx906-hc-ro.md) | own | rounding-only hyper-connection read (split-K `down`, norm folded in), ~+4% decode |
| [`perf/gfx906-dpp-shfl`](branches/perf-gfx906-dpp-shfl.md) | own | exact DPP / `ds_swizzle` xor shuffles in the HIP layer instead of `ds_bpermute`, ~+1.9% decode |
| [`gfx906/ab`](branches/gfx906-ab.md) | own | the benchmark harness, `bench/gfx906/ab.py` |
| [`gfx906/release`](branches/gfx906-release.md) | own | fork release tags, the image push and the Release workflow |
| `docs` | own | these docs and the README banner |

Not merged: `gfx906/dsv41` and `build/dsv41-serve` (DeepSeek V4.1 on gfx906; parked).
