# What was tried and not kept

Everything here was built and measured on the box described in the [start page](README.md) (one MI50, one
Qwen model, one request slot, MTP speculative decoding). A "no" here means "no on this box"; several of these
are real wins on other GPUs, on several GPUs, or with batching.

What *was* kept is documented on its own branch (`docs/gfx906/branches/<branch>.md`).

Every result was measured with the fork's A/B harness ([`bench/gfx906/ab.py`](branches/gfx906-ab.md)), each
variant against the build running at the time with a repeat of the base as the noise control. Kernel numbers (µs)
come from the parity tests' `--bench` modes.

**Legend.** Percentages are decode tok/s against the build running at the time, unless a row says otherwise.
The base moved from ~59 to ~83 tok/s over the period, so compare within a row.

- **N** neutral: inside the noise (about ±1%; each A/B repeats the base as a control)
- **X** no effect on gfx906: the code declines, is compiled out, or never runs on this path
- **H** harmful: measurably slower, or slower somewhere else (prefill, long context)
- **B** broken: wrong output or errors
- **exact**: the greedy output hashes did not change. **rounding**: only the order of floating-point sums changed.

## Forks and upstream PRs

| Source | What | Result | Why it was not taken |
| --- | --- | --- | --- |
| ryan4yin `develop` | `Q6K_V2`, `FUSE_PROJ`, `FUSE_ROUTEBELL`, `FUSE_FETCH` + `COMBINE` + `SHGATE` | N (±0.5%) | No gain. The merged build also changed output with every switch off (not bisected). |
| ryan4yin `develop` | `FUSE_SHPAIR`, `DPP_XOR`, `FUSE_QGEMV`, `FUSE_SHBEL`, `FUSE_SWIGLU_GU`, `EXPDEC`, `EXPDEC_V2` | X | They pair kernels gfx906 replaces with its wave64 MMVQ, need gfx10+/gfx12, or need `--batch`. |
| PR #3455 | 16-byte LDS staging in the expert gate/up loops (`NIB_GU_PROTO`) | H, −9..−14%, exact | VGPRs 77 → 116 drops occupancy from 3 to 2 waves/SIMD. With launch bounds it spills and is neutral. |
| PR #3455 | 4-bit slot transcode | X | CUDA only, and needs every expert resident. |
| PR #3820 | `STRATA_SELECT_LONG` | X | CUDA sm_60 only. |
| PR #3073 | `STRATA_SELECT_SIMT=2` | N | Bitwise equal to `SELECT_SIMT=1` (already upstream and in use), no faster end to end. |
| PR #4567 | `STRATA_QSA_ATTN_VARIANT=2` (32-cell attention chunks) | H, rounding | Attention −26% at short context but +1% overall (noise); prefill −1.3..−2.2%; at 80K context the drafter gets 9% slower. |
| PR #4158 | `PF_MMVQ` (MMVQ for short prefill projections) | N | It engages, but ~200-token prefill is flat, likely because our dense prompt projections already use int8 MMQ (not measured). |
| PR #4039 | `PF_HC_REGION` / `PF_GROUP_SCRATCH` | N, exact / B | The scratch variant gives non-finite logits after every prefill on this fork. |
| PR #4226 | disk conversation cache (spill dir, deltas, codec) | not taken | Still prices restores at the worst case, so it doesn't fix restores failing for RAM; 10.8K lines, CUDA-tested only, conflicts with #3565/#3251. Replaced by own delta session files. |
| PR #4258 | `--drop-file-cache` | X | Experts come from the resident arena, not mapped files, so nothing is dropped. |
| PR #4283 | `STRATA_PREFILL_ROUTED_STREAM=1` | X | Prefill here is compute bound: copies are done before the router needs them (wait copy ≤0.4% of prefill). |
| PR #4374 | `"system_reminders": "tail"` | H | Strata resumes only from a checkpoint, so moving a reminder inside the history forces the re-read it is meant to save. |
| sakuumomo | session file v3 append | H | As shipped it never appends (page-rounding gate). Patched: 1.2–2.5 s per save instead of 0.25, files grow to 1.5 GB after 4 turns, and restores of them get refused for RAM. Replaced by own delta session files. |
| svgop | `STRATA_DM=1` (fused wait + copy + combine) | N, +0.5% | Exact only on the host-plan route. |
| ramanujbhattacharjee93 | `STRATA_FZ_NOPCIE=1` | +3.8%, rounding | Taken as the idea for `perf/pcie-off-graph` (`--pcie-frac 0`), which gets the same output. |
| ramanujbhattacharjee93 | `FZ_ROUTE` / `FZ_PAIR` | N / X | `FZ_PAIR` only handles Q8_0; our shared experts are K-quants. |
| 0FL01 | HC BF16 clone / `Q5_HEAD_R2`, `Q8_OUT_R2` | X / N | The clone never fires; the two head kernels are covered by `MMVQ_WAVE_ROWS`. (0FL01's router rows were taken.) |
| dys0re | `--mtp-adaptive` | H, −0.8% | Needs `--suffix-draft 0` and min-p 0, which alone cost −1.4%. |
| xxDoman | DFlash2 drafting | X | 0 of 126 drafts accepted on gfx906. |

## Engine switches

Upstream switches and my own experiments that are in the code but left off.

| Switch | Result | Note |
| --- | --- | --- |
| `ADAPT_NOWAIT` + `ADAPT_LAG=2` | N | The wait moves into the draft step. |
| `FUSE_WAIT=2`, `HIP_ROUTER_FAST`, `GR_UP75`, `FUSE_CONVAB` + `FUSE_HITS`, `PLE_BATCH` | N | |
| `LFUSE` | X | Never engaged: it only runs when every expert is resident in VRAM, which never holds on this box. |
| `GDN_SPLIT=1`, `TSUM=1`, LDS-only GDN barriers + PR #3225 | N, exact | |
| `SH_STREAM=0` (the generic HIP default) | H, −0.8..−1.7% | The shared-expert stream helps on gfx906, unlike gfx1201/gfx1030. Left on. |
| `MTP_SHARED_BRANCH=0`, `ONE_TOKEN_COMMIT=0`, `HEAD_MIX_MULTI=0` | N, exact | The CUDA-side defaults are right here. |
| `MTP_HIP_PRIO=1`, `COPY_HIP_PRIO=1` (priority streams) | N | Queue priority does not arbitrate compute units on gfx906; they only move the wait around. |
| `VERIFY_DEVICE_PLAN=1` | H, −1.5% | Changes output. |
| `QFUSE=1` | B | Upstream bug: one-token windows lose their GDN state update (accept 0.71 → 0.645, −8..−9%). With `ONE_TOKEN_COMMIT=0` it is correct but neutral. |
| `SPEC_COST=1` | H, −7.3% | |
| `HC_REQ8=1` | H, −7.1% | The int8 path falls back to slower kernels; changes output. |
| `ADAPT_OVERLAP=1` | N, exact | Adapt wait −1.1 ms, verify +1.05 ms: the swap copies just move into the window. |
| `--adapt-async 1` + `ADAPT_ASYNC_WAIT=0` | H, −3.4% | Copies in flight slow verify's latency-bound PCIe round trips. |
| `ADAPT_LAG=2` | H, −0.8%, exact | |
| `ADAPT_ASYNC_BATCH=4` (with the async tier) | H, −5.9% | Swap rounds drag over many windows; hit rate 94 → 89%. |
| `GR_DOWN_TG` (enabled on HIP) | H, −1.1%, exact | The hc down stage gets ~10% slower, opposite to its kernel bench. |
| `--mtp-q4 head` | H, −0.6% | The Q4_0 draft head is faster per step but loses acceptance. |
| `DOORBELL_STORE`, `MMVQ_Q6K_R4`, `NIB_SHIFT` (own) | N | `Q6K_R4` and `NIB_SHIFT` benched 5–11% faster; it didn't carry end to end. Still on `perf/gfx906-decode-kernels`, off. |
| `GFX906_ROUTER_FUSED` (own) | N (kernel bench only, not A/B'd) | One launch instead of three; the top-10 is no longer slow, so nothing to win. |
| `ADAPT_COPY_BATCH` (own) | N | One launch per swap round ≈ one per swap. |
| 4-rows-per-warp expert down kernel (own) | H, −8..−55%, exact | |
| `HC_PERSIST` / `GR_SPLIT=1` | X / N (bench only) | `HC_PERSIST` is off on wave64; `GR_SPLIT` changes output and benched about even. |
| `PREFILL_STREAM_MIN=3072` (staging short chunks without the CPU share) | H | 2K follow-up turns 3.4% slower: the CPU share's gain is the CPU work, not the staging. |
| `HSA_ENABLE_SDMA=0` | N decode, H prefill (4K −6.8%) | |
| `GPU_MAX_HW_QUEUES=8` / `=2` | H, −23% / −4% | |
| `STRATA_DMA_BATCH` | X | CUDA 13+ only. |
| `-mllvm -amdgpu-sched-strategy=max-ilp` (compiler flag) | H, −6.6% | |

## Tuning

| Setting | Tried | Result |
| --- | --- | --- |
| `--spec` / `--spec-min-p` | 2, 3, 4, 5 × min-p 0.4–0.8 | **4 / 0.6 used** (+2.6% over 3 / 0.5); 4 / 0.7 was the steadiest runner-up (+1.2..+1.8%). Spec 5 −1.4..−3%, spec 2 −4.7%. |
| `--mtp-window` | 8192, 4096, 2048 (default 32768) | H at long context (acceptance 0.69 → 0.61, −4.7%), N short; changes output. |
| `--adapt-swaps` / `--adapt-every` | swaps 0–160, every 1/2/4 | **80 / 1 used.** Swaps 0 −22%, 8 −2.3%, every 2 −17%. |
| `--expert-cache` pinned at the old size | 11635 slots vs auto | +0.8% (noise) for 480 MiB less VRAM headroom. Not taken. |
| `--lookup-max-t 16` | | H, −1.6% on normal prompts. |
| `--prefill 4096` | vs auto | H, −18% prefill. |
| `STRATA_PREFILL_CPU_SHARE_MAX` / `CPU_TOKENS` | 4096 / 6144 / 8192, tokens 8 / 16 / 32 | **8192 / 16 used.** 6144 won as much on 4K prompts (−5.2%); 8192 covers more chunk sizes. 4096 and 32 neutral. |
| English-only draft vocabulary | 40.5K-token draft head instead of 106K | +1.8%, but every output changes and non-English text could lose acceptance. On hold. |

## Things learned along the way

- **Window cost grows with the number of tokens verified:** about 12.4 + 6.9 ms per token at the time. Each
  token's top-10 experts barely overlap, so a wider speculative window pays only once the per-token cost drops.
- **Draft-side changes are not output-neutral.** A different drafter gives different verify windows, and verify
  isn't batch-invariant across window sizes, so greedy output changes too. Check hashes on draft changes.
- **The third harness prompt's hash flips between runs** of the same build (the adaptive expert cache differs).
  Only the first pass of the first two prompts is stable enough to compare.
- **`rocprofv3 --kernel-trace` deadlocks the engine** (it holds completion signals while the verify graph spins on
  host doorbells). `--memory-copy-trace` alone is fine; time GPU work with events in the code instead.
- **A kernel bench is a hint, not a result.** Several kernels that benched 5–10% faster were neutral or slower
  end to end, and one (`ROUTER_W64`) did three times better than its bench predicted.
