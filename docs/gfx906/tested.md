# What was tried and not kept

Everything here was built and measured on the box described in the [start page](README.md) (one MI50, one
Qwen model, one request slot, MTP speculative decoding). A "no" here means "no on this box"; several of these
are real wins on other GPUs, on several GPUs, or with batching.

What *was* kept is documented on its own branch (`docs/gfx906/branches/<branch>.md`).

Every result was measured with the fork's A/B harness ([`bench/gfx906/ab.py`](branches/gfx906-ab.md), or before
2026-10-30 the in-house script it was cut from), each
variant against the build running at the time with a repeat of the base as the noise control. Kernel numbers (µs)
come from the parity tests' `--bench` modes.

**Legend.** Percentages are decode tok/s against the build running at the time, unless a row says otherwise.
Rows also name the sub-metrics that improved beyond the control arms' spread (stage times, the draft step, waits,
acceptance, tokens per window) even when tok/s didn't move; a row without them had none, or predates sub-metric
recording (before 2026-10-25). Host planning time, commit time and small adapt-wait changes swing as much between
identical arms, so they aren't quoted.
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
| PR #3073 | `STRATA_SELECT_SIMT=2` | N | Bitwise equal to `SELECT_SIMT=1` (already upstream and in use), no faster end to end. Improved: the KV-resolve stage −33%, a 0.01 ms stage. |
| PR #4567 | `STRATA_QSA_ATTN_VARIANT=2` (32-cell attention chunks) | H, rounding | Attention −26% at short context but +1% overall (noise); prefill −1.3..−2.2%; at 80K context the drafter gets 9% slower. |
| PR #4158 | `PF_MMVQ` (MMVQ for short prefill projections) | N | It engages, but ~200-token prefill is flat, likely because our dense prompt projections already use int8 MMQ (not measured). |
| PR #4039 | `PF_HC_REGION` / `PF_GROUP_SCRATCH` | N, exact / B | The scratch variant gives non-finite logits after every prefill on this fork. Re-tested 2026-11-03 on upstream's 10-31 rewrite of both: `PF_HC_REGION` N again (every prefill size inside the base's spread); `PF_GROUP_SCRATCH` still gives non-finite logits (on a 256-token chunk). |
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
| PR #4928 (lyesrock) | `STRATA_LOOKUP=1` (prompt-lookup windows up to 16 rows while a reply copies its context), built with `-DSTRATA_WINDOW_MAX_T=16` | N | Long windows almost never form: the 6-row suffix-lookup windows already get ~99% of their drafts accepted on copy prompts. Where windows did grow, each row past 8 cost ~12 ms against ~6.9 ms per row below that. The wide build changed the output hashes even with the knob off (not tracked down). |
| rajathvsm | `STRATA_PREWARM_GRAPHS=1` (capture the decode graphs before the server reports ready) | N, rounding | Saves ~150 ms on the first request after a restart and nothing after it. It didn't make A/B restarts steadier either: that noise comes from the adaptive expert cache and from the first arm running on a cold card. |
| bitaccesscomau | refills of lent expert slots on their own stream | X (not built) | It only changes the one-shot command-line prompt path; `--serve` refills slots through its own code. |
| samnolak | `PF_ROUTE_EVENT=1` (hand port) | N, exact | +0.3% at 16K prefill, the rest flat: the host already waited only on the router. |
| mgendre | `POOL_ONE_PHASE=1` | N, exact | Host CPU-expert time −5% in decode, hidden behind the wait for the GPU. The short decode right after a prompt may get slower. |
| bsorensen110 | `MOE_BUCKET_PLAN=1` | N, exact | Our windows' top-10 groups rarely hold more than one entry, so there is little to bucket; 4K prefill 1.7% slower. Improved: draft step −5.4%. |
| PR #4856 | `STRATA_SPEC_PCUM` (stop drafting once the drafts' cumulative probability falls below a bar; 0.3–0.5 and a spec-5 combination) | N | The drafter does stop earlier (draft time −2..−3%), but tokens per window fall with it; every setting landed inside the control's own −2.1% swing. Improved: acceptance +8.8% at 0.5. |
| bitaccesscomau | `QSA_ROWS56=1` (upstream's rows score kernel extended to 5–6 queries) | N, rounding | Live 80K ms/window −1.6% against −1.7% for `QSA_ROWS=1` alone (which is used): the 5–6-query windows are a minority and gain only ~13% in the kernel. |
| 3chet | `--aux-cpus auto` | N, exact | Window time flat (−0.1%). This box has one CPU socket, so the host thread doesn't share a core with driver threads the way it did on theirs. Improved: CPU experts per layer-window −12.7%, host CPU-expert time −11.9%, host per-layer −7.4%; the GPU wait absorbed it. |
| serboox | IQ3_S gate/up CPU rows with AVX-512 VBMI (`IQ_VBMI=1`) | N, rounding | 2.3× faster on one thread, but the 16-thread CPU pool already sits at the ~55 GB/s DRAM ceiling for the 1–4-token rows decode runs, so decode and prefill stay flat. |
| phoenixclyde | a per-shape MMVQ rows table (`MMVQ_ROWS_TABLE=1`, their method on our kernels) | N, exact | Out-proj −5.7% and shared-expert quant −2.9% per stage, but window time flat in the speed pass and no gain on the 80K live pass. |

## Engine switches

Upstream switches and my own experiments that are in the code but left off.

| Switch | Result | Note |
| --- | --- | --- |
| `ADAPT_NOWAIT` + `ADAPT_LAG=2` | N | The wait moves into the draft step. |
| `FUSE_WAIT=2`, `HIP_ROUTER_FAST`, `GR_UP75`, `FUSE_CONVAB` + `FUSE_HITS`, `PLE_BATCH` | N | |
| `LFUSE` | X | Never engaged: it only runs when every expert is resident in VRAM, which never holds on this box. |
| `GDN_SPLIT=1`, `TSUM=1`, LDS-only GDN barriers + PR #3225 | N, exact | |
| `SH_STREAM=0` (the generic HIP default) | H, −0.8..−1.7% | The shared-expert stream helps on gfx906, unlike gfx1201/gfx1030. Left on. With `MTP_SHARED_BRANCH=0` the draft step is 35% faster, but decode is slower overall. |
| `MTP_SHARED_BRANCH=0`, `ONE_TOKEN_COMMIT=0`, `HEAD_MIX_MULTI=0` | N, exact | The CUDA-side defaults are right here. Improved: the draft step −35% (`MTP_SHARED_BRANCH=0`, `HEAD_MIX_MULTI=0`) and −20% (`ONE_TOKEN_COMMIT=0`, plus GDN recurrence −7.6%), with decode tok/s flat. |
| `MTP_HIP_PRIO=1`, `COPY_HIP_PRIO=1` (priority streams) | N | Queue priority does not arbitrate compute units on gfx906; they only move the wait around. Measured: adapt wait −16% (−45% with a 32-block copy grid); the rest of the window absorbed it. |
| `VERIFY_DEVICE_PLAN=1` | H, −1.5% | Changes output. |
| `QFUSE=1` | B | Upstream bug: one-token windows lose their GDN state update (accept 0.71 → 0.645, −8..−9%). With `ONE_TOKEN_COMMIT=0` it is correct but neutral. |
| `SPEC_COST=1` | H, −7.3% | |
| `HC_REQ8=1` | H, −7.1% | The int8 path falls back to slower kernels; changes output. |
| `ADAPT_OVERLAP=1` | N, exact | Adapt wait −1.1 ms, verify +1.05 ms: the swap copies just move into the window. |
| `--adapt-async 1` + `ADAPT_ASYNC_WAIT=0` | H, −3.4% | Copies in flight slow verify's latency-bound PCIe round trips. |
| `ADAPT_LAG=2` | H, −0.8%, exact | |
| `ADAPT_ASYNC_BATCH=4` (with the async tier) | H, −5.9% | Swap rounds drag over many windows; hit rate 94 → 89%. Improved: adapt wait −86%, swaps per window −77%, GPU-reach wait −9%; the lost hits cost more. |
| `GR_DOWN_TG` (enabled on HIP) | H, −1.1%, exact | The hc down stage gets ~10% slower, opposite to its kernel bench. |
| `--mtp-q4 head` | H, −0.6% | The Q4_0 draft head is faster per step but loses acceptance. Improved: draft step −8.9%. |
| `DOORBELL_STORE`, `MMVQ_Q6K_R4`, `NIB_SHIFT` (own) | N | `Q6K_R4` and `NIB_SHIFT` benched 5–11% faster; it didn't carry end to end. Still on `perf/gfx906-decode-kernels`, off. `DOORBELL_STORE` also cut the GDN CPU wait 20%. |
| `GFX906_ROUTER_FUSED` (own) | N (kernel bench only, not A/B'd) | One launch instead of three; the top-10 is no longer slow, so nothing to win. |
| `ADAPT_COPY_BATCH` (own) | N | One launch per swap round ≈ one per swap. |
| 4-rows-per-warp expert down kernel (own) | H, −8..−55%, exact | |
| `HC_PERSIST` / `GR_SPLIT=1` | X / N (bench only) | `HC_PERSIST` is off on wave64; `GR_SPLIT` changes output and benched about even. |
| `PREFILL_STREAM_MIN=3072` (staging short chunks without the CPU share) | H | 2K follow-up turns 3.4% slower: the CPU share's gain is the CPU work, not the staging. |
| `HSA_ENABLE_SDMA=0` | N decode, H prefill (4K −6.8%) | Improved: adapt wait −89% (no SDMA queue to wait on). |
| `GPU_MAX_HW_QUEUES=8` / `=2` | H, −23% / −4% | |
| `STRATA_DMA_BATCH` | X | CUDA 13+ only. |
| `-mllvm -amdgpu-sched-strategy=max-ilp` (compiler flag) | H, −6.6% | |
| `MMVQ_IACC_OCC` (own: a waves-per-SIMD cap on the Q6_K head and projection gemv) | N, exact | Benched −5% at 3-token windows only, ~0 end to end. Raising occupancy spills on gfx906, and spilling is very slow here. Improved: hc norm stages −4..−11%, QSA scores −3.4%. |
| `SMALL_UNR=1` / `2` / `3` (upstream constant-trip router and GDN a/b reads) | X, exact | The GDN a/b stage is already at the 0.05 ms timer floor, and the router read on this fork goes through its own `router_gemv_906`, which the switch doesn't reach. Improved: shared-expert quant −4% (`=1`, `=3`) and draft step −6.8% (`=3`, one run). |
| `PF_SMALL=0` (upstream's small-card prompt rule, on by default) | X | The rule only applies to automatic prompt chunks under 8,192 tokens; this box's are 32,768. |
| `MTP_MULTI_COPY=0` (upstream's combined draft-copy kernel is the default) | N, exact | The old one-launch-a-copy path is no faster. Improved: several GPU stages −3% (hc0 down, qkv gemv, router) and swaps per window −12%; window time flat. |
| `QSA_FUSE_NR=0` (upstream's norm + rope fusion in the QSA prompt layers, on by default) | N | Same output on the decode prompts, prefill unchanged: the default stays. |
| `LOOKUP_MAX_T=0` (turns off upstream's 12-row lookup window default) | H on copy-heavy replies | −8.5% when a reply copies its context (a file copy −12%, an edit −6%, a re-copy −7%), neutral on normal prompts. The default is kept; it costs 38 MiB of VRAM. Rounding only. |
| `PENALTY_PROMPT=1` (upstream #4590's off switch) | not kept | Brings back the old penalty window: the last 64 tokens, which at the start of a reply are the prompt's. With the model card's presence penalty 1.5 (sent for thinking-off requests) that penalised the words the prompt asked for. Upstream now counts only the reply's tokens, as vLLM and SGLang do for presence and frequency penalties. It changed greedy outputs; speed is unchanged. |
| `MMVQ_IACC_PF` / `MMVQ_IACC_V2` (own: load prefetch / leaner decode of the same kernel) | H / X, exact | Prefetch doubles the VGPRs and halves occupancy (+35..+45% kernel time). V2 cut the decode ALU work and changed nothing at 1 token: that kernel is limited by memory latency, not ALU. |
| `MMQ_KQ=0` (turns off upstream's `MMQ_KQ`, the default on gfx906) | H | Prefill −24% at 200 tokens, −10.5% at 33K, follow-up turns −5.6%. The default stays. |
| `PREFILL_RING=192` / `768` (prompt staging depth) | N, exact | The ring isn't the limit. |
| `NO_ARENA_THP=1` (no transparent huge pages for the expert arena) | N | Huge pages on the 47 GiB arena are worth nothing measurable here. |
| `PF_W16_CACHE_MIB=N` (an FP16 weight cache for prompts) | X | No VRAM room above the 2 GiB reserve at this box's expert-cache size; with ~1,500 fewer slots it engaged and gained nothing. |
| `PARK_ASYNC=1` | X | Compiled out on gfx906 (`park_async_on()` is false there), as are `CKPT_ASYNC` and `CKPT_GATHER`. |
| `--lookup-chain 2` / `4` / `8` | N | Chained drafts are accepted on copying replies (268 of 276), but suffix lookup already covers them: −0.4..−1.3%. |
| `LM_HIP=1` + `PREFILL_LAYER_MAJOR=1` (layer-major prompt walk on one HIP GPU) | N, exact | Prefill within ±0.4% at 16K and 34K. |
| `IQ_STAGE_GRID_MMVQ=1` (own: LDS-staged codebook in the single-matrix IQ gemv) | N, exact | IQ3_XXS kernel −53% in its bench, but that kernel isn't on this pack's decode path: end to end flat, same hashes. |
| `ROUTE_TAIL_SKIP=7` (skip low-weight tail experts) | N, lowers precision | Inside the noise, and it drops experts without renormalising, so it is out by rule. The box sets `ROUTE_TAIL_SKIP=0` (see the start page). |
| `ADAPT_CU_MASK` (own: the expert-refill stream on 4 or 8 CUs only) | N, exact | The adapt wait drops 0.23 ms and the draft step grows 0.30 ms: the wait just moves. |
| `GR_UP_W2` (own: hc `up` at 2 waves per SIMD) | H, exact | It spills at every token count. |
| `OLD_Q80_SCALED=1`, `IL_PAD=0`, `AMD_COMMUNITY_TUNING=0`, `QSA_FUSE_GATE=0`, `PA_RPF=0`, `DRAFT_KEEP=0`, `MTP_KEEP=1` (off switches for upstream's 2026-11 defaults) | N | Each alone was speed-neutral; the defaults stay. |
| `MULTI_BLOCK_ARGMAX=1` (own port: multi-block draft argmax and top-prob kernels, compiled out on gfx906 before) | on hold, bitwise | ms/window −0.9%, verify head stage −7.6%, drafter argmax step 0.079 → 0.031 ms; decode tok/s inside the noise. |
| `--kv-grow` (own gfx906 port) + `KV_GROW_HOLD` | on hold, rounding / N | +1,629 expert slots and short-decode ms/window −1.8%, long context neutral, clean across restores and 6 interleaved conversations. It needs a flush after each remap: the amdgpu driver leaves a stale TLB translation when all VRAM is BAR-visible ([ROCm/TheRock#8744](https://github.com/ROCm/TheRock/issues/8744)). |

## Tuning

| Setting | Tried | Result |
| --- | --- | --- |
| `--spec` / `--spec-min-p` | 2, 3, 4, 5 × min-p 0.4–0.8 | **4 / 0.6 used** (+2.6% over 3 / 0.5); 4 / 0.7 was the steadiest runner-up (+1.2..+1.8%). Spec 5 −1.4..−3%, spec 2 −4.7%. Sub-metrics: min-p 0.7–0.8 raises acceptance 10–14% and cuts the draft step up to 21%, spec 4 over 3 gives +13% tokens per window; each costs window time. |
| `--mtp-window` | 8192, 4096, 2048 (default 32768) | H at long context (acceptance 0.69 → 0.61, −4.7%), N short; changes output. |
| `--adapt-swaps` / `--adapt-every` | swaps 0–160, every 1/2/4 | **80 / 1 used.** Swaps 0 −22%, 8 −2.3%, every 2 −17%. Re-tested 2026-11-01: 40 and 60 are neutral (the swap wait drops 33–47%, but the time moves to CPU experts and the draft). Swaps 8–16 cut the swap wait 76–93% and GPU-reach wait 3–6%, but hits drop. |
| `--adapt-decay` | 0.95, 0.99 | **0.97 used.** Both neutral. Both cut the swap wait 16–30% and swaps per window 8–12%. |
| `--expert-cache` pinned at the old size | 11635 slots vs auto | +0.8% (noise) for 480 MiB less VRAM headroom. Not taken. |
| `--lookup-max-t 16` | | H, −1.6% on normal prompts. 12, upstream's default since 2026-11, is kept: see `LOOKUP_MAX_T` above. |
| `--prefill 4096` | vs auto | H, −18% prefill. |
| `STRATA_PREFILL_CPU_SHARE_MAX` / `CPU_TOKENS` | 4096 / 6144 / 8192, tokens 8 / 16 / 32 | **8192 / 16 used.** 6144 won as much on 4K prompts (−5.2%); 8192 covers more chunk sizes. 4096 and 32 neutral. |
| English-only draft vocabulary | 40.5K-token draft head instead of 106K | +1.8% greedy, but every output changes and non-English text could lose acceptance; at temperature 1 it loses 0.9–1.9% (draft step +44%). On hold. At temperature 1 acceptance rises 5.9% and tokens per window 3–12%. |
| `--spec` / `--spec-min-p` at temperature 1 + thinking | spec 3, 5, 6; min-p 0.5, 0.7 | All inside the noise; **4 / 0.6 stays**. Sampled requests run ~9% slower than greedy because windows get narrower (tokens per window 2.30 → ~1.84), not because acceptance drops. Sub-metrics: min-p 0.7 acceptance +11%, spec 3 draft step −13..−30%, spec 6 tokens per window +7..+10%; window time moves with each. |
| Coupled / Gumbel draft sampling (`SPEC_COUPLED`, `SPEC_GUMBEL`) at temperature 1 | alone, with min-p 0.3 / 0.4, with spec 6 | Gumbel raises acceptance slightly with tok/s flat; coupled −3.5%, Gumbel + spec 6 −2.6%. Neither recovers the temperature-1 loss. Gumbel acceptance +8.6%; Gumbel + min-p 0.3 tokens per window +7..+13%. |
| `--expert-cache 7650` | 9,996 slots instead of 11,501 | H, −2.2% decode, −4.7% at long context: the cost of ~1,500 fewer slots. |
| rocBLAS tables | production auto-tune, the earlier hand-made table, two more tunes | Same speed (722–728 tok/s at 4K) and power. The tuner's picks differ run to run (94 of 120 agree) without mattering. |
| GPU clock pin `power_dpm_force_performance_level=high` | vs auto | X: under load the clock already sits at ~1,688 MHz either way. |
| HBM clock via `pp_table` | 1,050 and 1,180 MHz (stock 1,000) | 1,050: decode about +0.9% at most, prefill flat, not adopted. **1,180: an uncorrectable HBM ECC error and a failed GPU reset within 31 s, at idle.** Don't. |
| Unsloth UD-IQ3_XXS pack | instead of the IQ3_S pack | H, −2.3% decode (lower acceptance), −8% at 200-token prompts. Quality not measured. |

## Things learned along the way

- **Window cost grows with the number of tokens verified:** about 12.4 + 6.9 ms per token at the time. Each
  token's top-10 experts barely overlap, so a wider speculative window pays only once the per-token cost drops.
- **Draft-side changes are not output-neutral.** A different drafter gives different verify windows, and verify
  isn't batch-invariant across window sizes, so greedy output changes too. Check hashes on draft changes.
- **Two fresh starts of the same build drift apart.** They usually agree on the first decode prompt and then
  give other answers (the adaptive expert cache places experts on the GPU or the CPU by timing, and the two round
  differently). That moves draft acceptance, so decode tok/s varies ~0.4–1% between identical runs; ms per window
  and the stage times are steadier, so small exact changes are judged on those. Harness version 3 (2026-11-03) moved
  the long-context pass after every decode rep, which made the long-context numbers steady (±0.5%, was ±2.6%) but
  not the decode ones.
- **The third harness prompt's hash flips between runs** of the same build (the adaptive expert cache differs).
  Only the first pass of the first two prompts is stable enough to compare.
- **`rocprofv3 --kernel-trace` deadlocks the engine** (it holds completion signals while the verify graph spins on
  host doorbells). `--memory-copy-trace` alone is fine; time GPU work with events in the code instead.
- **A kernel bench is a hint, not a result.** Several kernels that benched 5–10% faster were neutral or slower
  end to end, and one (`ROUTER_W64`) did three times better than its bench predicted.
- **The gfx906 build holds verify windows of at most 8 rows** (`STRATA_WINDOW_MAX_T=auto`): it takes CMake's CUDA
  path, and the 16-row default belongs to the wave32 HIP backend. A feature that needs wider windows needs
  `-DSTRATA_WINDOW_MAX_T=16`, and the parity tests don't cover rows 9–16.
- **`STRATA_PREFILL_TIMING=1` changes prefill power.** Its timing events make the GPU draw less while reading a
  prompt, so power read with it doesn't hold without it: the rocBLAS table looked like −13% prefill power with the
  switch and about +20–25% without (about 155 → 190 W, read across separate runs, not isolated in one A/B).
  Compare power only between runs without it.
- **The instruction cache isn't flushed per dispatch.** On a raw AQL queue, AGENT and NONE fence scopes cost the
  same, and eight kernels in a row cost what one does; SYSTEM scope behaves as if it flushes. Overflowing the cache
  costs ~0.29 ns per byte of kernel code. Real kernels after a thrash run 0.3–1 µs slower (the hc read kernels the
  most), so a perfectly warm cache would be worth 0.5–2 ms per window at most.
- **The portable image is as fast as the native one on this box** (decode +0.8%, inside the noise; prefill and long
  context within ±2%).
