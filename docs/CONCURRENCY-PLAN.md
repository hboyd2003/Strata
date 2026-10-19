# Implementation plan: vLLM-style concurrency and checkpoint-tree KV in Strata

This is the build plan for the phases in [CONCURRENCY-RADIX.md](CONCURRENCY-RADIX.md). It changes
no code. Each phase below is opt-in; the default path stays byte-identical to the last release
(AGENTS.md), every file a phase touches gets built for CUDA, HIP (incl. this fork's gfx906
wave64 target), and SYCL where applicable before review, and every phase names the test that must
witness it plus the numbers it must produce. PRs are split one-change-each, per AGENTS.md.

Working tree for all of it: this checkout (gfx906 fork: `/ai/engines/gfx906/worktrees/xolotl-merge`,
branch `gfx906/xolotl-merge`, gfx906 build dir `build-906`, patched ggml at
`/ai/engines/gfx906/llama.cpp-xolotl` carrying `tools/gfx906/ggml/hip-cmake.patch`).
Host RAM is 125 GiB with no swap: every build and boot on this box runs behind the standing
`MemAvailable < 2 GiB -> kill` watchdog, and boots that need tens of GiB wait for the incumbent
workload to release memory.

## Order and why

```
P0 (measure what ships)  -->  G8 decision (policy, ~free)
        |                              |
        +----> P1 (window rows + parity tightening) <----+
        |                                                |
        +----> P3 (batched recurrent-state kernel) ------+  (P1's extended parity harness feeds P3)
        |
        +----> P2 (arena + checkpoint-tree radix)   [only if P0's census/reuse numbers justify]
        |
        +----> P4 (ported engines' per-session caches) [last; GLM/DS4.1 first boots ride T0-T5]
```

P0 is cheap, needs almost no code, and gates the expensive phases: it locates this box between
`docs/BATCHING.md`'s two regimes (all-experts-in-VRAM, where slots multiply throughput: 4 GPUs,
1/2/4/8 concurrent -> 120/113/205/360 tok/s, engine PR #559) and the 12 GB CPU-expert-card side
(where slots buy latency but cost up to -24% aggregate). The route-overlap census from P0 decides
whether P2 earns its keep; the per-step host-loop timing decides whether P3 is worth building at
all.

## Ground rules (apply to every PR)

- Opt-in only; default path byte-identical. A new flag defaults to today's behavior.
- Parity before performance: correctness claims ride exact-token (and exact-recurrent-state where
  applicable) comparison against solo behavior with `--pcie-frac 0`, `--adapt-every 1000000`,
  `--no-prefill-borrow` (BATCHING.md's full recipe), and on CPU-IQ quants with
  `STRATA_IQ_MT_MIN=1` (G8).
- ctest gate every PR against the known baseline (4 known failures on this fork:
  `split_lend_complement_test`, `mmvq_il_parity`, `verify_parity`, `ple_parity` - no new ones).
- gfx906 ISA gate: `hipcc --offload-arch=gfx906 -Xclang -mcode-object --save-temps` piped through
  `llvm-objdump-22 | grep -c "v_wmma"` must print 0 for any touched kernel (gfx906-build skill).
- HIP ggml builds require `STRATA_GGML_DIR` to carry `tools/gfx906/ggml/hip-cmake.patch`
  (enforced by `tools/ds4/cmake/ggml_cuda.cmake`).
- Every measured number lands in `benchmarks/gfx906/receipts/` with its build receipt, per
  `docs/COMMUNITY_BENCHMARKS.md` practice, and is written into the doc that cites it.

---

## Phase P0 - measure the shipped batch mode on this hardware (~1 week, ~0 new code)

**Prerequisite P0.0 - artifacts.** The mainline engine needs a mainline model artifact and a serve
config. Neither exists on this box yet (only GLM/DS4.1 ported-engine artifacts are complete;
`serve/` holds no model config). Steps: download/convert the Qwen3.8-Flash-Next Strata artifact
per `docs/MODEL-DOWNLOADS.md` + `setup.py`'s flow, generate its serve config, and record its
identity (sha, quant, ctx). This is operator-gated (disk/RAM): 869 GB free on `/ai` as of
2026-10-18. **Status 2026-10-19:** the first attempt stopped at setup's PC check - upstream's AMD allow-list
refuses gfx906 (`setup.py:5982`); the fork now accepts it behind
`STRATA_EXPERIMENTAL_GFX906=1` (mirroring upstream's own gfx1103 opt-in, `setup.py:2095`), gated by
`tools/test_setup_amd.py` (37 tests) and witnessed live on the 4x MI50 box (unset refuses 4/4, set
accepts 4/4). Re-run the same setup command with that env var set, behind the RAM watchdog.

**P0.1 - no-GPU server-side suite. DONE 2026-10-19.**
`python3 -m serve.test_parallel` from the repo root (NOT `python3 serve/test_parallel.py` - the
suite imports `serve.frontend`/`serve.server`, which needs the repo root on `sys.path`; BATCHING.md
does not record this). 30 cases over the protocol: BYIELD re-admission order,
solo<->slot transitions, per-lane stop semantics, fit handshake, SSE stream integrity (one
`[DONE]` per lane), `finish=length|stop` exactness, the batched sampler ignoring penalties,
`--batch-mtp` with 12 slots through 8-row windows. Witness recorded: **Ran 30 tests ... OK,
53.9 s, this checkout at `4b4a5b35`**, scripted fake engine behind the real `StrataEngine` +
`Service`, no GPU used.

**P0.2 - engine parity on gfx906 (GPU, minutes per run).**
`python3 tools/batch_test.py --exe build-906/strata --config <flash-next.json> --batch 4 --n 4`
then `--batch 8 --n 8`, then with `--extra "--batch-mtp"` (exactness recipe flags appended).
`tools/batch_interleave_test.py` for the prefill/decode interleaving (default `--long 5000`).
Witness: exact native-token parity (or a recorded G8-class divergence with its `STRATA_IQ_MT_MIN=1`
resolution). These runs also witness `INFO batch_slots=fit slot_cache=..` - the capacity handshake
- on MI50/HIP for the first time.

**P0.3 - route census (mainline change, ~1 day, opt-in).**
Add to the mainline router (`src/core/layer.cpp` path, same shape as the ported engines'
`--dump-routes`/`--rank`, `tools/v41/engine/ds41_dense.cpp:599,724-725`): an env/flag-gated
per-layer route histogram + pairwise lane-overlap counter, off by default. This is the only real
code in P0. Output: overlap % per window over a fixed script, comparable across quant sizes -
the gate input for P2. Prior numbers to compare against: 4 lanes disjoint cost 54 ms vs 20 ms per
window on an RTX 5070, 24 vs ~8 CPU-experts/layer (BATCHING.md); gfx906 numbers do not exist yet.

**P0.4 - batched tok/s at several sizes (GPU, ~half day).**
`"parallel": 1/2/4/8` (and 12 with `--batch-mtp`) on one MI50, then on the 4-card layer split with
`--batch-groups` default auto. Record against BATCHING.md's two regimes; locate the box.

**P0.5 - recurrent-state step cost (GPU, ~half day).**
Time the host-side slot loop in `src/core/verify.cpp:1208-1217, 1242-1255` at 1/2/4/8 slots
(engine-side timing behind the same opt-in flag; or `STRATA_BATCH_DECODE_SHARE=0` variants to
separate GDN work from attention work). This number alone decides P3: if the serial loop at 8
slots is far below the per-step budget, there is no 3-6-week kernel to build.

**P0.6 - checkpoint/restore cost curve (GPU, ~half day).**
The engine already logs `session restored ... in %.1f ms` (`generate.cpp:10665`): script
save/restore at 16K/64K/256K/1M ctx (the state is ~118 MB regardless, but the KV bytes scale), and
parking-cache restores (`--conversation-cache-mib`, note its 2,560 MiB RAM floor and that HIP
coverage is unevidenced per DETAILS.md - this PR witnesses it). Feeds P2's economics with real H2D
numbers, replacing the assumed ~120 ms.

**Exit criteria:** a `benchmarks/gfx906/receipts/` entry per step; route-overlap % and per-step
loop cost written into CONCURRENCY-RADIX.md replacing "UNMEASURED"; a go/no-go on P2 and P3.

## Phase G8 - the exactness decision (free, do first)

Decide and document ONE of: (a) batched serving guarantees exact solo parity by default (pay
`STRATA_IQ_MT_MIN=1` + `--pcie-frac 0` + `--adapt-every 1000000` + `--no-prefill-borrow` in
`"parallel"` configs); or (b) batched output is "same model, not bit-equal" and the exactness
recipe is a documented opt-in for parity-sensitive users. The repo has already chosen (b) in
effect (BATCHING.md documents the divergence); this phase makes it a decision instead of drift:
wire the recipe into CI's batch parity jobs and into the generated `"parallel"` configs.
Witness: CI job runs `tools/batch_test.py --extra "<recipe>"` green.

## Phase P1 - window rows for prefill, parity tightening (2-4 weeks)

One PR per item; each opt-in.

1. **Penalties under batch** - two options, pick in the issue first: apply them in the batched
   sampler (`src/core/verify.cpp:3218`, where the comment pins the ignore), or reject them at the
   server the way `logit_bias` already returns HTTP 400 under `--batch` (`docs/LOGIT_BIAS.md`).
   The 400 precedent is cheaper and already users' expectation; sampler support is the feature.
   Tests: `serve/test_parallel.py` case 22 must flip from "ignored" to the chosen behavior.
2. **Prefill chunks as window rows** - the last un-batched mainline path is the chunked-prefill
   GEN step (`generate.cpp:6947-7000`, `:5679-5681` pins "BGEN reads as a GEN 1"). Design: admit a
   prompt chunk as a row in the shared verify window (the row machinery, graph buckets, and
   per-row layouts already exist, `include/strata/core/verify.hpp:340-356`); park/resume stays
   BYIELD. Gate: P0.2 green first; extend `tools/batch_test.py` and `batch_interleave_test.py`
   with a `--prefill-rows` scenario (mixed chunk+decode rows, exact parity vs sequential).
3. **Failed-window blast radius** - today `ERR+exit` pins the slot and the server re-admits from
   checkpoints (server recovers, engine dies). Minimal honest step: engine-side recovery of the
   failed row (fail the row, keep the window) behind a flag, or explicitly keep exit+restart and
   document the restart cost with a measurement. No silent half-recovery.
4. **`--batch-mtp` drafts path in the harness** - extend `tools/batch_test.py` to compare
   drafted-vs-solo acceptance (not just tokens) so P3 inherits an acceptance-parity harness.

Exit: the three BATCHING.md "does not batch" paths are batched or explicitly exempted with
numbers; parity harness green on gfx906.

## Phase P3 - batched recurrent-state kernel (3-6 weeks, ONLY if P0.5 says the loop costs)

1. Batched GDN/PLE/indexer variants taking a state-stride grid dim
   (`include/strata/kernels/verify_kernels.hpp:41-44` shows the single-state signature today);
   wave64-safe; per-state independent so ordering cannot couple lanes.
2. Runtime window length: replace compile-time `kVerifyMaxT = 8`
   (`include/strata/kernels/verify_kernels.hpp:20`) with runtime-sized windows ONLY in the batched
   path; the solo pipeline keeps its static shapes (blast radius: solo buffers, `kHistSlots`,
   exact-T instantiations, CPU `MAXT`).
3. Parity: reuse the abcd parity contract (`tools/abcd-parity-contract-v2.json`,
   `abcd_parity_test.py`) extended to N-lane drafted acceptance from P1.4.
Exit: per-step GDN cost at 8 lanes <= measured solo-equivalent; exact-state parity green.

## Phase P2 - arena + checkpoint-tree radix (1-2 months, ONLY if P0 census/reuse justifies)

Sequence so each step is independently shippable and opt-in:

1. **P2a - arena allocator** (`src/core/vmm.cpp` HIP branch + a new frame arena): one large
   per-stage `cudaMalloc` with offset arithmetic, refcounts, LRU free list. HIP has no VMM
   (`vmm.cpp:154-157`), and no VA contiguity is needed because kernels address pages only through
   `page_table` (`qsa.hpp:70-77`; permuted-table tests at page_size 1/4/512). Default off:
   today's per-session pool allocation stays byte-identical. Test: allocator unit tests +
   `--max-context x N` VRAM reduction with unchanged tokens on `bench` runs.
2. **P2b - frozen-prefix nodes** - generalize the existing `pin=N` machinery (`vmm.cpp:284`) to
   refcounted shared frames, WITH the write-semantics decision settled by a rewind-path audit
   (copy-on-write vs private-suffix; this is the correctness crux, not the allocator).
   Parity gate: KV-pinning parity tests + byte-parity against solo with `--pcie-frac 0`.
3. **P2c - checkpoint tree** - nodes = {refcounted frozen pages, sparse GDN snapshot at node
   depth, token-key edges}, inserted ONLY at depths the chain already checkpoints (root/turn/pin/
   every-16K - `conversation_state.cpp:230-236`, `state.cpp:30-38`). Not page-granular: at 4-token
   pages the 118 MB state is ~2,300x the page's 4 x 12,672 B. Reuse gate: a node may be reused
   only under a token-equality check (its own tokens, not a hash) and a VRAM headroom check - the
   `can_fit` + parking-cache RAM-floor pattern already models admission
   (`generate.cpp:4573-4611`, `--conversation-cache-min-free-mib`).
   Prior art to build on, with its caveats carried: `docs/MESSAGE_BOUNDARY_CACHE.md`
   (first-edit -33.1%, reuse 22,016 -> 32,831; excludes layer splits, chunk-geometry FP caveat).
4. **P2d - cross-session shared prefixes** (system prompts, few-shot sets): multi-pin, LRU across
   sessions, `serve` side announces hits. Exit: two-session same-prefix followups reuse tokens and
   produce byte-identical continuations under the exactness recipe.

## Phase P4 - ported engines' per-session caches (last; follows the GLM/DS4.1 bring-up)

- First: the T0-T5 boots in `docs/gfx906/GLM-FIRST-BOOT.md` and the DS4.1 `--engram` smoke (both
  operator-gated on GPU + >= ~65 GiB free RAM; incumbent holds both today).
- Then DS4.1 (`tools/v41/engine/`): its caches are positional, so per-session page tables + the
  P2b/P2c pattern port nearly directly. GLM (`tools/glm/engine/`): KDA state per session first
  (`glm_session.hpp:16-18` already holds S/F per session), shared prefixes last - state
  save/restore at node depth is the same economics as Part 2 of the analysis.
- `glm_generate`/`ds41_generate` are single-device today; multi-card layer split and any
  replica-scale work (serve/`replicas.py`, whose per-process weights duplicate CPU RAM on a
  125 GiB box - a shared pinned arena is the same fix as P2a) come after.

## Cross-cutting

- Docs: each phase updates CONCURRENCY-RADIX.md's gap table (gap -> shipped, with the receipt
  path), BATCHING.md's limits, and DETAILS.md; no claims without a measurement.
- PR split table (one change each, reviewable alone):
  PR1 route census -> PR2 G8 CI wiring -> PR3 penalties -> PR4 prefill-rows -> PR5 window
  recovery -> PR6 batched-GDN -> PR7 runtime window -> PR8 arena -> PR9 frozen-prefix ->
  PR10 checkpoint-tree -> PR11 cross-session -> PR12+ ported-engine caches.
- This fork's identity rules: each PR records source identity; experiment paths stay behind the
  gfx906 gates; upstream-facing work rebases onto `gfx906/v0.30.0` per the standing rebase-park
  note.

## Risk register

| Risk | Mitigation |
| --- | --- |
| P0 shows low route overlap on real traffic -> P2 shares nothing | P0.3 is the cheap gate; P2 simply does not start |
| P0.5 shows the serial loop is not the limiter -> no P3 | P3 is conditional on the measurement by design |
| batch + layer split interaction (windows, `--pipeline-windows` off under `--batch`) caps split gains | P0.4 measures `--batch-groups` directly before any split-side code |
| HIP paths of shipped-but-unvalidated features (parking, KV-prefetch, conversation cache) misbehave on gfx906 | P0.2/P0.6 are explicitly first-witness events; failures become issues, not assumptions |
| Write-semantics of shared frames break a rewind path nobody thought of | P2b's rewind-path audit precedes any sharing; exact-parity gate blocks merge |
| RAM watchdog kills builds/boots mid-phase on the shared 125 GiB box | standing watchdog + schedule heavy boots in incumbent gaps; operator decides |

Effort summary: P0 ~1 week (mostly waiting on GPU/RAM windows), G8 ~0, P1 2-4 weeks, P3 3-6 weeks
(conditional), P2 1-2 months (conditional), P4 last and gated on the ported-engine bring-up.
The honest current position: P0.1 is DONE (30/30 OK, 2026-10-19, this checkout); everything else
waits on the P0.0
artifact and a GPU+RAM window on this box.
