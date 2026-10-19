# vLLM-style concurrency and Radix KV in Strata - source-based gap analysis

Investigation 2026-10-18, four read-only source sweeps of this tree (mainline engine `src/`,
`include/strata/`, server `serve/`, ported engines `tools/glm`, `tools/v41`). Every claim is a
file:line from this checkout or a number from an evidence doc, with that doc's own caveats
attached. Engineering effort figures are estimates, not commitments. Like every new feature here,
each phase below is opt-in and the default path stays byte-identical (AGENTS.md).

Headline: **the engine already contains multi-sequence batched decode, a concurrency-capable wire
protocol, grouped MoE batching, and kernel-verified page-table indirection.** What is missing is a
scheduler policy, capacity raised past 8 window rows, a global frame pool for sharing, and batched
recurrent-state kernels. vLLM-style concurrency here is a scheduler-and-capacity project on
existing machinery, not a rewrite; Radix KV is the same tree but a *checkpoint* tree, not a page
tree, because the recurrent state changes the unit economics.

## Part 1 - concurrency: what already exists

1. **Multi-slot batch mode.** `--batch N` advertises `INFO batch_slots=N`; `batch_step` builds one
   verify window from active `BSlot`s (`src/program/generate.cpp:9975-10150`); windows carry
   "SEVERAL SEQUENCES IN ONE WINDOW" (`include/strata/core/verify.hpp:183-203`); four-slot
   batch-MTP runs 2 rows/slot rotated (`generate.cpp:9981-10006`); CUDA graphs are captured per
   row-layout with LRU eviction (`verify.hpp:340-356`). Window rows are capped by
   `kVerifyMaxT = 8` (`include/strata/kernels/verify_kernels.hpp:21`).
2. **The line protocol already interleaves streams.** `BGEN <slot>` dynamic admission
   (`generate.cpp:10362-10398`), `BADM` continue, `BT/BDONE <slot>` per-slot token streams routed
   into per-slot server queues (`serve/server.py:984-1032`, `generate_batched` `:1544`), `BSTOP`,
   and `BYIELD`/`YIELDED` prompt preemption (`generate.cpp:11644-11712`). The "single turn" is a
   server-side fifo lock for solo requests (`serve/server.py:216, 3133`); batch requests bypass it.
3. **MoE decode already batches by expert union.** `run_multi`/`gpu_run_n`: "Each distinct expert is
   read/fetched once and applied to every token that routed to it" (`tools/v41/engine/ds41_moe.hpp:469-473`,
   dedup + grouping `ds41_moe.cpp:2856-2967`, kernel `native_expert_grouped`,
   `src/kernels/cuda/s2_expert_grouped.cu:817,879`). Prefill's grouped MMQ path is n-agnostic
   (`ds41_moe.hpp:485`, `include/strata/prefill/moe_mmq.hpp:54-67`). The router already scores n
   tokens in one pass (`src/core/layer.cpp:398-422`).
4. **Per-sequence isolation and three preemption tiers.** Each batch slot owns a full
   `SessionState` (`src/core/verify.cpp:2765`); preemption exists at three costs: verifier
   snapshot (rollback, us), `Ds41Dense::park/unpark` (host RAM, positional caches, single parked
   conversation `tools/v41/engine/ds41_dense.cpp:1685-1718`), and `SAVE`/`RESTORE` disk session
   files (`generate.cpp:10438-10448`).
5. **Prefix reuse is checkpoint-chain based.** Up to `--prompt-cache` 6 checkpoints at multiple
   depths (root, turn boundaries, pin, every 16,384 tokens, tail) in host RAM
   (`generate.cpp:639-658, 8447-8485`); resume at longest token-prefix match; `pin=N` marks one
   pinned anti-eviction prefix (`conversation_cache.hpp:24-28`); replicas route on chained
   64-token block hashes (`serve/replicas.py:32,123-159`).

Demonstrated ceiling so far: the fork's archived-binary evidence - P4 AB/CD, 4 lanes, 102K
context, 71.17/53.94 tok/s aggregate on **dual MI50** (`docs/gfx906/EVIDENCE.md`; archived binary,
parity open, not a causal claim across rows).

## Part 1b - the gaps, as work items

| # | gap | what it takes | est. |
|---|---|---|---|
| G1 | `kVerifyMaxT = 8` rows/window | raise the compile constant, size the buffers/graphs, capture-graph LRU pressure grows with layouts | S, 1-2 wk |
| G2 | admission is serialized (one prompt read at a time; BYIELD only yields) | scheduler policy that interleaves chunked-prefill rows with decode slots - the pump already runs `batch_step` between chunks and when stdin is idle (`generate.cpp:11720, 10353`) | M, 2-4 wk |
| G3 | recurrent-state kernels are sequence-serial | `gdn_step_norm_multi` takes ONE state pointer (`include/strata/kernels/verify_kernels.hpp:41-44`); batch windows loop slots host-side (`src/core/verify.cpp:1208-1247`). Past ~4 lanes needs a multi-state step kernel (state-stride grid dim) + parity gate | M, 3-6 wk |
| G4 | memory scales x N: ctx-sized KV pool + ~118 MB GDN state per sequence (`src/core/layer.cpp:670-688`, `conversation_state.cpp`) | interim: cap slots by ctx; real fix is Part 2 sharing | - |
| G5 | batch window greedy-only (header still says no MTP; code does 4-slot MTP - doc drift) | per-slot sampling params in `BGEN`, per-row sampler | S-M |
| G6 | MoE cache policy sized for 1 stream | prefetch budget `pf_b=1.43` experts/token, `kMaxPf=6` (`ds41_moe.hpp:280,344`); N streams multiply the per-step union; `predict_multi` + `STRATA_ROUTE_LOG` are the batch-census seeds | M, measure-driven |
| G7 | ported engines are single-global | GLM `L.kvc`/KDA state and DS4.1 comp/idxk caches have no sequence dim (`tools/glm/glm_dense.cpp:839-844`, `ds41_dense.cpp:1085-1094`) - per-session conversion over positional caches (park/unpark proves the swap primitive) | L, phase-3 |

Physics note that differentiates this from vLLM's version: the value of batching here is **expert-
union dedup and dense-pass amortization**, not tensor-core utilization. Concurrent lanes that route
to overlapping experts read each expert once (`run_multi` contract), so concurrency is cheaper
per-token than N serial streams; lanes with disjoint routes instead thrash the VRAM LRU. Route
overlap under real multi-agent traffic is the open measurement - the census (`--dump-routes`,
`STRATA_ROUTE_LOG`) answers it before any code above is justified.

## Part 2 - Radix KV: right idea, wrong granularity (as the papers define it)

What this tree already provides:

- **Real page tables, kernel-verified.** QSA layers: paged K/V with int32 `page_table`,
  `page_size = 4` cells (`src/core/layer.cpp:746-776`, `include/strata/kernels/qsa.hpp:113-115`);
  the attention kernel "does NOT assume the identity" and tests demand identical output under
  permuted tables at page_size 1/4/512 (`qsa.hpp:70-77`). The addressing spine for shared pages
  exists and is tested.
- **A page-granular eviction primitive**: the `--kv-resident` streaming mode runs a CLOCK
  second-chance sweep over residency slots (`src/kernels/cuda/kv_stream.hpp:6-15`).
- **A prefix index already computed**: chained 64-token block hashes (`serve/replicas.py`).
- **Multi-depth snapshots already kept**: the checkpoint chain (Part 1 item 5) IS a sparse
  token-prefix tree.

The three structural gaps (from the sweep, in ascending hardness):

1. **No global frame pool.** Each session's arena is one `cudaMalloc`
   (`generate.cpp:4418, 4454, 4576`); resident-mode page tables are identity; the elastic VMM maps
   whole-state ranges, not shareable frames. Refcounted cross-session pages need a device-side
   frame allocator + free list. The existing VMM machinery is the natural substrate.
2. **The recurrent state breaks the page-tree premise.** A branch point in a hybrid model must be
   able to restore the GDN state: 118 MB per snapshot (`conversation_state.cpp:165-186`) is ~9,300
   tokens' worth of int8 KV (12,672 B/token) - at a 4-token page granule the state is ~10^4x the
   page's cost. A page-granular radix tree is economically impossible here. What IS affordable: a
   **checkpoint tree** - a radix tree whose nodes sit only at depths that already get checkpoints
   (system-prompt boundary, turn boundaries, pin, every-16K cadence), each node = {refcounted KV
   pages, sparse GDN snapshot, token-keyed child edges}. At the 16K cadence a GDN snapshot adds
   ~57% over the node's int8 KV bytes - affordable exactly where the chain already pays it.
3. **Everything is conversation-sized.** SessionState, parked images, checkpoints - no refcountable
   object smaller than a conversation exists; eviction is LRU over whole conversations
   (`ConversationCache::evict_oldest`) or whole checkpoints. The radix unit (node) has to be born
   as a new object class with its own victim queue.

Plus a write-semantics discipline shift: cells are append-only per sequence but **rewind rewrites
old positions in place**, `conversation_kv_restore` overwrites a live pool wholesale
(`conversation_snapshot.cpp:204-224`), and rope-rotated K means a shared node is only shareable at
the same absolute position (`qsa.hpp:180-181`). Under sharing, every pre-existing rewrite path
becomes a corruption path: shared nodes must freeze (copy-on-write or private-suffix), which is
the one cross-cutting change with correctness teeth.

## Phased plan (each phase gates on measurements, per repo rules)

- **P0 - measure the batch mode that exists** (zero new code): `--batch 4` parity + tok/s on this
  box, route census for overlap, restore-cost timing for checkpoint chains. Answers whether G1-G6
  are worth opening at all. ~1 week.
- **P1 - continuous admission on existing 4-8 slots** (G2, G5): scheduler policy + per-slot
  sampling; LRU slot eviction over the existing park/SAVE tiers. 2-4 weeks.
- **P2 - checkpoint-tree radix** (Part 2): global frame allocator over VMM, refcounted frozen
  prefix pages, `pin=N` generalized to many pins = multi-root sharing. 1-2 months.
- **P3 - batched GDN multi-state kernel** (G3) to push past 8 rows, plus AB/CD comparator rewritten
  against the current `WindowContext` API (fixtures exist:
  `benchmarks/gfx906/fixtures/abcd-parity-contract-v2.json`). 3-6 weeks.
- **P4 - ported engines** (G7): GLM/DS4.1 per-session caches; DS4.1 first (positional caches, no
  recurrence), GLM's KDA state last (it carries Part 2's hardest node economics).

## What this analysis does NOT claim

No measurement on this box backs the effort estimates or the tok/s deltas; the 71.17 figure is
archived-binary evidence with the fork's own caveats. The parity gate the fork itself left open
(exact native-token and recurrent-state parity for batched lanes) is assumed, not resolved, in
every phase above. Route overlap under multi-agent traffic is unmeasured and could shrink Part 1's
throughput promise substantially - P0 exists precisely to find that out before months are spent.
