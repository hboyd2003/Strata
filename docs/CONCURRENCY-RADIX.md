# vLLM-style concurrency and Radix KV in Strata - source-based gap analysis

Investigation 2026-10-18, four read-only source sweeps of this tree (mainline engine `src/`,
`include/strata/`, server `serve/`, ported engines `tools/glm`, `tools/v41`), seven review passes
(framing, citations, completeness - the last three re-read every file fresh), and a reconciliation
against the shipped feature's own production docs (`docs/BATCHING.md`, `docs/MULTI_GPU.md`,
`docs/MESSAGE_BOUNDARY_CACHE.md`, `docs/DISJOINT_EXPERT_CACHE.md`, `docs/LOGIT_BIAS.md`,
`docs/KV_PREFETCH.md`, `docs/DETAILS.md`). Every claim is a
file:line from this checkout or a number from an evidence doc, with that doc's own caveats
attached. Engineering effort figures are estimates, not commitments. Like every new feature here,
each phase below is opt-in and the default path stays byte-identical (AGENTS.md).

Headline: **concurrent serving is not a to-do list here - it is a shipped, opt-in production
feature.** `"parallel": N` in a serve config (#465) runs real concurrent chats today through
multi-sequence batched windows, preempt-and-resume admission, grouped MoE batching, and a 30-case
server test suite (`serve/test_parallel.py`, fake engine, no GPU needed). What is actually missing
is narrower than vLLM-concurrency sounds: prefill run as rows of the shared window, windows past 8
rows, batched recurrent-state kernels, penalties in the batch sampler, any cross-session KV
*sharing* (Radix), and all of it in the ported engines. vLLM-style concurrency here means widening
a machine that already turns; Radix KV is the genuinely new part, and it must be a *checkpoint*
tree, not a page tree, because the recurrent state changes the unit economics.

Scope: the Qwen3.8-Flash-Next engine path (`src/program/generate.cpp` + `src/core/`). Qwen3.6-35B-A3B
dispatches to its own engine (`generate.cpp:1874-1886`, `src/qwen35/`) which has none of the
slot/checkpoint machinery described here - extending this analysis to it is separate work. (q36 has
its own `--serve` protocol, a one-conversation-deep prefix reuse (`q36_engine.cu:2274-2287`) and
mid-request STOP (`:2209-2211, 2313`); `--batch` degrades safely there - it skips unknown args and never
prints `INFO batch_slots=`, so the server keeps everything solo-fifo, `q36_engine.cu:2388-2390`,
`serve/server.py:965`.)

## Part 1 - concurrency: what already exists

1. **Multi-slot batch mode.** `--batch N` (alias `--slots N`) advertises `INFO batch_slots=N`;
   `batch_step` builds one verify window from active `BSlot`s (`src/program/generate.cpp:9975-10150`);
   windows carry "SEVERAL SEQUENCES IN ONE WINDOW" (`include/strata/core/verify.hpp:183-203`); opt-in
   `--batch-mtp` runs four slots x 2 rows rotated (`generate.cpp:581, 9981-10006`); on a layer split
   `--batch-groups G` pipelines the slots through the stages (default `auto` since 0.1.41;
   measured 8 clients: 166 against 86 tok/s on 4 GPUs, `docs/BATCHING.md`) - while the speculative
   `--pipeline-windows` overlap is switched OFF by `--batch` itself (`docs/MULTI_GPU.md`), so on a
   split the pipelining that survives batching is the group-per-stage one; CUDA graphs are captured per
   row-layout with LRU eviction (`verify.hpp:340-356`). Window rows are capped by
   `kVerifyMaxT = 8` (`include/strata/kernels/verify_kernels.hpp:20`).
2. **The line protocol already interleaves streams.** `BGEN <slot>` dynamic admission
   (`generate.cpp:10362-10398`), `BADM` continue, `BT/BDONE <slot>` per-slot token streams routed
   into per-slot server queues (`serve/server.py:984-1032`, `generate_batched` `:1544`), `BSTOP`,
   and `BYIELD`/`YIELDED` prompt preemption (`generate.cpp:11644-11712`). The "single turn" is a
   server-side fifo lock for solo requests (`serve/server.py:3133`); batch requests bypass it
   (`:4293-4302`).
   With `"parallel": N` this is the NORMAL `/v1/chat/completions` path, not a special endpoint
   (`parallel_args` `serve/server.py:2699-2718`; the fifo is never taken, `:4300-4302`), hardened
   through at least ten filed issues visible in-line (#656, #1012, #1527, #1603, #1869, #1994,
   ...; server #465, engine PR #559) and pinned by `serve/test_parallel.py` (30 cases).
3. **MoE decode already batches by expert union.** `run_multi`/`gpu_run_n`: "Each distinct expert is
   read/fetched once and applied to every token that routed to it" (`tools/v41/engine/ds41_moe.hpp:469-473`,
   dedup + grouping `ds41_moe.cpp:2856-2967`, kernel `native_expert_grouped`,
   `src/kernels/cuda/s2_expert_grouped.cu:817,879`). Prefill's grouped MMQ path is n-agnostic
   (`ds41_moe.hpp:485`, `include/strata/prefill/moe_mmq.hpp:54-67`). The router already scores n
   tokens in one pass (`src/core/layer.cpp:398-422`).
4. **Per-sequence isolation and three preemption tiers.** Each batch slot owns a full
   `SessionState` (`src/core/verify.cpp:2765`); preemption exists at three costs: verifier
   snapshot (device-side rollback; no cost is documented anywhere I found), `Ds41Dense::park/unpark` (host RAM, positional caches, single parked
   conversation `tools/v41/engine/ds41_dense.cpp:1685-1718`), and `SAVE`/`RESTORE` disk session
   files (`generate.cpp:10438-10448`) - but tier 3 is REFUSED exactly when it would be needed:
   slot save/restore returns 501 with `"parallel"`/`--batch` active, and even solo it needs
   `--slot-save-path DIR` (`serve/server.py:3209-3213`). Disk-tier preemption is absent precisely
   where concurrency runs. A fourth mechanism sits beside the tiers: the conversation parking
   cache (`--conversation-cache-mib/-slots`) restores a parked conversation on every stage before
   its admission - follow-ups reached first token in 0.53 s against 3.9 s for fresh reads on a
   4-GPU split (`docs/BATCHING.md`, `docs/DETAILS.md`); it is verified for layer splits by
   `tools/parking_test.py`, refuses itself under RAM pressure (2,560 MiB floor by default,
   `--conversation-cache-min-free-mib`), and its HIP and multi-GPU coverage is explicitly called
   out as needing separate evidence from Linux/CUDA (`docs/DETAILS.md`) - unvalidated on this box.
5. **Prefix reuse is checkpoint-chain based.** Up to `--prompt-cache` 6 checkpoints at multiple
   depths (root, turn boundaries, pin, every 16,384 tokens, tail) in host RAM
   (`generate.cpp:639-658, 8447-8485`); resume at longest token-prefix match; `pin=N` marks one
   pinned anti-eviction prefix (`conversation_cache.hpp:24-28`); replicas route on chained
   64-token block hashes (`serve/replicas.py:32,123-159`); turn-boundary prefix reuse is measured
   work: first-edit time fell 33.1%, reused prefix grew 22,016 -> 32,831 tokens, at prefixes
   >= 8,192 with a tail <= 1,024 - opt-in (`STRATA_CACHE_MESSAGE_BOUNDARY=1`) and LAYER-SPLIT
   EXECUTION IS EXCLUDED, which is this box's intended shape (`docs/MESSAGE_BOUNDARY_CACHE.md`).
6. **The admission state machine is production, not a sketch.** A waiting prompt announces its
   length; the reading request yields at the next chunk boundary and PARKS its part-read into a
   slot with matching checkpoints handed over (`copy_to_slot` `generate.cpp:9889, 11644-11712`,
   `YIELDED` `:13060`), and its later `BGEN` resumes from the parked prefix (`slot_source`
   `:11002-11026`). Between prompt chunks the pump decodes active slots for
   `STRATA_BATCH_DECODE_SHARE` (default 0.5) of the chunk time (`:11634-11640`); with stdin idle
   it pumps decode at all times (`:10353-10354`). A conversation prefers its old slot next turn
   (`serve/server.py:1878-1908`); a request that becomes alone upgrades back to solo GEN with MTP
   drafts (`:1537-1543`, session copy-back `generate.cpp:9927`); slot-full waits on a condition
   variable rather than erroring, with a stall witness and orphan-slot reaper (`:1645-1686,
   1462-1481`); only repeated engine `BGEN` refusals produce a retry-after (`:1735-1737`). Shared
   blast radius is real though: a failed window kills the engine process and every lane with it
   (`server.py:1041-1042`, `generate.cpp:10041-10044`) - the server-side behavior is pinned by
   `test_a_failed_window_names_its_error`; the engine has no recovery.

Demonstrated ceiling so far. The shipped feature's own measurements are the strongest evidence
and match this box's shape: a 4-GPU layer split (4x 16 GB, PCIe Gen3, IQ3_S,
`--batch 8 --batch-groups 4 --trim-stage-weights`) served 1/2/4/8 concurrent requests at
aggregate 120/113/205/360 tok/s (`docs/BATCHING.md`, engine PR #559). `--trim-stage-weights` (PR
#639) raised the VRAM expert share from 76-85% to 84-100%, with MI50 evidence in the same
direction (2x MI50 16 GB, Coder: 8,819 -> 10,626 experts in VRAM, 39.2 -> 41.7 tok/s,
`docs/MULTI_GPU.md`). The same doc's 12 GB card table shows the other side:
where most experts run on the CPU, slots buy waiting time, not speed (-11..-24%; the fourth of four
requests starts after 1.8 s there instead of 11.2 s queued). Older fork-specific evidence, the
archived AB/CD binary - 4 lanes, 102K context, 71.17/53.94 tok/s aggregate on MI50
(`docs/gfx906/EVIDENCE.md:12`; archived binary, parity open, not a causal claim across rows;
EVIDENCE.md records this row only as "archived MI50" and does not state its card count - its
"two MI50 cards" note belongs to the 2026-10-16 smoke receipt, not to this row).

## Part 1b - the gaps, as work items

| # | gap | what it takes | est. |
|---|---|---|---|
| G1 | window rows capped at 8 (`kVerifyMaxT`) - a WINDOW cap, not a slot cap: `--batch-mtp` already rotates >8 slots past it (`generate.cpp:4530-4533, 9981-10006`) | raising the constant is cross-cutting: it sizes solo `--pipeline-windows` buffers and graph rows (`:4635, 7832, 7926`), penalty history (`kHistSlots = 4096 x kVerifyMaxT`, `:7765`), static asserts and exact-T kernel instantiations (`verify_kernels.cu:1006-1008, 1129-1130`), CPU `MAXT` (`kq_avx2.cpp:28`) - it moves allocations on default solo+pipeline paths, which collides with the byte-identical-default rule; plus each captured row-layout is a graph pair in VRAM competing with expert slots on 32 GB (`verify.hpp:210-211`) | M, 2-3 wk |
| G2 | the serialized residue is ONE prompt read on the wire - and it is already chunk-interleaved with preempt-resume (item 6): waiting prompts force `BYIELD` at chunk boundaries, the part-read parks into a slot and resumes (`generate.cpp:11644-11712, 11002-11026`), slots keep decoding between chunks (`:11634-11640`) | the real gap: prompts read via a private path ("BGEN reads as a GEN 1", `:570, 10362`) instead of running as ROWS of the shared verify window, so a prefill chunk cannot share a step with other slots' decode | M, 2-4 wk |
| G3 | recurrent-state kernels are sequence-serial | `gdn_step_norm_multi` takes ONE state pointer (`include/strata/kernels/verify_kernels.hpp:41-44`); batch windows loop slots host-side (`src/core/verify.cpp:1208-1217, 1242-1255`). Whether that serial loop is the limiter at N lanes is UNMEASURED (small kernels x 36 layers x N per step); measure per-step host-loop time at 4 slots before building the multi-state kernel (state-stride grid dim) + parity gate | M, 3-6 wk |
| G4 | memory scales x N: every slot gets a FULL `--max-context` session pool (one `cudaMalloc` each, `generate.cpp:4564-4576`; pool bytes scale with ctx, `layer.cpp:660-688`) + ~118 MB running state per sequence (GDN + PLE + indexer tails; `generate.cpp:1502`, `docs/DETAILS.md:967`) | the cap SHIPS: the fit check allocates until OOM, self-reports `INFO batch_slots=fit slot_cache=..` and the server obeys (`:4573-4611`, `serve/server.py:962-968`); N = min(8 unless `--batch-mtp`, floor(free / session_bytes(full ctx))) per device; a slot past `max_context` ends `finish=length` (`generate.cpp:10105-10106`). Part 2 removes the ctx factor | - |
| G5 | per-request sampling IS plumbed (server `sampling_keys` -> `BGEN` -> `set_slot_sampling` at admission, `server.py:1195-1240`, `generate.cpp:11739-11747, 13082`; Philox seeded by position matches the solo draw, `verify.cpp:3219`; sampled rows documented as drawn row-by-row with the solo counter, `docs/BATCHING.md` Exactness) | what genuinely drifts: the batch sampler zeroes penalties (`sp.penalty_last_n = 0`, `verify.cpp:3218`) while the server keeps sending them (`server.py:1213-1227`) - same request, different sampling semantics solo vs batched. The limitation is DOCUMENTED (`docs/BATCHING.md` Limits: "Repetition / frequency / presence penalties are not applied in batch windows") but not enforced per-request - and the enforcement pattern exists next door: `logit_bias` returns HTTP 400 under `--batch` rather than being silently ignored (`docs/LOGIT_BIAS.md`) | S |
| G6 | expert-cache policy is per-stream, and cross-card duplication is real. NOT because batch windows ignore routes - they count them and adapt the VRAM tier every `--adapt-every` windows (`--adapt-async 1` measured 89.2 against 74.7 blocking against 45.2 frozen tok/s, `docs/BATCHING.md`) | the prefetch budget `pf_b=1.43` experts/token (`ds41_moe.hpp:288`, ported engine; `pcie_frac` comment `:280` "single-token decode only") is sized for one stream; disjoint-route cost is MEASURED: 4 lanes read ~24 CPU experts/layer against ~8 solo (54 ms against 20 ms per window, RTX 5070, `docs/BATCHING.md`) - why setup recommends `parallel` only where experts mostly fit VRAM; helper/primary duplicate slots measured ~1,900-2,000 with an ownership-guard fix (+15.5% historical, `docs/DISJOINT_EXPERT_CACHE.md`, upstream-GPU perf unvalidated); `predict_multi` + `STRATA_ROUTE_LOG` census seeds exist ONLY in the ported DS4.1 engine | M, measure-driven |
| G7 | ported engines are single-global | GLM `L.kvc`/KDA state and DS4.1 comp/idxk caches have no sequence dim (`tools/glm/glm_dense.cpp:839-844`, `ds41_dense.cpp:1085-1094`) - per-session conversion over positional caches (park/unpark proves the swap primitive) | L, phase-3 |
| G8 | batched output is NOT bit-equal to solo by default | the CPU IQ grouped kernels round a token's expert rows differently alone than in a group (#152: `src/kernels/cpu/native_expert.cpp:155-162`) so greedy can depend on how many drafts the window held, and kernel choice depends on CPU ISA (quant- and CPU-dependent: without AVX-512, IQ3_S already takes the multi-token kernel for every group by default and its rows no longer depend on drafting, `:162-168`); the equalizer `STRATA_IQ_MT_MIN=1` costs a measured -1..-3% decode on IQ3_S on AVX-512 boxes. Plus plain `--batch` runs NO drafts (1 row/slot): every lane loses MTP unless `--batch-mtp`. The full exactness recipe is documented (`docs/BATCHING.md`: `STRATA_IQ_MT_MIN=1`, `--pcie-frac 0`, `--adapt-every 1000000`, `--no-prefill-borrow` - the adaptive tier moving experts between runs and a prompt borrowing cache slots ALSO change rounding); without it, measured drift on RTX 5070: 3 of 4 lanes apart from solo from token 0/99/174. The comparator EXISTS: `tools/batch_test.py` compares alone-vs-together token by token (8 conversations, with and without pipeline) and `tools/batch_interleave_test.py` covers BYIELD/resume/solo-return. Decide the semantics, then wire the recipe into CI | S (a decision, not code) |

Physics note that differentiates this from vLLM's version - and bounds it. What batching amortizes
here is expert reads (lanes routing to overlapping experts read each expert once, the `run_multi`
contract) and the dense pass; that is the scarce-bandwidth currency, not tensor-core utilization.
What batching does NOT amortize: every lane reads its OWN attention KV at every step, so attention
traffic scales N x ctx - at long contexts (the AB/CD evidence ran at 102K) that term can dominate
and the dedup savings stop mattering. Lanes with disjoint expert routes thrash the VRAM LRU instead
of dedup'ing. Route overlap under real multi-agent traffic is partly measured already (BATCHING.md:
~24 against ~8 CPU experts per layer at 4 lanes on a CPU-expert card) but not on MI50 or past 4
lanes, and the census
tools are ported-engine-only (`STRATA_ROUTE_LOG` in tools/v41, `--dump-routes` in tools/glm; mainline
`src/` has neither) - measuring it in the mainline batch mode costs about a day of instrumentation,
the only new code P0 needs.

## Part 2 - Radix KV: right idea, wrong granularity (as the papers define it)

What this tree already provides:

- **Real page tables, kernel-verified.** QSA layers: paged K/V with int32 `page_table`,
  `page_size = 4` cells (`src/core/layer.cpp:746-776`, `include/strata/kernels/qsa.hpp:113-115`);
  the attention kernel reads "a gathered scratch and never a pool", tests demand identical output
  under permuted tables at page_size 1/4/512, and "nothing outside this header may assume the
  mapping" (`qsa.hpp:70-77`; the `page_table` "NOT assumed to be the identity" line is `:179-180`).
  The addressing spine for shared pages
  exists and is tested.
- **A page-granular eviction primitive**: the `--kv-resident` streaming mode runs a CLOCK
  second-chance sweep over residency slots (`include/strata/kernels/kv_stream.hpp:6-15`).
- **A prefix index already computed**: chained 64-token block hashes (`serve/replicas.py`).
- **Multi-depth snapshots already kept**: the checkpoint chain (Part 1 item 5) IS a sparse
  token-prefix tree.

The three structural gaps (from the sweep, in ascending hardness):

1. **No global frame pool.** Each session's arena is one `cudaMalloc`
   (`generate.cpp:4418, 4454, 4576`); resident-mode page tables are identity. Refcounted
   cross-session pages need a device-side frame allocator + free list. On CUDA the elastic VMM
   (`--kv-grow`) looks like the natural substrate - but **HIP has no VMM at all** (`vmm.cpp:3`
   excludes HIP; `:154`: "HIP: no virtual memory here; the elastic K/V stays off", `vmm_available()`
   returns false `:157`). On this gfx906 box the frame pool must therefore be an ARENA allocator:
   one big allocation, frames as offsets, refcounts, LRU free list. That is workable, not a
   blocker - kernels address pages ONLY through `page_table` (`qsa.hpp:70-77`), so a frame needs no
   VA contiguity and no driver VMM at all; the sharing rides on the table.
2. **The recurrent state breaks the page-tree premise.** A branch point in a hybrid model must be
   able to restore the running state (GDN + PLE + indexer tails): ~118 MB per snapshot (figure at
   `generate.cpp:1502`, `docs/DETAILS.md:967`; the copy set at `conversation_state.cpp:165-186`)
   is ~9,300 tokens' worth of int8 KV (12,672 B/token, `include/strata/kernels/kv_q8.hpp:8`) - at
   a 4-token page granule the state is ~2,300x the page's cost (118 MB / 4 x 12,672 B). Two economics notes that matter for
   the design: the snapshots live in HOST RAM (`generate.cpp:637-639`) while the shared pages are
   VRAM, so the tree's state cost and its page cost compete for DIFFERENT budgets; and a resume
   pays a ~118 MB H2D per node restore (the engine logs `session restored ... in %.1f ms`,
   `generate.cpp:10665`) - upload/compute overlap for RAM-resident KV already has a precedent
   (`STRATA_KV_PREFETCH=1`, prefill -5.3% at 32K, HIP unvalidated, `docs/KV_PREFETCH.md`). A
   page-granular radix tree is economically impossible here. What IS affordable: a
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
(`conversation_snapshot.cpp:204-224`), and rope-rotated K - stored already normed and rotated
(`qsa.hpp:180`) - means a shared node is shareable only at the same absolute position; that
sharing requirement is this doc's inference from the rotated-K semantics, not a header statement. Under sharing, every pre-existing rewrite path
becomes a corruption path: shared nodes must freeze (copy-on-write or private-suffix), which is
the one cross-cutting change with correctness teeth.

## Phased plan (each phase gates on measurements, per repo rules)

- **P0 - measure the shipped batch mode on this box** (~1 day of new code: a mainline route census -
  `STRATA_ROUTE_LOG`/`--dump-routes` are ported-engine tools only): first locate this box against
  `docs/BATCHING.md`'s two tables - the 4-GPU all-in-VRAM split (360 tok/s at 8 concurrent) is the
  expert-fit side, the 12 GB CPU-expert card (slots buy latency, -11..-24%) the other; then
  `"parallel": 4` tok/s, route overlap, restore-cost timing for checkpoint chains, and the
  engine-driving harnesses `tools/batch_test.py` / `batch_interleave_test.py` (need a GPU).
  `serve/test_parallel.py` (30 cases) runs WITHOUT a GPU - the cheapest gate on the list.
  Answers whether G1-G6 are
  worth opening at all. ~1 week.
- **P1 - prefill-as-window-rows + parity tightening** (G2 residue, G5, G8): run prompt chunks as
  rows of the shared verify window instead of a private GEN-style pass; apply penalties in the
  batch sampler or stop sending them (the `logit_bias` 400-under-batch pattern is the precedent);
  settle the solo-vs-batched equality semantics (G8) and extend `tools/batch_test.py` /
  `tools/batch_interleave_test.py` - which already compare alone-vs-together token by token - to
  the `--batch-mtp` drafts path. The remaining blast-radius item is
  engine-side recovery from a failed window (today: process death, all lanes with it). 2-4 weeks.
- **P2 - checkpoint-tree radix** (Part 2): global frame ARENA allocator (no VMM on HIP),
  refcounted frozen prefix pages, `pin=N` generalized to many pins = multi-root sharing.
  1-2 months.
- **P3 - batched GDN multi-state kernel** (G3) to push past 8 window rows, plus the AB/CD parity
  harness extended to the multi-state path (fixtures exist:
  `benchmarks/gfx906/fixtures/abcd-parity-contract-v2.json`). Note raising `kVerifyMaxT` touches
  the solo/pipeline path (G1) - keep solo window length a runtime choice so solo behavior stays
  byte-identical. 3-6 weeks.
- **P4 - ported engines** (G7): GLM/DS4.1 per-session caches; DS4.1 first (positional caches, no
  recurrence), GLM's KDA state last (it carries Part 2's hardest node economics).

## What this analysis does NOT claim

No measurement on this box backs the effort estimates or the tok/s deltas; the 71.17 figure is
archived-binary evidence with the fork's own caveats. The parity gate the fork itself left open
(exact native-token and recurrent-state parity for batched lanes) is assumed, not resolved, in
every phase above. Route overlap under multi-agent traffic is unmeasured and could shrink Part 1's
throughput promise substantially - P0 exists precisely to find that out before months are spent.
Two review corrections are part of this doc's record: an earlier draft UNDERSTATED the shipped
surface (it listed admission serialization, sampling plumbing and untested stop paths as gaps;
all three are production code with a test suite - Part 1 item 6), and batched lanes are NOT
bit-equal to solo by default (G8: grouped-kernel rounding, dropped penalties, no drafts without
`--batch-mtp`) - every parity or throughput claim above is conditional on those settings. A fourth
pass found a different class of gap: the analysis had not read the feature's own production docs,
and had consequently missed the shipped `--batch-groups` pipeline, its measured 4-GPU tables, the
batch-window tier adaptation, the parking cache, and the existing parity harnesses - corrected
throughout above, and a reminder that a source sweep is incomplete until it has read what the repo
already wrote down. Passes five to seven (fresh re-reads, no cached files) added the
`--pipeline-windows`-off-under-`--batch` interaction, the parking cache's RAM floor and its
unvalidated-HIP caveat, the message-boundary cache's layer-split EXCLUSION (this box's intended
shape), the MI50 trim-stage numbers, and the KV-prefetch overlap precedent; and they fixed line
drift (`server.py:3133` not `:215`; `DETAILS.md:967` not `:968`; the G3 loop range; the `qsa.hpp`
quote to the header's own words). A fresh audit's challenge to the ~2,300x page-cost ratio was
checked and REJECTED: 118 MB / (4 tokens x 12,672 B/token) = 2,328, consistent with the per-token
figure being across all 12 QSA layers (`kv_q8.hpp:8`) - the correct unit for a shared node's
savings. An auditor's arithmetic is itself an audit target. The build plan for these phases -
PR splits, gates, receipts, and the conditional go/no-go measurements - is
[CONCURRENCY-PLAN.md](CONCURRENCY-PLAN.md).
