# gfx906 upstream delta

This note maps gfx906-relevant features to public upstream commits observed on 2026-10-16 and 2026-10-17. It describes source history only; it does not identify the source commit used to build the archived container.

## Public fork updates

- [PR #11](https://github.com/cassettesgoboom/strata-gfx906/pull/11) adds the README's archived dual-MI50 cached multi-agent measurement table and links the canonical evidence and provenance. It changes documentation only; the archived image and engine are unchanged.
- [PR #12](https://github.com/cassettesgoboom/strata-gfx906/pull/12) introduced a CPU-only comparator for supplied
  serial-OFF / AB/CD-ON native traces. Its v1 singular-token record format is superseded by
  [PR #17](https://github.com/cassettesgoboom/strata-gfx906/pull/17), which uses `native_transaction_trace_v2`:
  ordered native token-ID vectors per committed transaction/window and persistent-state hashes only at window end.
  V1 records are rejected unless regenerated or migrated with original transaction-boundary evidence. The
  comparator output always states `runtime_parity_claim=false`; it cannot authenticate capture or establish runtime
  token/state parity, and does not port AB/CD or change the image.
- [PR #15](https://github.com/cassettesgoboom/strata-gfx906/pull/15) adds an unused host-side transaction lifecycle registry and focused C++ CPU test for slice ownership, snapshots, registry-bound handles, lifecycle transitions, accepted-prefix limits, and drain-before-reuse. Hosted CI passed 42 Python CPU tests and the C++20 transaction test. The registry is not integrated with the verifier or scheduler; this scaffolding does not prove GPU overlap, device completion/drain, or runtime token/state parity and does not change the image.
- [PR #17](https://github.com/cassettesgoboom/strata-gfx906/pull/17) updates the comparator and synthetic fixtures
  to v2 transaction-window token vectors and window-end state comparison. The current
  `python tools/gfx906.py test-cpu` suite passes 44 CPU tests. This remains a comparison contract, not runtime evidence.
- [PR #19](https://github.com/cassettesgoboom/strata-gfx906/pull/19) extends the unused CPU-side transaction
  registry with two-phase commit intent and result. `begin_commit` validates and copies accepted-prefix intent into
  registry-owned state, and its accessor returns a copy; the slice remains undrained while device commit may run.
  `finish_commit` records success only after the runtime observes it. A post-launch failure or abort requires
  explicit device drain followed by session recovery/discard acknowledgement before retirement. Validation is 44
  Python tests plus a separate focused C++20 transaction test. The contract remains unwired to `Verifier`/scheduler and implements no inference,
  QSA rollback, GPU transaction isolation, or parity.
- [PR #52](https://github.com/cassettesgoboom/strata-gfx906/pull/52) selectively ports upstream [commit
  `1d6fd9f97ab09da231b35cbbd84a2879adb6d77f`](https://github.com/Niko1221/Strata/commit/1d6fd9f97ab09da231b35cbbd84a2879adb6d77f)
  (#1752, KevinX8): on Linux, gfx906 defaults to the older full-read file-cache rule; other Linux builds retain the
  hot-read rule. `STRATA_FILE_CACHE_OLD_RULE=1` selects the older rule and `STRATA_FILE_CACHE_OLD_RULE=0` selects the
  hot-read rule. The existing
  `STRATA_UNBUFFERED_LOAD` force setting takes precedence. The checked-in HIP receipt is compile-only and records
  no GPU devices; no local MI50 runtime or performance comparison was run. This narrow file-tier policy change does
  not alter AB/CD or establish inference parity.

The earlier fork PRs document archived evidence and add CPU-side future-validation scaffolding. PR #52 is a selective
source port of one gfx906-specific upstream fix; it does not supply new source-to-image provenance. The
[current-main blockers in Issue #1](https://github.com/cassettesgoboom/strata-gfx906/issues/1) remain: the lifecycle
registry is not wired into `Verifier` or the scheduler; per-slice graph, staging, and poll ownership is not
integrated; and the registry's commit-intent/result and recovery gates are not connected to actual device work.
Ordinary rejection restores the QSA tail and re-appends accepted keys, but it is not general rollback for failed or
aborted same-session overlapping slices: indexer appends also mutate dead, pooled, and block-position state, and
GDN commit mutates persistent recurrence state in place. The registry requires drain and session recovery/discard
after a post-launch failure or abort, but it does not implement these actions or resolve that rollback gap. The
single next source gate is to integrate this lifecycle and recovery behavior in the current-main verifier/scheduler,
then use a pinned current-main build for isolated serial-versus-AB/CD GPU checks of ordered native token IDs and window-end state
hashes. Keep AB/CD opt-in until those runtime checks pass.

## Upstream sync to v0.2.2 (2026-10-22)

Fork `main` [`e063ec6cf2d9`](https://github.com/cassettesgoboom/strata-gfx906/commit/e063ec6cf2d9df86626b5746839f086cffbde78e)
was 1,445 commits behind upstream `main` [`011c9e600a93`](https://github.com/Niko1221/Strata/commit/011c9e600a93bbc9fd451c563501b5831c32bbca)
(tag `v0.2.2`, 2026-10-21); the previous merge base was `296b59d2f73c8f04a805df29c2fa97655adace78`. The sync is an
ordinary merge commit, so both histories are preserved. Conflict resolutions:

- **#1752 file-tier rule:** upstream now carries the gfx906 old-rule default itself. Behavior is unchanged from PR #52;
  the fork keeps its testable `file_cache_keeps_for_tier` helper, adds `file_cache_old_rule` for upstream's #3237
  drive probe (which runs only under the hot-read rule), and keeps upstream's `file_tier_prefers_direct` and its tests.
- **NVFP4 W4A4 on gfx906:** upstream #3172 builds `strata_mmq_w4a4` with `GGML_USE_HIP` as a "not available" stub.
  The fork's `if(NOT STRATA_HIP_GFX906)` exclusion is dropped in favor of the upstream equivalent.
- **Verifier window state:** the fork's `WindowContext` keeps row count, tokens, slots, positions and batch flag.
  Its `poll` sub-record is removed: upstream made the batch poll state atomic (`b_running_`, `b_k_`, `b_last_`) for
  `batch_serve` on a separate thread (#31, `STRATA_PIPED_CONCURRENT`), and those upstream members are used as-is.
- **Watchdog release control:** upstream added the overlapped split's hand-off wait (`hflag_in_h_`) to the flags
  `release_gpu_waits` raises. `VerifyReleaseControl::bind` takes it as an optional fourth flag, so the registered
  release path raises it too; `verify_release_control_test` covers it.
- **Correctness traces:** native trace IDs in `serve/` and `generate.cpp` are merged with upstream's queue deadlines,
  request telemetry and merged prompt windows. A merged first window reports `mtp_offered=0`, since its rows are prompt
  tokens, not drafts.
- **CI:** the gfx906 CPU workflow installs `regex==2026.9.10`, which upstream's `serve/server.py` now imports.

Validation of the merge tree (2026-10-22, local):

- Passed: `python tools/gfx906.py test-cpu` (78 tests); `python -m unittest serve.test_native_trace serve.test_replicas`
  (29); every `serve/test_*.py` (939, 15 skipped); the workflow's five standalone C++20 contracts; the synthetic-fixture
  rejection check; `platform_memory_test --policy-only` built with and without `STRATA_HIP_GFX906`.
- Passed: a CPU-only CMake build with `STRATA_BUILD_TESTS=ON` and its CTest suite, except `expert_parity` and
  `pool_test`, which need a model pack (`pack/full/experts.bin`) that was not present.
- Passed (compile only): `STRATA_HIP_GFX906=ON -DCMAKE_HIP_ARCHITECTURES=gfx906`, HIP 7.16 / AMD clang 24, targets
  `strata`, `platform_memory_test` and `verify_release_control_test`; `strata --families` lists `qwen4exp`.
- Passed (one MI50, serial): a short model-backed smoke of that build with the merge tree's server: two greedy chat
  requests with MTP speculation, conversation-cache reuse on the follow-up, verify graphs captured for 1-6 tokens, and
  a clean shutdown. With `STRATA_CORRECTNESS_TRACE=1` and `STRATA_CORRECTNESS_WINDOW_TRACE=1` the window records add
  up to the tokens produced and the greedy output is unchanged. See the
  [sanitized receipt](../../benchmarks/gfx906/receipts/2026-10-22-upstream-v0.2.2-sync-smoke.json).
- Not run: batching, pipeline windows, layer split or two cards, AB/CD, token/state parity against another build,
  or sustained performance. The open gates above are unchanged.

## janus-reith prefill port (2026-10-22)

[janus-reith/strata-gfx906](https://github.com/janus-reith/strata-gfx906) `b0f440d8` (upstream v0.1.45 plus 16 commits)
speeds up gfx906 prompt reading. Its gfx906 commits are cherry-picked with `-x` (author and original commit kept) onto
the v0.2.2 sync:

- `STRATA_PF_RP`: the prompt's MoE expert products through a repacked int8 GEMM adapted from
  [mx-llama.cpp](https://github.com/mxxm-t/mx-llama.cpp) (MIT, vendored in `src/prefill/rp/` with its licence): IQ4_NL
  experts, and IQ3_XXS / IQ3_S gate/up (IQ2_S with `STRATA_PF_RP_IQ3=2`), per product.
- `STRATA_DENSE_MMQ` on the gfx906 build with `STRATA_DENSE_MMQ_SKIP` (default `attn_k.,attn_v.`: K and V stay FP16 for
  the drafts' acceptance), and `STRATA_DENSE_RP`: dense Q8_0 / Q6_K / Q5_K / IQ4_XS / IQ4_NL through the repacked GEMM
  (Q4_K stays on MMQ). `dense_mmq_bench` and `rp_expert_bench` measure and check them.
- IQ4_NL gate/up experts on gfx906, the opt-in server option `system_to_user`, and the five prompt switches on by
  default in the gfx906 build (`=0` turns one off).

Resolved against v0.2.2: the gate/up format list keeps upstream's trellis formats and adds IQ4_NL; the kernel side of
janus's sub-16-row fix was already upstream, the launch side is ported; the repack decisions take upstream's new
`!wmma_l` guard; the dense skip list wraps upstream's cold-start trace. Not taken: janus's `d00a1018` (reverted: upstream
#3271 links `moe_f16_kq.cu` as a stub, the same behaviour), his Dockerfile and the workflow publishing his image, and his
README banner (the docs here describe the port instead). The two fixes of this fork that janus carried are already here.

Checked on one MI50 (native build, HIP 7.16): `native_grouped_parity` 0 failures (IQ4_NL gate/up included),
`rp_expert_bench` 0 failures (the repacked IQ4_NL products bitwise equal to MMQ, 23-81% faster per product),
`dense_mmq_bench` (int8 dense products 46-74% faster than the FP16 path at 4096 tokens; IQ4_XS repack bitwise equal to
MMQ, the Q6_K / Q5_K repack equal to it in error against an FP64 reference but summed in another order), every
`serve/test_*.py`, and `tools/gfx906.py test-cpu`.

In the engine, `STRATA_PF_RP_CHECK=1` compared 2,000 repacked products against MMQ on a 15,000-token prompt of a Swift
IQ3_S model: 0 differ. On a coding-agent workload with one slot and a 262K context, the port's image (native build)
read large prompts at 677 tok/s against 504 for the v0.2.2 image as it was and 621-624 with upstream's three prompt
switches on; one agent's session took 141.8 s against 185.7 and 156.8 s. Decode did not change measurably. See the
[receipt](../../benchmarks/gfx906/receipts/2026-10-22-janus-prefill-port.json).

## PR #4 current-source comparison (2026-10-17)

Public [PR #4](https://github.com/cassettesgoboom/strata-gfx906/pull/4) remains a draft with head `0fffc4412ebd1ba5e5f880fdabaeb840a1164adc`, based on archived `v0.1.41` commit `fb58e0dbc8399662c0e47c76578c6e878b14f6cf`. It changes 15 files (+1,781/−134) and reports no gfx906 build or exact native-token and persistent-state parity. The reviewed fork main is `f54111839951c0d7f64630064f840b359507fd01`; after `ef91497b71a9ec25a4a30b22695c7d23ffddc89e`, PR #35 changed only the README tag link. Upstream `v0.1.45` and `main` resolve to `477ac13cbe8e582c20484871ce459d20d82df09d`.

The complete PR #4 patch does not apply to either current tree. `git apply --check` fails on `CMakeLists.txt`, `serve/server.py`, `src/core/verify.cpp`, and `src/program/generate.cpp` against upstream v0.1.45; it also fails on `include/strata/core/verify.hpp` against fork main. The verifier and generator APIs have moved since the v0.1.41 base, so this is not a clean cherry-pick or a small context refresh.

A source-level recheck against upstream v0.1.45 found the PR #4 shared-draft-head assignment `dhead_type_ = shared->dhead_type_` is already present in `src/core/mtp.cpp`; that hunk is not a forward-port candidate. The PR #4 `dvocab_host_ = shared->dvocab_host_` assignment and per-slot `MropeScope` are absent from v0.1.45. The latter overlaps the per-slot MRoPE work still under upstream PR #1242; absence from upstream does not make either hunk safe to copy independently of the session/lifetime design below.

| PR #4 path | Disposition against current sources |
| --- | --- |
| `CMakeLists.txt` | **Incompatible:** old test-registration context does not apply; the vision and MRoPE test intent needs current build integration. |
| `docs/gfx906/EXPERIMENTAL-SOURCE.md` | **Obsolete:** records the historical v0.1.41 candidate and its open parity gates; it is not current-main guidance. |
| `include/strata/core/mtp.hpp` | **Still needed only as a separate fix:** moving `idle()` out of line permits device selection, but synchronizing the drafter stream alone does not provide bounded teardown. |
| `include/strata/core/session.hpp` | **Still needed as a capability:** per-slot MRoPE positions need session ownership; current v0.1.45 has no session MRoPE pointer. |
| `include/strata/core/verify.hpp` | **Mixed:** serial `commit_slot_prefixes(const int*, ...)` already exists; the historical `BatchTransaction` and per-slice verifier methods are incompatible with current APIs. |
| `include/strata/kernels/mrope.hpp` | **Still needed as a capability:** current code exposes a per-device global MRoPE table; the draft scope needs a current ownership and concurrency design. |
| `include/strata/platform/hip_compat/strata_hip.h` | **Obsolete:** current stream-priority paths use explicit HIP/CUDA APIs; these historical aliases are not needed by current call sites. |
| `include/strata/program/vision_records.hpp` | **Still needed only if retained:** this parser's record format has no current consumer. |
| `serve/server.py` | **Obsolete:** the audit-ID header and `RAW_DONE` logging are measurement instrumentation, not scheduler or commit logic. |
| `src/core/mtp.cpp` | **Mixed:** per-slot MRoPE selection and shared `dvocab_host_` mapping are separate capabilities/fixes. The `OnDevice` teardown hunk still synchronizes only one stream without a deadline. |
| `src/core/verify.cpp` | **Incompatible:** the old scratch-row and transaction scheduler targets a verifier implementation that has since changed. The existing serial accepted-prefix helper does not provide the missing AB/CD lifetime owner. |
| `src/program/generate.cpp` | **Incompatible:** the opt-in `STRATA_BATCH_ABCD=1` flow is wired to those old verifier APIs and cannot be copied onto current generation paths. |
| `tests/core/vision_records_test.cpp` | **Still needed only if retained:** it covers the historical parser, not scheduler correctness. |
| `tests/cuda/mrope_slot_graph_test.cpp` | **Still needed after redesign:** it checks captured per-slot MRoPE pointers, an API current sources do not expose. |
| `tools/batch_mtp_split_test.py` | **Obsolete:** it exercises serial split batch-MTP; current fork main has newer interleave and split-parity tooling, and this test does not validate AB/CD. |

Upstream v0.1.45 exposes `batch_launch`/`batch_poll`, `commit_slot_prefixes(const int*, ...)`, and `pl_launch`/`pl_commit_async`; it has no PR #4 per-slice transaction API ([verifier header](https://github.com/Niko1221/Strata/blob/v0.1.45/include/strata/core/verify.hpp)). Fork main has a separate CPU `BatchTransactionRegistry` with accepted-prefix intent and explicit completion, drain, and recovery acknowledgements ([PR #19](https://github.com/cassettesgoboom/strata-gfx906/pull/19)); upstream v0.1.45 does not have this fork-only registry. Its contract trusts a runtime owner to observe GPU completion and draining; it does not own verifier/session allocations, streams, graphs, events, or callbacks. [Issue #24](https://github.com/cassettesgoboom/strata-gfx906/issues/24) remains the gate: verifier release registration covers only selected streams, while graph work can still reference session-owned recurrence state and verifier destruction can block without a deadline. No scheduler subset from PR #4 is safe to port until that owner is implemented and connected; keep serial defaults and backends unchanged. Then port the opt-in accepted-prefix path against current APIs and prove exact token and recurrent-state parity before promotion.

## Feature-to-commit map

| Feature | Upstream commit (date) | Scope and relationship |
| --- | --- | --- |
| AMD gfx906 build support | [`9ff290b493a2`](https://github.com/Niko1221/Strata/commit/9ff290b493a2aaf6a784af69190c82f903ebf221) (2026-10-03) | Opt-in HIP support for gfx906 devices. This establishes public source support, not validation of a particular image or workload. |
| K8/V4 KV cache | [`3597c47ff8cd`](https://github.com/Niko1221/Strata/commit/3597c47ff8cd490dccfdafa2bd9d9b80ba9aa255) (2026-09-29) | Adds the hybrid K8/V4 cache and `--kv k8v4`. |
| Refreshed-residency batch servicing (#1102 equivalent) | [`7ef55e19599f`](https://github.com/Niko1221/Strata/commit/7ef55e19599f1ef009894b3260b40cdbe0d59ba9) (2026-10-07) | Upstream #1118 changes the batch graph/service decision to use per-window `ar_on()` residency. Its commit message identifies #1102 as the same fix for `run_slot_rows()` and `batch_poll()`; #1118 also updates `service()`. This is an upstream equivalent behavior, not evidence that the archived binary contains either commit. |
| Serial batch-MTP windows | [`a0116b2782a5`](https://github.com/Niko1221/Strata/commit/a0116b2782a5b82622358c04a06496485d12b2d6) (2026-10-12) | Adds/retains serial MTP windows with `STRATA_BATCH_MTP_SERIAL=1` or `--batch-groups 1`; the commit states that the default layer-split pipeline takes precedence over `--batch-mtp`. This is the current upstream behavior, not a source attribution for the archived image. |
| gfx906 Linux file-tier cache rule | [`1d6fd9f97ab0`](https://github.com/Niko1221/Strata/commit/1d6fd9f97ab09da231b35cbbd84a2879adb6d77f) (2026-10-15) | Upstream #1752 keeps the older full-read rule by default on gfx906, with `STRATA_FILE_CACHE_OLD_RULE=1/0`; fork PR #52 selectively ports this policy. Its compile receipt is not MI50 runtime or performance validation. |

## Relationship to #1242 and #1253

As observed on 2026-10-16, upstream PR [#1242](https://github.com/Niko1221/Strata/pull/1242) is open and concerns concurrent image-position isolation and batch-MTP draft paths; PR [#1253](https://github.com/Niko1221/Strata/pull/1253) is open and concerns serial multi-GPU batch MTP. The public upstream commit above documents serial windows and their current mode-selection policy, but does not establish that either open PR was merged unchanged or that it reproduces the archived experimental build. Treat the PRs and the current upstream commit as related source history, not interchangeable build provenance.

## Archived build boundary

The archived binary's exact coherent source commit is unknown. Public tag metadata, executable hashes, and current upstream feature presence do not establish which commits or patches produced it. The archived AB/CD path remains opt-in; its source is unpublished and its exact-token and recurrent-state parity remain open. Do not infer that current upstream contains that path or that the archived binary can be reproduced from this source tree.

## License

The [upstream LICENSE](https://github.com/Niko1221/Strata/blob/main/LICENSE) identifies Strata as MIT licensed. Retain its copyright and permission notice in redistributions. ROCm components have separate component-level licensing; see [AMD's ROCm license table](https://rocm.docs.amd.com/en/latest/about/license.html) and the caveat in [RELEASES.md](RELEASES.md).
