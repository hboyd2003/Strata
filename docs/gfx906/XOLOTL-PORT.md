# Xolotl port assessment: DeepSeek-V4.1-Flash and GLM-5.3-Flash on this fork

**Status: assessment only. Nothing here has been built or run for these two models; every number is
Xolotl's or upstream's, measured on hardware this fork does not have. No port, correctness, or
performance claim for gfx906 is made or implied.**

## What was investigated, and with what identities

- Subject: <https://github.com/Indras-Mirror/Xolotl>, a fork of Strata that adds serving for
  DeepSeek-V4.1-Flash and GLM-5.3-Flash.
- Studied as a blobless clone at `/ai/engines/gfx906/xolotl-study`, HEAD `f7c561b7`
  ("ds41: --tier-batch-pcie flag", 2026-10-18), with a `local` remote pointing at this checkout.
- Merge base with this fork's `main` (`e063ec6c`): `296b59d2` (2026-10-14, "Merge f-1441-perf ...
  into f-1441"). Since that base: Xolotl +1,389 commits (875 files changed), this fork's `main`
  +130 commits (81 files).
- `include/strata/hip_compat` and the `STRATA_HIP_GFX906` build option exist **at the merge base
  itself** — they are the shared gfx906 lineage, not Xolotl's work. Xolotl does not contain any of
  this fork's 130 later commits.

## What Xolotl actually is

Neither model is a Strata `ModelFamily`. `include/strata/artifact/model_family.hpp` is untouched
(it still knows only `qwen4exp` and `qwen35moe`); the GGUF architecture guard, the geometry
(`core/geometry.hpp`) and layout tables (`core/family_layout.hpp`), and the `strata` binary
(`src/program/generate.cpp`) never mention deepseek or glm. Both models are **standalone binaries
under `tools/` that link Strata's core libraries plus a patched llama.cpp/ggml**:

| Model | Engine | LOC | How it runs |
| --- | --- | ---: | --- |
| DeepSeek-V4.1-Flash | `tools/v41/engine/ds41_*` (main at `ds41_generate.cpp:242`) | ~12,600 | hybrid: the dense/attention half is a ggml-CUDA graph builder (`ds41_dense.cpp`); the MoE experts run on Strata's expert cache/streaming tier (`strata_engine`, `strata_kernels`, `strata_kernels_cpu`, `strata_artifact`) with Strata kernels |
| GLM-5.3-Flash | `tools/glm/glm_*` (main at `glm_generate.cpp:183`) | ~2,600 | ggml dense + the **legacy** `tools/ds4` expert tier unchanged, plus a baked rank-1 abliteration LoRA (`lora_bake.cpp`) and a soft-prune expert list (`reap_prune.py` -> `prune-ezct-0.25.txt`) |

Supporting pieces:

- **`tools/xolotl/ggml/xolotl-ggml.patch`** (1,201 lines, 48 hunks, auto-applied to the vendored
  llama.cpp by `CMakeLists.txt:1331-1365`): new ops `GGML_OP_DSV41_FAKE_QUANT` and
  `GGML_OP_DSV41_HC_MIXES` (CPU+CUDA implementations), fp4 (e2m1/NVFP4/MXFP4) quant/dequant and
  KV-fp4 helpers, lightning-indexer fixes, an MMQ rework (Q2_K/Q3_K/MXFP4, deterministic path) that
  **also matters for Strata's own `strata_mmq`/prefill build** (`CMakeLists.txt:1492-1516`), and
  `GGML_CUDA_GRAPH_MAX_TOKENS` graph gating (fixes GLM's `cudaGraphInstantiate` OOM on fresh long
  prompts; decode keeps its graphs).
- **Shared-core changes** (`src/`, `include/`): 63 files, +8,856/-251 versus the merge base. New
  quant formats for the GSQ files including the **PTQ1_0 ternary (type 143)** CPU kernels
  (`src/kernels/cpu/ptq1_0.cpp`); `src/kernels/cuda/fork/*.cuh` headers split out of
  `iq_kernels.cu`/`native_mmvq.cu`; per-model geometry/guard headers (`ds4_geometry.hpp`,
  `ds4_geometry_test.cpp`); `iq_kernels.cu` +148, `moe_mmq.cu` +116, `pool.cpp`,
  `native_expert.cpp`, `expert_cache.cpp` (+23) changes.
- **Not part of the DS/GLM path:** the large new `src/` files `dense_model.cpp`,
  `dense_main.cpp`, `dense_kernels.cu` are a Qwen3.8-27B **dense** engine (`option STRATA_DENSE`).
  A DS/GLM port does not need them.
- **Serve:** each model gets a clone of the Strata server pattern:
  `tools/{v41,glm}/serve/serve_{ds41,glm}.py` + `strata-*.json` configs + `ds41-engine.sh` /
  `glm-engine.sh` wrappers (memory guard, GPU lock, env defaults). These bypass `setup.py`'s
  catalog and `serve/server.py`'s `FAMILY_MODEL_NAMES`.
- **DeepSeek-V4.1-Flash's geometry** (from `docs/ds4/GGUF_INVENTORY.txt`): 43 layers, n_embd 4096,
  MLA-style attention (`q_lora_rank`/`output_lora_rank` 1024, 1 KV head, key_length 512), YaRN 16x
  over 64K -> 1M context, 256 experts top-6 + 1 shared, per-layer `compress_ratios`
  (`[0,0,4,128,...]`), `hash_layer_count = 3`. This is genuinely new architecture for Strata, not a
  Flash-Next shape.

## Merge mechanics against this fork: the easy part

- **Collision surface is 5 files** (of their 875 and our 81 changed files): `CMakeLists.txt`,
  `README.md`, `serve/server.py`, `src/core/verify.cpp`, `src/program/generate.cpp`. Our verify /
  generate edits (+208/-83 / +79) overlap theirs (+2 / +93): mechanical conflicts, not
  architectural ones.
- Xolotl's own `MERGE_SCOPE.md` maps their internal unification (~60 files, conflict hunk counts
  per file; `src/kernels/cuda/iq_kernels.cu` worst at 24 hunks). Their `tools/ds4/cmake/fork.cmake`
  already knows about this fork: it excludes the MMVQ test under `STRATA_HIP_GFX906`. They are
  aware of the gfx906 target but build only CUDA.
- A feature-branch `git merge` beats cherry-picking 1,389 commits. The new engines arrive behind
  their own CMake options, so the default `strata` path stays byte-identical to this fork's last
  release — the contributing rule is satisfied by construction, and a follow-up trim commit can
  drop the bulk of their `bench/` and `tools/` ecosystem without losing git's conflict
  resolution.

## The real work: CUDA -> HIP/gfx906

- **Engine side is mild.** Grep over `tools/v41/engine`, `tools/glm`, and the dense engine: no
  `cp.async`, no cooperative launches, no `__nanosleep`, no `cudaMallocAsync`. CUDA graphs appear
  in 2 files (`ds41_moe.cpp` tier staging), one dynamic-shared-memory attribute in 1 file, one
  `wmma` in `ds41_moe.cpp` (Vega needs an MFMA or scalar path). ROCm has graphs on the modern
  stack; whether they behave on gfx906 with this staging pattern is **untested here**.
- **The hard ~80% is ggml.** `xolotl-ggml.patch` is CUDA-only; its file list (fattn, mmq,
  cpy-utils, the two new ops' CUDA paths) must be mirrored into ggml's HIP backend — the same job
  this ecosystem already does for llama.cpp on gfx906, now extended with the new ops and fp4 KV.
  And because Strata compiles ggml MMQ sources for `strata_mmq`, the patch's MMQ instance list has
  to stay consistent with this fork's build.
- **De-risking precedent (their evidence, not ours):** HANDOFF S6 records
  `vincalkr/Strata-DSV4.1`, a gfx1100 (RX 7900 XTX) port of this code (+13-23% with the fused ops;
  graphs-off +6.6% on HIP), and antirez's DwarfStar ROCm discussion
  (`docs/ds4/RESEARCH_2026-10-10_DWARFSTAR.md`). A maintainer without gfx906 hardware would welcome
  a receipt from this box (`docs/COMMUNITY_BENCHMARKS.md`).

## Hardware and artifact budget (plan before building)

- Boot inputs per their docs: a GSQ GGUF (**96-152 GiB**; DeepSeek `IndrasMirror/DeepSeek-V4.1-Flash-GSQ(-v2)`,
  GLM `GLM-5.3-Flash-GSQ-RCO-3.0bit-q4kattn` 113.6 GB or the ~96 GB Maya-S-v2), **Engram tables
  ~203 GB** on SSD (`tools/v41/extract_engram.py`, `--engram a:b`), `rank.bin`, the DSpark draft
  sidecar (`V41-MTP-Vision.Q2_K.gguf`), an exported tokenizer dir, and the serve JSON/wrapper.
  So: ~350 GB NVMe and roughly >=128 GB RAM for the expert tier. This box's role split (2x MI50
  32 GB VRAM + RAM/SSD tiers) matches the design, and since the expert tier *is* Strata's
  `expert_source`/`expert_cache`, that part is home ground.
- **All their numbers come from one PC** — RTX 4090 24 GB, Ryzen 5700X, 90 GB DDR4, NVMe
  (`docs/ds4/V41_START_HERE.md`). Per this fork's rules none of them transfer as gfx906 claims;
  the port ships with its own receipts.
- **License check before download:** the DeepSeek GSQ repos are labelled "Uncensored" variants and
  the GLM files come from individual quant publishers; each model's own license governs.

## Their maturity caveats (from their own docs)

- "Experimental, single-developer, measured on one PC" (`docs/DEEPSEEK_V41.md` header).
- DeepSeek-V4.1: vision is not in the engine; a prefix-cache unpark bug is open; keep-cache never
  engages under `vram_lru`; long-context real-document decode is unproven; the 1017 quant is mid
  work.
- GLM: **no speculative decode at all** (KDA state cannot rewind); the runtime LoRA is "not
  implemented in the engine yet" (`docs/glm/PLAN.md:14`); the long-prompt graph OOM is worked
  around with `GGML_CUDA_GRAPH_MAX_TOKENS=64`, not fixed; the legacy `tools/ds4` "slice" engine is
  slated for deletion but GLM still links it.

## Validation gates for a merge candidate

1. CUDA build of the merged tree reproduces their per-engine check (`docs/xolotl/BENCHMARKS.md`
   claims DS4.1 logits bit-identical across their merge; `tools/v41/engine/compare_ds41.py`).
2. `ds4_geometry_test` and the parity harnesses pass (`src/kernels/ds4_expert_parity.cpp`, 417
   lines; `iq`/`mmvq` parity binaries with the fork headers).
3. This fork's gfx906 receipts and parity-contract fixtures for the untouched Flash-Next path still
   hold byte-identical (`benchmarks/gfx906/`).
4. After that, HIP/gfx906 port proceeds kernel-family by kernel-family, each gated on parity
   against the CUDA result before any perf number is written down; correctness, state parity, and
   performance stay separate claims.

## Target hardware: this server (measured on the box, 2026-10-18)

4x AMD Instinct MI50 32 GB (gfx906, `rocm-smi` product info, device 0x66a1 rev 0x02; the 32 GB
SKU is the dual-HBM2-stack Vega 20 at **~1 TB/s per card**, twice the 16 GB SKU's 512 GB/s), AMD
Threadripper Pro 3975WX (64 threads), 125 GiB DDR4-3200 across 8 channels (~205 GB/s theoretical),
Samsung 990 Pro 4 TB (the `/ai` LVM volume: 2.0 TiB, 697 GiB free at assessment time), ROCm
**7.2.0** active (`/opt/rocm` -> 7.2.0, 6.3.3 also installed), `hipcc` on PATH. This fork's gfx906
HIP receipts live in `benchmarks/gfx906/receipts/` (latest 2026-10-18).

Tier mapping the port inherits from Strata (no new machinery needed for Disk -> RAM -> VRAM):

| Tier | DeepSeek-V4.1-Flash | GLM-5.3-Flash | This box |
| --- | --- | --- | --- |
| SSD | GSQ GGUF 140-152 GiB read in place below full-RAM (file tier), **Engram tables ~203 GB** row-reads (`--engram a:b`), DSpark Q2_K sidecar | GSQ-RCO 113.6 GB or Maya-S-v2 ~96 GB GGUF | 990 Pro: ~650 GB for both models; **697 GiB free now** - free space or pick one model first |
| RAM | all experts pinned, or budgeted via `--resident-budget-gib` / their `--budget-gib --ternary-cap-at`; load-time PTQ1_0 -> Q2_0 transcode (`ds41_transcode.cu`) | same; soft-prune list shrinks the resident set | 125 GiB total; 205 GB/s is ~4x the bandwidth of the 4090 box that produced their numbers - the main CPU-expert lever |
| VRAM | dense/attention/mixers, KV, MTP, expert cache fills the rest | same | ~1 TB/s HBM2 per card, 128 GB aggregate - the GPU tier outranks the CPU tier 5:1 on bytes, which is the argument for eventually multi-device; but the engine is **single-device** (`CUDA::cudart`, device 0) -> phase 1 = one MI50 (32 GB), three wait |

Three constraints this box imposes, verified:

1. **RAM contention is live, not hypothetical.** `free` showed 117 of 125 GiB in use at
   assessment time (the box's serving workload). The resident-expert tier needs nearly all 128
   GB at full pin; plan to stop the incumbent server or boot these models in the budgeted file
   tier.
2. **Single-device engines.** Multi-GPU Strata (`--gpus`, `--peer-device`, `peer_experts.cpp` /
   `remote_experts.cpp`) exists on the Flash-Next path of this lineage but Xolotl's ds41/glm tier
   does not use it. Driving 4x MI50 for these models is a fork-side feature after the HIP port,
   not part of it.
3. **ROCm 7.2 on gfx906.** This fork already builds its wave64 target on this box (receipts), but
   Xolotl's patched ggml is built and patched for CUDA; the HIP mirror of `xolotl-ggml.patch` must
   be validated against the ROCm version this fork's receipts name, not assumed.

## Port log (this box, 2026-10-18)

Branch `gfx906/xolotl-merge` (worktree `/ai/engines/gfx906/worktrees/xolotl-merge`):

- `f366d550` merge of `xolotl/main` @ `f7c561b7` (conflicts: CMakeLists union, README fork banner;
  server.py / verify.cpp / generate.cpp auto-merged; their engine changes are flag-gated,
  default-off).
- `d43935cc` + `35cef805` keep it building on gfx906: `STRATA_DENSE` default ON->OFF, v41/glm
  CUDA targets gated on `NOT STRATA_HIP_GFX906`, `moe_mmq.cu` fallback inlined under HIP (its
  pre-merge body), new `hip_compat/cuda_profiler_api.h` shim.
- Gates, ROCm 7.2.0 / clang 22 / llama.cpp `3cf03257` (`-DSTRATA_HIP_GFX906=ON
  -DCMAKE_HIP_ARCHITECTURES=gfx906 -DSTRATA_GGML_DIR=/ai/engines/gfx906/llama.cpp-3cf03257`):
  all 192 targets build; `strata 0.1.44.1` links; ctest matches a pre-merge `main` baseline built
  on this box (same 4 pre-existing host failures: split_lend_complement, mmvq_il_parity,
  verify_parity, ple_parity; qsa/kv_stream pass serially). Logs: `logs/build-906-*.log`.
- Notable: the legacy `ds4_moe_cuda`/`ds4_generate` tier now COMPILES under HIP (only the
  profiler shim was missing). What is still gated: `ds41_transcode.cu`, and the patched-ggml
  CUDA dense halves (`ggml-cuda` mirror of `xolotl-ggml.patch`) - the remaining port.

## Suggested order

1. Read-only conflict probe (below), then the feature-branch merge, CUDA-gated.
2. Port the **GLM path first** (2.6k LOC, legacy tier, no spec-decode to validate), then the 12.6k
   LOC ds41 engine; the ggml HIP patch grows with each.
3. Decide later whether `deepseek4` should become a proper `ModelFamily` (registry guard +
   `layer_tensors` table for MLA/DSA geometry) instead of sidecar binaries.

Read-only probe — run from this checkout, changes nothing:

```bash
cd /ai/engines/gfx906/strata-gfx906
git fetch https://github.com/Indras-Mirror/Xolotl main:refs/remotes/xolotl/main --filter=blob:none
git merge-tree --write-tree --name-only main xolotl/main | head -40
```

Expected (from this assessment's same measurement): conflicts confined to `CMakeLists.txt`,
`README.md`, `serve/server.py`, `src/core/verify.cpp`, `src/program/generate.cpp`.

## Port log - 2026-10-18 (evening, while model downloads ran)

- `xolotl-ggml.patch` **applies cleanly** to the pinned llama.cpp clone at
  `/ai/engines/gfx906/llama.cpp-xolotl` (19 files, +720/-65); the **CPU** build of the patched
  ggml is green (exit 0, `-j2`, 6m16s, log `/tmp/ggmlcpu.log`). The **HIP/gfx906** `ggml-cuda`
  build is the one open decisive gate - held back by RAM only (needs `available >= ~10 GiB`;
  downloads + the incumbent workload leave 4.8 GiB right now).
- CUDA residue re-triage (line-level, added lines only): ggml patch has no hard CUDA blockers for
  HIP - masked shuffles exist in HIP, the NVIDIA-cc probes in `lightning-indexer.cu` evaluate
  false on gfx906 and take the existing fallback path. Engine side: `ds41_moe.cpp` is the only
  file with graph calls (12) and they are **already mapped** by `hip_compat/strata_hip.h`
  (`cudaGraph*` -> `hipGraph*`, both `cudaGraphInstantiate` overloads; ROCm 7.2's
  `hipGraphInstantiate` keeps the legacy 5-arg form the CUDA-12 `(NULL,NULL,0)` calls use).
  Syntax gates: `ds41_moe.cpp` (host, HIP defines) **exit 0**; `ds41_transcode.cu`
  (`hipcc -x hip --offload-arch=gfx906 -fsyntax-only`) **exit 0**.
- No engine `.cu/.cuh` under `tools/v41|glm|ds4` contains `wmma` or raw `cudaGraph` - the earlier
  CUDA-feature tally maps to: ds41_moe.cpp (shimmed), the ggml patch's NVIDIA-guarded wmma
  section (falls back on gfx906), and `dense_kernels.cu` (STRATA_DENSE, now OFF).
- **Maya-L kernel verdict for gfx906**: gate/up IQ3_S(=21) and down IQ4_XS(=23) are both in the
  AMD instantiation lists of `src/kernels/cuda/iq_kernels.cu` (gate/up {16,17,18,21,22,23,29,42,
  143,11}; down {20,23,42,143,11}); `is_iq` admits both; gate==up so the fork's mixed-pair
  {Q2_K,Q3_K} restriction does not apply; the AVX2 CPU fallback (this box is Zen2, no AVX-512)
  covers both types. Maya-S-v2's down types (IQ2_S=22, IQ3_XXS=18) are **not** in the gfx906
  down list - on this box the already-downloaded Maya-L is the better-supported file.
- New files: `docs/gfx906/GLM-FIRST-BOOT.md` (T0-T5 + six named first-boot unknowns),
  `tools/glm/serve/strata-glm-mayal-gfx906.json` (draft box config),
  `tools/glm/serve/glm-engine-906.sh` (`STRATA_BUILD_DIR -> build-906`).
