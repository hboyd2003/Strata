# GLM first boot on this box (Maya-L, gfx906) - checklist

Drafted 2026-10-18 on branch `gfx906/xolotl-merge`, **before any GLM boot**. Every T-step is one
paste-able block with the witness that proves it. Config: `tools/glm/serve/strata-glm-mayal-gfx906.json`.

## T0 - port gates (state as of 2026-10-18)

| gate | state |
| --- | --- |
| `xolotl-ggml.patch` applies to pinned llama.cpp (`llama.cpp-xolotl`) | **PASS** (+720/-65) |
| patched ggml **CPU** build | **PASS** exit 0, `-j2`, 6m16s (log `/tmp/ggmlcpu.log`) |
| engine tree + `ds4` tier compile under HIP gfx906 | **PASS** (192 targets, `build-906`) |
| expert kernels for Maya-L's types on gfx906: IQ3_S(21) gate/up, IQ4_XS(23) down | **PASS by instantiation table** (`src/kernels/cuda/iq_kernels.cu` AMD lists: gate/up {16,17,18,**21**,22,23,29,42,143,11}, down {20,**23**,42,143,11}) |
| patched ggml **HIP/gfx906** build (`ggml-cuda` as HIP) | **PASS 2026-10-18 19:25**, built as part of the engine tree (not the standalone `build-hip` sketch below). Artifacts on disk, re-checked 2026-10-21: `build-906/ggml/src/ggml-cuda/libggml-cuda.a` 74,266,036 B (143 objects, 144/144 TUs, `-j2`, 8m01s); engines linked against it: `build-906/glm_generate` 81,171,976 B, `build-906/ds41_generate` 77,374,200 B. Compilation is a build gate, not a runtime claim - the first GLM boot is still T4. (The standalone command kept for reference: `cmake -S /ai/engines/gfx906/llama.cpp-xolotl -B /ai/engines/gfx906/llama.cpp-xolotl/build-hip -DGGML_CUDA=ON -DCMAKE_CUDA_COMPILER=/opt/rocm/llvm/bin/clang++ -DGGML_CUDA_F16=ON -DLLAMA_CURL=OFF && cmake --build .../build-hip -j4`) |

## T1 - artifacts on disk

```bash
ls -la /ai/models/hf.co/peasantsmith/GLM-5.3-Flash-Maya-GGUF/Maya-L/ \
       /ai/models/hf.co/GCSA-AiLab/GLM-5.3-Flash-Abliterated-LoRA-GGUF/ \
       /ai/models/hf.co/zai-org/GLM-5.3-Flash/ \
       /ai/models/hf.co/peasantsmith/GLM-5.3-Flash-Maya-GGUF/vision/
```
Witness: 4 Maya-L shards (156.3 GB total, already verified), `GLM-5.3-Flash-Ablitered2-LoRA-v2.gguf`
(0.28 GB), tokenizer files, `vision/mmproj-GLM-5.3-Flash-F16.gguf`.

## T2 - tokenizer export

```bash
python3 /ai/engines/gfx906/worktrees/xolotl-merge/tools/glm/serve/export_tokenizer.py \
  /ai/models/hf.co/zai-org/GLM-5.3-Flash /ai/models/tok/glm-5.3-flash && ls /ai/models/tok/glm-5.3-flash
```
Witness: the export files the serve config's `tokenizer` dir expects (see script's own printout).

## T3 - RAM

`glm-engine.sh` runs the engine under `systemd-run -p MemoryMax=84G -p MemorySwapMax=0`. Free the
incumbent first (120 GiB used today): `free -h` must show **available >= ~90 GiB** before booting
with `--arena-gib 60`, or drop the arena in the config.

## T4 - direct engine smoke (no server)

```bash
cd /ai/engines/gfx906/worktrees/xolotl-merge && \
  STRATA_BUILD_DIR=$PWD/build-906 tools/glm/serve/glm-engine-906.sh --help | head -30
```
Then a 1-prompt run with the config's args plus whatever `--help` shows for prompt/max-tokens.
(Witness: model loads - geometry read, tier fill, no `native_expert_grouped has no kernel for
layer N's types` error, no shard-layout error - then 8 tokens out.)

## T5 - serve

Point Strata's server at `strata-glm-mayal-gfx906.json` the way `serve/` loads other model
configs; watch `--log /tmp/strata-glm-mayal-906.log`.

## Known unknowns (each one can bite at first boot; none refuted yet)

1. **4-shard split geometry** - Xolotl's split-aware tier code was tuned on S-v2's 3 shards
   ("shard 1 = blk.0-23 only"); Maya-L's 4-shard boundary arithmetic is untested here.
2. **hipGraph** - patched ggml's CUDA-graph prompt-chunk path uses `cudaGraph*` APIs; under HIP
   these need either name aliases or a shim (like the profiler one, `35cef805`).
   `GGML_CUDA_GRAPH_MAX_TOKENS=64` is the wrapper's default; `=0` disables graphs everywhere.
3. `--vram-margin 2.0` was tuned for a 24 GB card; MI50 32 GB may want different headroom.
4. gfx906 exp-mode defaults for these kernels (mode 1 is "gfx906's only" per `iq_kernels.cu`);
   first t/s number should be compared against the CPU-experts fallback (`--experts cpu`).
5. `--lora-exps` on IQ3_S experts - LoRA application path per type unverified.
6. Census file intentionally omitted (authors' copy is from their model copy); regenerate later.

## Rollback

Everything here is new files only (draft config + wrapper + this doc); the default path is
untouched. Deleting the config returns the box to its previous state.
