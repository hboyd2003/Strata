# `gfx906/own-mmq`

Keeps llama.cpp's MMQ prompt kernels for the gfx906 build. Based on upstream `main`; the first branch merged into the fork's `main`, and `rebased/janus-reith` sits on it.

## Why

Upstream 0.2.16 removed llama.cpp's MMQ build path and the vendored ggml-cuda tree. Its own prompt kernels are built on `mma.sync` and gfx11/gfx12 WMMA, which gfx906 does not have, so on the MI50 they skip and prompts would be read on the FP16 path.

## What the branch changes

- `engine/third_party/gfx906-mmq/` holds the two ggml-cuda folders as they were before the removal (llama.cpp, MIT; licence and a README beside them).
- The MoE MMQ prompt sources (`moe_mmq.cu`, `moe_iq4_w4a8.cu`, the NVFP4 MMQ files, `ggml_cuda_host.cu`) and their headers are restored unchanged.
- `engine/cmake/backends/gfx906_mmq.cmake` adds the CMake option `STRATA_MMQ` (default on under `STRATA_HIP_GFX906`, refused on every other backend) and builds only what the gfx906 build compiled before.
- A build-file test checks that the fork's MMQ stays gfx906-only.

## Switches

| Switch | What | Production | Output |
| --- | --- | --- | --- |
| `-DSTRATA_MMQ=ON` (CMake) | build the MMQ prompt kernels | on (the default on gfx906) | the fork's prompt path as before 0.2.16 |

## Measured (MI50, Qwen3.8-Flash-Next IQ3_S, `terminal/bench/gfx906/ab.py`)

Stock upstream 0.2.16 without MMQ against the fork's production build, at equal expert slots (64K context, 13,013 vs 12,876 slots):

| | upstream 0.2.16 | fork |
| --- | ---: | ---: |
| prefill ~200 tok/s | 160 | 240 |
| prefill ~4K tok/s | 517 | 742 |
| prefill ~16K tok/s | 579 | 830 |
| ~32K read tok/s | 570 | 803 |
| follow-up turn tok/s | 444 | 595 |

- That is about 30% slower prompt reading without the fork. The gap also includes the fork's other prompt changes (the rocBLAS table, prompt attention), so MMQ and the janus-reith stack are most of it, not all.
- Decode was 29% lower too, but that comes from the fork's decode branches, not from MMQ.
