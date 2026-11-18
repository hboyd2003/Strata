# third_party/gfx906-mmq: llama.cpp's MMQ prompt kernels for the gfx906 fork

`ggml-cuda/` and `ggml-cuda-tq/` are the two folders upstream Strata vendored at `third_party/strata-ggml/ggml/src/` (now under engine/) until 0.2.16
removed them with the MMQ build path (3e26b4e15, own-stack step 10). They are kept here, unchanged, for the HIP gfx906 build only
(cmake/backends/gfx906_mmq.cmake): upstream's own prompt kernels need tensor-core / WMMA instructions that gfx906 does not have.

- Source: https://github.com/ggml-org/llama.cpp, the `ggml/src/ggml-cuda` folder at the pin of `third_party/strata-ggml` (3cf03257f219afbe7334045ff7c6a06ac68c627d).
  `ggml-cuda-tq/` is Strata's patched copy of a few of those headers plus the trellis instances (see the 0.2.15 `third_party/strata-ggml/UPSTREAM.md`).
- One edit after the move: `ggml-cuda-tq/tq.cuh` names its two Strata headers by a path from this folder (two levels less deep).
- Licence: MIT, `LICENSE` beside this file (llama.cpp's).
- ggml.h / ggml-common.h and ggml-base still come from `third_party/strata-ggml`.
