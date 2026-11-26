# `gfx906/layout-0216`

Follows upstream's 0.2.16 repository layout (`engine/`, `terminal/`, `tools/docker/`) in the fork's own build and release tooling, and carries the build fixes later upstream code needs on gfx906. Stacked on `main`; the last branch merged.

## What the branch changes

- The gfx906 image build (`tools/docker/gfx906/Dockerfile`, `.dockerignore`), the release tooling and the benchmark harness's usage text name the new paths.
- The fork's release tooling reads the upstream version from `app/host/package.json`, and from `apps/host/package.json` for release tags older than 0.2.16.
- The Inkling-2-Mini engine is off by default on the gfx906 build. That engine needs nvcc or the wave32 HIP backend, so configuring it under `STRATA_HIP_GFX906` would stop the build. `-DSTRATA_BUILD_INKLING=ON` still asks for it.
- The MMQ prompt build also makes upstream's `strata_pf_fp4` stubs, as upstream's own prompt build does.
- Build fixes for upstream 0.2.20 and 0.2.21 code: three HIP name mappings, and the engine tools link `strata_kernels_cpu`.
- Upstream's serve tests are adjusted where they meet fork features: the internal-error fixture carries the client headers, and `auto_tls` provides what the fork's `_post` reads.

## Switches

| Switch | What | Production | Output |
| --- | --- | --- | --- |
| `-DSTRATA_BUILD_INKLING=ON` (CMake) | build the Inkling-2-Mini engine anyway | off | — |

No runtime switches; no speed change.
