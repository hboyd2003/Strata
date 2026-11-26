# `gfx906/local`

Build and image settings for the gfx906 Docker image, plus the code that a former merge commit carried.
Stacked: its commits sit on `main` at its merge slot.

## What it does

- **Native CPU build by default.** `STRATA_PORTABLE` is a build argument, `OFF` by default: the engine's CPU code and the
  picture readers get `-march=native` (AVX-512 VNNI on a Zen 4) instead of the AVX2 baseline. It is meant for the CPU
  expert path; the 10-22 measurement showed no measurable decode gain from it. The image then only runs on CPUs with
  the build host's instruction set;
  `--build-arg STRATA_PORTABLE=ON` restores the portable build. `BUILD.json` in the image records which one it is.
- **ROCm's `libomp`** is linked into `/usr/local/lib` for the native picture reader (it lives in `/opt/rocm/llvm/lib`,
  outside the loader's path).
- **Carried over from an old upstream merge:** the repacked prompt GEMM skips WMMA layers (no WMMA on gfx906, so
  nothing changes here).
- **Receipt** (`benchmarks/gfx906/receipts/2026-10-22-janus-prefill-port.json`) of the janus-reith prefill port
  (prefill 504 → 677 tok/s against the v0.2.2 image, one run per arm; 504 → 624 of it comes from three prompt switches
  alone).

## Building the image

```bash
docker build -f tools/docker/gfx906/Dockerfile --build-arg STRATA_COMMIT=$(git rev-parse HEAD) -t strata-gfx906:local .
```

It builds on `mixa3607/rocm-gfx906:10.1-complete` (pinned by digest), which ships rocBLAS/hipBLAS for gfx906.

- The entrypoint is the API server (`python3 -m serve.server` under `tini`), serving `/config/active.json` on port 8080.
  Nothing is downloaded at start.
- `/usr/local/bin/strata` is the engine; `strata-vision` (CPU) and `strata-vision-hip` (gfx906) read pictures; the
  engine's tools sit beside them.
- Volumes: `/models`, `/data`, `/config`, `/cache`. A `/health` healthcheck allows 10 minutes for the model to load.
- Auto sessions (`"auto_sessions": true`) need a writable directory as well, for example
  `-v ./data/strata/sessions:/sessions` with `"slot_save_path": "/sessions"` in the config.
