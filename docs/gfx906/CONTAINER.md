# Archived gfx906 container

The experimental image is available at
`docker.io/cassettesgoboom/strata-gfx906@sha256:a7527bd8efe2dcdb0a0a7fd6b96761507ce79f9d585ce1649ce5b91158258a29`.
The [release record](PROVENANCE.md) identifies the native binary recorded during an archived digest pullback. This
source checkout has not been shown to rebuild that binary. The image contains no model weights; provide a compatible
GGUF, tokenizer/pack, and model-specific data from your own storage. Hardware memory needs depend on the model and
host.

## Run the published image

The published digest's OCI configuration sets the entrypoint to `python3 -m serve.server` and its default arguments
to `--engine strata --config /config/active.json --host 0.0.0.0 --port 8080`. The example below uses that default
command, mounting a prepared server config and model data from the host. It does not run the source tree's setup
entrypoint. The config's engine paths must refer to paths inside the container (for example, `/usr/local/bin/strata`
and files under `/models` or `/data`); use the arguments required by your model pack.

Set these values to absolute paths that exist on the Docker host. `DATA_DIR` must contain the pack/tokenizer files
referenced by your config. `CONFIG_FILE` must be a prepared Strata server JSON config whose container-side paths
match these mounts.

```bash
export GPU0_RENDER=/dev/dri/renderDXXX
export GPU1_RENDER=/dev/dri/renderDYYY
export MODEL_DIR=/path/to/gguf-files
export DATA_DIR=/path/to/model-pack-and-tokenizer
export CONFIG_FILE=/path/to/config.json

docker run --rm --name strata-gfx906-exp \
  --device=/dev/kfd --device="$GPU0_RENDER" --device="$GPU1_RENDER" \
  --ulimit memlock=-1 \
  -p 127.0.0.1:8080:8080 \
  --mount "type=bind,src=$MODEL_DIR,dst=/models,readonly" \
  --mount "type=bind,src=$DATA_DIR,dst=/data,readonly" \
  --mount "type=bind,src=$CONFIG_FILE,dst=/config/active.json,readonly" \
  docker.io/cassettesgoboom/strata-gfx906@sha256:a7527bd8efe2dcdb0a0a7fd6b96761507ce79f9d585ce1649ce5b91158258a29
```

This example keeps the mounted config and model data read-only, binds the API to localhost, and uses the published
image's default server command. The `/dev/dri/renderDXXX` names depend on the host; map the render nodes belonging
to the intended GPUs. The archived smoke used external config and model data with a read-only `/data` mount, but did
not establish compatibility with every model, ROCm installation, or host. Validate those requirements on the target
system. If a model config needs writable runtime data, provide an explicit writable mount for that path.

## Build the gfx906 image from this source (ROCm 10.1)

[`docker/gfx906/Dockerfile`](../../docker/gfx906/Dockerfile) builds the engine from this checkout on
[mixa3607/ML-gfx906](https://github.com/mixa3607/ML-gfx906)'s `docker.io/mixa3607/rocm-gfx906:10.1-complete` image (TheRock
ROCm 10.1 with gfx906 rocBLAS/hipBLAS; HIP 7.16, AMD clang 24), pinned by digest. It keeps the published image's layout,
so a config prepared for that image runs unchanged:

- the entrypoint is the API server (`python3 -m serve.server`, under `tini`), and the default command serves
  `/config/active.json`; nothing is downloaded or set up at start
- `/usr/local/bin/strata` is the engine; `strata-vision` (CPU) and `strata-vision-hip` (gfx906) read pictures; the
  engine's tools are beside them (`strata-gguf`, `strata-plan`, `strata-dequant`, `strata-device`, `strata-load`,
  `strata-overlap`, `strata-concurrent`, `strata-q36ref`), with `BUILD.json`
- `/opt/strata` holds the server, `setup.py`, every script in `tools/` (`iq_pack.py`, `mtp_fetch.py`, `mtp_pack.py`,
  `mtp_rt.py`, `gfx906.py`, ...), `data/` (draft vocabularies, expert profiles), llama.cpp at the pinned commit
  (`STRATA_GGUF_PY` points at its `gguf-py`), and a venv with the server, monitor and NVFP4-converter dependencies
  (PyTorch's CPU build)
- volumes `/models`, `/data`, `/config`, `/cache`; a `/health` healthcheck

The engine's sources, docs and benchmark results are not copied into the runtime stage. AB/CD is not part of this
source tree, so the image has no `STRATA_BATCH_ABCD` path.

```bash
docker build -f docker/gfx906/Dockerfile --build-arg STRATA_COMMIT=$(git rev-parse HEAD) -t strata-gfx906:local .
```

`--build-arg CMAKE_HIP_FLAGS="-mllvm -amdgpu-sched-strategy=max-ilp"` passes extra HIP flags to the engine and the GPU
picture reader (mixa3607's build uses this one; it has not been measured here).

The CPU code is built for the CPU that builds the image (`STRATA_PORTABLE=OFF`, the default: `-march=native`, AVX-512
on a Zen 4). The CPU computes every expert the GPU cache misses, and the portable build ran ggml's AVX2 kernels for
them. Such an image runs only on CPUs with the build host's instruction set; `--build-arg STRATA_PORTABLE=ON` builds the
AVX2 baseline for other machines. A native picture reader uses OpenMP, so the image links ROCm's `libomp.so` into
`/usr/local/lib`.

The gfx906 prompt path ported from [janus-reith/strata-gfx906](https://github.com/janus-reith/strata-gfx906) is on by
default in this build: `STRATA_PF_RP` (repacked expert GEMM), `STRATA_DENSE_MMQ` (dense projections in int8, K/V kept
in FP16 by `STRATA_DENSE_MMQ_SKIP`), `STRATA_MMQ_KQ`, `STRATA_DENSE_RP` and `STRATA_HIP_PROMPT_F16`. Set one to `0` in
the config's `env` to turn that part off. See [UPSTREAM-DELTA.md](UPSTREAM-DELTA.md#janus-reith-prefill-port-2026-10-22).

To move a compose setup from the archived image:

- change the image tag;
- hide any non-gfx906 GPU: since upstream #1728 the engine refuses to start when it sees a wave32 card next to the
  MI50 (an integrated Radeon included). Set `HIP_VISIBLE_DEVICES` to the MI50s, for example `"0"`;
- replace `/opt/rocm/core-7.14/lib` in the config's `lib_dirs` with `/opt/rocm/lib` (the 10.1 base has no 7.14 tree).

A short single-MI50 smoke of this image with a 155K-context Swift IQ3_S config (pack built by the v0.1.41 tools) is
recorded in the [receipt](../../benchmarks/gfx906/receipts/2026-10-22-rocm10.1-image-smoke.json).

Conversation disk parking (`auto_sessions`, see [UPSTREAM-DELTA.md](UPSTREAM-DELTA.md#disk-parking-for-conversations-2026-10-23))
needs a writable directory beside the read-only mounts, for example `-v ./data/strata/sessions:/sessions` with
`"slot_save_path": "/sessions"` and `"auto_sessions": true` in the config.

## Source image distinction

Upstream's own images ([`docker/Dockerfile`](../../docker/Dockerfile), and the root `Dockerfile` for CUDA) use a
different entrypoint: `docker/entrypoint.sh` runs `strata launch` from settings such as `MODEL`, `CONTEXT` and
`MODEL_PATH`, keeps everything in one `/data` volume, and installs the engine as a component package. Their
`/usr/local/bin/strata` is the Terminal's CLI, not the engine. A server config written for the gfx906 layout above does
not run there, and the published-digest command must not be copied to those images.

AB/CD remains experimental and disabled by default. For a separately validated four-slot batch-MTP config on a
two-GPU layer split, `-e STRATA_BATCH_ABCD=1` opts in on the archived image only. Exact native-token and
recurrent-state parity remain open; the current source checkout has not reproduced that binary.
