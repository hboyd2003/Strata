# Strata for AMD gfx906 (Instinct MI50 / MI60, Radeon VII), built FROM SOURCE on mixa3607's ROCm 10.1 gfx906 image
# (https://github.com/mixa3607/ML-gfx906: TheRock ROCm with the gfx906 rocBLAS / hipBLAS kernels that AMD's own releases no
# longer ship). The context is the repository root:
#
#   docker build -f docker/gfx906/Dockerfile --build-arg STRATA_COMMIT=$(git rev-parse HEAD) -t strata-gfx906:local .
#
# The image keeps the layout of the fork's published images (cassettesgoboom/strata-gfx906): the API server is the
# entrypoint and starts the engine from a prepared config, nothing is downloaded or set up at start.
#
#   /usr/local/bin/strata            the engine (STRATA_HIP_GFX906=ON), the "exe" of the config
#   /usr/local/bin/strata-vision     the picture reader on the CPU; strata-vision-hip on the GPU
#   /usr/local/bin/strata-*          the engine's command-line tools (strata-gguf, strata-plan, strata-dequant, ...)
#   /opt/strata                      the server, the tools (iq_pack, mtp_fetch / mtp_pack / mtp_rt, ...), data/ (draft
#                                    vocabularies, expert profiles), llama.cpp at the pinned commit (gguf-py), a venv
#   /models /data /config /cache     volumes: model files, prepared packs and MTP files, the server config, caches
#
#   docker run --device /dev/kfd --device /dev/dri --group-add video --ipc host --ulimit memlock=-1 -p 127.0.0.1:8080:8080 \
#     -v ./models:/models:ro -v ./data:/data:ro -v ./config:/config:ro strata-gfx906:local      # serves /config/active.json
#
# Model preparation runs from the same image with another entrypoint (sh, python3 tools/iq_pack.py ...); see
# docs/gfx906/CONTAINER.md. AB/CD is not in this source tree. Plain legacy-builder features only (no heredocs, no --mount).
#
# The CPU code is built for the CPU that builds the image (STRATA_PORTABLE=OFF: -march=native, AVX-512 on a Zen 4), so
# the image runs only on CPUs with that instruction set; the CPU computes the experts the GPU cache misses.
# --build-arg STRATA_PORTABLE=ON builds the AVX2 baseline instead, for other machines.

ARG ROCM_IMAGE=docker.io/mixa3607/rocm-gfx906:10.1-complete@sha256:b1d72163863a56b6b2bf498d159c1dc17627f8751aed6c8b8de2ebe9882972b4

# ---- 1. compile the engine, its tools and the picture readers
FROM ${ROCM_IMAGE} AS build
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends build-essential ca-certificates git python3 python3-venv \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY . .
# cmake and ninja at the versions requirements-build.txt pins (Ubuntu's are older)
RUN python3 -m venv /buildenv && /buildenv/bin/pip install --no-cache-dir -r requirements.txt -r requirements-build.txt
ENV PATH=/buildenv/bin:$PATH
# llama.cpp at the pinned commit (ggml for the engine, mtmd for pictures, gguf-py for the tools), as setup.py fetches it
RUN python3 -c "import setup; print(setup.get_llama_cpp())"
# extra HIP compiler flags for the engine and the GPU picture reader (empty: none), e.g. "-mllvm -amdgpu-sched-strategy=max-ilp"
ARG CMAKE_HIP_FLAGS=
ARG STRATA_PORTABLE=OFF
ARG ENGINE_TOOLS="strata-concurrent strata-dequant strata-device strata-gguf strata-load strata-overlap strata-plan strata-q36ref"
RUN set -e; \
    cmake -G Ninja -S . -B build -DCMAKE_BUILD_TYPE=Release \
        -DSTRATA_HIP_GFX906=ON -DCMAKE_HIP_ARCHITECTURES=gfx906 "-DCMAKE_HIP_FLAGS=${CMAKE_HIP_FLAGS}" \
        -DCMAKE_C_COMPILER="$ROCM_PATH/llvm/bin/clang" -DCMAKE_CXX_COMPILER="$ROCM_PATH/llvm/bin/clang++" \
        -DSTRATA_BUILD_TESTS=OFF -DSTRATA_PORTABLE=${STRATA_PORTABLE} -DSTRATA_GGML_DIR=/src/third_party/llama.cpp; \
    cmake --build build --target strata $ENGINE_TOOLS -j "$(nproc)"
RUN set -e; \
    cmake -G Ninja -S tools/vision -B build-vision -DCMAKE_BUILD_TYPE=Release -DLLAMA_DIR=/src/third_party/llama.cpp \
        -DCMAKE_C_COMPILER="$ROCM_PATH/llvm/bin/clang" -DCMAKE_CXX_COMPILER="$ROCM_PATH/llvm/bin/clang++" \
        -DSTRATA_VISION_CUDA=OFF -DSTRATA_PORTABLE=${STRATA_PORTABLE}; \
    cmake --build build-vision --target strata-vision -j "$(nproc)"; \
    cmake -G Ninja -S tools/vision -B build-vision-hip -DCMAKE_BUILD_TYPE=Release -DLLAMA_DIR=/src/third_party/llama.cpp \
        -DCMAKE_C_COMPILER="$ROCM_PATH/llvm/bin/clang" -DCMAKE_CXX_COMPILER="$ROCM_PATH/llvm/bin/clang++" \
        -DCMAKE_HIP_COMPILER="$ROCM_PATH/llvm/bin/clang++" -DCMAKE_HIP_COMPILER_ROCM_ROOT="$ROCM_PATH" \
        -DCMAKE_PREFIX_PATH="$ROCM_PATH" -DCMAKE_HIP_ARCHITECTURES=gfx906 "-DCMAKE_HIP_FLAGS=${CMAKE_HIP_FLAGS}" \
        -DSTRATA_VISION_HIP=ON -DSTRATA_VISION_HIP_ARCHS=gfx906 -DSTRATA_PORTABLE=${STRATA_PORTABLE}; \
    cmake --build build-vision-hip --target strata-vision -j "$(nproc)"
# /out/bin is what goes to /usr/local/bin; BUILD.json beside the engine says what it is (setup.py and the tools read it)
RUN set -e; \
    mkdir -p /out/bin; \
    cp build/strata /out/bin/; \
    for t in $ENGINE_TOOLS; do cp "build/$t" /out/bin/; done; \
    cp build-vision/bin/strata-vision /out/bin/strata-vision; \
    cp build-vision-hip/bin/strata-vision-hip /out/bin/; \
    python3 -c "import json, setup; print(json.dumps({'source': 'docker build', 'backend': 'hip', 'version': setup.source_version(), 'archs': ['gfx906'], 'vision': 'gpu', 'vision_archs': ['gfx906'], 'vision_gpu': setup.HVEXE, 'vision_cpu': setup.VEXE, 'lib_dirs': ['/opt/rocm/lib'], 'portable': '${STRATA_PORTABLE}' == 'ON', 'src': setup.source_hash(setup.ENGINE_SOURCES), 'vision_src': setup.source_hash(setup.VISION_SOURCES)}, indent=1))" \
        > /out/bin/BUILD.json; \
    cat /out/bin/BUILD.json

# ---- 2. the runtime: the server, every tool and the data files, without the engine's sources, docs and benchmark results
FROM ${ROCM_IMAGE}
ARG STRATA_COMMIT=
ARG STRATA_VERSION=dev
ARG ROCM_IMAGE
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates python3 python3-venv tini \
    && rm -rf /var/lib/apt/lists/* /var/cache/apt/*
WORKDIR /opt/strata
COPY --from=build /src/setup.py /src/strata.py /src/chat.py /src/LICENSE /src/README.md /opt/strata/
COPY --from=build /src/requirements.txt /src/requirements-build.txt /src/requirements-monitor.txt \
     /src/requirements-nvfp4-local.txt /opt/strata/
COPY --from=build /src/serve /opt/strata/serve
COPY --from=build /src/tools /opt/strata/tools
COPY --from=build /src/data /opt/strata/data
COPY --from=build /src/ref /opt/strata/ref
COPY --from=build /src/packaging /opt/strata/packaging
COPY --from=build /src/strata_control /opt/strata/strata_control
COPY --from=build /src/strata_cli /opt/strata/strata_cli
COPY --from=build /src/third_party /opt/strata/third_party
COPY --from=build /src/benchmarks/gfx906 /opt/strata/benchmarks/gfx906
COPY --from=build /src/tests/gfx906 /opt/strata/tests/gfx906
COPY --from=build /out/bin/ /usr/local/bin/
# the server and tools, the monitor's terminal UI, and the NVFP4 checkpoint converter with PyTorch's CPU build (it runs on the
# CPU; the default CUDA wheels would add several GB this image cannot use)
RUN set -e; \
    python3 -m venv /opt/strata/venv; \
    /opt/strata/venv/bin/pip install --no-cache-dir --disable-pip-version-check -r requirements.txt -r requirements-monitor.txt; \
    torch=$(sed -n 's/^torch==\([^;[:space:]]*\);[[:space:]]*python_version[[:space:]]*<[[:space:]]*"3\.14".*/\1/p' requirements-nvfp4-local.txt); \
    /opt/strata/venv/bin/pip install --no-cache-dir --disable-pip-version-check --index-url https://download.pytorch.org/whl/cpu "torch==$torch"; \
    /opt/strata/venv/bin/pip install --no-cache-dir --disable-pip-version-check -r requirements-nvfp4-local.txt; \
    find /opt/strata -name __pycache__ -prune -exec rm -rf {} +
ENV PATH=/opt/strata/venv/bin:$PATH PYTHONPATH=/opt/strata PYTHONUNBUFFERED=1 \
    STRATA_GGUF_PY=/opt/strata/third_party/llama.cpp/gguf-py
# every binary's libraries resolve in this image (the ROCm runtime and its gfx906 kernels are the base image's), and the server,
# the preparation tools and the converter import
RUN set -e; \
    for f in /usr/local/bin/strata /usr/local/bin/strata-*; do \
      head -c4 "$f" | grep -q ELF || continue; \
      if ldd "$f" | grep 'not found'; then echo "MISSING LIBRARIES: $f"; exit 1; fi; \
    done; \
    python3 -c "import serve.server, serve.prometheus, textual, torch, transformers, safetensors; print('imports ok, torch', torch.__version__)"; \
    for t in iq_pack mtp_fetch mtp_pack mtp_rt nvfp4_local_pack; do python3 tools/$t.py --help > /dev/null; done; \
    python3 setup.py --help > /dev/null; \
    python3 tools/gfx906.py --help > /dev/null
VOLUME ["/models", "/data", "/config", "/cache"]
EXPOSE 8080
LABEL org.opencontainers.image.title="Strata gfx906" \
      org.opencontainers.image.description="Strata for AMD gfx906 (MI50 / MI60 / Radeon VII) on ROCm 10.1 (mixa3607/ML-gfx906); serial and batch slots, no AB/CD" \
      org.opencontainers.image.source="https://github.com/cassettesgoboom/strata-gfx906" \
      org.opencontainers.image.revision="${STRATA_COMMIT}" \
      org.opencontainers.image.version="${STRATA_VERSION}" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.base.name="${ROCM_IMAGE}"
# /health answers before the API key check; the port opens once the model is loaded (minutes)
HEALTHCHECK --interval=30s --timeout=5s --start-period=10m --retries=3 \
    CMD ["/opt/strata/venv/bin/python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/health', timeout=4)"]
STOPSIGNAL SIGTERM
ENTRYPOINT ["/usr/bin/tini", "--", "python3", "-m", "serve.server"]
CMD ["--engine", "strata", "--config", "/config/active.json", "--host", "0.0.0.0", "--port", "8080"]
