# syntax=docker/dockerfile:1.7
# Strata (Qwen3.8-Flash-Next engine) for gfx906 (MI50 / MI60 / Radeon VII), built from
# this checkout. Build context is the repository root:
#
#   docker build -f docker/gfx906/Dockerfile -t strata-gfx906 .
#
# The recipe is upstream's own (docs/AMD_HIP.md, "gfx906 ... built from source"):
# -DSTRATA_HIP_GFX906=ON on mixa3607/rocm-gfx906 (HIP 7.14 with gfx906 rocBLAS),
# llama.cpp at setup.py's LLAMA_CPP_COMMIT for ggml / MMQ / mtmd, and the CPU
# image encoder from tools/vision (setup.py's build_vision_cpu). docker build has
# no GPU and needs none: the engine is compiled for gfx906 only.
#
# The image encoder is portable (AVX2 baseline) by default, so an image built on
# one machine (e.g. CI) runs on another; VISION_PORTABLE=OFF builds it with
# -march=native for the build machine, as setup.py does.

ARG ROCM_BASE=mixa3607/rocm-gfx906@sha256:622c025824d465faa0f5e59eea4f723845032011df273c3b11e25031965fc428

############
# llama.cpp at the commit setup.py pins
############
FROM ${ROCM_BASE} AS llama
ARG LLAMA_CPP_COMMIT=3cf03257f219afbe7334045ff7c6a06ac68c627d
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN git init -q /src/llama.cpp && cd /src/llama.cpp \
    && git fetch -q --depth 1 https://github.com/ggml-org/llama.cpp "${LLAMA_CPP_COMMIT}" \
    && git checkout -q FETCH_HEAD && rm -rf .git

############
# Build stage
############
FROM ${ROCM_BASE} AS build
ARG VISION_PORTABLE=ON
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        cmake ninja-build build-essential git python3 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=llama /src/llama.cpp /src/llama.cpp
COPY . /src/strata
WORKDIR /src/strata

# Engine: upstream's documented gfx906 configure line.
RUN cmake -S . -B build-906 -G Ninja -DCMAKE_BUILD_TYPE=Release \
        -DSTRATA_HIP_GFX906=ON -DCMAKE_HIP_ARCHITECTURES=gfx906 \
        -DSTRATA_BUILD_TESTS=OFF \
        -DSTRATA_GGML_DIR=/src/llama.cpp \
        -DCMAKE_C_COMPILER=/opt/rocm/llvm/bin/clang \
        -DCMAKE_CXX_COMPILER=/opt/rocm/llvm/bin/clang++ \
    && ninja -C build-906 strata

# CPU image encoder, as setup.py builds it for AMD (--vision cpu).
RUN cmake -S tools/vision -B build-vision -G Ninja -DCMAKE_BUILD_TYPE=Release \
        -DLLAMA_DIR=/src/llama.cpp -DSTRATA_VISION_CUDA=OFF -DSTRATA_PORTABLE=${VISION_PORTABLE} \
    && ninja -C build-vision strata-vision

# Optional GPU parity/bench tests (e.g. TEST_TARGETS="native_grouped_parity"), in a build
# dir of their own so the engine above is configured exactly as without them.
ARG TEST_TARGETS=""
RUN mkdir -p /out/tests && if [ -n "$TEST_TARGETS" ]; then \
        cmake -S . -B build-906-tests -G Ninja -DCMAKE_BUILD_TYPE=Release \
            -DSTRATA_HIP_GFX906=ON -DCMAKE_HIP_ARCHITECTURES=gfx906 -DSTRATA_BUILD_TESTS=ON \
            -DSTRATA_GGML_DIR=/src/llama.cpp \
            -DCMAKE_C_COMPILER=/opt/rocm/llvm/bin/clang -DCMAKE_CXX_COMPILER=/opt/rocm/llvm/bin/clang++ \
        && ninja -C build-906-tests $TEST_TARGETS \
        && for t in $TEST_TARGETS; do cp "$(find build-906-tests -type f -name "$t" -perm -u+x | head -1)" /out/tests/; done; \
    fi

RUN mkdir -p /out && cp build-906/strata build-vision/bin/strata-vision /out/ \
    && sha256sum /out/strata /out/strata-vision > /out/SHA256SUMS

##############
# Runtime stage
##############
FROM ${ROCM_BASE}
ARG SRC_REV=unknown
ARG SOURCE_URL=https://github.com/Niko1221/Strata
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 \
    PATH=/opt/strata/venv/bin:/usr/local/bin:/opt/rocm/bin:/usr/local/sbin:/usr/sbin:/usr/bin:/sbin:/bin \
    PYTHONPATH=/opt/strata \
    STRATA_GGUF_PY=/opt/strata/third_party/llama.cpp/gguf-py
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-venv libgomp1 libatomic1 \
    && rm -rf /var/lib/apt/lists/*

# Python side: server, prep tools (iq_pack, mtp_*), data (expert profiles).
COPY requirements.txt /opt/strata/requirements.txt
RUN python3 -m venv /opt/strata/venv \
    && /opt/strata/venv/bin/pip install --no-cache-dir -r /opt/strata/requirements.txt
COPY serve /opt/strata/serve
COPY tools /opt/strata/tools
COPY data /opt/strata/data
COPY setup.py chat.py /opt/strata/
COPY --from=llama /src/llama.cpp/gguf-py /opt/strata/third_party/llama.cpp/gguf-py
COPY --from=build /out/strata /out/strata-vision /usr/local/bin/
COPY --from=build /out/SHA256SUMS /usr/share/strata/SHA256SUMS
COPY --from=build /out/tests /opt/strata/tests-bin

LABEL org.opencontainers.image.title="Strata gfx906" \
      org.opencontainers.image.description="Strata (Qwen3.8-Flash-Next engine) built for AMD gfx906 (MI50/MI60/Radeon VII)" \
      org.opencontainers.image.source="${SOURCE_URL}" \
      org.opencontainers.image.revision="${SRC_REV}" \
      org.opencontainers.image.licenses="MIT"

WORKDIR /opt/strata
VOLUME ["/models", "/data", "/config"]
EXPOSE 8080
ENTRYPOINT ["python3", "-m", "serve.server"]
CMD ["--engine", "strata", "--config", "/config/active.json", "--host", "0.0.0.0", "--port", "8080"]
