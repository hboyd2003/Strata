# cmake/backends/gfx906_mmq.cmake - gfx906 fork: llama.cpp's MMQ prompt kernels, kept for the HIP gfx906 build only.
#
# Upstream 0.2.16 removed llama.cpp's MMQ build path and the vendored ggml-cuda (3e26b4e15, own-stack step 10): its own prompt kernels
# (moe_fused*.cu) are built on NVIDIA mma.sync / gfx11-12 WMMA and do not run on gfx906, so a gfx906 build without MMQ reads prompts on
# the FP16 path (~30% slower prefill on the MI50, A/B 20261118-up0216-64k). This file is the old strata_cuda_mmq macro cut down to what the
# gfx906 build compiled (vendored ggml, trellis overlay on, the K-quant instances in strata_mmq, no pointer table, no libstrata_kq, no
# weight-stationary experts). The ggml-cuda sources live in third_party/gfx906-mmq (the llama.cpp pin of third_party/strata-ggml, MIT).
# The engine side is unchanged upstream: moe_mmq.hpp and the STRATA_PREFILL_MMQ paths of the prompt code are still there; mmq_absent.cpp
# (the no-MMQ stubs) is simply not compiled when this macro runs.

option(STRATA_MMQ "gfx906 fork: build llama.cpp's MMQ prompt kernels (third_party/gfx906-mmq); HIP gfx906 builds only" ${STRATA_HIP_GFX906})
if(STRATA_MMQ AND NOT STRATA_HIP_GFX906)
  message(FATAL_ERROR "STRATA_MMQ (the gfx906 fork's MMQ) builds only with STRATA_HIP_GFX906=ON")
endif()

# strata_pf_own (Strata's own prompt kernels) with strata_mmq's include directories (in their order, the trellis overlay's
# first), definitions and options: the same command line as the files had when they were part of strata_mmq
macro(strata_pf_own_mirror)
  foreach(_strata_p INCLUDE_DIRECTORIES COMPILE_DEFINITIONS COMPILE_OPTIONS)
    get_target_property(_strata_v strata_mmq ${_strata_p})
    if(_strata_v)
      set_property(TARGET strata_pf_own PROPERTY ${_strata_p} "${_strata_v}")
    endif()
  endforeach()
  target_include_directories(strata_pf_own INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
endmacro()

macro(strata_gfx906_mmq)
    set(_strata_gg ${_strata_ggml_root}/ggml)                                      # ggml.h, ggml-common.h, ggml-base
    set(_strata_mmq ${CMAKE_CURRENT_SOURCE_DIR}/third_party/gfx906-mmq/ggml-cuda)
    set(_strata_tq_overlay_dir ${CMAKE_CURRENT_SOURCE_DIR}/third_party/gfx906-mmq/ggml-cuda-tq)   # patched mmq.cuh etc., first on the path
    set(_strata_mmq_types q2_0 iq2_xxs iq2_xs iq2_s iq3_xxs iq3_s iq4_nl iq4_xs q8_0
                          q4_k q5_k q5_1 q6_k)   # + the K-quants: the dense GGUF projections (STRATA_DENSE_MMQ) and K-quant experts
    set(_strata_mmq_sources src/prefill/moe_mmq.cu src/prefill/moe_iq4_w4a8.cu src/prefill/rp_moe.cu src/prefill/mmq_nvfp4_w4a8.cu
                            src/prefill/ggml_cuda_host.cu ${_strata_tq_overlay_dir}/quantize.cu)
    foreach(_t IN LISTS _strata_mmq_types)
      list(APPEND _strata_mmq_sources ${_strata_mmq}/template-instances/mmq-instance-${_t}.cu)
    endforeach()
    foreach(_t tq2_t tqk6 tqk7 tqp5 tqp6 tqp7)
      list(APPEND _strata_mmq_sources ${_strata_tq_overlay_dir}/template-instances/mmq-instance-${_t}.cu)
    endforeach()
    # upstream 0.2.18+: the Blackwell FP4 / FP8 prompt kernels and the hardware FP4 decode sit in strata_pf_fp4, which strata_kernels
    # and strata_pf_own link; strata_cuda_pf_own builds only their _absent stubs under STRATA_HIP_GFX906, so this build does the same
    add_library(strata_pf_fp4 STATIC src/prefill/moe_fp4_tc_absent.cpp src/prefill/dense_fp4_tc_absent.cpp src/prefill/moe_fp8_mxf_tc_absent.cpp
                src/kernels/cuda/nvfp4_decode_sm120_absent.cpp)
    target_include_directories(strata_pf_fp4 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/third_party/ggml PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_link_libraries(strata_pf_fp4 PUBLIC strata_quant ${_strata_gpu_runtime_target})
    target_link_libraries(strata_kernels PUBLIC strata_pf_fp4)
    add_library(strata_mmq STATIC ${_strata_mmq_sources})
    # Strata's own prompt kernels: built as on every other card (they report themselves unavailable on gfx906), without mmq_absent.cpp
    set(_strata_pf_own_sources src/prefill/moe_fused.cu src/prefill/moe_fused_iq.cu src/prefill/moe_helpers.cu
                               src/prefill/moe_fused_iq_q36.cu src/prefill/moe_f16_kq.cu src/prefill/moe_p1a.cu)
    add_library(strata_pf_own STATIC ${_strata_pf_own_sources})
    # NVFP4 W4A4 (RTX 50's FP4 MMA): only reports itself unavailable here; moe_mmq.cu links it
    add_library(strata_mmq_w4a4 STATIC src/prefill/mmq_nvfp4_w4a4.cu)
    target_include_directories(strata_mmq_w4a4 PRIVATE ${_strata_gg}/include ${_strata_gg}/src ${_strata_mmq})
    target_link_libraries(strata_mmq_w4a4 PUBLIC ggml-base ${_strata_gpu_runtime_target})
    target_link_libraries(strata_mmq PRIVATE strata_mmq_w4a4)
    target_include_directories(strata_mmq BEFORE PRIVATE ${_strata_tq_overlay_dir} ${_strata_tq_overlay_dir}/include)
    target_include_directories(strata_mmq PRIVATE ${_strata_gg}/include ${_strata_gg}/src ${_strata_mmq}
                               PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_link_libraries(strata_mmq PUBLIC ggml-base ${_strata_gpu_runtime_target})
    target_compile_definitions(strata_mmq PRIVATE STRATA_MMQ_KQUANTS=1 STRATA_MMQ_TQ=1 STRATA_MMQ_TQP=1)
    strata_pf_own_mirror()
    target_link_libraries(strata_pf_own PUBLIC strata_pf_fp4 strata_quant ggml-base ${_strata_gpu_runtime_target})
    target_link_libraries(strata_mmq PUBLIC strata_pf_own)
    target_link_libraries(strata_prefill PUBLIC strata_mmq)
    target_compile_definitions(strata_prefill PUBLIC STRATA_PREFILL_MMQ=1 STRATA_PREFILL_FUSED=1)
    set(_strata_pf_tests_lib strata_mmq)
    # HIP gfx906 compiles every .cu as HIP: src/ is globbed (strata_hip_gfx906_sources), the ggml-cuda files are named here
    set_source_files_properties(${_strata_mmq_sources} src/prefill/mmq_nvfp4_w4a4.cu ${_strata_pf_own_sources} PROPERTIES LANGUAGE HIP)
    foreach(_strata_t strata_mmq strata_pf_own strata_mmq_w4a4)
      target_compile_definitions(${_strata_t} PRIVATE GGML_USE_HIP)
    endforeach()
    message(STATUS "Strata (gfx906 fork): llama.cpp's MMQ prompt kernels from third_party/gfx906-mmq")
    # gfx906: mx-llama.cpp's repacked IQ4_NL expert GEMM (rp_moe) against MMQ, correctness and time
    add_executable(rp_expert_bench src/prefill/rp_expert_bench.cpp)
    target_link_libraries(rp_expert_bench PRIVATE strata_mmq ggml-base CUDA::cudart)
    # gfx906: the dense projections as today (dequant + hipBLAS FP16) against int8 MMQ and the repacked GEMM
    add_executable(dense_mmq_bench src/prefill/dense_mmq_bench.cu)
    target_link_libraries(dense_mmq_bench PRIVATE strata_prefill strata_mmq ggml-base ggml-cpu CUDA::cudart CUDA::cublas)
endmacro()
