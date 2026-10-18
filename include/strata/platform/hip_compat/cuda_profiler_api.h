// cuda_profiler_api.h for a STRATA_HIP_GFX906 build (docs/AMD_HIP.md gfx906).  The only use in this tree is the
// profiler toggle in tools/ds4/ds4_moe.cpp (cudaProfilerStart/Stop around a capture); HIP's own equivalents are
// deprecated but present (hip/hip_runtime_api.h: "use roctracer/rocTX instead"), so the names map to them.
#pragma once
#include "strata_hip.h"

#define cudaProfilerStart hipProfilerStart
#define cudaProfilerStop hipProfilerStop
