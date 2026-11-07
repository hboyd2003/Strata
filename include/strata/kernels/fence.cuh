// One-sided memory fences for device code: strata_hip.h's on gfx906, a full __threadfence elsewhere (same guarantees).
#pragma once
#if defined(STRATA_HIP_GFX906)
#define STRATA_FENCE_RELEASE() strata_fence_release()
#define STRATA_FENCE_ACQUIRE() strata_fence_acquire()
#define STRATA_FENCE_SYSTEM_RELEASE() strata_fence_system_release()
#else
#define STRATA_FENCE_RELEASE() __threadfence()
#define STRATA_FENCE_ACQUIRE() __threadfence()
#define STRATA_FENCE_SYSTEM_RELEASE() __threadfence_system()
#endif
