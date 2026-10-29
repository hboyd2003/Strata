#pragma once
#include <cstdint>

namespace strata::kernels {
// Configure before session capture. Existing graphs retain their selected kernels.
// This experiment is off by default and does not modify the legacy router.
void native_router_set_enabled(bool enabled);
bool native_router_enabled();

// Pinned CUDA topk-moe contract for ONE token, 512 experts, top 10, softmax,
// no selection bias, lower normalization clamp 2^-14, and scale 1.
// Reads 512 finite F32 logits; writes 10 I32 IDs and 10 F32 weights. Equal
// computed probabilities select the lower expert index. All spans must be
// four-byte aligned and outputs disjoint from each other and the input.
// Requires a nonnull ordered CUDA stream. No allocation or synchronization.
void native_router_top10(const float* logits, int32_t* ids, float* weights, void* stream);
/// n_tok rows at once (logits [n,512], ids/weights [n,10]); each row exactly as the single call.
void native_router_top10_multi(const float* logits, int32_t* ids, float* weights, int n_tok, void* stream);
/// gfx906: native_router_top10_multi's rows as one 64-lane wave per token, bitwise (STRATA_GFX906_ROUTER_W64=1 routes
/// the two calls above here).  false (nothing launched) on other builds.
bool native_router_top10_w64(const float* logits, int32_t* ids, float* weights, int n_tok, void* stream);
/// doubr's route_wave64 (VlDubr/strata-mi50 c3d0ca379c), the same contract: a bench / parity arm.
bool native_router_top10_w64_doubr(const float* logits, int32_t* ids, float* weights, int n_tok, void* stream);
/// gfx906 STRATA_GFX906_ROUTER_FUSED=1: the 512 x 2560 BF16 router GEMV (x [n][2560] -> logits [n][512]), the top-10
/// (ids / weights [n][10]) and doorbell_publish_res (ids_out, x_out on a miss, then *seq += 1) in ONE launch, every
/// output bitwise the three calls'.  counter: one device uint32 that starts at zero (the kernel leaves it at zero).
/// false (nothing launched) when off, not built, n outside 1..4, or x / x_out not 16-byte aligned.
struct RouterFused906 {
    const float* x = nullptr;
    const uint16_t* w = nullptr;
    float* logits = nullptr;
    int32_t* ids = nullptr;
    float* weights = nullptr;
    const int32_t* d_res = nullptr;   // the layer's residency table (nullptr: every expert a miss, as the doorbell)
    int n_expert = 512;
    float* x_out = nullptr;
    int32_t* ids_out = nullptr;
    uint32_t* seq = nullptr;
    uint32_t* counter = nullptr;
};
bool router_fused_906_enabled();
bool router_fused_906(const RouterFused906& args, int n_tok, void* stream);
/// The same contract for a model family's (experts, top-k): 512 / 10 (Flash-Next: the two calls above, unchanged) and
/// 256 / 8 (Qwen3.6-35B-A3B).  logits [n_tok, n_expert], ids / weights [n_tok, k].  CUDA / HIP.
bool native_router_topk_supported(int n_expert, int k);
void native_router_topk(const float* logits, int32_t* ids, float* weights, int n_expert, int k, int n_tok, void* stream);
/// STRATA_ROUTE_RESIDENT: swap non-resident tail-rank experts for resident ones within `margin` logits (see the .cu).
/// `stats` is nullptr or 4 device uint64 counters: tail entries seen, swaps, non-resident before, non-resident after.
void native_route_resident(const float* logits, int32_t* ids, float* weights, const int32_t* res_layer, int n_tok, float margin,
                           int rank_lo, int rank_hi, unsigned long long* stats, void* stream);
}
