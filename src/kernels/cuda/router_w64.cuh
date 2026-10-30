// src/kernels/cuda/router_w64.cuh - gfx906: the 512-expert top-10 as one 64-lane wave per token, bitwise
// native_router.cu's route / route_multi.  Shared by native_router.cu (route_w64, STRATA_GFX906_ROUTER_W64=1) and
// native_bf16.cu (the router GEMV with the top-10 and the doorbell in its last block, STRATA_GFX906_ROUTER_FUSED=1).
//
// Lane L = s + 32 h holds experts s + 32 (8 h + j), j = 0..7, i.e. route_multi lane s's values[8 h + j].
//   * the max is order-independent (fmaxf), done over order-preserving uint keys with DPP;
//   * the softmax sum keeps route_multi's order: lane s sums j = 0..7 serially, lane s + 32 continues it with
//     values 8..15, then the same xor butterfly over the 32 slots (ds_swizzle xor = __shfl_xor's partner);
//   * a rank's argmax is a total order (larger p, then the smaller id), so any reduction tree gives route_multi's
//     winner: a DPP max of the p keys, then the lowest id among the lanes that hold it;
//   * selected_sum accumulates per slot s = id & 31 in rank order and reduces with route_multi's butterfly.
#pragma once
#if defined(STRATA_HIP_GFX906)
#include <cuda_runtime.h>
#include <cfloat>
#include <cstdint>

namespace strata::kernels::w64 {
__device__ __forceinline__ uint32_t w64_key(float f) {   // order-preserving: a < b  <=>  key(a) < key(b) (no NaN)
    const uint32_t u = __float_as_uint(f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}
__device__ __forceinline__ float w64_unkey(uint32_t k) { return __uint_as_float((k & 0x80000000u) ? (k & 0x7fffffffu) : ~k); }
template <int CTRL, int ROW_MASK, int BANK_MASK>
__device__ __forceinline__ uint32_t w64_dpp(uint32_t v) {   // lanes outside the masks read 0 (the identity of a max)
    return (uint32_t) __builtin_amdgcn_update_dpp(0, (int) v, CTRL, ROW_MASK, BANK_MASK, false);
}
__device__ __forceinline__ uint32_t w64_max_u32(uint32_t v) {   // the wave's max, uniform
    v = max(v, w64_dpp<0xb1, 0xf, 0xf>(v));    // quad_perm [1,0,3,2]
    v = max(v, w64_dpp<0x4e, 0xf, 0xf>(v));    // quad_perm [2,3,0,1]
    v = max(v, w64_dpp<0x141, 0xf, 0xf>(v));   // row_half_mirror
    v = max(v, w64_dpp<0x140, 0xf, 0xf>(v));   // row_mirror: row maxima
    v = max(v, w64_dpp<0x142, 0xa, 0xf>(v));   // row_bcast15
    v = max(v, w64_dpp<0x143, 0xc, 0xf>(v));   // row_bcast31: lane 63 holds the max
    return (uint32_t) __builtin_amdgcn_readlane((int) v, 63);
}
template <int M>
__device__ __forceinline__ float w64_xor32(float v) {   // __shfl_xor(v, M, 32)'s value without an address (M < 32)
    return __int_as_float(__builtin_amdgcn_ds_swizzle(__float_as_int(v), 0x1f | (M << 10)));
}
__device__ __forceinline__ float w64_sum32(float v) {   // warp_sum's butterfly within each 32-lane half
    v += w64_xor32<16>(v);
    v += w64_xor32<8>(v);
    v += w64_xor32<4>(v);
    v += w64_xor32<2>(v);
    v += w64_xor32<1>(v);
    return v;
}
// one token's top-10 by the calling 64-lane wave (lane = threadIdx.x & 63); ids_sm (nullable): the 10 ids again
__device__ __forceinline__ void route_w64_token(const float* __restrict__ logits, int32_t* __restrict__ ids,
                                                float* __restrict__ weights, int32_t* ids_sm) {
    const int lane = (int) threadIdx.x & 63, s = lane & 31, h = lane >> 5;
    float v[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) v[j] = logits[s + 32 * (8 * h + j)];
    uint32_t km = 0;
#pragma unroll
    for (int j = 0; j < 8; ++j) km = max(km, w64_key(v[j]));
    const float maximum = w64_unkey(w64_max_u32(km));
#pragma unroll
    for (int j = 0; j < 8; ++j) v[j] = expf(v[j] - maximum);
    float part = 0.0f;   // lane s: 0 + values[0] + ... + values[7], route_multi's order
    if (h == 0) {
#pragma unroll
        for (int j = 0; j < 8; ++j) part += v[j];
    }
    float sum = __int_as_float(__builtin_amdgcn_ds_bpermute(s << 2, __float_as_int(part)));
    if (h == 1) {   // ... + values[8] + ... + values[15]
#pragma unroll
        for (int j = 0; j < 8; ++j) sum += v[j];
    }
    sum = w64_sum32(sum);
    const float reciprocal = 1.0f / __int_as_float(__builtin_amdgcn_readlane(__float_as_int(sum), 32));
    uint32_t key[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        float p = v[j] * reciprocal;
        if (__isnanf(p)) p = -FLT_MAX;
        key[j] = w64_key(p);
    }
    constexpr uint32_t removed = 0x007fffffu;   // w64_key(-INFINITY), what route_multi writes over a winner
    float selected = 0.0f, selected_sum = 0.0f;
    int my_id = 0;
    for (int rank = 0; rank < 10; ++rank) {
        uint32_t lm = key[0];
#pragma unroll
        for (int j = 1; j < 8; ++j) lm = max(lm, key[j]);
        const uint32_t m = w64_max_u32(lm);
        int cand = 0x7fffffff;
#pragma unroll
        for (int j = 7; j >= 0; --j)
            if (key[j] == m) cand = s + 32 * (8 * h + j);
        const unsigned long long holders = __builtin_amdgcn_ballot_w64(lm == m);
        int expert;
        if ((holders & (holders - 1)) == 0) {   // one lane holds the max: its lowest j
            expert = __builtin_amdgcn_readlane(cand, (int) __builtin_ctzll(holders));
        } else {   // a tie across lanes: the smallest id
            expert = (int) ~w64_max_u32(~(uint32_t) cand);
        }
        const float best = w64_unkey(m);
        const int owner = (expert & 31) + 32 * ((expert >> 8) & 1), jw = (expert >> 5) & 7;
#pragma unroll
        for (int j = 0; j < 8; ++j)
            if (lane == owner && j == jw) key[j] = removed;
        if (lane == (expert & 31)) selected_sum += best;   // route_multi's per-slot accumulation, in rank order
        if (rank == lane) { selected = best; my_id = expert; }
    }
    selected_sum = max(__int_as_float(__builtin_amdgcn_readlane(__float_as_int(w64_sum32(selected_sum)), 0)), 6.103515625e-5f);
    const float inverse_selected_sum = 1.0f / selected_sum;
    if (lane < 10) {
        ids[lane] = my_id;
        weights[lane] = selected * inverse_selected_sum;
        if (ids_sm != nullptr) ids_sm[lane] = my_id;
    }
}
}  // namespace strata::kernels::w64
#endif
