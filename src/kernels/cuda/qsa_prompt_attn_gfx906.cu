// src/kernels/cuda/qsa_prompt_attn_gfx906.cu - the prompt path's QSA attention for the gfx906 (wave64) build.
//
// Why this exists: `qsa_prompt_attn_batch` (qsa_prompt_attn.cu) is written around m16n8k16 MMAs and cp.async, neither of
// which this card has, so the prompt falls back to `qsa_decode_attn_batch` - the split-K decode kernel, whose partials
// for one query are n_chunks x 24 heads x 256 floats (811 KB at a 32K context) written to global scratch and read back
// by a second (merge) kernel, once per 32-query batch.
//
// This kernel keeps that kernel's per-cell arithmetic (the same int8 codes and per-64 scales, the same query-LDS
// swizzle, the same butterfly reduction for the 12 heads of a KV head) and changes only the shape of the traversal: ONE
// BLOCK walks all of a query's selected cells for one KV head with a running (online) softmax, so the split-K partials
// stay in registers and LDS and neither the scratch write nor the merge kernel exists.
//
// Consequence: not bitwise equal to the decode kernel. The scores and the per-dimension value products are the same
// operands in the same order; what differs is when the exponentials are taken (relative to the running maximum instead
// of each chunk's own maximum, with the merge's rescale folded in), which moves the last bits. The parity harness bounds
// it and the prompt quality gate (KL, top-1, MTP acceptance) is the gate, as for every other non-bit-identical prompt
// change on this box.
//
// The query-LDS swizzle is applied unconditionally here, unlike the decode kernel's STRATA_GFX906_ATTN_QUERY_SWIZZLE
// switch: it is a bit-identical permutation of the tile (proved by bench/gfx906_qsa_query_swizzle_probe), and the
// kernel as a whole is already the opt-in.
#include "strata/kernels/qsa_prompt_attn.hpp"
#include "strata/core/geometry.hpp"
#include "strata/kernels/kv_q8.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cfloat>
#include <cmath>
#include <cstdlib>

namespace strata::kernels {
#if defined(STRATA_HIP_GFX906)
namespace {

constexpr int HD = 256;           // head_dim
constexpr int G = 12;             // query heads per KV head (24 / 2)
static_assert(HD == strata::core::kCompiledGeometry.head_dim &&
                  G == strata::core::kCompiledGeometry.n_head / strata::core::kCompiledGeometry.n_head_kv,
              "qsa_prompt_attn_gfx906 is compiled for core/geometry.hpp");
constexpr int CHUNK = 64;         // selected cells per pass
constexpr int THREADS = 256;      // 8 warps: scores warp-per-cell, values thread-per-dimension
constexpr int WARPS = THREADS / 32;

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}

// The 12 heads' lane sums reduced with one butterfly (16 exchanges instead of 12 x 5) in the pairing order of twelve
// warp_sum calls - bit-identical sums, the form qsa_decode_attn.cu's gfx906 specialization uses.
__device__ __forceinline__ float reduce12(const float (&part)[16], int lane) {
    float r8[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const bool hi = (lane & 16) != 0;
        r8[i] = (hi ? part[i + 8] : part[i]) + __shfl_xor_sync(0xffffffffu, hi ? part[i] : part[i + 8], 16);
    }
    float r4[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const bool hi = (lane & 8) != 0;
        r4[i] = (hi ? r8[i + 4] : r8[i]) + __shfl_xor_sync(0xffffffffu, hi ? r8[i] : r8[i + 4], 8);
    }
    float r2[2];
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const bool hi = (lane & 4) != 0;
        r2[i] = (hi ? r4[i + 2] : r4[i]) + __shfl_xor_sync(0xffffffffu, hi ? r4[i] : r4[i + 2], 4);
    }
    const bool hi2 = (lane & 2) != 0;
    const float r1 = (hi2 ? r2[1] : r2[0]) + __shfl_xor_sync(0xffffffffu, hi2 ? r2[0] : r2[1], 2);
    return r1 + __shfl_xor_sync(0xffffffffu, r1, 1);
}
__device__ __forceinline__ int head_of_lane(int lane) {
    return ((lane >> 4) & 1) * 8 + ((lane >> 3) & 1) * 4 + ((lane >> 2) & 1) * 2 + ((lane >> 1) & 1);
}

// 8 consecutive K codes of one cell for this KV head, dequantized by their per-64 scale (int8) or read as halves.
template <int KV_MODE>
__device__ __forceinline__ void load_k8(const QsaAttnPools& p, long long row, int d0, float* out) {
    if constexpr (KV_MODE == 1) {
        const uint2 raw = *reinterpret_cast<const uint2*>(p.k_q + row * HD + d0);
        const int8_t* c = reinterpret_cast<const int8_t*>(&raw);
        const float sc = __half2float(__ushort_as_half(p.k_scale[row * (HD / KV_Q8_GROUP) + d0 / KV_Q8_GROUP]));
#pragma unroll
        for (int j = 0; j < 8; ++j) out[j] = (float) c[j] * sc;
    } else {
        const uint4 raw = *reinterpret_cast<const uint4*>(p.k_pool + row * HD + d0);
        const __half2* h2 = reinterpret_cast<const __half2*>(&raw);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float2 f = __half22float2(h2[j]);
            out[2 * j] = f.x;
            out[2 * j + 1] = f.y;
        }
    }
}
// One value dimension of one cell (thread t owns dimension t, as in the decode kernel's value phase).
template <int KV_MODE>
__device__ __forceinline__ float load_v(const QsaAttnPools& p, long long row, int t) {
    if constexpr (KV_MODE == 1) {
        const float sc = __half2float(__ushort_as_half(p.v_scale[row * (HD / KV_Q8_GROUP) + t / KV_Q8_GROUP]));
        return (float) p.v_q[row * HD + t] * sc;
    } else {
        return __half2float(__ushort_as_half(p.v_pool[row * HD + t]));
    }
}

// grid (n_q, n_kv_heads): blockIdx.x = query, blockIdx.y = KV head; the block's 12 query heads share its KV cells.
template <int KV_MODE>
__global__ void __launch_bounds__(THREADS) prompt_attn_kernel(const float* __restrict__ q, QsaAttnPools p,
                                                             const int32_t* __restrict__ ids,
                                                             const int32_t* __restrict__ step, int64_t cap,
                                                             int n_kv_heads, int page_size, float scale,
                                                             float* __restrict__ attn) {
    // Per QUERY, not per call: the width of a query's selection grows with its position, so a chunk's first and last
    // query differ by up to a chunk's worth of cells (the probe at ctx 4096 vs 32768 catches this, the 32K case does not)
    const int n_ids = __ldg(step + (size_t) blockIdx.x * kStepCount + kStepWidth);
    const int kvh = blockIdx.y, t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const size_t row_stride = (size_t) (n_kv_heads * G) * HD;
    const int32_t* my_ids = ids + (size_t) blockIdx.x * (size_t) cap;
    const float* my_q = q + (size_t) blockIdx.x * row_stride + (size_t) kvh * G * HD;
    __shared__ __align__(16) float sq[G][HD];   // 12 KB: this KV head's 12 query heads, swizzled as the decode kernel does
    __shared__ float sp[G][CHUNK];              // scores, then this pass's probabilities
    __shared__ long long srow[CHUNK];           // pool row of each selected cell (-1: masked / not resident)
    __shared__ float sm[G], sl[G], sc[G];       // running maximum, running sum, this pass's rescale, per head
    for (int i = t; i < G * HD; i += THREADS) {
        const int d = i % HD;
        sq[i / HD][d ^ ((d >> 3) & 4)] = my_q[i];
    }
    if (t < G) { sm[t] = -FLT_MAX; sl[t] = 0.0f; sc[t] = 1.0f; }
    float acc[G];
#pragma unroll
    for (int h = 0; h < G; ++h) acc[h] = 0.0f;
    __syncthreads();
    if (n_ids <= 0) {   // no selected cells (never in the engine, which always selects at least the current cell)
        float* out = attn + (size_t) blockIdx.x * row_stride + (size_t) kvh * G * HD + t;
#pragma unroll
        for (int h = 0; h < G; ++h) out[(size_t) h * HD] = 0.0f;
        return;
    }
    for (int c0 = 0; c0 < n_ids; c0 += CHUNK) {
        const int n_here = min(CHUNK, n_ids - c0);
        if (t < CHUNK) {
            long long r = -1;
            if (t < n_here) {
                const int cell = my_ids[c0 + t];
                const long long page = (long long) p.page_table[cell / page_size];
                // a page the KV streaming could not make resident keeps -1: its cells are masked below
                if (page >= 0) r = (page * n_kv_heads + kvh) * page_size + (cell % page_size);
            }
            srow[t] = r;
        }
        __syncthreads();
        // scores: warp w takes cells w, w+8, ...; each lane holds 8 of the 256 dimensions, the 12 heads reduced together
        for (int c = warp; c < CHUNK; c += WARPS) {
            if (c >= n_here || srow[c] < 0) {
                if (lane < G) sp[lane][c] = -FLT_MAX;
                continue;
            }
            float k8[8];
            load_k8<KV_MODE>(p, srow[c], lane * 8, k8);
            float part[16];
            const int qflip = lane & 4;
#pragma unroll
            for (int h = 0; h < G; ++h) {
                const float4 qa = *reinterpret_cast<const float4*>(&sq[h][lane * 8 + qflip]);
                const float4 qb = *reinterpret_cast<const float4*>(&sq[h][lane * 8 + (qflip ^ 4)]);
                part[h] = k8[0] * qa.x + k8[1] * qa.y + k8[2] * qa.z + k8[3] * qa.w +
                          k8[4] * qb.x + k8[5] * qb.y + k8[6] * qb.z + k8[7] * qb.w;
            }
#pragma unroll
            for (int h = G; h < 16; ++h) part[h] = 0.0f;
            const float score = reduce12(part, lane);
            const int head = head_of_lane(lane);
            if ((lane & 1) == 0 && head < G) sp[head][c] = score * scale;
        }
        __syncthreads();
        // running softmax: warp w owns heads w and w+WARPS. The maximum only grows, so a pass rescales the running
        // accumulators once by exp(m_old - m_new) - the job the decode path's merge kernel does per chunk, folded in.
        for (int h = warp; h < G; h += WARPS) {
            const float a = sp[h][lane], b = sp[h][lane + 32];
            const float mo = sm[h];
            const float m = fmaxf(mo, warp_max(fmaxf(a, b)));
            const float corr = m > mo ? __expf(mo - m) : 1.0f;
            const float ea = (lane < n_here && srow[lane] >= 0) ? __expf(a - m) : 0.0f;
            const float eb = (lane + 32 < n_here && srow[lane + 32] >= 0) ? __expf(b - m) : 0.0f;
            sp[h][lane] = ea;
            sp[h][lane + 32] = eb;
            const float lc = warp_sum(ea + eb);
            if (lane == 0) {
                sm[h] = m;
                sc[h] = corr;
                sl[h] = sl[h] * corr + lc;
            }
        }
        __syncthreads();
        // values: thread t owns dimension t for all 12 heads of this KV head
#pragma unroll
        for (int h = 0; h < G; ++h) acc[h] = acc[h] * sc[h];
        for (int c = 0; c < n_here; ++c) {
            if (srow[c] < 0) continue;   // masked above, weight 0
            const float v = load_v<KV_MODE>(p, srow[c], t);
#pragma unroll
            for (int h = 0; h < G; ++h) acc[h] = fmaf(sp[h][c], v, acc[h]);
        }
        __syncthreads();   // sp / srow are rewritten by the next pass
    }
    float* out = attn + (size_t) blockIdx.x * row_stride + (size_t) kvh * G * HD + t;
#pragma unroll
    for (int h = 0; h < G; ++h) out[(size_t) h * HD] = sl[h] > 0.0f ? acc[h] / sl[h] : 0.0f;   // as attn_merge_kernel
}

// int8 (1) / fp16 (0) pools this kernel serves; -1 = another path (Q4_0 KV must stay on the decode kernel, which
// dequantizes it with the caller's rotate-back).
int gfx906_prompt_attn_mode(const QsaAttnPools& p) {
    if (p.k_q4 != nullptr) return -1;
    if (p.k_q != nullptr) return p.v_q != nullptr ? 1 : -1;
    return p.k_pool != nullptr ? 0 : -1;
}

bool prompt_attn_enabled() {
    // Opt-in in the engine: `=1` turns it on, anything else (unset included) leaves the decode path. That is what the
    // gfx906 deployment does (docker-compose.yml), because both halves of the gate have been run: it is not bitwise the
    // decode kernel, but the prompt gate reads top-1 100 % at KL 0.0136 / 0.00062 and MTP acceptance does not move
    // (72.2 -> 72.0 % over 1,550 drafts) - see the header comment and
    // docs/gfx906-results/20261103-qsa-prompt-attn/README.md.
    static const bool on = [] {
        const char* v = std::getenv("STRATA_PROMPT_ATTN_GFX906");
        return v != nullptr && v[0] == '1';
    }();
    return on;
}

}  // namespace

bool qsa_prompt_attn_gfx906_batch(const float* q, const QsaAttnPools& pools, const int32_t* ids,
                                  const int32_t* steps, int64_t cap, const QsaShapes& s, float* attn, int64_t n_q,
                                  void* stream) {
    if (!prompt_attn_enabled() || n_q <= 0 || cap <= 0) return false;
    if (s.n_head != 24 || s.n_head_kv != 2 || s.head_dim != HD || s.page_size <= 0) return false;
    const int mode = gfx906_prompt_attn_mode(pools);
    if (mode < 0) return false;
    const dim3 grid((unsigned) n_q, (unsigned) s.n_head_kv);
    const float scale = 1.0f / sqrtf((float) HD);
    hipStream_t st = reinterpret_cast<hipStream_t>(stream);
    if (mode == 1)
        prompt_attn_kernel<1><<<grid, THREADS, 0, st>>>(q, pools, ids, steps, cap, (int) s.n_head_kv,
                                                        (int) s.page_size, scale, attn);
    else
        prompt_attn_kernel<0><<<grid, THREADS, 0, st>>>(q, pools, ids, steps, cap, (int) s.n_head_kv,
                                                        (int) s.page_size, scale, attn);
    return true;
}

#else   // not the gfx906 build: the entry point exists so the dispatch stays one line, and it says no
bool qsa_prompt_attn_gfx906_batch(const float*, const QsaAttnPools&, const int32_t*, const int32_t*, int64_t,
                                  const QsaShapes&, float*, int64_t, void*) {
    return false;
}
#endif
}  // namespace strata::kernels
