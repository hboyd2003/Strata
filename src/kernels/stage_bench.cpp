// stage_bench: what each latency-bound verify-window kernel of the FFN half costs on this GPU, in a captured graph.
// Synthetic data (timing only, no model).  Each case captures REPS dependent repetitions of a launch sequence and
// reports us per repetition (kernel time + one dependent node each).  Run with the production env (STRATA_MMVF_ROWS=1,
// STRATA_GFX906_ROUTER_T2_EXACT=1, STRATA_GFX906_ROUTER_T1_ROWS=1, STRATA_MMVQ_WAVE_ROWS=2).
//     stage_bench [T=2]
#include "strata/kernels/bf16_gemv.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/native_moe.hpp"
#include "strata/kernels/native_router.hpp"
#include "strata/kernels/s2_expert_grouped.hpp"
#include "strata/kernels/shared_expert.hpp"
#include "strata/kernels/verify_kernels.hpp"
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <random>
#include <vector>

namespace k = strata::kernels;
static void ck(cudaError_t e, const char* w) {
    if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", w, cudaGetErrorString(e)); std::exit(2); }
}
template<class T> static T* dalloc(size_t n) { T* p = nullptr; ck(cudaMalloc(&p, n * sizeof(T) + 256), "malloc"); return p; }
static void* dbytes(size_t n, std::mt19937& rng, bool small_halves = true) {
    std::vector<uint8_t> h(n);
    for (auto& b : h) b = (uint8_t) rng();
    // keep fp16 scales finite-ish: clear the top exponent bit of every odd byte
    if (small_halves) for (size_t i = 1; i < n; i += 2) h[i] &= 0x3b;
    void* d = nullptr; ck(cudaMalloc(&d, n + 256), "malloc"); ck(cudaMemcpy(d, h.data(), n, cudaMemcpyHostToDevice), "h2d");
    return d;
}

static cudaStream_t s;
static const int REPS = 96;   // two windows' worth of layers
// rotate between NB weight copies so L2 (4 MB) does not serve the repeats
static double time_seq(const char* name, const std::function<void(int)>& seq) {
    cudaGraph_t g; cudaGraphExec_t ge;
    ck(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal), "begin");
    for (int r = 0; r < REPS; ++r) seq(r);
    ck(cudaStreamEndCapture(s, &g), name);
    ck(cudaGraphInstantiate(&ge, g, nullptr, nullptr, 0), "inst");
    size_t nn = 0; cudaGraphGetNodes(g, nullptr, &nn);
    for (int i = 0; i < 3; ++i) ck(cudaGraphLaunch(ge, s), "launch");
    ck(cudaStreamSynchronize(s), name);
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    const int it = 20;
    cudaEventRecord(a, s);
    for (int i = 0; i < it; ++i) cudaGraphLaunch(ge, s);
    cudaEventRecord(b, s);
    ck(cudaEventSynchronize(b), name);
    float ms = 0; cudaEventElapsedTime(&ms, a, b);
    const double us = 1000.0 * ms / it / REPS;
    std::printf("  %-52s %7.2f us/layer  (%zu nodes/layer)  -> %.2f ms/48 layers\n", name, us, nn / REPS, us * 48 / 1000.0);
    cudaGraphExecDestroy(ge); cudaGraphDestroy(g);
    return us;
}

int main(int argc, char** argv) {
    const int T = argc > 1 ? std::atoi(argv[1]) : 2;
    const int N = 2560, NE = 512, K = 10, FF = 640, NB = 8;
    const int gu_type = argc > 2 ? std::atoi(argv[2]) : 12;   // Q4_K gate/up by default
    const int d_type = argc > 3 ? std::atoi(argv[3]) : 20;    // IQ4_NL down
    ck(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking), "stream");
    std::mt19937 rng(7);
    std::printf("stage_bench T=%d gate/up type %d down type %d (REPS %d)\n", T, gu_type, d_type, REPS);

    float* x = dalloc<float>((size_t) 8 * N);
    {
        std::vector<float> hx((size_t) 8 * N);
        std::uniform_real_distribution<float> d(-1.f, 1.f);
        for (auto& v : hx) v = d(rng);
        cudaMemcpy(x, hx.data(), hx.size() * 4, cudaMemcpyHostToDevice);
    }
    std::vector<uint16_t*> wr(NB);
    for (auto& w : wr) {
        std::vector<uint16_t> h((size_t) NE * N);
        for (auto& v : h) v = (uint16_t) (0x3c00 + (rng() & 0x3ff) - 0x200) ;   // bf16 around +-1e-2..1
        w = dalloc<uint16_t>(h.size());
        cudaMemcpy(w, h.data(), h.size() * 2, cudaMemcpyHostToDevice);
    }
    float* logits = dalloc<float>((size_t) 8 * NE);
    int32_t* ids = dalloc<int32_t>(8 * K);
    float* wts = dalloc<float>(8 * K);
    // mapped doorbell regions
    float* h_x; int32_t* h_ids; uint32_t* h_seq; int32_t* h_plan; float* h_rows; uint32_t* h_flag;
    ck(cudaHostAlloc((void**) &h_x, 8 * N * 4, cudaHostAllocMapped), "ha");
    ck(cudaHostAlloc((void**) &h_ids, 8 * K * 4, cudaHostAllocMapped), "ha");
    ck(cudaHostAlloc((void**) &h_seq, 64, cudaHostAllocMapped), "ha");
    ck(cudaHostAlloc((void**) &h_plan, 4096 * 4, cudaHostAllocMapped), "ha");
    ck(cudaHostAlloc((void**) &h_rows, (size_t) 8 * K * N * 4, cudaHostAllocMapped), "ha");
    ck(cudaHostAlloc((void**) &h_flag, 64, cudaHostAllocMapped), "ha");
    std::memset(h_plan, 0, 4096 * 4); std::memset(h_rows, 0, (size_t) 8 * K * N * 4);
    h_flag[0] = 1u << 30; h_seq[0] = 0;
    float *m_x, *m_rows; int32_t *m_ids, *m_plan; uint32_t *m_seq, *m_flag;
    cudaHostGetDevicePointer((void**) &m_x, h_x, 0); cudaHostGetDevicePointer((void**) &m_ids, h_ids, 0);
    cudaHostGetDevicePointer((void**) &m_seq, h_seq, 0); cudaHostGetDevicePointer((void**) &m_plan, h_plan, 0);
    cudaHostGetDevicePointer((void**) &m_rows, h_rows, 0); cudaHostGetDevicePointer((void**) &m_flag, h_flag, 0);
    int32_t* d_res = dalloc<int32_t>(NE);
    {   // every expert resident except a few, so the publish copies x (a miss in the window, as in prod at 94% hits)
        std::vector<int32_t> r(NE, 1);
        for (int i = 0; i < NE; i += 7) r[i] = -1;
        cudaMemcpy(d_res, r.data(), NE * 4, cudaMemcpyHostToDevice);
    }

    std::printf("== router stage (stamp 32 -> 17 in the profile: router+plan+bell)\n");
    time_seq("router gemv bf16 512x2560 (mmvf_multi)", [&](int r) {
        k::bf16_gemv_fp32_mmvf_multi(x, N, wr[r % NB], logits, NE, N, NE, T, s);
    });
    time_seq("router top10 (native_router_top10_multi)", [&](int) { k::native_router_top10_multi(logits, ids, wts, T, s); });
    time_seq("doorbell_publish_res", [&](int) {
        k::doorbell_publish_res(x, ids, d_res, NE, (int64_t) T * N, (int64_t) T * K, m_x, m_ids, m_seq, s);
    });
    time_seq("router gemv + top10 + doorbell", [&](int r) {
        k::bf16_gemv_fp32_mmvf_multi(x, N, wr[r % NB], logits, NE, N, NE, T, s);
        k::native_router_top10_multi(logits, ids, wts, T, s);
        k::doorbell_publish_res(x, ids, d_res, NE, (int64_t) T * N, (int64_t) T * K, m_x, m_ids, m_seq, s);
    });

    std::printf("== shared expert + experts' quant (stamp 17 -> 18: shared+quant)\n");
    const size_t gu_bytes = k::native_mmvq_weight_bytes(gu_type, N, FF), d_bytes = k::native_mmvq_weight_bytes(d_type, FF, N);
    std::vector<void*> wg(NB), wu(NB), wd(NB);
    for (int i = 0; i < NB; ++i) { wg[i] = dbytes(gu_bytes, rng); wu[i] = dbytes(gu_bytes, rng); wd[i] = dbytes(d_bytes, rng); }
    std::vector<uint16_t*> wgi(NB);
    for (auto& w : wgi) { std::vector<uint16_t> h(N, 0x3c00); w = dalloc<uint16_t>(N); cudaMemcpy(w, h.data(), N * 2, cudaMemcpyHostToDevice); }
    void* q8 = dalloc<uint8_t>((size_t) 8 * (N / 32) * 36);
    void* q8b = dalloc<uint8_t>((size_t) 8 * (N / 32) * 36);
    float* gate = dalloc<float>((size_t) 8 * FF); float* up = dalloc<float>((size_t) 8 * FF);
    float* sg = dalloc<float>(8); float* shared = dalloc<float>((size_t) 8 * N);
    uint16_t* xbf = dalloc<uint16_t>((size_t) 8 * N);
    std::printf("  (gate/up %zu B, down %zu B per layer)\n", gu_bytes, d_bytes);
    time_seq("quantize q8_1 (x)", [&](int) { k::native_quantize_q8_1(x, q8, N, T, s); });
    time_seq("mmvq gate 2560->640", [&](int r) { k::native_mmvq(gu_type, wg[r % NB], q8, gate, N, FF, T, s); });
    time_seq("mmvq gate + up (2 launches)", [&](int r) {
        k::native_mmvq(gu_type, wg[r % NB], q8, gate, N, FF, T, s);
        k::native_mmvq(gu_type, wu[r % NB], q8, up, N, FF, T, s);
    });
    time_seq("swiglu + q8_1", [&](int) { k::native_swiglu_quantize_q8_1(gate, up, q8b, FF, T, s); });
    time_seq("mmvq down 640->2560", [&](int r) { k::native_mmvq(d_type, wd[r % NB], q8b, shared, FF, N, T, s); });
    // gate GEMV of 1 row as the batch path does it: one row of bf16 (n_out 1) - via mmvf_multi
    time_seq("shared gate gemv bf16 1x2560", [&](int r) { k::bf16_gemv_fp32_mmvf_multi(x, N, wgi[r % NB], sg, 1, N, 1, T, s); });
    k::NativeSharedWeights nw;
    nw.gate_type = gu_type; nw.up_type = gu_type; nw.down_type = d_type;
    time_seq("shared_expert_multi (whole)", [&](int r) {
        nw.gate_data = wg[r % NB]; nw.up_data = wu[r % NB]; nw.down_data = wd[r % NB]; nw.q8_1 = q8;
        k::shared_expert_multi(T, x, xbf, nw, wgi[r % NB], gate, up, sg, shared, N, FF, s);
    });
    time_seq("quantize_q8_1_rows (experts' copy of x)", [&](int) { k::quantize_q8_1_rows(x, T, N, q8b, s); });
    time_seq("shared_expert_multi + experts' quant", [&](int r) {
        nw.gate_data = wg[r % NB]; nw.up_data = wu[r % NB]; nw.down_data = wd[r % NB]; nw.q8_1 = q8;
        k::shared_expert_multi(T, x, xbf, nw, wgi[r % NB], gate, up, sg, shared, N, FF, s);
        k::quantize_q8_1_rows(x, T, N, q8b, s);
    });
    time_seq("QDEDUP: experts' quant first, shared reads it", [&](int r) {
        nw.gate_data = wg[r % NB]; nw.up_data = wu[r % NB]; nw.down_data = wd[r % NB]; nw.q8_1 = q8;
        k::quantize_q8_1_rows(x, T, N, q8b, s);
        k::shared_expert_multi(T, x, xbf, nw, wgi[r % NB], gate, up, sg, shared, N, FF, s, q8b);
    });

    std::printf("== post: plan wait/copy, CPU rows, hit add, combine (flags already raised)\n");
    const int cap = 5 * K;   // max_t 5
    const int plan_i32 = (int) ((((4 + (cap + 1) + 2 * cap) + 1) & ~1) + 4 * cap + (cap + 1) + 1);
    int32_t* plan = dalloc<int32_t>(4096);
    {   // counts: [0] GPU groups, [1] GPU rows (dst list), ...; dst = rows the GPU computed
        std::vector<int32_t> hp(4096, 0);
        hp[1] = T * K - 2;   // all but 2 rows are hits
        for (int i = 0; i < T * K; ++i) hp[4 + (cap + 1) + i] = i + 2;   // dst list
        std::memcpy(h_plan, hp.data(), 4096 * 4);
        cudaMemcpy(plan, hp.data(), 4096 * 4, cudaMemcpyHostToDevice);
    }
    float* parts = dalloc<float>((size_t) 8 * K * N); float* hit = dalloc<float>((size_t) 8 * K * N);
    float* bo = dalloc<float>((size_t) 8 * N);
    cudaMemset(parts, 0, (size_t) 8 * K * N * 4); cudaMemset(hit, 0, (size_t) 8 * K * N * 4);
    std::printf("  (plan %d ints)\n", plan_i32);
    time_seq("wait_flag_ge (raised)", [&](int) { k::wait_flag_ge(m_flag, 1, s); });
    time_seq("copy_i32_from_mapped (plan)", [&](int) { k::copy_i32_from_mapped(plan, m_plan, plan_i32, s); });
    time_seq("wait + copy plan (2 launches)", [&](int) {
        k::wait_flag_ge(m_flag, 1, s);
        k::copy_i32_from_mapped(plan, m_plan, plan_i32, s);
    });
    time_seq("wait_copy_i32_from_mapped (FUSE_WAIT form)", [&](int) {
        k::wait_copy_i32_from_mapped(m_flag, 1, plan, m_plan, plan_i32, s);
    });
    const int32_t* p_dst = plan + 4 + (cap + 1);
    const int32_t* p_cnt1 = plan + 1;
    time_seq("copy_rows_from_mapped (T*K rows, 2 CPU)", [&](int) {
        k::copy_rows_from_mapped(parts, m_rows, (int64_t) T * K, N, p_dst, p_cnt1, s);
    });
    time_seq("moe_hit_add", [&](int) { k::moe_hit_add(parts, hit, p_dst, p_cnt1, (int64_t) T * K, N, s); });
    time_seq("native_moe_combine_multi", [&](int) { k::native_moe_combine_multi(parts, wts, shared, bo, N, K, T, s); });
    time_seq("post chain: wait+plan, wait+rows, hit_add, combine", [&](int) {
        k::wait_flag_ge(m_flag, 1, s);
        k::copy_i32_from_mapped(plan, m_plan, plan_i32, s);
        k::wait_flag_ge(m_flag, 1, s);
        k::copy_rows_from_mapped(parts, m_rows, (int64_t) T * K, N, p_dst, p_cnt1, s);
        k::moe_hit_add(parts, hit, p_dst, p_cnt1, (int64_t) T * K, N, s);
        k::native_moe_combine_multi(parts, wts, shared, bo, N, K, T, s);
    });
    std::printf("doorbell seq after: %u\n", h_seq[0]);
    return 0;
}
