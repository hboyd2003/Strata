// src/kernels/router_fused_parity.cpp - gfx906 STRATA_GFX906_ROUTER_FUSED: router_fused_906 (the router GEMV, the
// top-10 and doorbell_publish_res in one launch) against the three launches it replaces, bitwise: the logits, ids,
// weights, the published ids, the published x (on a miss; untouched when every expert is resident) and the ring.
// Then both chains timed in a graph of 48 layers (distinct weights).
//
//     STRATA_GFX906_ROUTER_FUSED=1 build/router_fused_parity [--bench]
// Leave STRATA_GFX906_ROUTER_W64 / _GEMV unset: the reference must be the old kernels.
#include "strata/kernels/bf16_gemv.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/native_router.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace K = strata::kernels;

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", #x, cudaGetErrorString(e_)); std::exit(1); } } while (0)

namespace {
constexpr int N = 2560, NE = 512, KK = 10, MAXT = 4, L = 48;
uint16_t bf16(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    return (uint16_t) ((u + 0x7fff + ((u >> 16) & 1)) >> 16);
}
struct Pub {   // mapped host memory, as verify's doorbell
    float* h_x; int32_t* h_ids; uint32_t* h_seq;
    float* m_x; int32_t* m_ids; uint32_t* m_seq;
    void alloc() {
        CK(cudaHostAlloc((void**) &h_x, (size_t) MAXT * N * 4, cudaHostAllocMapped));
        CK(cudaHostAlloc((void**) &h_ids, (size_t) MAXT * KK * 4, cudaHostAllocMapped));
        CK(cudaHostAlloc((void**) &h_seq, 64, cudaHostAllocMapped));
        CK(cudaHostGetDevicePointer((void**) &m_x, h_x, 0));
        CK(cudaHostGetDevicePointer((void**) &m_ids, h_ids, 0));
        CK(cudaHostGetDevicePointer((void**) &m_seq, h_seq, 0));
    }
    void reset() {
        std::memset(h_x, 0x5a, (size_t) MAXT * N * 4);
        std::memset(h_ids, 0x5a, (size_t) MAXT * KK * 4);
        h_seq[0] = 7;
    }
};
}  // namespace

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    bool bench = false;
    for (int i = 1; i < argc; ++i) bench |= std::strcmp(argv[i], "--bench") == 0;
    if (std::getenv("STRATA_GFX906_ROUTER_W64") || std::getenv("STRATA_GFX906_ROUTER_GEMV")) {
        std::fprintf(stderr, "unset STRATA_GFX906_ROUTER_W64 / _GEMV: the reference must be the old kernels\n");
        return 2;
    }
    if (!K::router_fused_906_enabled()) { std::printf("set STRATA_GFX906_ROUTER_FUSED=1 (or not a gfx906 build)\n"); return 0; }
    std::mt19937 rng(11);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<uint16_t*> dw(L);
    {
        std::vector<uint16_t> w((size_t) NE * N);
        for (int l = 0; l < L; ++l) {
            for (auto& v : w) v = bf16(0.05f * nd(rng));
            CK(cudaMalloc((void**) &dw[l], w.size() * 2));
            CK(cudaMemcpy(dw[l], w.data(), w.size() * 2, cudaMemcpyHostToDevice));
        }
    }
    float *dx, *dl0, *dl1, *dw0, *dw1;
    int32_t *di0, *di1, *dres_all, *dres_some;
    uint32_t* dcnt;
    CK(cudaMalloc((void**) &dx, (size_t) MAXT * N * 4));
    CK(cudaMalloc((void**) &dl0, (size_t) MAXT * NE * 4)); CK(cudaMalloc((void**) &dl1, (size_t) MAXT * NE * 4));
    CK(cudaMalloc((void**) &dw0, (size_t) MAXT * KK * 4)); CK(cudaMalloc((void**) &dw1, (size_t) MAXT * KK * 4));
    CK(cudaMalloc((void**) &di0, (size_t) MAXT * KK * 4)); CK(cudaMalloc((void**) &di1, (size_t) MAXT * KK * 4));
    CK(cudaMalloc((void**) &dres_all, NE * 4)); CK(cudaMalloc((void**) &dres_some, NE * 4));
    CK(cudaMalloc((void**) &dcnt, 4)); CK(cudaMemset(dcnt, 0, 4));
    {
        std::vector<int32_t> r(NE);
        for (int i = 0; i < NE; ++i) r[i] = i;
        CK(cudaMemcpy(dres_all, r.data(), NE * 4, cudaMemcpyHostToDevice));
        for (int i = 0; i < NE; i += 7) r[i] = -1;
        CK(cudaMemcpy(dres_some, r.data(), NE * 4, cudaMemcpyHostToDevice));
    }
    Pub p0, p1;
    p0.alloc(); p1.alloc();
    cudaStream_t s;
    CK(cudaStreamCreate(&s));
    auto ref = [&](int l, int T, const int32_t* res, Pub& p) {
        K::bf16_gemv_fp32_mmvf_multi(dx, N, dw[l], dl0, NE, N, NE, T, s);
        K::native_router_top10_multi(dl0, di0, dw0, T, s);
        K::doorbell_publish_res(dx, di0, res, NE, (int64_t) T * N, (int64_t) T * KK, p.m_x, p.m_ids, p.m_seq, s);
    };
    auto fused = [&](int l, int T, const int32_t* res, Pub& p) {
        K::RouterFused906 a;
        a.x = dx; a.w = dw[l]; a.logits = dl1; a.ids = di1; a.weights = dw1; a.d_res = res; a.n_expert = NE;
        a.x_out = p.m_x; a.ids_out = p.m_ids; a.seq = p.m_seq; a.counter = dcnt;
        return K::router_fused_906(a, T, s);
    };
    int fails = 0, checked = 0;
    for (int it = 0; it < 64; ++it) {
        const int T = 1 + it % MAXT, l = it % L;
        const int32_t* res = (it / MAXT) % 3 == 0 ? nullptr : (it / MAXT) % 3 == 1 ? dres_all : dres_some;
        std::vector<float> xh((size_t) MAXT * N);
        for (auto& v : xh) v = nd(rng) * (it % 5 == 4 ? 30.f : 1.f);
        CK(cudaMemcpy(dx, xh.data(), xh.size() * 4, cudaMemcpyHostToDevice));
        CK(cudaMemset(dl0, 0x11, (size_t) MAXT * NE * 4)); CK(cudaMemset(dl1, 0x11, (size_t) MAXT * NE * 4));
        CK(cudaMemset(di0, 0x11, (size_t) MAXT * KK * 4)); CK(cudaMemset(di1, 0x11, (size_t) MAXT * KK * 4));
        CK(cudaMemset(dw0, 0x11, (size_t) MAXT * KK * 4)); CK(cudaMemset(dw1, 0x11, (size_t) MAXT * KK * 4));
        p0.reset(); p1.reset();
        ref(l, T, res, p0);
        if (!fused(l, T, res, p1)) { std::printf("router_fused_906 declined\n"); return 1; }
        CK(cudaStreamSynchronize(s));
        auto same = [&](const void* a, const void* b, size_t n) { return std::memcmp(a, b, n) == 0; };
        std::vector<float> l0((size_t) MAXT * NE), l1((size_t) MAXT * NE), w0((size_t) MAXT * KK), w1((size_t) MAXT * KK);
        std::vector<int32_t> i0((size_t) MAXT * KK), i1((size_t) MAXT * KK);
        CK(cudaMemcpy(l0.data(), dl0, l0.size() * 4, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(l1.data(), dl1, l1.size() * 4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(w0.data(), dw0, w0.size() * 4, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(w1.data(), dw1, w1.size() * 4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(i0.data(), di0, i0.size() * 4, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(i1.data(), di1, i1.size() * 4, cudaMemcpyDeviceToHost));
        uint32_t cnt = 1;
        CK(cudaMemcpy(&cnt, dcnt, 4, cudaMemcpyDeviceToHost));
        const char* bad = nullptr;
        if (!same(l0.data(), l1.data(), l0.size() * 4)) bad = "logits";
        else if (!same(i0.data(), i1.data(), i0.size() * 4)) bad = "ids";
        else if (!same(w0.data(), w1.data(), w0.size() * 4)) bad = "weights";
        else if (!same(p0.h_ids, p1.h_ids, (size_t) MAXT * KK * 4)) bad = "published ids";
        else if (!same(p0.h_x, p1.h_x, (size_t) MAXT * N * 4)) bad = "published x";
        else if (p0.h_seq[0] != 8 || p1.h_seq[0] != 8) bad = "ring";
        else if (cnt != 0) bad = "counter not reset";
        ++checked;
        if (bad) {
            ++fails;
            std::printf("  it %d T %d res %s: %s differ\n", it, T, res == nullptr ? "null" : res == dres_all ? "all" : "some", bad);
        }
    }
    std::printf("router_fused: %d of %d launches differ\n", fails, checked);
    if (bench) {
        cudaEvent_t e0, e1;
        CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
        for (int T = 1; T <= MAXT; ++T)
            for (int which = 0; which < 2; ++which) {
                cudaGraph_t g;
                cudaGraphExec_t ge;
                CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
                for (int l = 0; l < L; ++l) {
                    if (which) fused(l, T, dres_some, p1);
                    else ref(l, T, dres_some, p0);
                }
                CK(cudaStreamEndCapture(s, &g));
                CK(cudaGraphInstantiate(&ge, g, nullptr, nullptr, 0));
                for (int i = 0; i < 5; ++i) CK(cudaGraphLaunch(ge, s));
                CK(cudaStreamSynchronize(s));
                std::vector<float> ms;
                for (int r = 0; r < 30; ++r) {
                    CK(cudaEventRecord(e0, s)); CK(cudaGraphLaunch(ge, s)); CK(cudaEventRecord(e1, s));
                    CK(cudaEventSynchronize(e1));
                    float m = 0;
                    CK(cudaEventElapsedTime(&m, e0, e1));
                    ms.push_back(m);
                }
                std::sort(ms.begin(), ms.end());
                std::printf("T %d %-5s : %6.2f us per layer (gemv + top10 + doorbell)\n", T, which ? "fused" : "3 launches",
                            1e3 * ms[ms.size() / 2] / L);
                cudaGraphExecDestroy(ge);
                cudaGraphDestroy(g);
            }
    }
    return fails ? 1 : 0;
}
