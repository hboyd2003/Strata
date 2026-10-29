// src/kernels/router_w64_parity.cpp - gfx906: native_router_top10_w64 (one 64-lane wave per token) against
// native_router_top10_multi / native_router_top10 (the pinned 32-lane kernels), bitwise on ids and weights, then
// both timed in a graph of 48 layers.  Run WITHOUT STRATA_GFX906_ROUTER_W64 (the reference must be the old kernel).
//
//     build/router_w64_parity [--selftest] [--bench]
#include "strata/kernels/native_router.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace K = strata::kernels;

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", #x, cudaGetErrorString(e_)); std::exit(1); } } while (0)

int main(int argc, char** argv) {
    bool bench = false;
    for (int i = 1; i < argc; ++i) bench |= std::strcmp(argv[i], "--bench") == 0;
    if (std::getenv("STRATA_GFX906_ROUTER_W64")) { std::fprintf(stderr, "unset STRATA_GFX906_ROUTER_W64\n"); return 2; }
    constexpr int NE = 512, KK = 10, MAXT = 64;
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::uniform_int_distribution<int> ui(0, NE - 1);
    float* dl;
    int32_t *di0, *di1;
    float *dw0, *dw1;
    CK(cudaMalloc((void**) &dl, (size_t) MAXT * NE * 4));
    CK(cudaMalloc((void**) &di0, (size_t) MAXT * KK * 4)); CK(cudaMalloc((void**) &di1, (size_t) MAXT * KK * 4));
    CK(cudaMalloc((void**) &dw0, (size_t) MAXT * KK * 4)); CK(cudaMalloc((void**) &dw1, (size_t) MAXT * KK * 4));
    cudaStream_t s;
    CK(cudaStreamCreate(&s));
    // cases: 0 gaussian, 1 gaussian x 4 (peaky), 2 logits on a 1/4 grid (many equal p), 3 all equal, 4 a few equal
    // maxima planted at random experts, 5 large magnitudes (underflowing p), 6 two values only
    int fails_all = 0;
    for (int arm = 0; arm < 2; ++arm) {
    std::mt19937 rng(7);
    int fails = 0, checked = 0;
    for (int c = 0; c < 7; ++c) {
        for (int it = 0; it < 40; ++it) {
            const int T = it < 8 ? it + 1 : 1 + (int) (rng() % MAXT);
            std::vector<float> h((size_t) T * NE);
            for (int t = 0; t < T; ++t) {
                float* r = h.data() + (size_t) t * NE;
                for (int e = 0; e < NE; ++e) {
                    float x = nd(rng);
                    if (c == 1) x *= 4.f;
                    if (c == 2) x = std::round(x * 4.f) / 4.f;
                    if (c == 3) x = 0.5f;
                    if (c == 5) x *= 60.f;
                    if (c == 6) x = (rng() & 1) ? 1.f : -1.f;
                    r[e] = x;
                }
                if (c == 4) {
                    const float top = 5.f;
                    for (int k = 0; k < 14; ++k) r[ui(rng)] = top - (float) (k / 4);
                }
            }
            CK(cudaMemcpy(dl, h.data(), h.size() * 4, cudaMemcpyHostToDevice));
            CK(cudaMemset(di0, 0x11, (size_t) MAXT * KK * 4)); CK(cudaMemset(di1, 0x22, (size_t) MAXT * KK * 4));
            CK(cudaMemset(dw0, 0x11, (size_t) MAXT * KK * 4)); CK(cudaMemset(dw1, 0x22, (size_t) MAXT * KK * 4));
            if (T == 1) K::native_router_top10(dl, di0, dw0, s);
            else K::native_router_top10_multi(dl, di0, dw0, T, s);
            const bool built = arm == 0 ? K::native_router_top10_w64(dl, di1, dw1, T, s) : K::native_router_top10_w64_doubr(dl, di1, dw1, T, s);
            if (!built) { std::printf("w64 not built here\n"); return 0; }
            CK(cudaStreamSynchronize(s));
            std::vector<int32_t> i0((size_t) T * KK), i1((size_t) T * KK);
            std::vector<float> w0((size_t) T * KK), w1((size_t) T * KK);
            CK(cudaMemcpy(i0.data(), di0, i0.size() * 4, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(i1.data(), di1, i1.size() * 4, cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(w0.data(), dw0, w0.size() * 4, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(w1.data(), dw1, w1.size() * 4, cudaMemcpyDeviceToHost));
            ++checked;
            if (std::memcmp(i0.data(), i1.data(), i0.size() * 4) || std::memcmp(w0.data(), w1.data(), w0.size() * 4)) {
                ++fails;
                for (int t = 0; t < T; ++t)
                    for (int k = 0; k < KK; ++k) {
                        const size_t q = (size_t) t * KK + k;
                        if (i0[q] != i1[q] || std::memcmp(&w0[q], &w1[q], 4)) {
                            std::printf("  case %d T %d tok %d rank %d: id %d vs %d, w %.9g vs %.9g\n", c, T, t, k, i0[q], i1[q], w0[q], w1[q]);
                            t = T;
                            break;
                        }
                    }
            }
        }
    }
    std::printf("router %s: %d of %d launches differ\n", arm ? "w64_doubr" : "w64", fails, checked);
    fails_all += fails;
    }
    const int fails = fails_all;
    if (bench) {
        constexpr int L = 48, reps = 30;
        std::vector<float> h((size_t) MAXT * NE);
        for (auto& x : h) x = nd(rng);
        CK(cudaMemcpy(dl, h.data(), h.size() * 4, cudaMemcpyHostToDevice));
        cudaEvent_t e0, e1;
        CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
        for (int T : {1, 2, 3, 4, 8}) {
            for (int which = 0; which < 3; ++which) {
                cudaGraph_t g;
                cudaGraphExec_t ge;
                CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
                for (int l = 0; l < L; ++l) {
                    if (which == 2) K::native_router_top10_w64_doubr(dl, di1, dw1, T, s);
                    else if (which) K::native_router_top10_w64(dl, di1, dw1, T, s);
                    else if (T == 1) K::native_router_top10(dl, di0, dw0, s);
                    else K::native_router_top10_multi(dl, di0, dw0, T, s);
                }
                CK(cudaStreamEndCapture(s, &g));
                CK(cudaGraphInstantiate(&ge, g, nullptr, nullptr, 0));
                for (int i = 0; i < 5; ++i) CK(cudaGraphLaunch(ge, s));
                CK(cudaStreamSynchronize(s));
                std::vector<float> ms;
                for (int r = 0; r < reps; ++r) {
                    CK(cudaEventRecord(e0, s)); CK(cudaGraphLaunch(ge, s)); CK(cudaEventRecord(e1, s));
                    CK(cudaEventSynchronize(e1));
                    float m = 0;
                    CK(cudaEventElapsedTime(&m, e0, e1));
                    ms.push_back(m);
                }
                std::sort(ms.begin(), ms.end());
                std::printf("T %d %-6s : %6.2f us per layer\n", T, which == 2 ? "doubr" : which ? "w64" : "old", 1e3 * ms[ms.size() / 2] / L);
                cudaGraphExecDestroy(ge);
                cudaGraphDestroy(g);
            }
        }
    }
    return fails ? 1 : 0;
}
