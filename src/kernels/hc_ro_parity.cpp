// src/kernels/hc_ro_parity.cpp - the gfx906 STRATA_GR_RO=1 hyper-connection read (fused_gr_read_ro: split-K down,
// fixed-order reduce, up) against the default multi read (fused_gr_read_multi), then both timed with cold weights.
//
// Real geometry (n_embd 2560, hc 4, hc_lr 320), every T = 1..8, three call shapes: the verifier's read (pending write
// applied, inject rows), a read without the pending write, and the final mixer (no inject rows).  The RO read sums in
// another order, so it is bounded, not bitwise: every output (mixed, inject, the written R, lo, rs) must be finite and
// within 1e-4 of the default read relative to its largest magnitude, and two RO runs must agree bit for bit.
// Off gfx906 the RO read reports that it did not run and the test passes as a skip.
//     hc_ro_parity [iters] [cold sets]      (timing: T 1..5, default read vs RO read, weights rotated over the sets)
// GPU, synthetic, no model.
#include "strata/kernels/fused_gr.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace k = strata::kernels;

namespace {
constexpr int N = 2560, HC = 4, D = N * HC, LR = 320, TM = 8;
void ck(cudaError_t e, const char* w) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", w, cudaGetErrorString(e));
        std::exit(2);
    }
}
uint16_t bf16(float x) {
    uint32_t u;
    std::memcpy(&u, &x, 4);
    return (uint16_t) ((u + 0x7fff + ((u >> 16) & 1)) >> 16);
}
template <typename T>
T* up(const std::vector<T>& h) {
    void* p = nullptr;
    ck(cudaMalloc(&p, h.size() * sizeof(T)), "malloc");
    ck(cudaMemcpy(p, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice), "h2d");
    return (T*) p;
}
template <typename T>
T* dev(size_t n) {
    void* p = nullptr;
    ck(cudaMalloc(&p, n * sizeof(T)), "malloc");
    return (T*) p;
}
}  // namespace

int main(int argc, char** argv) {
    const int iters = argc > 1 ? std::atoi(argv[1]) : 300;
    const int sets = argc > 2 ? std::atoi(argv[2]) : 8;
    k::fused_gr_check();   // the default read's variant, as the verifier picks it

    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> R((size_t) TM * D), bo((size_t) TM * N), inj((size_t) TM * HC), wn(D);
    for (auto& x : R) x = nd(rng);
    for (auto& x : bo) x = 0.5f * nd(rng);
    for (auto& x : inj) x = nd(rng);
    for (auto& x : wn) x = 1.f + 0.3f * nd(rng);
    std::vector<uint16_t> wd((size_t) LR * D), wu((size_t) D * LR), wi((size_t) HC * D);
    for (auto& x : wd) x = bf16(0.02f * nd(rng));
    for (auto& x : wu) x = bf16(0.05f * nd(rng));
    for (auto& x : wi) x = bf16(0.02f * nd(rng));
    float* dR = dev<float>(R.size());
    float* dbo = up(bo);
    float* dinj = up(inj);
    float* dwn = up(wn);
    uint16_t* dwd = up(wd);
    uint16_t* dwu = up(wu);
    uint16_t* dwi = up(wi);
    float* dxn = dev<float>((size_t) TM * D);
    float* dlo = dev<float>((size_t) TM * LR);
    float* drs = dev<float>((size_t) TM * HC);
    float* dio = dev<float>((size_t) TM * HC);
    float* dmix = dev<float>((size_t) TM * N);

    auto make = [&](k::FusedGrArgs* a, int T, bool apply, bool inject, const uint16_t* w_d, const uint16_t* w_u,
                    const uint16_t* w_i) {
        for (int t = 0; t < T; ++t) {
            a[t] = k::FusedGrArgs{};
            a[t].R = dR + (size_t) t * D;
            a[t].R_out = dR + (size_t) t * D;
            a[t].apply = apply;
            a[t].bo_prev = dbo + (size_t) t * N;
            a[t].inj_prev = dinj + (size_t) t * HC;
            a[t].w_norm = dwn;
            a[t].w_down = w_d;
            a[t].w_up = w_u;
            a[t].w_inject = inject ? w_i : nullptr;
            a[t].eps = 1e-6f;
            a[t].lo = dlo + (size_t) t * LR;
            a[t].rs = drs + (size_t) t * HC;
            a[t].inject_out = dio + (size_t) t * HC;
            a[t].mixed = dmix + (size_t) t * N;
        }
    };

    const struct { bool apply, inject; const char* what; } shapes[] = {
        {true, true, "verifier read (pending write + inject)"}, {false, true, "no pending write"}, {false, false, "final mixer (no inject)"}};
    int bad = 0, ran = 0;
    for (const auto& sh : shapes) {
        for (int T = 1; T <= TM; ++T) {
            k::FusedGrArgs a[TM];
            make(a, T, sh.apply, sh.inject, dwd, dwu, dwi);
            // 0: the default read, 1 and 2: the RO read twice; each output is mixed | inject | R | lo | rs
            const size_t per = (size_t) N + HC + D + LR + HC;
            std::vector<float> out[3];
            bool used = true;
            for (int w = 0; w < 3; ++w) {
                ck(cudaMemcpy(dR, R.data(), R.size() * 4, cudaMemcpyHostToDevice), "R");
                ck(cudaMemset(dmix, 0xff, (size_t) TM * N * 4), "mix");
                ck(cudaMemset(dio, 0, (size_t) TM * HC * 4), "io");
                ck(cudaMemset(dlo, 0xff, (size_t) TM * LR * 4), "lo");
                ck(cudaMemset(drs, 0xff, (size_t) TM * HC * 4), "rs");
                ck(cudaDeviceSynchronize(), "sync");
                if (w == 0) k::fused_gr_read_multi(a, T, dxn, nullptr);
                else used = k::fused_gr_read_ro(a, T, dxn, nullptr) && used;
                ck(cudaDeviceSynchronize(), "read");
                out[w].resize((size_t) T * per);
                float* o = out[w].data();
                ck(cudaMemcpy(o, dmix, (size_t) T * N * 4, cudaMemcpyDeviceToHost), "d2h mix");
                ck(cudaMemcpy(o + (size_t) T * N, dio, (size_t) T * HC * 4, cudaMemcpyDeviceToHost), "d2h io");
                ck(cudaMemcpy(o + (size_t) T * (N + HC), dR, (size_t) T * D * 4, cudaMemcpyDeviceToHost), "d2h R");
                ck(cudaMemcpy(o + (size_t) T * (N + HC + D), dlo, (size_t) T * LR * 4, cudaMemcpyDeviceToHost), "d2h lo");
                ck(cudaMemcpy(o + (size_t) T * (N + HC + D + LR), drs, (size_t) T * HC * 4, cudaMemcpyDeviceToHost), "d2h rs");
                if (w == 1 && !used) break;
            }
            if (!used) {
                std::printf("%s T=%d: the RO read did not run (skip)\n", sh.what, T);
                continue;
            }
            ++ran;
            // relative to each output block's largest magnitude
            const size_t offs[6] = {0, (size_t) T * N, (size_t) T * (N + HC), (size_t) T * (N + HC + D),
                                    (size_t) T * (N + HC + D + LR), (size_t) T * per};
            const char* names[5] = {"mixed", "inject", "R", "lo", "rs"};
            bool finite = true;
            double worst = 0.0;
            const char* worst_name = "";
            for (int b = 0; b < 5; ++b) {
                if (b == 1 && !sh.inject) continue;
                double mref = 0.0, md = 0.0;
                for (size_t i = offs[b]; i < offs[b + 1]; ++i) {
                    finite = finite && std::isfinite(out[1][i]);
                    mref = std::fmax(mref, std::fabs((double) out[0][i]));
                    md = std::fmax(md, std::fabs((double) out[1][i] - out[0][i]));
                }
                const double rel = md / (mref > 0 ? mref : 1.0);
                if (rel > worst) worst = rel, worst_name = names[b];
            }
            const bool same = std::memcmp(out[1].data(), out[2].data(), out[1].size() * 4) == 0;
            const bool ok = finite && worst < 1e-4 && same;
            std::printf("%-40s T=%d: worst relative difference %.3g (%s), %s, two runs %s  %s\n", sh.what, T, worst,
                        worst_name, finite ? "finite" : "NON-FINITE", same ? "bitwise equal" : "DIFFER", ok ? "ok" : "FAIL");
            if (!ok) ++bad;
        }
    }
    if (ran == 0) std::printf("SKIP: the RO read is not built for this card\n");
    else std::printf("%s: %d reads checked\n", bad ? "FAIL" : "PASS", ran);

    if (ran > 0 && iters > 0 && sets > 0) {   // timing: the verifier's read shape, weights cold over `sets` copies
        std::vector<uint16_t*> cd(sets), cu(sets), ci(sets);
        for (int s = 0; s < sets; ++s) cd[s] = up(wd), cu[s] = up(wu), ci[s] = up(wi);
        cudaEvent_t e0, e1;
        ck(cudaEventCreate(&e0), "ev");
        ck(cudaEventCreate(&e1), "ev");
        std::printf("timing (verifier read shape, weights cold over %d sets, %d iters), us per read:\n", sets, iters);
        for (int T = 1; T <= 5; ++T) {
            std::vector<k::FusedGrArgs> as((size_t) sets * TM);
            for (int s = 0; s < sets; ++s) make(&as[(size_t) s * TM], T, true, true, cd[s], cu[s], ci[s]);
            double us[2];
            for (int v = 0; v < 2; ++v) {
                auto run = [&](int i) {
                    const k::FusedGrArgs* a = &as[(size_t) (i % sets) * TM];
                    if (v == 0) k::fused_gr_read_multi(a, T, dxn, nullptr);
                    else k::fused_gr_read_ro(a, T, dxn, nullptr);
                };
                for (int i = 0; i < 10; ++i) run(i);
                ck(cudaEventRecord(e0), "rec");
                for (int i = 0; i < iters; ++i) run(i);
                ck(cudaEventRecord(e1), "rec");
                ck(cudaEventSynchronize(e1), "sync");
                float ms = 0.f;
                ck(cudaEventElapsedTime(&ms, e0, e1), "elapsed");
                us[v] = 1e3 * ms / iters;
            }
            // the RO read's launches alone: down, down + reduce (up = all - that)
            double part_us[2];
            for (int p = 0; p < 2; ++p) {
                k::fused_gr_ro_set_parts(p == 0 ? 1 : 3);
                for (int i = 0; i < 10; ++i) k::fused_gr_read_ro(&as[(size_t) (i % sets) * TM], T, dxn, nullptr);
                ck(cudaEventRecord(e0), "rec");
                for (int i = 0; i < iters; ++i) k::fused_gr_read_ro(&as[(size_t) (i % sets) * TM], T, dxn, nullptr);
                ck(cudaEventRecord(e1), "rec");
                ck(cudaEventSynchronize(e1), "sync");
                float ms = 0.f;
                ck(cudaEventElapsedTime(&ms, e0, e1), "elapsed");
                part_us[p] = 1e3 * ms / iters;
            }
            k::fused_gr_ro_set_parts(7);
            std::printf("  T=%d  default %6.1f   ro %6.1f   (%+.1f us, %+.0f%%)   ro split: down %5.1f  reduce %4.1f  up %5.1f\n",
                        T, us[0], us[1], us[1] - us[0], 100.0 * (us[1] / us[0] - 1.0), part_us[0],
                        part_us[1] - part_us[0], us[1] - part_us[1]);
        }
    }
    return bad ? 1 : 0;
}
