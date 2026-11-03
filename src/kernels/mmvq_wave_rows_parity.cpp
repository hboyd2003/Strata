// src/kernels/mmvq_wave_rows_parity.cpp - gfx906's wave64 MMVQ with 2 or 4 rows a wavefront, bitwise vs 1 row.
//
//     build-906/mmvq_wave_rows_parity            parity only
//     build-906/mmvq_wave_rows_parity --bench    + the time of each rows choice and its weight bandwidth
//
// STRATA_MMVQ_WAVE_ROWS (native_mmvq_set_wave_rows) changes how many rows one wavefront of native_mmvq_wave_kernel
// takes.  Each row's sum is computed by the same lanes in the same order either way, so every output must be the
// R = 1 kernel's bit for bit.  This checks that on the shapes and types of the Flash-Next dense projections (the
// qkv, gate and out projections of a decode window) at 1-5 columns, on the same weights and the same Q8_1 bytes.
// The data is mmvq_multi_parity's: random weight bytes with sane fp16 scales, normal activations.
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/native_mmvq.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

struct Case {
    const char* name;
    int type;
    int n_in;
    int n_out;
    int block_elems;
    int block_bytes;
    int scale_at[2];
};

const Case CASES[] = {
    {"Q6_K qkv", 14, 2560, 10240, 256, 210, {208, -1}},
    {"Q5_K qkv", 13, 2560, 10240, 256, 176, {0, 2}},
    {"Q4_K gate", 12, 2560, 6144, 256, 144, {0, 2}},
    {"Q6_K gate", 14, 2560, 6144, 256, 210, {208, -1}},
    {"Q6_K out", 14, 6144, 2560, 256, 210, {208, -1}},
    {"IQ4_XS", 23, 2560, 6144, 256, 136, {0, -1}},
    {"Q6_K q", 14, 2560, 12288, 256, 210, {208, -1}},
    {"Q6_K head", 14, 2560, 248320, 256, 210, {208, -1}},
    {"Q6_K dhead", 14, 2560, 106299, 256, 210, {208, -1}},   // the MTP draft head over rt/draft_vocab.bin
    {"Q6_K dhead en", 14, 2560, 40525, 256, 210, {208, -1}},   // ... over draft_vocab_en.bin
};
const int WIDTHS[] = {1, 2, 3, 4, 5, 6};
// exp/tier1: rows a wavefront, STRATA_MMVQ_MUL24, STRATA_MMVQ_Q6K_R4; #5033: LDS (native_mmvq_set_wave_lds: 1 XLDS,
// 2 Q6X4, 4 WPB 8, 8 Q4X4, 16 IACC_XLDS).  Every wave config bitwise config 0's; the iacc configs (IACC 1, rounding-only
// vs the wave kernel) bitwise the first iacc config's (CFG_IACC).
const int ROWS[] = {1, 2, 4, 1, 2, 2, 2, 1, 2, 1, 1, 1, 0, 0};
const int MUL24[] = {0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0};
const int Q6R4[] = {0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0};
const int LDS[] = {0, 0, 0, 0, 0, 0, 0, 1, 1, 11, 15, 15, 0, 16};
const int IACC[] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 7};
constexpr int NCFG = 14;
constexpr int CFG_PROD_WAVE = 4;   // prod's wave config (WAVE_ROWS 2, MUL24); prod takes iacc for the K-quants
constexpr int CFG_IACC = 12;
const char* cfg_name(int ri) {
    static const char* names[NCFG] = {"R1", "R2", "R4", "R1m24", "R2m24", "R2q6r4", "R2m24q6r4", "R1xl", "R2m24xl",
                                      "x4w4", "x4w8m24", "x4w8", "iacc", "iaccxl"};
    return names[ri];
}

uint16_t sane_half(std::mt19937& rng) {
    const uint32_t r = rng();
    return (uint16_t) (((r >> 31) << 15) | ((5u + (r >> 10) % 5u) << 10) | (r & 0x3ffu));
}

bool ck(cudaError_t e, const char* what) {
    if (e == cudaSuccess) return true;
    std::printf("CUDA: %s: %s\n", what, cudaGetErrorString(e));
    return false;
}

// 0 ok, 1 differs, -1 could not run
int run_case(const Case& c, int T, bool bench, cudaStream_t s) {
    if (!strata::kernels::native_mmvq_supported(c.type)) return -1;
    const std::size_t wbytes = strata::kernels::native_mmvq_weight_bytes(c.type, c.n_in, c.n_out);
    const std::size_t qall = strata::kernels::native_q8_1_bytes(c.n_in, T);
    const std::size_t n_blocks = (std::size_t) c.n_out * (std::size_t) (c.n_in / c.block_elems);
    if (n_blocks * (std::size_t) c.block_bytes != wbytes) {
        std::printf("%-10s: %zu blocks of %d bytes are not the %zu weight bytes native_mmvq expects\n", c.name,
                    n_blocks, c.block_bytes, wbytes);
        return -1;
    }
    std::mt19937 rng(4321u + (unsigned) c.type * 97u + (unsigned) T);
    std::vector<uint8_t> w(wbytes);
    for (auto& b : w) b = (uint8_t) (rng() & 0xff);
    for (std::size_t k = 0; k < n_blocks; ++k)
        for (int at : c.scale_at)
            if (at >= 0) {
                const uint16_t h = sane_half(rng);
                std::memcpy(&w[k * (std::size_t) c.block_bytes + (std::size_t) at], &h, 2);
            }
    std::vector<float> x((std::size_t) T * c.n_in);
    std::normal_distribution<float> nd(0.f, 1.f);
    for (auto& v : x) v = nd(rng);

    void *dw = nullptr, *xq = nullptr;
    float *dx = nullptr, *dy = nullptr;
    const std::size_t ybytes = (std::size_t) T * c.n_out * 4;
    if (!ck(cudaMalloc(&dw, wbytes), "malloc w") || !ck(cudaMalloc(&xq, qall), "malloc xq") ||
        !ck(cudaMalloc(&dx, x.size() * 4), "malloc x") || !ck(cudaMalloc(&dy, ybytes), "malloc y"))
        return -1;
    if (!ck(cudaMemcpy(dw, w.data(), wbytes, cudaMemcpyHostToDevice), "copy w") ||
        !ck(cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice), "copy x"))
        return -1;
    strata::kernels::quantize_q8_1_rows(dx, T, c.n_in, xq, s);

    std::vector<uint32_t> ref((std::size_t) T * c.n_out), got(ref.size());
    int verdict = 0;
    double us[NCFG] = {};
    for (int ri = 0; ri < NCFG; ++ri) {
        strata::kernels::native_mmvq_set_wave_rows(ROWS[ri]);
        strata::kernels::native_mmvq_set_mul24(MUL24[ri]);
        strata::kernels::native_mmvq_set_q6k_r4(Q6R4[ri]);
        strata::kernels::native_mmvq_set_wave_lds(LDS[ri]);
        strata::kernels::native_mmvq_set_iacc(IACC[ri], 0, 0);
        try {
            cudaMemsetAsync(dy, 0xff, ybytes, s);
            strata::kernels::native_mmvq(c.type, dw, xq, dy, c.n_in, c.n_out, T, s);
        } catch (const std::exception& e) {
            std::printf("%-10s T=%d R=%d: %s\n", c.name, T, ROWS[ri], e.what());
            return -1;
        }
        if (!ck(cudaStreamSynchronize(s), "mmvq")) return -1;
        const bool is_ref = ri == 0 || ri == CFG_IACC;
        std::vector<uint32_t>& dst = is_ref ? ref : got;
        if (!ck(cudaMemcpy(dst.data(), dy, ybytes, cudaMemcpyDeviceToHost), "read y")) return -1;
        if (!is_ref) {
            long long diff = 0, nonfinite = 0;
            for (std::size_t i = 0; i < ref.size(); ++i) {
                float f;
                std::memcpy(&f, &ref[i], 4);
                if (!std::isfinite(f)) ++nonfinite;
                if (ref[i] != got[i]) ++diff;
            }
            if (diff != 0 || nonfinite != 0) {
                std::printf("%-10s T=%d cfg %d (%s): %lld of %zu outputs differ from %s, %lld non-finite  FAIL\n",
                            c.name, T, ri, cfg_name(ri), diff, ref.size(), ri > CFG_IACC ? "iacc" : "R1", nonfinite);
                verdict = 1;
            }
        }
        if (bench) {
            cudaEvent_t a, b;
            cudaEventCreate(&a);
            cudaEventCreate(&b);
            const int iters = 200;
            for (int i = 0; i < 20; ++i) strata::kernels::native_mmvq(c.type, dw, xq, dy, c.n_in, c.n_out, T, s);
            std::vector<float> samples;
            for (int rep = 0; rep < 5; ++rep) {
                cudaEventRecord(a, s);
                for (int i = 0; i < iters; ++i) strata::kernels::native_mmvq(c.type, dw, xq, dy, c.n_in, c.n_out, T, s);
                cudaEventRecord(b, s);
                cudaEventSynchronize(b);
                float ms = 0;
                cudaEventElapsedTime(&ms, a, b);
                samples.push_back(ms * 1000.f / iters);
            }
            std::sort(samples.begin(), samples.end());
            us[ri] = samples[samples.size() / 2];
            cudaEventDestroy(a);
            cudaEventDestroy(b);
        }
    }
    strata::kernels::native_mmvq_set_wave_rows(0);
    strata::kernels::native_mmvq_set_mul24(-1);
    strata::kernels::native_mmvq_set_q6k_r4(-1);
    strata::kernels::native_mmvq_set_wave_lds(-1);
    strata::kernels::native_mmvq_set_iacc(-1, 0, 0);
    if (bench) {
        const bool kq = c.type == 12 || c.type == 13 || c.type == 14;   // prod: iacc for the K-quants
        const double prod = kq ? us[CFG_IACC] : us[CFG_PROD_WAVE];
        std::printf("BENCH %-13s T=%d %6.2f MB", c.name, T, wbytes / 1e6);
        for (int ri = 0; ri < NCFG; ++ri) std::printf(" | %s %6.1f", cfg_name(ri), us[ri]);
        std::printf(" | vs prod(%s %.1fus): R1xl %+5.1f%% R2m24xl %+5.1f%% x4w4 %+5.1f%% x4w8 %+5.1f%% iaccxl %+5.1f%%%s\n",
                    kq ? "iacc" : "R2m24", prod, 100.0 * (us[7] / prod - 1.0), 100.0 * (us[8] / prod - 1.0),
                    100.0 * (us[9] / prod - 1.0), 100.0 * (us[10] / prod - 1.0), 100.0 * (us[13] / prod - 1.0),
                    verdict ? "  FAIL" : "");
    }
    else if (verdict == 0)
        std::printf("%-13s T=%d: every wave config bitwise R=1, iacc xl bitwise iacc (%zu outputs)\n", c.name, T, ref.size());
    cudaFree(dw); cudaFree(xq); cudaFree(dx); cudaFree(dy);
    return verdict;
}

}  // namespace

int main(int argc, char** argv) {
    const bool bench = argc > 1 && std::string(argv[1]) == "--bench";
    cudaStream_t s;
    if (!ck(cudaStreamCreate(&s), "stream create")) return 1;
    int bad = 0, skipped = 0;
    const char* only = std::getenv("STRATA_BENCH_CASE");   // a substring of the case names to run (all: unset)
    for (const Case& c : CASES)
        for (int T : WIDTHS) {
            if (only != nullptr && std::strstr(c.name, only) == nullptr) continue;
            const int r = run_case(c, T, bench, s);
            if (r < 0) ++skipped;
            else bad += r;
        }
    if (bad != 0 || skipped != 0) {
        std::printf("\nmmvq_wave_rows_parity: FAILED (%d case(s) differ, %d did not run)\n", bad, skipped);
        return 1;
    }
    std::printf("\nmmvq_wave_rows_parity: ok (every rows choice bitwise equal to one row a wavefront)\n");
    return 0;
}
