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
};
const int WIDTHS[] = {1, 2, 3, 4, 5};
// exp/tier1: rows a wavefront, STRATA_MMVQ_MUL24, STRATA_MMVQ_Q6K_R4 - every config bitwise the first's
const int ROWS[] = {1, 2, 4, 1, 2, 2, 2};
const int MUL24[] = {0, 0, 0, 1, 1, 0, 1};
const int Q6R4[] = {0, 0, 0, 0, 0, 1, 1};
constexpr int NCFG = 7;

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
        try {
            cudaMemsetAsync(dy, 0xff, ybytes, s);
            strata::kernels::native_mmvq(c.type, dw, xq, dy, c.n_in, c.n_out, T, s);
        } catch (const std::exception& e) {
            std::printf("%-10s T=%d R=%d: %s\n", c.name, T, ROWS[ri], e.what());
            return -1;
        }
        if (!ck(cudaStreamSynchronize(s), "mmvq")) return -1;
        std::vector<uint32_t>& dst = ri == 0 ? ref : got;
        if (!ck(cudaMemcpy(dst.data(), dy, ybytes, cudaMemcpyDeviceToHost), "read y")) return -1;
        if (ri > 0) {
            long long diff = 0, nonfinite = 0;
            for (std::size_t i = 0; i < ref.size(); ++i) {
                float f;
                std::memcpy(&f, &ref[i], 4);
                if (!std::isfinite(f)) ++nonfinite;
                if (ref[i] != got[i]) ++diff;
            }
            if (diff != 0 || nonfinite != 0) {
                std::printf("%-10s T=%d cfg %d (R=%d m24 %d q6r4 %d): %lld of %zu outputs differ from R=1, %lld non-finite  FAIL\n",
                            c.name, T, ri, ROWS[ri], MUL24[ri], Q6R4[ri], diff, ref.size(), nonfinite);
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
    if (bench) {
        std::printf("BENCH %-10s T=%d %6.2f MB", c.name, T, wbytes / 1e6);
        for (int ri = 0; ri < NCFG; ++ri)
            std::printf(" | R%d%s%s %6.1fus", ROWS[ri], MUL24[ri] ? "m24" : "", Q6R4[ri] ? "q4" : "", us[ri]);
        std::printf(" | R2: m24 %+5.1f%% q6r4 %+5.1f%% both %+5.1f%%%s\n", 100.0 * (us[1] / us[4] - 1.0),
                    100.0 * (us[1] / us[5] - 1.0), 100.0 * (us[1] / us[6] - 1.0), verdict ? "  FAIL" : "");
    }
    else if (verdict == 0)
        std::printf("%-10s T=%d: R=2/R=4/MUL24/Q6K_R4 bitwise equal to R=1 (%zu outputs)\n", c.name, T, ref.size());
    cudaFree(dw); cudaFree(xq); cudaFree(dx); cudaFree(dy);
    return verdict;
}

}  // namespace

int main(int argc, char** argv) {
    const bool bench = argc > 1 && std::string(argv[1]) == "--bench";
    cudaStream_t s;
    if (!ck(cudaStreamCreate(&s), "stream create")) return 1;
    int bad = 0, skipped = 0;
    for (const Case& c : CASES)
        for (int T : WIDTHS) {
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
