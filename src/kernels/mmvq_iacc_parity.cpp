// src/kernels/mmvq_iacc_parity.cpp - exp/iacc: the integer-accumulating gfx906 wave64 MMVQ (STRATA_MMVQ_IACC) against
// the default wave kernel and an fp64 reference, on real model weights.
//
//     mmvq_iacc_parity <dir> [--bench]
//
// <dir> holds raw GGUF tensors named <name>.<TYPE>.<n_in>x<n_out>.bin (TYPE Q6_K / Q4_K / Q5_K / IQ4_XS).  The
// activations are synthetic (normal, with log-normal per-channel scales so a few channels dominate, like the residual
// stream), quantized to Q8_1 by the engine's own kernel.  The reference is the fp64 sum of the dequantized weights times
// the dequantized Q8_1 values, so it measures only the kernels' arithmetic (both read the same bytes).  Per case it
// prints, for the old and the new kernel: ||y - ref|| / ||ref||, max |y - ref| / rms(ref), and mean |y - ref| / rms(ref).
// Pass: the new kernel's rel_l2 and mean errors within 2x the old kernel's (or below 1e-6 relative), finite outputs, and every column of
// a T-column call bitwise equal to a 1-column call on that column (the batch-invariance the wave kernel also keeps).
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/native_mmvq.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <dirent.h>
#include <random>
#include <string>
#include <vector>

namespace {

double h2d(uint16_t h) {
    const int s = h >> 15, e = (h >> 10) & 31, m = h & 1023;
    double v;
    if (e == 0) v = std::ldexp(double(m), -24);
    else if (e == 31) v = m ? NAN : INFINITY;
    else v = std::ldexp(double(m | 1024), e - 25);
    return s ? -v : v;
}
uint16_t rd16(const uint8_t* p) { uint16_t v; std::memcpy(&v, p, 2); return v; }

void scale_min_k4(int j, const uint8_t* q, int& d, int& m) {
    if (j < 4) { d = q[j] & 63; m = q[j + 4] & 63; }
    else { d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4); m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4); }
}
const int8_t kIQ4NL[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};

// ggml's dequantize_row_* in fp64
void dequant_block(int type, const uint8_t* b, double* y) {
    if (type == 14) {   // Q6_K
        const uint8_t* ql = b; const uint8_t* qh = b + 128; const int8_t* sc = reinterpret_cast<const int8_t*>(b + 192);
        const double d = h2d(rd16(b + 208));
        for (int n = 0; n < 256; n += 128) {
            for (int l = 0; l < 32; ++l) {
                const int is = l / 16;
                const int q1 = ((ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                const int q2 = ((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                const int q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                const int q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                y[n + l] = d * sc[is + 0] * q1;
                y[n + l + 32] = d * sc[is + 2] * q2;
                y[n + l + 64] = d * sc[is + 4] * q3;
                y[n + l + 96] = d * sc[is + 6] * q4;
            }
            ql += 64; qh += 32; sc += 8;
        }
    } else if (type == 12 || type == 13) {   // Q4_K / Q5_K
        const double d = h2d(rd16(b)), mn = h2d(rd16(b + 2));
        const uint8_t* scales = b + 4;
        const uint8_t* qh = b + 16;
        const uint8_t* q = type == 12 ? b + 16 : b + 48;
        int is = 0, u1 = 1, u2 = 2;
        for (int j = 0; j < 256; j += 64) {
            int sc, m;
            scale_min_k4(is + 0, scales, sc, m);
            const double d1 = d * sc, m1 = mn * m;
            scale_min_k4(is + 1, scales, sc, m);
            const double d2 = d * sc, m2 = mn * m;
            for (int l = 0; l < 32; ++l) {
                const int h1 = type == 13 && (qh[l] & u1) ? 16 : 0, h2 = type == 13 && (qh[l] & u2) ? 16 : 0;
                y[j + l] = d1 * ((q[l] & 0xF) + h1) - m1;
                y[j + l + 32] = d2 * ((q[l] >> 4) + h2) - m2;
            }
            q += 32; is += 2; u1 <<= 2; u2 <<= 2;
        }
    } else if (type == 23) {   // IQ4_XS
        const double d = h2d(rd16(b));
        const uint16_t sh = rd16(b + 2);
        const uint8_t* sl = b + 4; const uint8_t* qs = b + 8;
        for (int ib = 0; ib < 8; ++ib) {
            const int ls = ((sl[ib / 2] >> 4 * (ib % 2)) & 0xf) | (((sh >> 2 * ib) & 3) << 4);
            const double dl = d * (ls - 32);
            for (int j = 0; j < 16; ++j) {
                y[32 * ib + j] = dl * kIQ4NL[qs[16 * ib + j] & 0xf];
                y[32 * ib + j + 16] = dl * kIQ4NL[qs[16 * ib + j] >> 4];
            }
        }
    }
}

struct Tensor { std::string name; int type, n_in, n_out, bbytes; std::vector<uint8_t> w; };

bool ck(cudaError_t e, const char* what) {
    if (e == cudaSuccess) return true;
    std::printf("CUDA: %s: %s\n", what, cudaGetErrorString(e));
    return false;
}

struct Err { double rel_l2, max_rms, mean_rms; long long nonfinite; };
Err errors(const std::vector<float>& y, const std::vector<double>& ref) {
    double se = 0, sr = 0, mx = 0, sa = 0;
    long long nf = 0;
    for (std::size_t i = 0; i < y.size(); ++i) {
        if (!std::isfinite(y[i])) { ++nf; continue; }
        const double e = std::fabs(double(y[i]) - ref[i]);
        se += e * e; sr += ref[i] * ref[i]; sa += e; mx = std::max(mx, e);
    }
    const double rms = std::sqrt(sr / double(y.size()));
    return {std::sqrt(se / sr), mx / rms, sa / double(y.size()) / rms, nf};
}

int run(const Tensor& t, int T, bool bench, cudaStream_t s, void* dw, int mask) {
    const std::size_t qbytes = strata::kernels::native_q8_1_bytes(t.n_in, T);
    std::mt19937 rng(777u + unsigned(T) * 31u + unsigned(t.n_out));
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> chs(t.n_in);
    for (auto& c : chs) c = std::exp(0.8f * nd(rng));   // log-normal per-channel scale
    std::vector<float> x(std::size_t(T) * t.n_in);
    for (int j = 0; j < T; ++j)
        for (int k = 0; k < t.n_in; ++k) x[std::size_t(j) * t.n_in + k] = nd(rng) * chs[k];
    void* xq = nullptr; float *dx = nullptr, *dy = nullptr, *d1 = nullptr;
    const std::size_t ybytes = std::size_t(T) * t.n_out * 4;
    if (!ck(cudaMalloc(&xq, qbytes), "xq") || !ck(cudaMalloc(&dx, x.size() * 4), "x") ||
        !ck(cudaMalloc(&dy, ybytes), "y") || !ck(cudaMalloc(&d1, std::size_t(t.n_out) * 4), "y1"))
        return -1;
    cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice);
    strata::kernels::quantize_q8_1_rows(dx, T, t.n_in, xq, s);
    std::vector<uint8_t> hq(qbytes);
    cudaStreamSynchronize(s);
    cudaMemcpy(hq.data(), xq, qbytes, cudaMemcpyDeviceToHost);
    // the dequantized Q8_1 columns (36-byte blocks: half d, half s, 32 int8)
    const int nb = t.n_in / 32;
    std::vector<double> xd(std::size_t(T) * t.n_in);
    for (int j = 0; j < T; ++j)
        for (int b = 0; b < nb; ++b) {
            const uint8_t* blk = hq.data() + (std::size_t(j) * nb + b) * 36;
            const double d = h2d(rd16(blk));
            for (int k = 0; k < 32; ++k) xd[std::size_t(j) * t.n_in + 32 * b + k] = d * double(int8_t(blk[4 + k]));
        }
    std::vector<double> ref(std::size_t(T) * t.n_out);
    std::vector<double> wr(t.n_in);
    const int bpr = t.n_in / 256;
    for (int r = 0; r < t.n_out; ++r) {
        for (int b = 0; b < bpr; ++b)
            dequant_block(t.type, t.w.data() + (std::size_t(r) * bpr + b) * t.bbytes, wr.data() + 256 * b);
        for (int j = 0; j < T; ++j) {
            double acc = 0;
            const double* xc = xd.data() + std::size_t(j) * t.n_in;
            for (int k = 0; k < t.n_in; ++k) acc += wr[k] * xc[k];
            ref[std::size_t(j) * t.n_out + r] = acc;
        }
    }
    std::vector<float> yo(std::size_t(T) * t.n_out), yn(yo.size()), y1(t.n_out);
    auto call = [&](int m, int ncols, const void* xin, float* out) {
        strata::kernels::native_mmvq_set_iacc(m);
        strata::kernels::native_mmvq(t.type, dw, xin, out, t.n_in, t.n_out, ncols, s);
    };
    call(0, T, xq, dy);
    cudaStreamSynchronize(s);
    cudaMemcpy(yo.data(), dy, ybytes, cudaMemcpyDeviceToHost);
    cudaMemsetAsync(dy, 0xff, ybytes, s);
    call(mask, T, xq, dy);
    if (!ck(cudaStreamSynchronize(s), "iacc")) return -1;
    cudaMemcpy(yn.data(), dy, ybytes, cudaMemcpyDeviceToHost);
    // batch invariance: the last column alone
    const std::size_t qcol = strata::kernels::native_q8_1_bytes(t.n_in, 1);
    call(mask, 1, static_cast<const uint8_t*>(xq) + std::size_t(T - 1) * qcol, d1);
    cudaStreamSynchronize(s);
    cudaMemcpy(y1.data(), d1, std::size_t(t.n_out) * 4, cudaMemcpyDeviceToHost);
    long long coldiff = 0, bitsame = 0;
    for (int r = 0; r < t.n_out; ++r)
        if (std::memcmp(&y1[r], &yn[std::size_t(T - 1) * t.n_out + r], 4) != 0) ++coldiff;
    for (std::size_t i = 0; i < yo.size(); ++i) bitsame += std::memcmp(&yo[i], &yn[i], 4) == 0;
    const Err eo = errors(yo, ref), en = errors(yn, ref);
    const bool ok = en.nonfinite == 0 && coldiff == 0 && (en.rel_l2 <= 2.0 * eo.rel_l2 || en.rel_l2 < 1e-6) &&
                    (en.mean_rms <= 2.0 * eo.mean_rms || en.mean_rms < 1e-6);
    // max |err| is one sample of n_out * T and reported only (a 640-row case swings 0.5-2.5x between runs)
    std::printf("ERR %-32s T=%d old rel_l2 %.3e max %.3e mean %.3e | iacc rel_l2 %.3e max %.3e mean %.3e | ratio l2 %.2f "
                "mean %.2f max %.2f | same bits %.1f%% | col-invariant %s%s\n",
                t.name.c_str(), T, eo.rel_l2, eo.max_rms, eo.mean_rms, en.rel_l2, en.max_rms, en.mean_rms,
                en.rel_l2 / eo.rel_l2, en.mean_rms / eo.mean_rms, en.max_rms / eo.max_rms, 100.0 * double(bitsame) / double(yo.size()),
                coldiff ? "NO" : "yes", ok ? "" : "  FAIL");
    if (bench) {
        constexpr int NC = 4;   // old (STRATA_MMVQ_WAVE_ROWS), iacc default, iacc R2, iacc R4
        const int cfg_mask[NC] = {0, mask, mask, mask};
        const int cfg_rows[NC] = {0, 0, 2, 4};
        const int cfg_k[NC] = {0, 0, 4, 4};
        double us[NC];
        cudaEvent_t a, b;
        cudaEventCreate(&a); cudaEventCreate(&b);
        for (int c = 0; c < NC; ++c) {
            strata::kernels::native_mmvq_set_iacc(cfg_mask[c], cfg_rows[c], cfg_k[c]);
            for (int i = 0; i < 20; ++i) strata::kernels::native_mmvq(t.type, dw, xq, dy, t.n_in, t.n_out, T, s);
            std::vector<float> smp;
            for (int rep = 0; rep < 7; ++rep) {
                const int iters = 100;
                cudaEventRecord(a, s);
                for (int i = 0; i < iters; ++i) strata::kernels::native_mmvq(t.type, dw, xq, dy, t.n_in, t.n_out, T, s);
                cudaEventRecord(b, s);
                cudaEventSynchronize(b);
                float ms = 0;
                cudaEventElapsedTime(&ms, a, b);
                smp.push_back(ms * 1000.f / iters);
            }
            std::sort(smp.begin(), smp.end());
            us[c] = smp[smp.size() / 2];
        }
        cudaEventDestroy(a); cudaEventDestroy(b);
        const double mb = double(t.w.size());
        std::printf("BENCH %-32s T=%d %7.2f MB | old %6.1fus %4.0fGB/s", t.name.c_str(), T, mb / 1e6, us[0], mb / us[0] / 1e3);
        double best = us[1];
        for (int c = 1; c < NC; ++c) {
            std::printf(" | R%dK%d %6.1fus %4.0f", cfg_rows[c], cfg_k[c], us[c], mb / us[c] / 1e3);
            best = std::min(best, us[c]);
        }
        std::printf(" | best vs old %+5.1f%%\n", 100.0 * (best / us[0] - 1.0));
    }
    strata::kernels::native_mmvq_set_iacc(-1);
    cudaFree(xq); cudaFree(dx); cudaFree(dy); cudaFree(d1);
    return ok ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) { std::printf("usage: mmvq_iacc_parity <dir> [--bench] [--tmax N]\n"); return 2; }
    const std::string dir = argv[1];
    bool bench = false;
    int tmax = 5;
    for (int i = 2; i < argc; ++i) {
        if (std::string(argv[i]) == "--bench") bench = true;
        if (std::string(argv[i]) == "--tmax" && i + 1 < argc) tmax = std::atoi(argv[++i]);
    }
    std::vector<std::string> files;
    if (DIR* d = opendir(dir.c_str())) {
        while (dirent* e = readdir(d)) {
            const std::string n = e->d_name;
            if (n.size() > 4 && n.substr(n.size() - 4) == ".bin") files.push_back(n);
        }
        closedir(d);
    }
    std::sort(files.begin(), files.end());
    cudaStream_t s;
    cudaStreamCreate(&s);
    int bad = 0, ran = 0;
    for (const auto& f : files) {
        // <name>.<TYPE>.<n_in>x<n_out>.bin
        const std::size_t p3 = f.rfind('.'), p2 = f.rfind('.', p3 - 1), p1 = f.rfind('.', p2 - 1);
        const std::string ty = f.substr(p1 + 1, p2 - p1 - 1), dims = f.substr(p2 + 1, p3 - p2 - 1);
        Tensor t;
        t.name = f.substr(0, p1);
        t.n_in = std::atoi(dims.c_str());
        t.n_out = std::atoi(dims.c_str() + dims.find('x') + 1);
        int mask = 0;
        if (ty == "Q6_K") { t.type = 14; t.bbytes = 210; mask = 1; }
        else if (ty == "Q4_K") { t.type = 12; t.bbytes = 144; mask = 2; }
        else if (ty == "Q5_K") { t.type = 13; t.bbytes = 176; mask = 4; }
        else if (ty == "IQ4_XS") { t.type = 23; t.bbytes = 136; mask = 0; }
        else continue;
        t.name += "." + ty;
        FILE* fp = std::fopen((dir + "/" + f).c_str(), "rb");
        if (!fp) continue;
        t.w.resize(std::size_t(t.n_out) * (t.n_in / 256) * t.bbytes);
        const std::size_t got = std::fread(t.w.data(), 1, t.w.size(), fp);
        std::fclose(fp);
        if (got != t.w.size()) { std::printf("%s: short read\n", f.c_str()); ++bad; continue; }
        void* dw = nullptr;
        if (!ck(cudaMalloc(&dw, t.w.size()), "w")) return 1;
        cudaMemcpy(dw, t.w.data(), t.w.size(), cudaMemcpyHostToDevice);
        for (int T = 1; T <= tmax; ++T) {
            if (mask == 0 && !bench) break;
            const int r = run(t, T, bench, s, dw, mask);
            if (r < 0) return 1;
            bad += r; ++ran;
        }
        cudaFree(dw);
    }
    std::printf("\nmmvq_iacc_parity: %s (%d cases, %d failed)\n", bad ? "FAILED" : "ok", ran, bad);
    return bad ? 1 : 0;
}
