// Times strata's native_*_mmvq on gfx906 at this model's decode shapes and reports GB/s of weight bytes.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <random>
#include <string>
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/iq_kernels.hpp"
using namespace strata::kernels;
static void iq3s_mmvq(const void* w, const void* x, float* y, int n_in, int n_out, int nc, void* s) { iq_mmvq(21, w, x, y, n_in, n_out, nc, s); }
static void iq2xs_mmvq(const void* w, const void* x, float* y, int n_in, int n_out, int nc, void* s) { iq_mmvq(17, w, x, y, n_in, n_out, nc, s); }
#define CK(x) do { hipError_t e = (x); if (e != hipSuccess) { std::printf("%s: %s\n", #x, hipGetErrorString(e)); std::exit(1); } } while (0)
using Fn = void (*)(const void*, const void*, float*, int, int, int, void*);
struct Fmt { const char* name; Fn fn; int bpb; int epb; std::vector<int> scale_offs; };

__global__ void read_kernel(const uint4* p, size_t n, float* out) {
    uint32_t acc = 0;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x) { uint4 v = p[i]; acc ^= v.x ^ v.y ^ v.z ^ v.w; }
    if (acc == 0x12345678u) out[0] = 1.f;
}

int main(int argc, char** argv) {
    int iters = 200;
    FILE* dump = argc > 1 && std::strcmp(argv[1], "-") ? std::fopen(argv[1], "wb") : nullptr;
    hipStream_t s; CK(hipStreamCreate(&s));
    hipEvent_t a, b; CK(hipEventCreate(&a)); CK(hipEventCreate(&b));
    float* dummy; CK(hipMalloc(&dummy, 4));
    if (!std::getenv("MB_ROOF")) {} else { // the read roofline (MB_ROOF=1)
        size_t bytes = size_t(1) << 30; uint4* p; CK(hipMalloc(&p, bytes)); CK(hipMemset(p, 1, bytes));
        for (int w = 0; w < 3; ++w) read_kernel<<<64 * 40, 256, 0, s>>>(p, bytes / 16, dummy);
        CK(hipEventRecord(a, s)); for (int i = 0; i < 20; ++i) read_kernel<<<64 * 40, 256, 0, s>>>(p, bytes / 16, dummy);
        CK(hipEventRecord(b, s)); CK(hipEventSynchronize(b)); float ms; CK(hipEventElapsedTime(&ms, a, b));
        std::printf("read roofline: %.0f GB/s\n", 20.0 * bytes / (ms * 1e6)); CK(hipFree(p));
    }
    std::vector<Fmt> fmts = {
        {"Q6_K", native_q6_k_mmvq, 210, 256, {208}},
        {"Q4_K", native_q4_k_mmvq, 144, 256, {0, 2}},
        {"IQ4_XS", native_iq4_xs_mmvq, 136, 256, {0}},
        {"IQ4_NL", native_iq4_nl_mmvq, 18, 32, {0}},
        {"Q5_K", native_q5_k_mmvq, 176, 256, {0, 2}},
        {"Q4_0", native_q4_0_mmvq, 18, 32, {0}},
        {"Q3_K", native_q3_k_mmvq, 110, 256, {108}},
        {"Q2_0", native_q2_0_mmvq, 18, 64, {0}},
        {"Q5_0", native_q5_0_mmvq, 22, 32, {0}},
        {"Q8_0", native_q8_0_mmvq, 34, 32, {0}},
        {"IQ3_S", iq3s_mmvq, 110, 256, {0}},
    };
    struct Case { const char* what; const char* fmt; int n_in, n_out; };
    std::vector<Case> cases = {
        {"dhead q6", "Q6_K", 2560, 106299}, {"dhead q4_0", "Q4_0", 2560, 106299}, {"dhead xs", "IQ4_XS", 2560, 106299}, {"dhead q4k", "Q4_K", 2560, 106299},
        {"GDN qkv", "Q6_K", 2560, 10240}, {"GDN out", "Q6_K", 6144, 2560}, {"GDN z", "Q4_K", 2560, 6144},
        {"QSA out", "Q4_K", 6144, 2560}, {"QSA q", "IQ4_XS", 2560, 12288}, {"shexp dn", "IQ4_NL", 640, 2560}, {"shexp gate", "Q4_K", 2560, 640},
        {"GDN qkv5", "Q5_K", 2560, 10240}, {"shexp q6", "Q6_K", 2560, 640}, {"shexp xs", "IQ4_XS", 2560, 640},
        {"qkv q3k", "Q3_K", 2560, 10240}, {"gate q3k", "Q3_K", 2560, 6144}, {"out q3k", "Q3_K", 6144, 2560}, {"shexp q3k", "Q3_K", 2560, 640},
        {"shexp q2", "Q2_0", 2560, 640}, {"shdn q2", "Q2_0", 640, 2560}, {"qkv q2", "Q2_0", 2560, 10240},
        {"shdn q4_0", "Q4_0", 640, 2560}, {"shdn q5_0", "Q5_0", 640, 2560}, {"shdn q8_0", "Q8_0", 640, 2560},
        {"gate iq3s", "IQ3_S", 2560, 6144}, {"out iq3s", "IQ3_S", 6144, 2560}, {"shexp iq3s", "IQ3_S", 2560, 640}, {"qkv iq3s", "IQ3_S", 2560, 10240},
    };
    // argv[2]: only the cases whose name or format contains it
    if (argc > 2) { std::vector<Case> k; for (auto& c : cases) if (std::strstr(c.what, argv[2]) || std::strstr(c.fmt, argv[2])) k.push_back(c); cases = k; }
    std::mt19937 rng(1);
    for (auto& c : cases) {
        const Fmt* f = nullptr; for (auto& x : fmts) if (!std::strcmp(x.name, c.fmt)) f = &x;
        size_t nb = size_t(c.n_out) * (c.n_in / f->epb), wbytes = nb * f->bpb;
        const int copies = std::max<int>(1, int((256u << 20) / wbytes));       // rotate copies so L2 cannot hold them
        std::vector<uint8_t> h(wbytes); for (auto& v : h) v = uint8_t(rng());
        for (size_t i = 0; i < nb; ++i) for (int o : f->scale_offs) { uint16_t d = 0x1419; std::memcpy(&h[i * f->bpb + o], &d, 2); }
        std::vector<void*> w(copies); for (auto& p : w) { CK(hipMalloc(&p, wbytes)); CK(hipMemcpy(p, h.data(), wbytes, hipMemcpyHostToDevice)); }
        float* x; void* xq; float* y;
        CK(hipMalloc(&x, size_t(c.n_in) * 8 * 4)); CK(hipMalloc(&xq, native_q8_1_bytes(c.n_in, 8))); CK(hipMalloc(&y, size_t(c.n_out) * 8 * 4));
        std::vector<float> hx(size_t(c.n_in) * 8); std::normal_distribution<float> nd(0, 1); for (auto& v : hx) v = nd(rng);
        CK(hipMemcpy(x, hx.data(), hx.size() * 4, hipMemcpyHostToDevice));
        native_quantize_q8_1(x, xq, c.n_in, 8, s);
        std::printf("%-9s %-6s %5dx%-5d %6.2f MB:", c.what, c.fmt, c.n_out, c.n_in, wbytes / 1e6);
        for (int nc : {1, 2, 3, 4}) {
            for (int i = 0; i < 5; ++i) f->fn(w[i % copies], xq, y, c.n_in, c.n_out, nc, s);
            CK(hipEventRecord(a, s));
            for (int i = 0; i < iters; ++i) f->fn(w[i % copies], xq, y, c.n_in, c.n_out, nc, s);
            CK(hipEventRecord(b, s)); CK(hipEventSynchronize(b)); float ms; CK(hipEventElapsedTime(&ms, a, b));
            if (dump) { f->fn(w[0], xq, y, c.n_in, c.n_out, nc, s); CK(hipStreamSynchronize(s)); std::vector<float> hy(size_t(c.n_out) * nc);
                        CK(hipMemcpy(hy.data(), y, hy.size() * 4, hipMemcpyDeviceToHost)); std::fwrite(hy.data(), 4, hy.size(), dump); }
            double us = ms * 1e3 / iters; std::printf("  T%d %6.1f us %4.0f GB/s", nc, us, wbytes / (us * 1e3));
        }
        std::printf("\n");
        for (auto& p : w) CK(hipFree(p)); CK(hipFree(x)); CK(hipFree(xq)); CK(hipFree(y));
    }
}
