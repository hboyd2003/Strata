// Standalone HIP probe for the gfx906 prompt attention (qsa_prompt_attn_gfx906.cu): accuracy against the decode kernel
// it replaces, and the time both take on the same selection geometry.
//
// usage: gfx906_qsa_prompt_attn_probe [nq=2048] [ctx=32768] [reps=10] [candidate.bin] [--masked-pages]
//   nq    queries in one call (the engine's prompt chunk is 2048)
//   ctx   context the last nq queries sit at (each query selects up to its own position)
//   reps  timed repetitions per path (medians printed)
// The candidate's output is written to the fourth argument (or STRATA_QSA_PROBE_OUT) for byte-level A/B; the decode
// path's output is compared against it and the tolerance is printed rather than asserted.
// Run the two paths with different environments in SEPARATE processes: the dispatcher caches its decisions.
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/qsa_decode_attn.hpp"
#include "strata/kernels/qsa_prompt_attn.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <numeric>
#include <vector>

namespace k = strata::kernels;
namespace {
constexpr int64_t kBatch = 32;      // qsa_decode_attn_batch's prompt batching (prefill.cpp's attn_batch)
constexpr size_t kGuard = 64;
constexpr uint64_t kSeed = 0x7161736150524f42ull;

[[noreturn]] void fail(const char* why) {
    std::fprintf(stderr, "qsa_prompt_attn_probe: %s\n", why);
    std::exit(2);
}
void ck(hipError_t e, const char* what) {
    if (e != hipSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, hipGetErrorString(e));
        std::exit(2);
    }
}
#define HIP_CHECK(x) ck((x), #x)

struct Rng {
    uint64_t state = kSeed;
    uint32_t next() {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        return uint32_t((state * 2685821657736338717ull) >> 32);
    }
    size_t below(size_t n) { return size_t(next()) % n; }
    template <class T> void shuffle(std::vector<T>& v) {
        for (size_t n = v.size(); n > 1; --n) std::swap(v[n - 1], v[below(n)]);
    }
};

template <class T> struct Device {
    T* p = nullptr;
    explicit Device(size_t n) { HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&p), n * sizeof(T))); }
    explicit Device(const std::vector<T>& v) : Device(v.size()) {
        HIP_CHECK(hipMemcpy(p, v.data(), v.size() * sizeof(T), hipMemcpyHostToDevice));
    }
    ~Device() { if (p) (void) hipFree(p); }
    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;
};

int64_t positive(const char* text) {
    errno = 0;
    char* end = nullptr;
    const long long n = std::strtoll(text, &end, 10);
    if (errno || end == text || *end || n <= 0 || n > INT32_MAX) fail("nq, ctx and reps must be positive int32 values");
    return int64_t(n);
}
const char* env(const char* name) {
    const char* value = std::getenv(name);
    return value ? value : "<unset>";
}
float median_ms(std::vector<float> t) {
    std::sort(t.begin(), t.end());
    return t.size() % 2 ? t[t.size() / 2] : 0.5f * (t[t.size() / 2 - 1] + t[t.size() / 2]);
}
}  // namespace

int main(int argc, char** argv) {
    try {
        std::vector<const char*> args;
        bool masked = false;
        for (int i = 1; i < argc; ++i) {
            if (std::strcmp(argv[i], "--masked-pages") == 0) masked = true;
            else args.push_back(argv[i]);
        }
        if (args.size() > 4) fail("usage: probe [nq] [ctx] [reps] [candidate.bin] [--masked-pages]");
        const int64_t nq = args.size() > 0 ? positive(args[0]) : 2048;
        const int64_t ctx = args.size() > 1 ? positive(args[1]) : 32768;
        const int reps = args.size() > 2 ? int(positive(args[2])) : 10;
        const char* output = args.size() > 3 ? args[3] : std::getenv("STRATA_QSA_PROBE_OUT");
        if (ctx < nq) fail("ctx must be >= nq");
        const k::QsaShapes s = k::qsa_real_shapes();
        if (s.n_head != 24 || s.n_head_kv != 2 || s.head_dim != 256 || s.page_size != 4)
            fail("unexpected model geometry");
        const int64_t hd = s.head_dim, nh = s.n_head, nkvh = s.n_head_kv, ps = s.page_size;
        const int64_t pages = (ctx + ps - 1) / ps, rows = pages * nkvh * ps;
        const int64_t cap = k::qsa_selection_width(ctx, s);
        const size_t count = size_t(nq) * size_t(nh * hd);
        Rng rng;
        std::vector<int8_t> kq(size_t(rows * hd)), vq(kq.size());
        std::vector<uint16_t> ks(size_t(rows * (hd / 64))), vs(ks.size());
        for (auto* v : {&kq, &vq})
            for (auto& x : *v) x = int8_t(int(rng.next() % 255) - 127);
        for (auto* v : {&ks, &vs})
            for (auto& x : *v) x = uint16_t(0x1c00u + rng.next() % 0x0c00u);
        std::vector<int32_t> table(static_cast<size_t>(pages));
        std::iota(table.begin(), table.end(), 0);
        rng.shuffle(table);
        const auto physical = table;
        if (masked)
            for (int64_t page = 1; page < pages; page += 3) table[size_t(page)] = -1;
        std::vector<int32_t> ids(size_t(nq * cap) + kGuard, 0);
        std::vector<int32_t> steps(size_t(nq * k::kStepCount), 0);
        std::vector<float> query(count);
        for (auto& x : query) x = float(int(rng.next() % 4096) - 2048) * (1.0f / 1024.0f);
        std::vector<int32_t> old_cells;
        for (int64_t i = 0; i < nq; ++i) {
            const int64_t pos = ctx - nq + i, nkv = pos + 1;
            const int64_t width = k::qsa_selection_width(nkv, s);
            k::qsa_step_fill(steps.data() + i * k::kStepCount, pos, s);
            int32_t* sel = ids.data() + i * cap;
            if (width == nkv) {
                std::iota(sel, sel + width, 0);
            } else {
                constexpr int64_t recent = 512;   // the selection shape qsa_prompt_attn_parity uses
                const int64_t older = width - recent;
                if (int64_t(old_cells.size()) != older) {
                    std::vector<int32_t> all(size_t(nkv - recent));
                    std::iota(all.begin(), all.end(), 0);
                    rng.shuffle(all);
                    old_cells.assign(all.begin(), all.begin() + older);
                } else {
                    for (int64_t r = 0; r < older / 32; ++r) {
                        const int32_t cell = int32_t(rng.below(size_t(nkv - recent)));
                        if (std::find(old_cells.begin(), old_cells.end(), cell) == old_cells.end())
                            old_cells[rng.below(size_t(older))] = cell;
                    }
                }
                std::copy(old_cells.begin(), old_cells.end(), sel);
                std::iota(sel + older, sel + width, int32_t(nkv - recent));
                std::sort(sel, sel + width);
            }
            bool valid_page = false;
            for (int64_t c = 0; c < width; ++c) {
                if (sel[c] < 0 || sel[c] >= nkv || (c && sel[c - 1] >= sel[c]))
                    fail("selection must be unique, sorted and causal");
                valid_page |= table[size_t(sel[c] / ps)] >= 0;
            }
            if (!valid_page) table[size_t(sel[0] / ps)] = physical[size_t(sel[0] / ps)];
        }

        // ---- device side
        Device<int8_t> dk(kq), dv(vq);
        Device<uint16_t> dks(ks), dvs(vs);
        Device<int32_t> dtable(table), dids(ids), dsteps(steps);
        Device<float> dq(query), dref(count), dnew(count);
        const uint64_t stride = k::qsa_decode_attn_scratch_floats(cap, s);
        Device<float> scratch(size_t(std::min(kBatch, nq)) * size_t(stride));
        k::QsaAttnPools pools;
        pools.k_q = dk.p; pools.v_q = dv.p; pools.k_scale = dks.p; pools.v_scale = dvs.p;
        pools.page_table = dtable.p;
        HIP_CHECK(hipMemset(dref.p, 0, count * sizeof(float)));
        HIP_CHECK(hipMemset(dnew.p, 0, count * sizeof(float)));
        // reference: the decode kernel in kBatch-query batches - what prefill.cpp runs on gfx906 without this kernel
        auto ref_launch = [&] {
            for (int64_t t = 0; t < nq; t += kBatch)
                k::qsa_decode_attn_batch(dq.p + t * nh * hd, pools, dids.p + t * cap,
                    dsteps.p + t * k::kStepCount, cap, s, scratch.p, dref.p + t * nh * hd,
                    std::min(kBatch, nq - t), nullptr);
            HIP_CHECK(hipGetLastError());
        };
        // candidate: one call over the whole chunk (qsa_prompt_attn_batch -> the gfx906 kernel on this build)
        auto new_launch = [&] {
            const bool ok = k::qsa_prompt_attn_batch(dq.p, pools, dids.p, dsteps.p, cap, s, dnew.p, nq, nullptr);
            HIP_CHECK(hipGetLastError());
            return ok;
        };
        ref_launch();
        HIP_CHECK(hipDeviceSynchronize());
        const bool took = new_launch();
        HIP_CHECK(hipDeviceSynchronize());
        if (!took) {
            std::printf("prompt kernel DECLINED (a geometry/pool it does not serve, or STRATA_PROMPT_ATTN_GFX906=0): "
                        "the decode path is what the engine would run here\n");
            return 3;
        }
        std::vector<float> ref(count), got(count);
        HIP_CHECK(hipMemcpy(ref.data(), dref.p, count * sizeof(float), hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(got.data(), dnew.p, count * sizeof(float), hipMemcpyDeviceToHost));
        double max_abs = 0.0, max_rel = 0.0, rms = 0.0;
        size_t bitwise = 0;
        for (size_t i = 0; i < count; ++i) {
            if (!std::isfinite(got[i])) fail("candidate produced a non-finite output");
            const double d = std::fabs(double(got[i]) - double(ref[i]));
            max_abs = std::max(max_abs, d);
            rms += d * d;
            max_rel = std::max(max_rel, d / std::max(1e-30, std::fabs(double(ref[i]))));
            uint32_t a = 0, b = 0;
            std::memcpy(&a, &got[i], 4);
            std::memcpy(&b, &ref[i], 4);
            bitwise += a == b;
        }
        rms = std::sqrt(rms / double(count));
        // ---- timing: medians of `reps`, each path warmed three times
        auto time_it = [&](bool candidate) {
            std::vector<float> t(static_cast<size_t>(reps));
            for (int i = 0; i < 3; ++i) {
                if (candidate) (void) new_launch();
                else ref_launch();
            }
            HIP_CHECK(hipDeviceSynchronize());
            hipEvent_t b = nullptr, e = nullptr;
            HIP_CHECK(hipEventCreate(&b)); HIP_CHECK(hipEventCreate(&e));
            for (int i = 0; i < reps; ++i) {
                HIP_CHECK(hipEventRecord(b, nullptr));
                if (candidate) (void) new_launch();
                else ref_launch();
                HIP_CHECK(hipEventRecord(e, nullptr));
                HIP_CHECK(hipEventSynchronize(e));
                HIP_CHECK(hipEventElapsedTime(&t[size_t(i)], b, e));
            }
            HIP_CHECK(hipEventDestroy(b)); HIP_CHECK(hipEventDestroy(e));
            return median_ms(t);
        };
        const float ref_ms = time_it(false), new_ms = time_it(true);
        int device = 0;
        hipDeviceProp_t prop{};
        HIP_CHECK(hipGetDevice(&device)); HIP_CHECK(hipGetDeviceProperties(&prop, device));
        std::printf("device=%s arch=%s nq=%lld ctx=%lld cap=%lld batch=%lld reps=%d\n", prop.name, prop.gcnArchName,
                    (long long) nq, (long long) ctx, (long long) cap, (long long) kBatch, reps);
        std::printf("STRATA_PROMPT_ATTN_GFX906=%s masked_pages=%zu\n", env("STRATA_PROMPT_ATTN_GFX906"),
                    size_t(std::count(table.begin(), table.end(), -1)));
        std::printf("accuracy vs the decode kernel: max_abs=%.3e max_rel=%.3e rms=%.3e bitwise=%zu/%zu\n", max_abs,
                    max_rel, rms, bitwise, count);
        std::printf("time: decode %.4f ms | prompt %.4f ms | %.3fx | us/query %.3f -> %.3f\n", ref_ms, new_ms,
                    ref_ms / new_ms, 1000.0 * ref_ms / double(nq), 1000.0 * new_ms / double(nq));
        if (output && *output) {
            std::FILE* f = std::fopen(output, "wb");
            if (!f) fail("cannot open the output file");
            const size_t written = std::fwrite(got.data(), sizeof(float), count, f);
            const int closed = std::fclose(f);
            if (written != count || closed != 0) fail("cannot write the output file");
        }
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "qsa_prompt_attn_probe: %s\n", e.what());
        return 2;
    }
}
