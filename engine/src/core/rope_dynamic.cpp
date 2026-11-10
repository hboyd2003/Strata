// src/core/rope_dynamic.cpp - see include/strata/core/rope_dynamic.hpp.
#include "strata/core/rope_dynamic.hpp"

#include "strata/kernels/kv_stream.hpp"
#include "strata/kernels/mrope.hpp"
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/rope_dynamic.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <thread>
#include <vector>

namespace strata::core {

double RopeDynamic::factor_for(int64_t positions) const {
    if ((double) positions <= orig) return 1.0;
    const double f = std::ceil((double) positions / orig / step - 1e-9) * step;
    return on_grid(std::max(1.0, std::min(max_factor, f)));
}

strata::kernels::RopeScaling RopeDynamic::config(double f) const {
    if (f <= 1.0) {
        strata::kernels::RopeScaling none;
        none.freq_base = base.freq_base;
        return none;
    }
    strata::kernels::RopeScaling s = base;
    s.type = strata::kernels::RopeScalingType::YaRN;
    s.factor = f;
    return s;
}

bool rope_rerotate_state(const QsaState& st, int64_t cells, bool index, const strata::kernels::RopeScaling& from,
                         const strata::kernels::RopeScaling& to, void* stream, std::string& error) {
    using namespace strata::kernels;
    if (cells <= 0) return true;
    if (st.kv_mode != 0 && !st.host.present()) { error = "dynamic rope: a streamed K/V without its host copy"; return false; }
    if (st.kv_q4 || st.kv_hybrid) { error = "dynamic rope: --kv q4_0 / k8v4 is not supported"; return false; }
    if (cells > st.max_cells) { error = "dynamic rope: more cells than the state holds"; return false; }
    const QsaShapes s = qsa_real_shapes();
    const RopeRates a = rope_rates((int) s.n_rot, from), b = rope_rates((int) s.n_rot, to);
    RopeKPool k;
    if (st.kv_mode != 0) {   // streamed or a ring: the host copy is the K/V (identity layout); the slots are refetched
        if (st.kv_int8) { k.k_q = st.host.k_q; k.k_scale = st.host.k_scale; }
        else k.k_pool = st.host.k_pool;
    } else {
        if (st.kv_int8) { k.k_q = st.k_q; k.k_scale = st.k_scale; }
        else k.k_pool = st.k_pool;
        k.page_table = st.page_table;
    }
    k.n_kv_heads = (int) s.n_head_kv;
    k.page_size = (int) s.page_size;
    k.head_dim = (int) s.head_dim;
    k.hadamard = st.kv_rot;
    rope_rerotate_k(k, cells, a, b, stream);
    if (st.kv_mode == 1) kv_stream_reset(st.map, stream);   // every slot fetched again from the rotated host copy
    if (index && st.idx_pooled != nullptr)
        rope_rerotate_pooled(st.idx_pooled, st.idx_dead, cells / s.idx_block, (int) s.idx_block, (int) s.idx_dim, a, b,
                             stream);
    return true;
}

namespace {
bool same_table(const strata::kernels::RopeScaling& a, const strata::kernels::RopeScaling& b) {
    return a.type == b.type && a.freq_base == b.freq_base && a.factor == b.factor && a.freq_scale_in == b.freq_scale_in &&
           a.orig_ctx == b.orig_ctx && a.ext_factor == b.ext_factor && a.attn_factor == b.attn_factor &&
           a.beta_fast == b.beta_fast && a.beta_slow == b.beta_slow;
}
// Two pinned host copies of the table, each remembering the scaling it holds: a switch back to a factor still held
// (a long conversation and a short one taking turns) is only an upload; otherwise the copy not holding the table on
// the device is rebuilt.  Pinned once: no zeroed vectors and page faults per switch, and the upload runs at full speed.
struct HostTable {
    float* cos = nullptr;
    float* sin = nullptr;
    size_t n = 0;
    bool valid = false;
    strata::kernels::RopeScaling scaling;
    uint64_t used = 0;
};
HostTable g_host[2];
uint64_t g_clock = 0;
}  // namespace

bool rope_table_rebuild(const QsaState& owner, const strata::kernels::RopeScaling& to, std::string& error,
                        RopeTableTiming* timing) {
    using namespace strata::kernels;
    using clk = std::chrono::steady_clock;
    auto ms = [](clk::time_point a, clk::time_point b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
    if (owner.cos_tab == nullptr || owner.sin_tab == nullptr || owner.max_cells <= 0) {
        error = "dynamic rope: no rope table";
        return false;
    }
    const auto t0 = clk::now();
    const QsaShapes s = qsa_real_shapes();
    const size_t n = (size_t) owner.max_cells * (size_t) (s.n_rot / 2);
    HostTable* h = nullptr;
    for (HostTable& c : g_host)
        if (c.valid && c.n == n && same_table(c.scaling, to)) h = &c;
    const bool hit = h != nullptr;
    if (!hit) {   // the least recently used copy
        h = g_host[0].used <= g_host[1].used ? &g_host[0] : &g_host[1];
        if (h->n != n) {
            if (h->cos) cudaFreeHost(h->cos);
            if (h->sin) cudaFreeHost(h->sin);
            h->cos = h->sin = nullptr;
            h->n = 0;
            h->valid = false;
            if (cudaMallocHost((void**) &h->cos, n * 4) != cudaSuccess ||
                cudaMallocHost((void**) &h->sin, n * 4) != cudaSuccess) {
                (void) cudaGetLastError();
                error = "dynamic rope: no pinned host memory for the table";
                return false;
            }
            h->n = n;
        }
        h->valid = false;
    }
    const auto t1 = clk::now();
    if (!hit) {
        const int threads = (int) std::max(1u, std::min(16u, std::thread::hardware_concurrency()));
        build_rope_table_mt((int) s.n_rot, to, (int) owner.max_cells, h->cos, h->sin, threads);
        h->scaling = to;
        h->valid = true;
    }
    h->used = ++g_clock;
    const auto t2 = clk::now();
    if (cudaDeviceSynchronize() != cudaSuccess) {
        error = "dynamic rope: the device failed before the table upload";
        return false;
    }
    const auto t3 = clk::now();
    if (cudaMemcpy(owner.cos_tab, h->cos, n * 4, cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(owner.sin_tab, h->sin, n * 4, cudaMemcpyHostToDevice) != cudaSuccess) {
        error = "dynamic rope: the table upload failed";
        return false;
    }
    const auto t4 = clk::now();
    rope_table_set(owner.cos_tab, owner.sin_tab, (int) owner.max_cells, to);
    rope_scaling_set(to);
    if (timing) *timing = {hit, ms(t0, t1), ms(t1, t2), ms(t2, t3), ms(t3, t4)};
    return true;
}

}  // namespace strata::core
