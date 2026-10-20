#pragma once
// P0.3 route census (docs/CONCURRENCY-PLAN.md, PR1): opt-in aggregation of per-step expert routing that
// answers the one question P2's economics hang on - when N lanes decode in one batched window, do they
// RE-READ the same experts (the batch amortizes the expert reads, `run_multi`'s contract) or DISJOINT
// ones (every lane reads its own set and the VRAM LRU thrashes)?  The ported engines have had route
// tooling for a while - `STRATA_ROUTE_LOG` (tools/v41/engine/ds41_moe.cpp), `--dump-routes` and
// `--census` (tools/glm/glm_generate.cpp) - and mainline `src/` had neither (docs/CONCURRENCY-RADIX.md,
// "the census tools are ported-engine-only").  This is the mainline instrument.
//
// OFF BY DEFAULT, AND OFF MEANS ABSENT: nothing constructs this class unless `STRATA_ROUTE_CENSUS` names
// a file, so the default decode path stays byte-identical (AGENTS.md).  When on it is a measurement run:
// the numbers it collects are set relations, not timings, and it is deliberately not on the hot path.
//
// Three accumulators, dumped cumulatively so a killed run still holds its data:
//   H  per (layer, expert) token hits          - which experts the workload's decode touches;
//   U  per layer: distinct-expert sum vs lane-expert sum.  Their ratio is the READ FACTOR: 1.0 means the
//      lanes are disjoint and batching dedups nothing, k/lane against `n` at ratio ~1/n... concretely
//      the ratio times the lane count is the average number of lanes that share each read;
//   J  per (layer, lane pair): summed |A intersect B| and |A union B| - the pairwise Jaccard, so a
//      reader can see WHICH lanes diverge (e.g. a helper lane with a different system prompt).
//
// Batched decode lays ids out token-major, `ids[token * k + j]`, the same layout `moe_route_window`
// writes (src/core/layer.cpp).  A lane is a token here; the hook calls `record` with the window's lanes.
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace strata::core {

class RouteCensus {
public:
    // `n_lanes` sizes the pairwise storage; `record` refuses overlap work for windows wider than it
    // (the histogram is still kept).  Sized by the caller from the batch width, not by the model.
    RouteCensus(int n_layer, int n_expert, int n_lanes)
        : n_layer_(n_layer), n_expert_(n_expert), n_lanes_(n_lanes),
          words_(((size_t) n_expert + 63) / 64),
          hist_((size_t) n_layer * n_expert, 0),
          union_sum_((size_t) n_layer, 0), total_sum_((size_t) n_layer, 0), steps_((size_t) n_layer, 0),
          inter_sum_((size_t) n_layer * n_lanes * n_lanes, 0), pair_union_((size_t) n_layer * n_lanes * n_lanes, 0),
          lane_bits_((size_t) n_lanes * words_) {}

    // One layer of one step/window: `n` lanes' top-k ids, token-major.  `overlap` says whether the n
    // tokens are concurrent lanes (batch decode: yes; a prefill chunk is ONE stream over n tokens, so
    // there its ids feed the histogram only - counting prefill tokens as lanes would fake the ratio).
    void record(int layer, const int32_t* ids, int n, int k, bool overlap) {
        if (layer < 0 || layer >= n_layer_ || n <= 0 || k <= 0) return;
        for (int t = 0; t < n; ++t)
            for (int j = 0; j < k; ++j) {
                const int32_t e = ids[(size_t) t * k + j];
                if (e >= 0 && e < n_expert_) ++hist_[(size_t) layer * n_expert_ + e];
            }
        if (!overlap || n > n_lanes_) return;
        std::fill(lane_bits_.begin(), lane_bits_.end(), 0);
        for (int t = 0; t < n; ++t)
            for (int j = 0; j < k; ++j) {
                const int32_t e = ids[(size_t) t * k + j];
                if (e >= 0 && e < n_expert_)
                    lane_bits_[(size_t) t * words_ + (size_t) e / 64] |= (uint64_t) 1 << (e % 64);
            }
        int64_t distinct = 0;
        for (int t = 0; t < n; ++t)
            for (size_t w = 0; w < words_; ++w)
                distinct += __builtin_popcountll(lane_bits_[(size_t) t * words_ + w]);
        // the window's union: under per-expert grouped execution a shared expert is READ once for the
        // whole window, so the union is what the window costs and `distinct` (the per-lane sum) is what
        // serial lanes would cost.  OR into scratch so the per-lane bits stay intact for the pairs.
        union_acc_.assign(words_, 0);
        for (int t = 0; t < n; ++t)
            for (size_t w = 0; w < words_; ++w)
                union_acc_[w] |= lane_bits_[(size_t) t * words_ + w];
        int64_t window_union = 0;
        for (uint64_t u : union_acc_) window_union += __builtin_popcountll(u);
        union_sum_[layer] += (uint64_t) window_union;
        total_sum_[layer] += (uint64_t) distinct;  // == n*k when every id is distinct within its lane
        ++steps_[layer];
        for (int a = 0; a < n; ++a)
            for (int b = a + 1; b < n; ++b) {
                int64_t inter = 0, un = 0;
                for (size_t w = 0; w < words_; ++w) {
                    const uint64_t x = lane_bits_[(size_t) a * words_ + w], y = lane_bits_[(size_t) b * words_ + w];
                    inter += __builtin_popcountll(x & y);
                    un += __builtin_popcountll(x | y);
                }
                inter_sum_[((size_t) layer * n_lanes_ + a) * n_lanes_ + b] += (uint64_t) inter;
                pair_union_[((size_t) layer * n_lanes_ + a) * n_lanes_ + b] += (uint64_t) un;
            }
    }

    // Cumulative dump, appended.  Called on a cadence by the hook (so Ctrl-C keeps data) and again in
    // the destructor; the reader takes the LAST value of each line.  Keys are fixed-width so
    // `sort -k2,2n`-style tools work without padding care.
    void dump(const char* path) const {
        std::FILE* f = std::fopen(path, "a");
        if (f == nullptr) return;
        std::fprintf(f, "# census layer=%d expert=%d lanes=%d\n", n_layer_, n_expert_, n_lanes_);
        for (int l = 0; l < n_layer_; ++l)
            for (int e = 0; e < n_expert_; ++e)
                if (hist_[(size_t) l * n_expert_ + e])
                    std::fprintf(f, "H %d %d %" PRIu64 "\n", l, e, hist_[(size_t) l * n_expert_ + e]);
        for (int l = 0; l < n_layer_; ++l)
            if (steps_[l])
                std::fprintf(f, "U %d %" PRIu64 " %" PRIu64 " %" PRIu64 "\n", l, union_sum_[l], total_sum_[l],
                             steps_[l]);
        for (int l = 0; l < n_layer_; ++l)
            for (int a = 0; a < n_lanes_; ++a)
                for (int b = a + 1; b < n_lanes_; ++b) {
                    const size_t i = ((size_t) l * n_lanes_ + a) * n_lanes_ + b;
                    if (inter_sum_[i] || pair_union_[i])
                        std::fprintf(f, "J %d %d %d %" PRIu64 " %" PRIu64 "\n", l, a, b, inter_sum_[i], pair_union_[i]);
                }
        std::fflush(f);
        std::fclose(f);
    }

    const char* path() const { return path_.c_str(); }
    void set_path(std::string p) { path_ = std::move(p); }

    // test/introspection accessors
    uint64_t hits(int layer, int expert) const { return hist_[(size_t) layer * n_expert_ + expert]; }
    uint64_t union_sum(int layer) const { return union_sum_[layer]; }
    uint64_t total_sum(int layer) const { return total_sum_[layer]; }
    uint64_t steps(int layer) const { return steps_[layer]; }
    uint64_t pair_inter(int layer, int a, int b) const {
        return inter_sum_[((size_t) layer * n_lanes_ + a) * n_lanes_ + b];
    }
    uint64_t pair_union(int layer, int a, int b) const {
        return pair_union_[((size_t) layer * n_lanes_ + a) * n_lanes_ + b];
    }

    ~RouteCensus() { if (!path_.empty()) dump(path_.c_str()); }

private:
    int n_layer_, n_expert_, n_lanes_;
    size_t words_;
    std::vector<uint64_t> hist_, union_sum_, total_sum_, steps_, inter_sum_, pair_union_, lane_bits_, union_acc_;
    std::string path_;
};

}  // namespace strata::core
