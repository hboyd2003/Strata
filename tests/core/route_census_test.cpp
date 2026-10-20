// P0.3 route census (PR1): the accumulators behind the "do lanes re-read or read disjoint experts" gate.
// Header-only, CPU - no GPU, no weights: the census is pure arithmetic over host-visible ids.
#include "strata/core/route_census.hpp"

#include <cstdio>
#include <vector>

static int failures = 0;
static void check(bool ok, const char* what) {
    if (!ok) { std::printf("FAIL: %s\n", what); ++failures; }
}

int main() {
    using strata::core::RouteCensus;

    // ---- disjoint lanes: the window pays for every lane, dedup saves nothing
    {
        RouteCensus c(2, 64, 4);
        const int32_t ids[8] = {0, 1, 2, 3, 4, 5, 6, 7};  // lane 0 = 0..3, lane 1 = 4..7
        c.record(0, ids, 2, 4, true);
        check(c.union_sum(0) == 8, "disjoint: union is 8");
        check(c.total_sum(0) == 8, "disjoint: total is 8");
        check(c.union_sum(0) == c.total_sum(0), "disjoint: read factor 1.0");
        check(c.pair_inter(0, 0, 1) == 0 && c.pair_union(0, 0, 1) == 8, "disjoint: J=0/8");
        check(c.steps(0) == 1, "disjoint: one step");
        check(c.union_sum(1) == 0, "disjoint: other layer untouched");
    }

    // ---- fully shared lanes: the read factor collapses to 1/n
    {
        RouteCensus c(1, 64, 4);
        const int32_t ids[8] = {7, 8, 9, 10, 7, 8, 9, 10};
        c.record(0, ids, 2, 4, true);
        check(c.union_sum(0) == 4, "shared: union is 4");
        check(c.total_sum(0) == 8, "shared: total is 8");
        check(c.pair_inter(0, 0, 1) == 4 && c.pair_union(0, 0, 1) == 4, "shared: |A∩B|=|A∪B|=4");
    }

    // ---- partial overlap and accumulation across steps
    {
        RouteCensus c(1, 64, 2);
        const int32_t a[8] = {0, 1, 2, 3, 2, 3, 4, 5};  // inter 2, union 6
        c.record(0, a, 2, 4, true);
        const int32_t b[8] = {0, 1, 2, 3, 0, 1, 2, 3};  // inter 4, union 4
        c.record(0, b, 2, 4, true);
        check(c.union_sum(0) == 10, "accum: 6+4 union");
        check(c.total_sum(0) == 16, "accum: 8+8 total");
        check(c.pair_inter(0, 0, 1) == 6, "accum: 2+4 inter");
        check(c.pair_union(0, 0, 1) == 10, "accum: 6+4 pair union");
        check(c.steps(0) == 2, "accum: two steps");
    }

    // ---- histogram counts tokens, not windows: two lanes on expert 5 is two hits
    {
        RouteCensus c(2, 64, 2);
        const int32_t ids[4] = {5, 6, 5, 7};
        c.record(0, ids, 2, 2, true);
        check(c.hits(0, 5) == 2, "hist: expert 5 has 2 hits");
        check(c.hits(0, 6) == 1 && c.hits(0, 7) == 1, "hist: singles");
        check(c.hits(1, 5) == 0, "hist: layer 1 untouched");
    }

    // ---- a prefill chunk is ONE stream, not N lanes: overlap=false keeps the histogram but no U/J
    {
        RouteCensus c(1, 64, 4);
        const int32_t ids[8] = {0, 1, 2, 3, 4, 5, 6, 7};
        c.record(0, ids, 2, 4, false);
        check(c.hits(0, 0) == 1 && c.hits(0, 7) == 1, "prefill: hist still recorded");
        check(c.steps(0) == 0 && c.union_sum(0) == 0 && c.total_sum(0) == 0, "prefill: no window stats");
        check(c.pair_inter(0, 0, 1) == 0 && c.pair_union(0, 0, 1) == 0, "prefill: no pairs");
    }

    // ---- out-of-range and negative ids are ignored everywhere (a host mirror glitch must not index OOB)
    {
        RouteCensus c(1, 64, 2);
        const int32_t ids[4] = {-1, 64, 3, 1000};
        c.record(0, ids, 2, 2, true);
        check(c.hits(0, 3) == 1, "oob: only the in-range id counted");
        check(c.union_sum(0) == 1 && c.total_sum(0) == 1, "oob: window sees one real expert");
    }

    // ---- a window wider than the census lanes still records the histogram, just not the pairs
    {
        RouteCensus c(1, 64, 2);
        std::vector<int32_t> ids(3 * 2);
        for (size_t i = 0; i < ids.size(); ++i) ids[i] = (int32_t) i;
        c.record(0, ids.data(), 3, 2, true);
        check(c.hits(0, 5) == 1, "wide: hist kept past the lane cap");
        check(c.steps(0) == 0, "wide: no window stats beyond the lane cap");
    }

    // ---- the dump parses back: 'H l e n', 'U l union total steps', 'J l a b inter union'
    {
        const char* path = "route_census_test_dump.txt";
        std::remove(path);
        {
            RouteCensus c(1, 64, 2);
            c.set_path(path);
            const int32_t ids[4] = {0, 1, 2, 3};
            c.record(0, ids, 2, 2, true);
            c.dump(path);  // explicit dump in addition to the destructor's
        }
        std::FILE* f = std::fopen(path, "r");
        check(f != nullptr, "dump: file exists");
        bool h = false, u = false, j = false;
        char line[256];
        long l, x, y, z, w;
        if (f) {
            while (std::fgets(line, sizeof(line), f)) {
                if (line[0] == 'H' && std::sscanf(line, "H %ld %ld %ld", &l, &x, &y) == 3 && l == 0 && x == 0 && y == 1)
                    h = true;
                else if (line[0] == 'U' && std::sscanf(line, "U %ld %ld %ld %ld", &l, &x, &y, &z) == 4 && x == 4 &&
                         y == 4 && z == 1)
                    u = true;
                else if (line[0] == 'J' &&
                         std::sscanf(line, "J %ld %ld %ld %ld %ld", &l, &x, &y, &z, &w) == 5 && x == 0 && y == 1)
                    j = true;
            }
            std::fclose(f);
        }
        check(h && u && j, "dump: H/U/J lines parse with expected numbers");
        std::remove(path);
    }

    std::printf("%s\n", failures == 0 ? "ok: route_census" : "route_census FAILED");
    return failures == 0 ? 0 : 1;
}
