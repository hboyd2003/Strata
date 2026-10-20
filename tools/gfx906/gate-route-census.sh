#!/bin/bash
# P0.3 route census gate: build (HIP gfx906 wave64) + unit test + full ctest vs the 4-failure baseline.
set -u
WT=/ai/engines/gfx906/worktrees/xolotl-merge
cd "$WT" || exit 1

# standing RAM watchdog: this box has no swap; kill the build if MemAvailable drops under 2 GiB
watchdog() {
  while sleep 5; do
    av=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
    if [ "${av:-99999}" -lt 2048 ]; then
      echo "WATCHDOG: MemAvailable=${av}MiB < 2048MiB - killing build" >&2
      pkill -9 -P $$ -g 2>/dev/null; kill -9 0 2>/dev/null; exit 124
    fi
  done
}
watchdog & WD=$!
trap 'kill $WD 2>/dev/null' EXIT

echo "=== [1/3] cmake/ninja: route_census_test + strata (wave64 HIP, -j4) ==="
cmake --build build-906 --target route_census_test strata -j4 || { echo "GATE FAIL: build"; exit 1; }

echo "=== [2/3] unit: route_census_test ==="
build-906/route_census_test || { echo "GATE FAIL: route_census_test"; exit 1; }

echo "=== [3/3] full ctest (baseline = the 4 known pre-existing failures) ==="
cd build-906 && ctest -j4 > /tmp/ctest-route-census.log 2>&1; rc=$?
tail -8 /tmp/ctest-route-census.log
FAILED=$(grep -E "^The following tests FAILED" -A 10 /tmp/ctest-route-census.log | grep -oE "[0-9]+\s*-\s*\S+" | awk '{print $2}' | sort | tr '\n' ' ')
echo "failed set: ${FAILED:-NONE}"
EXPECT="mmvq_il_parity ple_parity split_lend_complement_test verify_parity "
# NOTE: the 4-failure baseline names come from the fork's baseline ctest; if the set differs, this
# prints it and the caller compares against docs/gfx906/START-HERE.md before judging.
if [ "$FAILED" = "$EXPECT" ]; then echo "ctest: exactly the baseline failures - OK"; else echo "ctest FAILED SET DIFFERS from expected - inspect /tmp/ctest-route-census.log"; fi
echo "ctest exit: $rc"
echo "GATE DONE"
