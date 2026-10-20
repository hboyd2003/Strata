#!/bin/bash
# P0.4 (task #8): batched decode sweep 1/2/4/8 WITH the route census (U read-factor, J Jaccard) that gates P2.
# Same exactness recipe as P0.2 (BATCHING.md:191-199). DIFFERENT EXE than P0.2:
#   --exe build-906/strata  (census-capable; engine/strata @ 20261019-2034 predates commit 683c92f5)
# RULE: run ONLY when every other inference engine is down (SVM-resident-limit crash, daily 2026-10-20).
# Operator launches this; capture: bash tools/gfx906/run-p0.4-sweep.sh 2>&1 | tee run-p0.4-sweep.log
set -u
cd "$(dirname "$0")/../.."
TS=$(date +%Y%m%d-%H%M%S)
REC=benchmarks/gfx906/receipts
mkdir -p logs "$REC"
EXTRA="--gpu 1 --pcie-frac 0 --adapt-every 1000000 --no-prefill-borrow"
PY=/root/.pyenv/versions/venv312/bin/python3
AVAIL=$(free -g | awk '/Mem/{print $7}')
echo "P0.4 sweep $(date +%T) RAM avail ${AVAIL} GiB - need >= 65"
[ "$AVAIL" -lt 65 ] && { echo "REFUSING: not enough RAM for the single engine (crash rule)"; exit 3; }
rc=0
# B=1 is not a batch: the engine refuses a 1-slot group (BGEN, witnessed 2026-10-20);
# the solo rate is already measured by batch_test.py's per-slot solo runs.
for B in 2 4 8; do
  echo "=== batch $B start $(date +%T) (RAM avail $(free -g | awk '/Mem/{print $7}') GiB) ==="
  BATCH_TEST_LOG="logs/p0.4-b${B}-${TS}.log" \
  STRATA_ROUTE_CENSUS="$REC/census-p0.4-b${B}-${TS}.txt" \
  $PY tools/batch_test.py --exe build-906/strata --config strata-swift-iq3_xxs.json \
      --batch $B --n $B --max-new 64 --extra "$EXTRA" 2>&1 | tee "logs/p0.4-b${B}-stdout-${TS}.log"
  code=${PIPESTATUS[0]}
  echo "batch $B exit=$code"
  [ "$code" -ne 0 ] && rc=$code
done
echo "=== P0.4 sweep done rc=$rc census files: ==="
ls -la "$REC"/census-p0.4-*-${TS}.txt 2>/dev/null
exit $rc
