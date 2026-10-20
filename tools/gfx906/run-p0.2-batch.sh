#!/bin/bash
# P0.2 (task #6): first HIP batched-decode measurement, Swift IQ3_XXS, gfx906.
# Exactness recipe (BATCHING.md:191-199): STRATA_IQ_MT_MIN=1 (batch_test.py default --mt-min 1),
# --pcie-frac 0, --adapt-every 1000000, --no-prefill-borrow. Route census (P0.3 instrument) via
# inherited env. Device 1: the live server's engine (config gpu 0) sits on rocm-smi GPU[2].
set -u
cd "$(dirname "$0")/../.."
TS=$(date +%Y%m%d-%H%M%S)
REC=benchmarks/gfx906/receipts
mkdir -p logs "$REC"
EXTRA="--gpu 1 --pcie-frac 0 --adapt-every 1000000 --no-prefill-borrow"
PY=/root/.pyenv/versions/venv312/bin/python3
rc=0
for B in 4 8; do
  echo "=== batch $B start $(date +%T) (RAM avail $(free -g | awk '/Mem/{print $7}') GiB) ==="
  BATCH_TEST_LOG="logs/p0.2-b${B}-${TS}.log" \
  STRATA_ROUTE_CENSUS="$REC/census-b${B}-${TS}.txt" \
  $PY tools/batch_test.py --exe engine/strata --config strata-swift-iq3_xxs.json \
      --batch $B --n $B --max-new 64 --extra "$EXTRA" 2>&1 | tee "logs/p0.2-b${B}-stdout-${TS}.log"
  code=${PIPESTATUS[0]}
  echo "batch $B exit=$code"
  [ "$code" -ne 0 ] && rc=$code
done
echo "=== P0.2 sweep done rc=$rc census files: ==="
ls -la "$REC"/census-*-${TS}.txt 2>/dev/null
exit $rc
