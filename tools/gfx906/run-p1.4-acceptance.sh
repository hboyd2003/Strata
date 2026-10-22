#!/bin/bash
# P1.4 (task #16): does --batch-mtp draft as well in the windows as the same request does alone?
# tools/batch_test.py --acceptance reads the tallies the engine already logs and prints the pair that
# means the same thing on both sides: a slot verifies ONE proposal per window (keep the proposal only
# when the target picked it, generate.cpp:10154, summed at :10168 and said at :10283), and a solo
# window's FIRST proposal is the spec-depth ladder's depth 0 (counted at :12894, said at :13354 when
# STRATA_SPEC_DEPTH is set, which --acceptance sets).  The tool's own gate: the difference must sit
# inside --acceptance-tol percentage points AND inside the counting-noise band (2 sigma, pooled).
# WHY THIS AND NOT A TOKEN TEST: token parity already passes on gfx906 (P0.2 receipt, 12/12
# IDENTICAL on the exactness recipe), and a plain --batch lane verifies no proposals at all - it
# silently loses MTP (BATCHING.md), so only a --batch-mtp run says anything about drafted serving.
# Tokens identical + acceptance within the band = the slots draft the way solo does.
# ONE ENGINE AT A TIME (SVM crash rule): refuses while any strata/vllm engine is alive, and under 65 GiB free.
# Operator launches: bash tools/gfx906/run-p1.4-acceptance.sh 2>&1 | tee run-p1.4-acceptance.log
set -u
cd "$(dirname "$0")/../.."
TS=$(date +%Y%m%d-%H%M%S)
mkdir -p logs
if pgrep -f "engine/strata|vllm" >/dev/null 2>&1; then
    echo "refusing: another engine is alive - one HIP engine at a time (SVM crash rule)"; exit 3
fi
AVAIL=$(awk '/MemAvailable/{print $2}' /proc/meminfo)
if [ "$AVAIL" -lt $((65 * 1024 * 1024)) ]; then
    echo "refusing: $((AVAIL / 1024 / 1024)) GiB available, the engine load wants >= 65 GiB"; exit 3
fi
# the exactness recipe (BATCHING.md): STRATA_IQ_MT_MIN=1 comes from batch_test.py's own --mt-min 1 default
EXTRA="--gpu 1 --pcie-frac 0 --adapt-every 1000000 --no-prefill-borrow --batch-mtp"
PY=/root/.pyenv/versions/venv312/bin/python3
rc=0
for B in 2 4; do
  echo "=== batch-mtp acceptance, batch $B, start $(date +%T) (RAM avail $(free -g | awk '/Mem/{print $7}') GiB) ==="
  BATCH_TEST_LOG="logs/p1.4-b${B}-${TS}.log" \
  $PY tools/batch_test.py --exe engine/strata --config strata-swift-iq3_xxs.json \
      --batch $B --n $B --max-new 64 --acceptance --extra "$EXTRA" \
      2>&1 | tee "logs/p1.4-b${B}-stdout-${TS}.log"
  code=${PIPESTATUS[0]}
  echo "batch $B exit=$code  (expect: every slot IDENTICAL and 'ACCEPTANCE OK')"
  echo "--- witnesses for this arm: ---"
  grep -c "MTP proposals accepted" "logs/p1.4-b${B}-${TS}.log" \
      || echo "ZERO tally lines in the engine log - --batch-mtp was left off (see the --batch-mtp is off warning)"
  grep -c "spec depth" "logs/p1.4-b${B}-${TS}.log" \
      || echo "ZERO spec-depth lines - STRATA_SPEC_DEPTH did not reach the engine"
  grep -E "ACCEPTANCE (OK|OFF)" "logs/p1.4-b${B}-stdout-${TS}.log" || echo "no ACCEPTANCE verdict line"
  [ "$code" -ne 0 ] && rc=$code
done
echo "=== P1.4 acceptance witness done rc=$rc logs: logs/p1.4-*-${TS}.log ==="
exit $rc
