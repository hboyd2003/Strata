#!/bin/bash
# P0.5 (task #9): recurrent-state per-step cost = the P3 go/no-go number.
# NO new engine code: STRATA_VERIFY_PROFILE (#610) already stamps every verify stage per window,
# the gfx906 stamp path is real (verify_kernels.cu:1399 wall_clock64()*40; this card reports
# hipDeviceAttributeWallClockRate=25000 kHz = the 25 MHz the *40 assumes), and the batch path
# collects it (verify.cpp:3092/3156 run_slot_rows, :3314 batch_poll) and prints
# "strata batch GPU stages ... GDN layers: conv X ab Y z Z rec W ..." (generate.cpp:10240).
# The conv stamp interval brackets the batch_rec_ per-slot conv loop, rec brackets the per-slot
# recurrence loop (verify.cpp:1210-1221, 1242-1255) - exactly the serial loops P3 would fuse.
#
# DECISION RULE (recorded into the receipt): read conv+ab+z+rec ms/window at 2/4/8 lanes vs the
# solo "strata decode GPU stages" lines. If the GDN stages at 8 lanes stay small vs the window
# total (~131 ms at b8 per P0.2) and grow near-linearly with rows, the 3-6-week multi-state
# kernel has nothing to win -> P3 NO-GO. CALIBRATION: profile "total" ms/window should be near
# the host-measured "run" ms/window of the same run's "strata batch:" line; a constant factor
# off means the clock scale is wrong - report, do not compare absolute values across scales.
# CAVEAT (DETAILS.md #610): the profiler slows decode a little - use these to COMPARE stages,
# not as speed numbers; P0.4 already holds the speed numbers.
#
# RULE: run ONLY when every other inference engine is down (SVM crash, daily 2026-10-20).
# Operator launches: bash tools/gfx906/run-p0.5-profile.sh 2>&1 | tee run-p0.5-profile.log
set -u
cd "$(dirname "$0")/../.."
TS=$(date +%Y%m%d-%H%M%S)
mkdir -p logs
EXTRA="--gpu 1 --pcie-frac 0 --adapt-every 1000000 --no-prefill-borrow"
PY=/root/.pyenv/versions/venv312/bin/python3
AVAIL=$(free -g | awk '/Mem/{print $7}')
echo "P0.5 profile sweep $(date +%T) RAM avail ${AVAIL} GiB - need >= 65"
[ "$AVAIL" -lt 65 ] && { echo "REFUSING: not enough RAM for the single engine (crash rule)"; exit 3; }
rc=0
for B in 2 4 8; do
  echo "=== batch $B start $(date +%T) ==="
  BATCH_TEST_LOG="logs/p0.5-b${B}-${TS}.log" STRATA_VERIFY_PROFILE=1 \
  $PY tools/batch_test.py --exe build-906/strata --config strata-swift-iq3_xxs.json \
      --batch $B --n $B --max-new 64 --extra "$EXTRA" 2>&1 | tee "logs/p0.5-b${B}-stdout-${TS}.log"
  code=${PIPESTATUS[0]}
  echo "batch $B exit=$code"
  echo "--- GPU stage lines seen (witness): ---"
  grep -c "GPU stages" "logs/p0.5-b${B}-${TS}.log" || echo "ZERO - profiler did NOT report (A770-style failure? see receipt)"
  [ "$code" -ne 0 ] && rc=$code
done
echo "=== P0.5 sweep done rc=$rc ==="
exit $rc
