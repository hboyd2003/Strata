#!/bin/bash
# P1.2 (task #14) economics probe: does a prompt read through VERIFY WINDOW ROWS ever keep up with the
# chunked prefill read, on THIS box?  It must, before --prefill-rows can pass its gate ("slot tok/s AND
# prompt tok/s >= time-share control", docs/CONCURRENCY-PLAN.md P1.2): a mixed window pays the measured
# row cost (8-row window = 131.24 ms, so ~16 ms/row, P0.2 b8 log), and its rows compete with the
# decodes' rows (kVerifyMaxT = 8 total).  The engine's own comment says the chunked path costs ~300 ms
# per run however few tokens and a window ~16 ms a token (generate.cpp:11405-11409).
#
# Two arms of tools/batch_interleave_test.py (A, B decode while C's ~250-token prompt is read):
#   arm 1 "chunked"  - today's time-share: C's read is one Prefill::run chunk, slots get decode_share
#                      of wall time between (STRATA_BATCH_DECODE_SHARE, default 0.5).
#   arm 2 "windowed" - --short-read 999999 makes C's read go through the verify windows instead
#                      (read_windows: teacher-forced T tokens per window, commit-all) - the row cost
#                      a --prefill-rows read would pay.
# DECIDE: compare arm 2's C read tok/s against arm 1's (the "read in ... ms (... tok/s)" and "the prompt
# was read in N parts" lines in each engine log; the harness prints "admissions Xs; N slot tokens
# arrived while ...").  If arm 2's prompt read is far slower (a perfect --prefill-rows cannot beat the
# row cost its rows pay), report NO-GO with these numbers and do not build the engine feature.  If they
# are within ~1.5x, the mixed window only adds the decodes' service and P1.2 is worth building.
# Parity lines must read IDENTICAL in both arms (window reads are the shipped short-read path; exact
# on this box per the P0.2 receipt).
# ONE ENGINE AT A TIME (SVM crash rule): refuses while any strata/vllm engine is alive, and under 65 GiB free.
# Operator launches: bash tools/gfx906/run-p1.2-economics.sh 2>&1 | tee run-p1.2-economics.log
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
# the exactness recipe (BATCHING.md), so both arms run the same expert tiers as every P0/P1 witness
RECIPE="--gpu 1 --pcie-frac 0 --adapt-every 1000000 --no-prefill-borrow"
PY=/root/.pyenv/versions/venv312/bin/python3
rc=0
for ARM in chunked windowed; do
    EXTRA="$RECIPE"
    [ "$ARM" = windowed ] && EXTRA="$RECIPE --short-read 999999"
    echo "=== P1.2 economics arm=$ARM start $(date +%T) (RAM avail $(free -g | awk '/Mem/{print $7}') GiB) ==="
    BATCH_TEST_LOG="logs/p1.2-econ-$ARM-$TS.log" \
    $PY tools/batch_interleave_test.py --exe engine/strata --config strata-swift-iq3_xxs.json \
        --extra "$EXTRA" 2>&1 | tee "logs/p1.2-econ-$ARM-stdout-$TS.log"
    code=${PIPESTATUS[0]}
    echo "arm=$ARM exit=$code"
    echo "--- witnesses ($ARM): ---"
    grep -E "read in|the prompt was read in" "logs/p1.2-econ-$ARM-$TS.log" | tail -4 || echo "no read lines"
    grep -E "admissions|IDENTICAL|DIFFERS" "logs/p1.2-econ-$ARM-stdout-$TS.log" || echo "no harness lines"
    [ "$code" -ne 0 ] && rc=$code
done
echo "=== P1.2 economics done rc=$rc - compare arm=windowed's long-prompt read tok/s against arm=chunked's;"
echo "    far slower => NO-GO with these numbers, do not build --prefill-rows; within ~1.5x => build it."
echo "    logs: logs/p1.2-econ-*-$TS.log ==="
exit $rc
