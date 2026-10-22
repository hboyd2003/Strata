#!/usr/bin/env bash
# run-p5.1-probe.sh - operator-launched P5.1 interconnect probe (task #26).
#
# The probe maps ~640 MB across the 4 cards (< 1% of one card): a different risk class from an
# engine, but the standing one-engine rule still applies, so this script REFUSES while any
# engine is alive (vLLM, strata-server, glm/ds41 engines). Numbers are a design input for the
# P5.3 relay transport - not a speed claim (AGENTS.md).
#
# Usage (free window, everything else stopped):
#   cd /ai/engines/gfx906/worktrees/xolotl-merge && tools/gfx906/run-p5.1-probe.sh
set -u
cd "$(dirname "$0")/../.."
TS=$(date +%Y%m%d-%H%M%S)
mkdir -p logs
LOG="logs/p5.1-probe-${TS}.log"

if pgrep -f "engine/strata|vllm|glm_generate|ds41_generate" >/dev/null 2>&1; then
    echo "refusing: another engine is alive - one HIP engine at a time (SVM crash rule)"; exit 3
fi
AVAIL=$(( $(awk '/MemAvailable/{print $2}' /proc/meminfo) / 1048576 ))
[ "$AVAIL" -ge 10 ] || { echo "refusing: ${AVAIL} GiB available < 10 (probe needs ~1 GiB pinned + headroom)"; exit 3; }

echo "=== p5.1 probe start $(date +%T) (avail ${AVAIL} GiB) ===" | tee "$LOG"
echo "--- sysfs link state (authority for link claims):" | tee -a "$LOG"
for b in 23:00.0 26:00.0 43:00.0 46:00.0; do
    echo "  $b cur=$(cat /sys/bus/pci/devices/0000:$b/current_link_speed) x$(cat /sys/bus/pci/devices/0000:$b/current_link_width)" \
         "max=$(cat /sys/bus/pci/devices/0000:$b/max_link_speed) x$(cat /sys/bus/pci/devices/0000:$b/max_link_width)" | tee -a "$LOG"
done

BIN=build-906/probe-p2p
if [ ! -x "$BIN" ] || [ tools/gfx906/probe-p2p.cu -nt "$BIN" ]; then
    echo "--- building probe:" | tee -a "$LOG"
    /opt/rocm/bin/hipcc --offload-arch=gfx906 -O2 -o "$BIN" tools/gfx906/probe-p2p.cu 2>&1 | tee -a "$LOG"
fi

"$BIN" 2>&1 | tee -a "$LOG"
rc=${PIPESTATUS[0]}
echo "=== p5.1 probe rc=$rc log=$LOG ===" | tee -a "$LOG"
if [ "$rc" -ne 0 ]; then
    echo "failure ladder: re-run with HIP_LAUNCH_BLOCKING=1 and report the first FATAL line;"
    echo "peer columns being 0 with rc=0 is NOT a failure - it is the measurement (staged transport)."
fi
exit $rc
