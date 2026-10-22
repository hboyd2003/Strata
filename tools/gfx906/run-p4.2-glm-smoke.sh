#!/bin/bash
# P4.2 (task #20): GLM-5.3-Flash Maya-L first HIP smoke on ONE MI50 (gfx906).  OPERATOR launches:
#   bash tools/gfx906/run-p4.2-glm-smoke.sh 2>&1 | tee run-p4.2-glm-smoke.log
# (a run_and_wait window of ~15 min: the arena pins ~60 GiB before the first token).
#
# Direct-engine witness (no --serve): greedy -n 64 must produce readable text.  Runs through
# tools/glm/serve/glm-engine-906.sh so the serve-path plumbing (build dir, GPU lock, systemd scope
# + MemoryMax=84G) is witnessed on the same route the server will take.
#
# Deliberate deltas vs the serve config strata-glm-mayal-gfx906.json (first boot = plainest path,
# one thing at a time): NO --prune/--prune-penalty, NO --route-bias, NO --lora/--lora-exps.  Those
# return only in P4.3 after this is green.
#
# Failure ladder (GLM-FIRST-BOOT.md): (1) base, (2) GGML_CUDA_GRAPH_MAX_TOKENS=0 (CUDA graphs off
# everywhere), (3) --experts cpu (dense-only GPU; a PASS here does NOT witness the GPU expert path).
#
# ONE ENGINE AT A TIME (SVM crash rule): refuses while any engine is alive, or below MIN_AVAIL_GIB
# (default arena+25 GiB - a placeholder; P4.1 #19 replaces the heuristic with a computed floor).
set -u
cd "$(dirname "$0")/../.."
TS=$(date +%Y%m%d-%H%M%S)
mkdir -p logs
if pgrep -f "engine/strata|vllm|glm_generate|ds41_generate" >/dev/null 2>&1; then
    echo "refusing: another engine is alive - one HIP engine at a time (SVM crash rule)"; exit 3
fi
ARENA_GIB=${ARENA_GIB:-60}
MIN_AVAIL_GIB=${MIN_AVAIL_GIB:-$((ARENA_GIB + 25))}
AVAIL=$(( $(awk '/MemAvailable/{print $2}' /proc/meminfo) / 1048576 ))
[ "$AVAIL" -ge "$MIN_AVAIL_GIB" ] || { echo "refusing: ${AVAIL} GiB available < ${MIN_AVAIL_GIB}"; exit 3; }

PY=/root/.pyenv/versions/venv312/bin/python3
IDS=logs/p4.2-ids-${TS}.i32
PYTHONPATH="$PWD/tools" $PY - "$IDS" <<'EOF' || exit 1
import sys
sys.path.insert(0, "tools/glm/serve")
import glm_tok
import strata_tokenizer as ST
ST.PRE_PATTERNS.setdefault("glm4", [glm_tok.GLM4_PATTERN])   # what serve_glm.py:37 does before loading
import numpy as np
tok = glm_tok.load("/ai/models/tok/glm-5.3-flash")
ids = tok.encode("What is a mixture-of-experts model? Answer in two sentences.")
np.array(ids, dtype=np.int32).tofile(sys.argv[1])
print(f"prompt ids: {len(ids)} tokens -> {sys.argv[1]}")
EOF

MODEL=/ai/models/hf.co/peasantsmith/GLM-5.3-Flash-Maya-GGUF/Maya-L/GLM-5.3-Flash-Maya-L-IQ3_S-00001-of-00004.gguf
ARGS=(-m "$MODEL" --backend cuda --experts gpu --ids-file "$IDS" -n 64
      --ctx 32768 --arena-gib "$ARENA_GIB" --arena-skip-resident --prefill-chunk 4096 --chunk-mmq
      --vram-margin 2.0 --arena-adapt --pcie auto --stop 154820,154827,154829)

run_step() {
    local name=$1; shift
    local log=logs/p4.2-${name}-${TS}.log
    echo "=== p4.2/$name start $(date +%T) (avail ${AVAIL} GiB) ==="
    "$@" >"$log" 2>&1
    local rc=$?
    echo "--- tail $log:"; tail -25 "$log"
    echo "=== p4.2/$name rc=$rc log=$log ==="
    return $rc
}

if run_step base bash tools/glm/serve/glm-engine-906.sh "${ARGS[@]}"; then
    echo "P4.2 WITNESS: PASS (base path - CUDA graphs on, GPU experts)"
elif run_step nograph env GGML_CUDA_GRAPH_MAX_TOKENS=0 bash tools/glm/serve/glm-engine-906.sh "${ARGS[@]}"; then
    echo "P4.2 WITNESS: PASS (graphs-off ladder rung - record it, graphs cost ~17% decode elsewhere)"
elif run_step cpuexp env GGML_CUDA_GRAPH_MAX_TOKENS=0 bash tools/glm/serve/glm-engine-906.sh \
        "${ARGS[@]/--experts gpu/--experts cpu}"; then
    echo "P4.2 WITNESS: PASS-CPU-ONLY - the GPU expert path is NOT witnessed; do not proceed to serve"
else
    echo "P4.2 WITNESS: FAIL (all three rungs; logs logs/p4.2-*-${TS}.log - attach them to the issue)"
    exit 1
fi
