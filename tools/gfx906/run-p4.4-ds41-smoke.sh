#!/bin/bash
# P4.4 (task #22): DeepSeek-V4.1-Flash engram smoke on ONE MI50 (gfx906).  OPERATOR launches:
#   bash tools/gfx906/run-p4.4-ds41-smoke.sh 2>&1 | tee run-p4.4-ds41-smoke.log
# (a run_and_wait window of ~20 min: engram headers gate first, then the arena pins ~64 GiB; the
# 95+95 GiB tables themselves stream from NVMe at row level, they are never read whole here.)
#
# Direct-engine witness (no --serve, no DSpark): greedy -n 128 with the engram tables wired in
# (--engram l1:l14) must produce readable text - that is the engram path running end to end.
# Runs through tools/v41/serve/ds41-engine-906.sh (build dir + scope + MemoryMax=84G + DSpark off).
#
# Deliberate deltas vs strata-ds41-906.json / the authors' serve args: no --route-bias*, no --kv-fp4,
# no --prompt-batch, DSpark off (the 906 wrapper default) - plainest path first, per plan.
#
# Gate before any GPU work: check_engram_tables.py (the byte-vs-source verify_engram.py cannot run -
# the FP8 shards were deleted; this mirrors the engine's own loader expectations).  A FAIL here costs
# seconds, not a wedged KFD: exit 3.
#
# ONE ENGINE AT A TIME (SVM crash rule): refuses while any engine is alive, or below MIN_AVAIL_GIB
# (floor computed per docs/gfx906/P4-RAM-BUDGET.md in P4.1; default arena+25, override MIN_AVAIL_GIB=).
set -u
cd "$(dirname "$0")/../.."
TS=$(date +%Y%m%d-%H%M%S)
mkdir -p logs
if pgrep -f "engine/strata|vllm|glm_generate|ds41_generate" >/dev/null 2>&1; then
    echo "refusing: another engine is alive - one HIP engine at a time (SVM crash rule)"; exit 3
fi
ARENA_GIB=${ARENA_GIB:-64}
# Floor per docs/gfx906/P4-RAM-BUDGET.md: arena + dense 8 + scratch 3 + 10 margin >= 85; default arena+25 is stricter
MIN_AVAIL_GIB=${MIN_AVAIL_GIB:-$((ARENA_GIB + 25))}
AVAIL=$(( $(awk '/MemAvailable/{print $2}' /proc/meminfo) / 1048576 ))
[ "$AVAIL" -ge "$MIN_AVAIL_GIB" ] || { echo "refusing: ${AVAIL} GiB available < ${MIN_AVAIL_GIB}"; exit 3; }

M=/ai/models/hf.co/IndrasMirror/DeepSeek-V4.1-Flash-GSQ-Uncensored
PY=/root/.pyenv/versions/venv312/bin/python3

echo "=== engram gate (pre-boot) ==="
$PY tools/gfx906/check_engram_tables.py "$M/V41-GSQ.gguf" "$M/engram/l1" "$M/engram/l14" || {
    echo "refusing: engram gate FAILED before touching the GPU"; exit 3; }

IDS=logs/p4.4-ids-${TS}.i32
$PY - "$IDS" <<EOF || exit 1
import sys, json
sys.path.insert(0, "tools")
import strata_tokenizer as ST, numpy as np
tdir = "$M/tokenizer"
vocab = json.load(open(tdir + "/vocab.json"))
tokens = [None] * len(vocab)
for t, i in vocab.items():
    tokens[i] = t
merges = open(tdir + "/merges.txt").read().split("\\n")
types = json.load(open(tdir + "/token_type.json"))
tok = ST.Tokenizer(tokens, merges, types, pre="joyai-llm")   # the server's own pre-split (serve_ds41.py:45)
ids = tok.encode("What is a mixture-of-experts model? Answer in two sentences.")
np.array(ids, dtype=np.int32).tofile(sys.argv[1])
print(f"prompt ids: {len(ids)} tokens -> {sys.argv[1]}")
EOF

# no --ctx: P4.1 verdict is the engine's own auto ctx (max(4096, prompt+n+64 rounded to 256),
# ds41_generate.cpp:286) - passing 32768 contradicted the RAM-budget doc and only grew KV
ARGS=(-m "$M/V41-GSQ.gguf" --backend cuda --experts gpu
      --engram "$M/engram/l1:$M/engram/l14" --rank "$M/rank.bin"
      --ids-file "$IDS" -n 128 --arena-gib "$ARENA_GIB" --arena-adapt --vram-margin 1.0)

run_step() {
    # SIGABRT'd rungs release VRAM asynchronously; the run-2 cpuexp rung cudaMalloc'd over a
    # not-yet-freed card and OOM'd on the 6.8 GiB dense half. Let the driver drain first.
    sleep 10
    local name=$1; shift
    local log=logs/p4.4-${name}-${TS}.log
    echo "=== p4.4/$name start $(date +%T) (avail ${AVAIL} GiB) ==="
    "$@" >"$log" 2>&1
    local rc=$?
    echo "--- tail $log:"; tail -25 "$log"
    # P4.1 witness (P4-RAM-BUDGET.md): ds41_moe prints "arena: pinned (cudaHostRegister PORTABLE ok; ...)";
    # "N slices pinned" or an absent PORTABLE = degraded registration = the UNREGISTERED fallback was tested
    grep -m4 -iE "arena: |PORTABLE ok|slices pinned|MemAvailable" "$log" 2>/dev/null | sed 's/^/[arena] /'
    echo "=== p4.4/$name rc=$rc log=$log ==="
    return $rc
}

# ds41_generate defaults experts=gpu (ds41_generate.cpp:73) and this ARGS carries no --experts at all, so the
# old "${ARGS[@]/--experts gpu/--experts cpu}" matched nothing and the rung silently re-ran base. Append the flag.
if run_step base bash tools/v41/serve/ds41-engine-906.sh "${ARGS[@]}"; then
    echo "P4.4 WITNESS: PASS (engram tables served a greedy continuation)"
elif run_step noenc bash tools/v41/serve/ds41-engine-906.sh "${ARGS[@]}" --no-enc-skip; then
    echo "P4.4 WITNESS: PASS-NO-ENC-SKIP - the enc-skip prefill path is the suspect; do not proceed to serve"
elif run_step cpuexp bash tools/v41/serve/ds41-engine-906.sh "${ARGS[@]}" --experts cpu --backend cpu -n 8; then
    # --backend cpu is REQUIRED here (run-3 refuse-reason print, ds41_generate.cpp:926): with
    # --backend cuda the layer-major path always passes device activations and run_chunk_dev
    # refuses cpu_only - ds41 has no GPU-dense/CPU-experts combo, unlike GLM. -n 8 (last -n
    # wins, :106): CPU dense would take hours for 128 tokens; the witness is tokens flowing.
    echo "P4.4 WITNESS: PASS-CPU-ONLY - the GPU expert path is NOT witnessed; do not proceed to serve"
else
    echo "P4.4 WITNESS: FAIL (both rungs; logs logs/p4.4-*-${TS}.log - attach them to the issue)"
    exit 1
fi
