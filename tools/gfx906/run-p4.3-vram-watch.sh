#!/usr/bin/env bash
# run-p4.3-vram-watch.sh - task #32: WHO EATS THE VRAM during GLM serve decode.
# Samples the engine's KFD VRAM (rocm-smi --showpids) every 5 s and prints the curve.
#
#   tools/gfx906/run-p4.3-vram-watch.sh direct   # glm_generate -n 200 with the exact SERVE args
#   tools/gfx906/run-p4.3-vram-watch.sh serve    # serve.server + one budget-cancel request (the crashing curl)
#
# direct answers the gate question: does the growth/crash need serve at all (cancel, reuse),
# or is it per-decoded-token engine behaviour? serve reproduces the witness path.
# Read-only wrt models/configs; refuses under the standing 65 GiB RAM guard.
set -u
cd "$(dirname "$0")/../.."
MODE="${1:-direct}"
[ "$MODE" = direct ] || [ "$MODE" = serve ] || { echo "usage: $0 [direct|serve]"; exit 2; }

mkdir -p logs
STAMP=$(date +%Y%m%d-%H%M%S)
LOG="logs/p4.3-vram-$MODE-$STAMP.log"
VLOG="$LOG.vram"
IDS=$(ls -t logs/p4.2-ids-*.i32 2>/dev/null | head -1)
[ -n "$IDS" ] || { echo "no prompt ids file (logs/p4.2-ids-*.i32) - run the P4.2 smoke first"; exit 1; }
AVAIL=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)
[ "$AVAIL" -ge 65 ] || { echo "MemAvailable ${AVAIL} GiB < 65 - refuse (standing P4 guard)"; exit 1; }
pgrep -f "glm_generate|ds41_generate" >/dev/null && { echo "another HIP engine is running - one engine at a time"; exit 1; }

MODEL=/ai/models/hf.co/peasantsmith/GLM-5.3-Flash-Maya-GGUF/Maya-L/GLM-5.3-Flash-Maya-L-IQ3_S-00001-of-00004.gguf
LORA=/ai/models/hf.co/GCSA-AiLab/GLM-5.3-Flash-Abliterated-LoRA-GGUF/GLM-5.3-Flash-Ablitered2-LoRA-v2.gguf

sam() { # sample "<secs> <vram_bytes>" lines for the engine pid until it is gone or SAM_STOP exists
  local pat="$1" t0 pid b
  t0=$(date +%s)
  while [ ! -e "$LOG.stop" ]; do
    pid=$(pgrep -x "$pat" | head -1)   # -x by exe NAME: the wrapper/systemd scope reshuffles argv, -f missed it
    if [ -n "$pid" ]; then
      b=$(rocm-smi --showpids 2>/dev/null | awk -v p="$pid" '$1==p{gsub(/[^0-9]/,"",$4); print $4; exit}')
      echo "$(( $(date +%s) - t0 )) ${b:--1}" >> "$VLOG"
    fi
    sleep 5
  done
}

echo "=== p4.3 vram-watch ($MODE) -> $LOG (samples: $VLOG) avail ${AVAIL} GiB ==="
: > "$VLOG"

if [ "$MODE" = direct ]; then
  echo "--- direct: glm_generate with the EXACT serve args (prune+route-bias+lora+lora-exps), -n 200 ---"
  SAM_PAT='build-906/glm_generate'
  STRATA_NO_LARGEPAGES=1 build-906/glm_generate \
    -m "$MODEL" --backend cuda --experts gpu --arena-gib 60 --arena-skip-resident \
    --ctx 32768 --prefill-chunk 4096 --chunk-mmq --vram-margin 4.0 \
    --prune tools/glm/serve/prune-ezct-0.25.txt --prune-penalty 0.05 --route-bias 0.25 \
    --arena-adapt --pf-b 1.0 --pcie auto --stop 154820,154827,154829 \
    --lora "$LORA" --lora-exps \
    --ids-file "$IDS" -n 200 > "$LOG" 2>&1 &
  ENG=$!
  sam 'glm_generate' & SAM=$!
  wait "$ENG"; RC=$?
  touch "$LOG.stop"; wait "$SAM" 2>/dev/null
  echo "--- direct rc=$RC (134/1 = the serve death class; 0 = survive: then serve-only trigger) ---"
else
  echo "--- serve: server + the crashing curl (thinking budget cuts at 128 -> cancel -> wrap-up) ---"
  CFG=tools/glm/serve/strata-glm-mayal-gfx906.json
  if [ -n "${2:-}" ]; then   # optional 2nd arg: --vram-margin override (temp config, original untouched)
    CFG="logs/p4.3-cfg-margin$2-$STAMP.json"
    python3 - "$2" "$CFG" <<'PYEOF'
import json, sys
margin, out = sys.argv[1], sys.argv[2]
cfg = json.load(open("tools/glm/serve/strata-glm-mayal-gfx906.json"))
a = cfg["args"]; i = a.index("--vram-margin") + 1
a[i] = margin
cfg["_comment"] = (cfg.get("_comment", "") + f"  TEMP (#32 vram-watch): --vram-margin {margin}.")
json.dump(cfg, open(out, "w"), indent=2)
PYEOF
    echo "--- margin override $2 -> $CFG ---"
  fi
  STRATA_NO_LARGEPAGES=1 python3 -m serve.server --engine strata \
    --config "$CFG" --port 8147 > "$LOG" 2>&1 &
  SRV=$!
  for i in $(seq 1 90); do grep -q "ready: http" "$LOG" 2>/dev/null && break; sleep 2; done
  grep -q "ready: http" "$LOG" || { echo "server never became ready; see $LOG"; kill -9 "$SRV" 2>/dev/null; exit 1; }
  sam 'glm_generate' & SAM=$!
  curl -s --max-time 900 http://127.0.0.1:8147/v1/chat/completions -H 'Content-Type: application/json' \
    -d '{"model":"glm","messages":[{"role":"user","content":"Say hello in three words."}],"max_tokens":256}' > "$LOG.curl"
  echo "--- curl -> $LOG.curl ---"
  touch "$LOG.stop"; wait "$SAM" 2>/dev/null
  kill -INT "$SRV" 2>/dev/null
  for i in $(seq 1 15); do kill -0 "$SRV" 2>/dev/null || break; sleep 1; done
  kill -9 "$SRV" 2>/dev/null   # engine already exited on its own; SIGKILL to the idle server touches no GPU state
fi

echo "--- VRAM curve (secs, GiB) ---"
awk '{printf "%6s s  %8.3f GiB\n", $1, ($2<0?0:$2)/1073741824}' "$VLOG"
echo "--- engine tail ---"
if [ "$MODE" = direct ]; then tail -6 "$LOG"; else tail -6 /tmp/strata-glm-mayal-906.log; fi
echo "DONE log=$LOG vram=$VLOG"
