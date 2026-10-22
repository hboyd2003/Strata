#!/usr/bin/env bash
# P4 heap-corruption repro under AddressSanitizer (operator-launched).
# Why: 2026-10-22 runs 2+3, BOTH ported engines abort `malloc(): invalid size (unsorted)`
# (glibc heap-corruption) on the GPU-experts path after a clean load; DS4.1's --no-enc-skip
# rung crashed too, so enc-skip is exonerated and the corruption sits on the common
# GPU-experts prefill path. The CPU-experts rung (GLM) runs clean, which brackets it.
# ASan instruments the HOST allocations of the engine TUs and names the writing line and
# stack at the corrupting store, instead of at some later unrelated malloc.
#
# Uses build-asan-906 (RelWithDebInfo + -fsanitize=address, host code only; HIP device code
# is uninstrumented). Default-path artifacts (build-906) are NOT touched.
#
# ASAN_OPTIONS: detect_leaks=0 because the HIP runtime leaks by design; abort_on_error=0 so
# the report prints in full before exit; the first ERROR line IS the answer.
set -uo pipefail
cd "$(dirname "$0")/../.."
MODEL="${1:-ds41}"   # ds41 | glm
LOGDIR=logs; mkdir -p "$LOGDIR"; STAMP="$(date +%Y%m%d-%H%M%S)"
if pgrep -f "glm_generate|ds41_generate|vllm" >/dev/null; then
    echo "an engine is already running; one HIP engine at a time"; exit 3
fi
AVAIL_KB=$(awk '/MemAvailable/{print $2}' /proc/meminfo)
[ "${AVAIL_KB:-0}" -lt $((70 * 1024 * 1024)) ] && { echo "refusing: only $((AVAIL_KB/1024/1024)) GiB"; exit 2; }
BIN=build-asan-906
export STRATA_BUILD_DIR="$PWD/$BIN"
export ASAN_OPTIONS=detect_leaks=0:abort_on_error=0:print_stacktrace=1:symbolize=1
export STRATA_NO_LARGEPAGES=1   # same as the smokes (box has vm.nr_hugepages=0)
IDS=logs/p4-ids-$STAMP.i32
printf '151644 856 279 315 264 279 1294 315 82 4072 29 82 29 9446' | tr ' ' '\n' |
    xxd -r -p -c 4 > "$IDS" 2>/dev/null ||
    python3 -c "import struct,sys; ids=[151644,856,279,315,264,279,1294,315,82,4072,29,82,29,9446]
open('$IDS','wb').write(b''.join(struct.pack('<i',i) for i in ids))"

if [ "$MODEL" = ds41 ]; then
    LOG=$LOGDIR/p4.4-asan-$STAMP.log
    echo "=== ds41 GPU-experts repro under ASan -> $LOG ==="
    tools/v41/serve/ds41-engine-906.sh \
        -m /ai/models/hf.co/IndrasMirror/DeepSeek-V4.1-Flash-GSQ-Uncensored/V41-GSQ.gguf \
        --backend cuda --experts gpu \
        --engram "/ai/models/hf.co/IndrasMirror/DeepSeek-V4.1-Flash-GSQ-Uncensored/engram/l1:/ai/models/hf.co/IndrasMirror/DeepSeek-V4.1-Flash-GSQ-Uncensored/engram/l14" \
        --rank /ai/models/hf.co/IndrasMirror/DeepSeek-V4.1-Flash-GSQ-Uncensored/rank.bin \
        --ids-file "$IDS" -n 8 --arena-gib 64 --arena-adapt --vram-margin 1.0 \
        > "$LOG" 2>&1
    RC=$?
else
    LOG=$LOGDIR/p4.2-asan-$STAMP.log
    echo "=== glm GPU-experts repro under ASan -> $LOG ==="
    tools/glm/serve/glm-engine-906.sh \
        -m /ai/models/hf.co/peasantsmith/GLM-5.3-Flash-Maya-GGUF/Maya-L/GLM-5.3-Flash-Maya-L-IQ3_S-00001-of-00004.gguf \
        --backend cuda --experts gpu --ids-file "$IDS" -n 8 \
        --ctx 32768 --arena-gib 60 --arena-skip-resident --prefill-chunk 4096 --chunk-mmq \
        --vram-margin 4.0 --arena-adapt --pcie auto --stop 154820,154827,154829 \
        > "$LOG" 2>&1
    RC=$?
fi
echo "rc=$RC log=$LOG"
if grep -q "AddressSanitizer" "$LOG"; then
    echo "=== FIRST ASAN REPORT (the writer's stack) ==="
    grep -A 30 "ERROR: AddressSanitizer" "$LOG" | head -40
    echo "P4 ASAN WITNESS: REPORT CAPTURED - paste the block above into the issue"
else
    echo "no ASan report (rc=$RC); if rc=134 again with no report the corruption is in an"
    echo "uninstrumented allocator (HIP runtime / ggml-cuda host side) - next step is"
    echo "instrumenting the ggml archive or a MALLOC_CHECK_=3 run"
fi
exit 0
