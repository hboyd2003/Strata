#!/usr/bin/env bash
# glm-engine.sh - the GLM engine as Strata's server starts it (`<exe> --serve <args>`), with memguard's protections
# in the FOREGROUND (memguard.sh backgrounds its command, which would give it /dev/null as stdin and break the
# stdin/stdout protocol): ComfyUI check, the shared GPU lock (held while the server runs), a 84 GiB RAM cap, no swap.
set -u
cd "$(dirname "$0")/../../.."
if [ "${MEMGUARD_ALLOW_COMFY:-0}" != 1 ] && ss -ltn 2>/dev/null | grep -q ':8188 '; then
    echo "glm-engine: refusing - ComfyUI is running on :8188 (RAM)" >&2; exit 3
fi
LOCK="${DS4_LOCK:-$HOME/.quetza-data/conductor/ds4-gpu.lock}"
# flock opens the lock file but does not create its parent directory - make sure it exists
# (no-op where the conductor tree already lives, which is every existing deployment)
mkdir -p "$(dirname "$LOCK")"
export GGML_NO_BACKTRACE=1
# s6: prompt chunks run without CUDA graphs (instantiating a chunk's graph ran out of VRAM on fresh ~30K prompts);
# decode keeps them (graphs off everywhere fixed the crash but cost ~17% decode). =0 restores graphs everywhere.
export GGML_CUDA_GRAPH_MAX_TOKENS="${GGML_CUDA_GRAPH_MAX_TOKENS:-64}"
# xolotl: the unified tree builds every engine into build-xolotl; STRATA_BUILD_DIR overrides it, and the
# pre-merge build dir (build-glm-gpu) is still found so an old checkout of this script keeps working
BUILD_DIR="${STRATA_BUILD_DIR:-$PWD/build-xolotl}"
[ -x "$BUILD_DIR/glm_generate" ] || [ ! -x "$PWD/build-glm-gpu/glm_generate" ] || BUILD_DIR="$PWD/build-glm-gpu"
# 'systemd-run --user' needs a systemd USER manager: desktop sessions have one, a root ssh session
# without XDG_RUNTIME_DIR does not. Witnessed 2026-10-22 on this fork's gfx906 box: --user exits 1
# ("Failed to connect to bus"), a --system scope works and its child sees MemoryMax=84G exactly
# (90194313216 in memory.max); scope keeps the caller's stdin/stdout, so the serve protocol is intact.
# Probe --user first so existing deployments keep the previous command byte-for-byte.
SCOPE=--user
if ! systemd-run --user --scope -q --unit="glm-probe-$$" true 2>/dev/null; then
    echo "glm-engine: no systemd user manager - using the system scope (same MemoryMax=84G, no swap)" >&2
    SCOPE=--system
fi
exec flock "$LOCK" systemd-run "$SCOPE" --scope -q --unit="glm-serve-$$" -p MemoryMax=84G -p MemorySwapMax=0 \
    "$BUILD_DIR/glm_generate" "$@"
