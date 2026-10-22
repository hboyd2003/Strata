#!/usr/bin/env bash
# ds41-engine-906.sh - THIS BOX's wrapper (gfx906 fork), mirrors glm-engine-906.sh: point the authors'
# serve wrapper at this build dir, and make the author-machine defaults local.  The authors'
# ds41-engine.sh stays untouched; the server config (strata-ds41-906.json) names THIS file.
#
# Two things differ from the authors' box, both wrong-path defaults that would break the boot here:
#   1. build dir:  this repo's HIP build lives in build-906 (their tree builds build-xolotl)
#   2. DSpark:     their default sidecar is /media/mal/NVME1TB/... - no such path here.  The local
#      sidecar IS on disk (/ai/models/hf.co/IndrasMirror/DeepSeek-V4.1-Flash-GSQ-Uncensored/
#      V41-MTP-Vision.Q2_K.gguf) but P4.5 runs serve WITHOUT drafts first; enable after the
#      exactness witness:  STRATA_SERVE_DSPARK=<that path> ./ds41-engine-906.sh --serve ...
#      ('${VAR-default}' in ds41-engine.sh substitutes only when UNSET, so exporting empty = off.)
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
export STRATA_BUILD_DIR="${STRATA_BUILD_DIR:-$HERE/../../../build-906}"
export STRATA_SERVE_DSPARK="${STRATA_SERVE_DSPARK-}"
exec "$HERE/ds41-engine.sh" "$@"
