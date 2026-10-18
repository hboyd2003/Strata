#!/usr/bin/env bash
# glm-engine-906.sh - this box's wrapper (gfx906 fork): points the Xolotl serve wrapper at OUR build dir.
# The upstream glm-engine.sh defaults to build-xolotl; this checkout builds to build-906.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
export STRATA_BUILD_DIR="${STRATA_BUILD_DIR:-$HERE/../../../build-906}"
exec "$HERE/glm-engine.sh" "$@"
