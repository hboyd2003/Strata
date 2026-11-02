#!/usr/bin/env bash
# Cut a fork release from the MI50 box: push the gfx906 image to ghcr.io, then push the version tag. The tag starts
# the "gfx906 release" workflow, which checks it and publishes the GitHub Release with its notes.
#
#   docker/gfx906/release.sh [--fork X.Y] [--image IMAGE | --build] [--portable-image IMAGE | --no-portable] [--dry-run]
#
# Run it from a clean checkout of origin/main. The version is v<upstream>+<fork>-gfx906: the upstream part comes from
# apps/host/package.json, the fork part stays the latest release's unless --fork X.Y raises it (do that when the fork's
# own changes changed, not for a catch-up that only moves the upstream base).
#
# Image: by default the one the running `strata` container uses, i.e. what production already runs and was smoke-tested
# on. --image names another local image; either must carry an org.opencontainers.image.revision label whose code
# (all but the docs, CI, bench/ and release tooling) equals HEAD's. --build builds a new one from HEAD instead; the build uses
# every core, so don't run it beside a busy server on a RAM-tight box.
#
# Portable image: the same release also gets a STRATA_PORTABLE=ON build (AVX2 baseline, for other CPUs), pushed as
# :<version>-portable and :latest-portable before the tag; the tag's message names it and the notes then list both pulls.
# By default it is built from HEAD (same caveat as --build: the script refuses while the `strata` container runs);
# --portable-image names an existing one (checked against HEAD like --image), --no-portable leaves it out.
#
# Needs `docker login ghcr.io` with a token that has write:packages (for example
# `gh auth refresh -s write:packages && gh auth token | docker login ghcr.io -u <user> --password-stdin`).
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
NOTES=docker/gfx906/release_notes.py
# files that never reach the image: the docs, the CI, the benchmark harness and this release tooling
NOT_IMAGE=(':!*.md' ':!.agents' ':!.github' ':!bench' ':!docker/gfx906/release.sh' ':!docker/gfx906/release_notes.py')
fork="" src="" build=0 dry=0 psrc="" portable=1
while [ $# -gt 0 ]; do
  case $1 in
    --fork) fork=$2; shift 2 ;;
    --image) src=$2; shift 2 ;;
    --build) build=1; shift ;;
    --portable-image) psrc=$2; shift 2 ;;
    --no-portable) portable=0; shift ;;
    --dry-run) dry=1; shift ;;
    *) sed -n '2,21p' "$0"; exit 2 ;;
  esac
done
run() { if [ $dry = 1 ]; then echo "+ $*"; else "$@"; fi; }

git fetch -q origin main
git fetch -q --tags origin
[ -z "$(git status --porcelain --untracked-files=no)" ] || { echo "uncommitted changes; commit them first" >&2; exit 1; }
[ "$(git rev-parse HEAD)" = "$(git rev-parse origin/main)" ] || { echo "HEAD is not origin/main ($(git rev-parse --short origin/main))" >&2; exit 1; }

if [ -n "$fork" ]; then tag="v$(python3 -c 'import json; print(json.load(open("apps/host/package.json"))["version"])')+$fork-gfx906"
else tag=$(python3 $NOTES next); fi
eval "$(python3 $NOTES check "$tag")"   # sets upstream=, fork=, image= (the ghcr name:tag)
echo "release $tag  (upstream $upstream, fork $fork)  ->  $image"

# check_image IMAGE true|false: built from HEAD's code (revision label) and the expected kind (BUILD.json "portable")
check_image() {
  local img=$1 want=$2 rev got
  rev=$(docker image inspect -f '{{index .Config.Labels "org.opencontainers.image.revision"}}' "$img")
  [ -n "$rev" ] || { echo "$img has no revision label; build it instead" >&2; exit 1; }
  git cat-file -e "$rev^{commit}" 2>/dev/null || { echo "$img was built from $rev, which this clone doesn't have" >&2; exit 1; }
  if ! git diff --quiet "$rev" HEAD -- . "${NOT_IMAGE[@]}"; then
    echo "$img was built from $(git rev-parse --short $rev), whose code differs from HEAD:" >&2
    git diff --stat "$rev" HEAD -- . "${NOT_IMAGE[@]}" | tail -5 >&2
    echo "deploy a build of HEAD first, or build it instead" >&2; exit 1
  fi
  got=$(docker run --rm --entrypoint cat "$img" /usr/local/bin/BUILD.json |
        python3 -c 'import json, sys; print(str(json.load(sys.stdin).get("portable", False)).lower())')
  [ "$got" = "$want" ] || { echo "$img: BUILD.json says portable=$got, expected $want" >&2; exit 1; }
  echo "image: $img (built from $(git rev-parse --short $rev); same code as HEAD; portable=$got)"
}
# build_image TAG [docker build args...]: refuses beside a running server (a build uses every core and lots of RAM)
build_image() {
  local t=$1; shift
  if [ $dry = 0 ] && [ "$(docker inspect -f '{{.State.Running}}' strata 2>/dev/null)" = true ]; then
    echo "the strata container is running; stop it before building (or pass --image / --portable-image)" >&2; exit 1
  fi
  run docker build -f docker/gfx906/Dockerfile --build-arg STRATA_COMMIT="$(git rev-parse HEAD)" "$@" -t "$t" .
}

stamp=$(date +%Y%m%d)-$(git rev-parse --short HEAD)
if [ $build = 1 ]; then
  src=strata-gfx906:release-$stamp
  build_image "$src"
else
  [ -n "$src" ] || src=$(docker inspect -f '{{.Config.Image}}' strata)
  check_image "$src" false
fi
if [ $portable = 1 ]; then
  if [ -n "$psrc" ]; then
    check_image "$psrc" true
  else
    psrc=strata-gfx906:release-portable-$stamp
    build_image "$psrc" --build-arg STRATA_PORTABLE=ON
  fi
fi

latest=${image%:*}:latest
run docker tag "$src" "$image"
run docker tag "$src" "$latest"
run docker push "$image"
run docker push "$latest"
msg="Fork release $fork on upstream v$upstream ($image)"
if [ $portable = 1 ]; then
  pimage=$image-portable
  run docker tag "$psrc" "$pimage"
  run docker tag "$psrc" "$latest-portable"
  run docker push "$pimage"
  run docker push "$latest-portable"
  msg+=$'\n\n'"Portable image: $pimage"   # release_notes.py reads this line
fi
run git tag -a "$tag" -m "$msg"
run git push origin "refs/tags/$tag"
echo "pushed $tag; the gfx906 release workflow publishes the GitHub Release"
