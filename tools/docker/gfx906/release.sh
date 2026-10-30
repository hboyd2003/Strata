#!/usr/bin/env bash
# Cut a fork release from the MI50 box: push the gfx906 image to ghcr.io, then push the version tag. The tag starts
# the "gfx906 release" workflow, which checks it and publishes the GitHub Release with its notes.
#
#   docker/gfx906/release.sh [--fork X.Y] [--image IMAGE | --build] [--dry-run]
#
# Run it from a clean checkout of origin/main. The version is v<upstream>+<fork>-gfx906: the upstream part comes from
# apps/host/package.json, the fork part is the next one after the highest v*-gfx906 tag (or --fork X.Y).
#
# Image: by default the one the running `strata` container uses, i.e. what production already runs and was smoke-tested
# on. --image names another local image; either must carry an org.opencontainers.image.revision label whose code
# (all but the docs, CI, bench/ and release tooling) equals HEAD's. --build builds a new one from HEAD instead; the build uses
# every core, so don't run it beside a busy server on a RAM-tight box.
#
# Needs `docker login ghcr.io` with a token that has write:packages (for example
# `gh auth refresh -s write:packages && gh auth token | docker login ghcr.io -u <user> --password-stdin`).
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
NOTES=docker/gfx906/release_notes.py
# files that never reach the image: the docs, the CI, the benchmark harness and this release tooling
NOT_IMAGE=(':!*.md' ':!.agents' ':!.github' ':!bench' ':!docker/gfx906/release.sh' ':!docker/gfx906/release_notes.py')
fork="" src="" build=0 dry=0
while [ $# -gt 0 ]; do
  case $1 in
    --fork) fork=$2; shift 2 ;;
    --image) src=$2; shift 2 ;;
    --build) build=1; shift ;;
    --dry-run) dry=1; shift ;;
    *) sed -n '2,16p' "$0"; exit 2 ;;
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

if [ $build = 1 ]; then
  src=strata-gfx906:release-$(date +%Y%m%d)-$(git rev-parse --short HEAD)
  run docker build -f docker/gfx906/Dockerfile --build-arg STRATA_COMMIT="$(git rev-parse HEAD)" -t "$src" .
else
  [ -n "$src" ] || src=$(docker inspect -f '{{.Config.Image}}' strata)
  rev=$(docker image inspect -f '{{index .Config.Labels "org.opencontainers.image.revision"}}' "$src")
  [ -n "$rev" ] || { echo "$src has no revision label; use --build" >&2; exit 1; }
  git cat-file -e "$rev^{commit}" 2>/dev/null || { echo "$src was built from $rev, which this clone doesn't have" >&2; exit 1; }
  if ! git diff --quiet "$rev" HEAD -- . "${NOT_IMAGE[@]}"; then
    echo "$src was built from $(git rev-parse --short $rev), whose code differs from HEAD:" >&2
    git diff --stat "$rev" HEAD -- . "${NOT_IMAGE[@]}" | tail -5 >&2
    echo "deploy a build of HEAD first, or use --build" >&2; exit 1
  fi
  echo "image: $src (built from $(git rev-parse --short $rev); same code as HEAD)"
fi

latest=${image%:*}:latest
run docker tag "$src" "$image"
run docker tag "$src" "$latest"
run docker push "$image"
run docker push "$latest"
run git tag -a "$tag" -m "Fork release $fork on upstream v$upstream ($image)"
run git push origin "refs/tags/$tag"
echo "pushed $tag; the gfx906 release workflow publishes the GitHub Release"
