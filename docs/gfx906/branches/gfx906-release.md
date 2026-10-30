# `gfx906/release`

Fork releases. Own branch on upstream.

## Versions

A release is tagged `v<upstream version>+<fork version>-gfx906`, for example `v0.2.8+0.1-gfx906`:

- **Upstream version:** the upstream Strata app version `main` is built on (`apps/host/package.json`).
- **Fork version:** counts every fork release, across upstream versions: `0.1`, `0.2`, … (then `1.0` for a big
  change). A new upstream base doesn't reset it, so the fork version alone orders all releases.

The image for a release is `ghcr.io/hboyd2003/strata-gfx906:<upstream>-<fork>` (Docker tags can't hold `+`), and
`:latest` follows the newest release.

## Cutting a release

On the MI50 box, from a clean checkout of `origin/main`:

```bash
docker/gfx906/release.sh --dry-run     # shows the version, the image and every step
docker/gfx906/release.sh               # pushes the image, then the tag
```

- By default it releases the image production runs (the `strata` container's), after checking that the image's
  `org.opencontainers.image.revision` label has the same code as `HEAD`; the docs, the CI and the release tooling don't
  count, nor does `bench/`. `--image IMAGE` picks another local image under the same check, and `--build` builds a fresh one from `HEAD`.
- `--fork X.Y` sets the fork version instead of taking the next one.
- Pushing needs `docker login ghcr.io` with a token that has `write:packages`.

Pushing the tag starts the **gfx906 release** workflow (`.github/workflows/gfx906-release.yml`). It checks the tag (the
upstream part must match the tagged commit's app version, the fork part must be above every earlier release, and the
commit must be on `main`), and publishes the GitHub Release with notes from `docker/gfx906/release_notes.py`: the
upstream base, the image, the branches merged, and the branches added or dropped since the previous release.

## Why the image is built on the box

The image's CPU code is built with `-march=native` on the box's Zen 4 (AVX-512 VNNI), which GitHub's runners don't
have, and the ROCm base image is larger than a hosted runner's free disk. Releasing the production image also means
every release is a build that has already been smoke-tested and served.
