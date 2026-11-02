# How this fork is maintained

**Short version:** `main` is generated, not developed on. Every branch and `main` itself get force-pushed
whenever upstream moves. Pin a commit if you depend on one.

## What `main` is

`main` = the latest upstream `main` + one `--no-ff` merge per branch, always in the same order
(the list is in the [start page](README.md#branches-merged-into-main)). No commit is ever made on `main`
directly; every change lives on a branch, and `main` only holds the merges.

Each branch is one of these kinds:

| Kind | Branch names | What it holds | Sits on |
| --- | --- | --- | --- |
| Fork copy | `rebased/<fork>` | another fork's branch, rebased onto upstream | upstream |
| PR copy | `rebased/pr-<N>` | an upstream PR's commits, rebased onto upstream | upstream |
| Own | `gfx906/*`, `perf/*`, `serve/*`, `fix/*`, `docs` | my changes, independent of the others | upstream (or the fork/PR copy it extends) |
| Stacked | same prefixes | my changes that need code from earlier merges | `main` as it was at that branch's slot |
| Parked | `gfx906/dsv41`, `build/dsv41-serve` | kept rebased but not merged (DeepSeek work, paused) | `main` |

Fork and PR copies change only to follow their source. Fixes and tuning go on my own branches instead,
so a copy can always be compared with what its author published. Bug fixes to the copies' code collect on
[`fix/rebased-review`](branches/fix-rebased-review.md), merged after my other branches.

## A catch-up (rebuilding `main`)

When upstream moves:

1. Back up every branch head.
2. Rebase each fork copy, PR copy and own branch onto the new upstream `main`. Fork and PR sources are
   checked for new commits or force-pushes first, and those are pulled in.
3. Rebuild `main` from scratch: start at upstream and merge each branch with `--no-ff`, in order.
   A stacked branch is re-created on the new `main` at its slot (its own commits cherry-picked) and then merged.
4. Build the image, A/B it against the running one, and deploy only if it is no slower.
5. Force-push `main` and every rebased branch.

Conflicts are resolved in favour of keeping the gfx906 code. When upstream has reorganised a file that a
branch only adds text to (READMEs, docs), the branch's text is re-applied to upstream's new layout rather
than hand-merging the old one.

## What that means for you

- **Hashes do not survive a catch-up.** `main` and every branch are rewritten, typically every few days.
- To follow the fork: `git fetch` and then `git reset --hard origin/main` (or the branch you track).
  `git pull` will try to merge the old history with the new and make a mess.
- To build on it: start from a release tag (those are never moved), or from a commit you keep around yourself.
  Rebase your work onto the new `main` after a catch-up; don't merge it.

## Releases

Releases are tagged `v<upstream version>+<fork version>-gfx906`, for example `v0.2.8+0.1-gfx906`. The upstream part
is the upstream Strata version `main` is built on; the fork part is the version of the fork's own changes. It goes up
only when the fork itself changes, so a catch-up that only moves the upstream base keeps it (`v0.2.8+0.1-gfx906` →
`v0.2.9+0.1-gfx906`). Tags are the only refs that are never rewritten, so a release tag is the thing to pin.

Each release has a GitHub Release and an image, `ghcr.io/hboyd2003/strata-gfx906:<upstream>-<fork>` (plus `:latest`).
The image is the one production runs on the MI50 box, pushed from there; the tag then starts a workflow that checks it
and writes the notes. Details: [branches/gfx906-release.md](branches/gfx906-release.md).

The fork carries no upstream tags. Upstream's own releases are at [Niko1221/Strata](https://github.com/Niko1221/Strata/releases).

## How a change gets in

Every candidate (an upstream PR, a commit from another fork, or my own idea) goes through the same steps:

1. **Check it can reach this box at all**: gfx906 wave64, one request slot, MTP speculative decoding,
   these quant types, one GPU. Many switches decline or compile out on gfx906 (see [tested.md](tested.md)).
2. **Branch** off `main` as an experiment. Experiments stay local.
3. **Build** in the ROCm container. For kernel changes, check VGPRs and occupancy before timing anything.
4. **Parity tests** for the touched kernels, on the MI50.
5. **A/B** with [`bench/gfx906/ab.py`](branches/gfx906-ab.md), against the running build, with a repeat of the
   base as a noise control: decode tok/s and ms per window, every decode stage, prefill at several lengths,
   long-context turns where relevant, and the greedy output hashes. On the box a small wrapper restarts the server
   with each variant's config and runs the harness. (Before 2026-10-30 the harness and the restarts were one in-house
   script; the published tool is that script's measuring part, unchanged in what it measures.)
6. **Classify the output effect**: exact, rounding only, or quality-reducing. Quality-reducing changes are
   not taken, whatever the speed.
7. **Keep or drop.** A keeper gets a real branch name, is merged into `main` and deployed, and is documented
   in `docs/gfx906/branches/<branch>.md` on that branch. Everything else is recorded in [tested.md](tested.md).

## Where the docs live

- `docs/gfx906/branches/<branch>.md`: written on that branch, so the page travels with the code. Each branch
  only ever touches its own page, so merging never conflicts on docs.
- Everything else (this page, the start page, what was tried and dropped) lives on the `docs` branch,
  which is merged last.
- The working notes these pages are written from live outside the repo. The docs are updated from them
  before each push, so the pushed docs can lag the box by a day or so.
