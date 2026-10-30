#!/usr/bin/env python3
"""Fork release versions: v<upstream version>+<fork version>-gfx906, for example v0.2.8+0.1-gfx906.

The upstream version is the app version `main` is built on (apps/host/package.json); the fork version counts every
fork release, across upstream versions (0.1, 0.2, ... 0.10, then 1.0 for a big change).

  release_notes.py check <tag>   check the tag against the checkout at HEAD; print upstream=, fork=, image= lines
  release_notes.py next          print the next tag for HEAD (the fork version after the highest v*-gfx906 tag)
  release_notes.py notes <tag>   print the release notes (Markdown) for the tag

Run inside a checkout with the history and the tags (fetch-depth: 0)."""
import json, re, subprocess, sys

TAG_RE = re.compile(r"^v(\d+(?:\.\d+){2,3})\+(\d+)\.(\d+)-gfx906$")
IMAGE = "ghcr.io/hboyd2003/strata-gfx906"
REPO = "https://github.com/hboyd2003/Strata"
BRANCH_RE = re.compile(r"\b(rebased/[\w.-]+|(?:gfx906|perf|serve|fix|ci)/[\w.-]+|docs)\b")


def git(*a):
    return subprocess.run(["git", *a], check=True, capture_output=True, text=True).stdout.strip()


def upstream_version(rev="HEAD"):
    return json.loads(git("show", f"{rev}:apps/host/package.json"))["version"]


def parse(tag):
    m = TAG_RE.match(tag)
    if not m:
        sys.exit(f"{tag!r} is not a fork version tag (v<upstream>+<major>.<minor>-gfx906, e.g. v0.2.8+0.1-gfx906)")
    return m.group(1), (int(m.group(2)), int(m.group(3)))


def image_tag(tag):
    up, (a, b) = parse(tag)
    return f"{up}-{a}.{b}"   # Docker tags can't hold '+'


def fork_tags(exclude=None):
    """Every v*-gfx906 tag, highest fork version first."""
    tags = [t for t in git("tag", "--list", "v*-gfx906").split() if TAG_RE.match(t) and t != exclude]
    return sorted(tags, key=lambda t: parse(t)[1], reverse=True)


def upstream_base(rev):
    """main is upstream + --no-ff merges, so below its merges the first-parent chain reaches upstream's head: the first
    one-parent commit after a merge (one-parent commits above the merges, which main shouldn't have, are skipped)."""
    merged = False
    for line in git("rev-list", "--first-parent", "--parents", rev).splitlines():
        h, *parents = line.split()
        if len(parents) > 1:
            merged = True
        elif merged:
            return h
    sys.exit("no upstream base found")


def merged_branches(rev):
    """(branch, merge subject) for each --no-ff merge on main's first-parent chain, in merge order."""
    base = upstream_base(rev)
    out = []
    for line in git("log", "--first-parent", "--merges", "--reverse", "--format=%s", f"{base}..{rev}").splitlines():
        m = BRANCH_RE.search(line)
        out.append((m.group(1) if m else line, line))
    return out


def check(tag, rev="HEAD"):
    up, fork = parse(tag)
    have = upstream_version(rev)
    if up != have:
        sys.exit(f"{tag}: the upstream part is {up}, but {rev} is built on upstream {have} (apps/host/package.json)")
    for t in fork_tags(exclude=tag):
        if parse(t)[1] >= fork:
            sys.exit(f"{tag}: fork version {fork[0]}.{fork[1]} is not above the existing {t}")
        break
    print(f"upstream={up}\nfork={fork[0]}.{fork[1]}\nimage={IMAGE}:{image_tag(tag)}")


def next_tag(rev="HEAD"):
    tags = fork_tags()
    a, b = parse(tags[0])[1] if tags else (0, 0)
    print(f"v{upstream_version(rev)}+{a}.{b + 1}-gfx906")


def notes(tag, rev="HEAD"):
    up, (a, b) = parse(tag)
    base = upstream_base(rev)
    branches = merged_branches(rev)
    prev = next(iter(fork_tags(exclude=tag)), None)
    lines = [
        f"**Fork release {a}.{b}** of the gfx906 (AMD Instinct MI50) fork, built on upstream Strata **v{up}** "
        f"([`{base[:8]}`](https://github.com/Niko1221/Strata/commit/{base})).",
        "",
        "Read the [warning in the README](" + REPO + "#readme) first: this fork is tuned and tested on one machine, "
        "and its history is force-pushed.",
        "",
        "## Image",
        "",
        "```bash",
        f"docker pull {IMAGE}:{image_tag(tag)}",
        "```",
        "",
        "The image's CPU code is built for the build machine (`-march=native` on a Zen 4: AVX-512 VNNI). It needs a CPU "
        "with the same instruction set; build it yourself with `--build-arg STRATA_PORTABLE=ON` for other CPUs "
        f"([gfx906/local]({REPO}/blob/{tag}/docs/gfx906/branches/gfx906-local.md)).",
        "",
        "## What is in it",
        "",
        f"Upstream v{up} plus these branches, merged in this order "
        f"([start page]({REPO}/blob/{tag}/docs/gfx906/README.md) has what each does and what production sets):",
        "",
    ]
    lines += [f"- `{br}`" for br, _ in branches]
    if prev:
        old = [br for br, _ in merged_branches(prev)]
        new = [br for br, _ in branches]
        added = [x for x in new if x not in old]
        dropped = [x for x in old if x not in new]
        lines += ["", f"## Since {prev}", ""]
        prev_up = parse(prev)[0]
        lines.append(f"- Upstream: v{prev_up} → v{up}" if prev_up != up else f"- Upstream: unchanged (v{up})")
        lines.append("- Added: " + (", ".join(f"`{x}`" for x in added) if added else "none"))
        lines.append("- Dropped: " + (", ".join(f"`{x}`" for x in dropped) if dropped else "none"))
        lines.append("- The branches are rebased on every catch-up, so compare their docs pages rather than commit hashes.")
    print("\n".join(lines))


if __name__ == "__main__":
    cmd, *args = sys.argv[1:] or ["help"]
    if cmd == "check" and len(args) == 1:
        check(args[0])
    elif cmd == "next" and not args:
        next_tag()
    elif cmd == "notes" and len(args) == 1:
        notes(args[0])
    else:
        sys.exit(__doc__)
