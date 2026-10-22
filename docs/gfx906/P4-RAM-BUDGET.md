# P4 RAM / arena budget (P4.1, task #19) — computed 2026-10-22, no GPU touched

Scope: what one ported engine (GLM-5.3-Flash or DeepSeek-V4.1-Flash) costs in **system RAM**
while it boots and serves on this box, and the MemAvailable floors the P4.2-P4.5 witness
scripts refuse under. Numbers derive from the GGUF breakdowns and engine source (cited); the
two components marked *estimate* are the first things a boot log can replace with measurements.
Correctness/performance are NOT claimed here (AGENTS.md split).

## The box (facts, measured 2026-10-22)

- 125 GiB total RAM, **no swap** — an over-commit is an OOM-kill or worse, never a slowdown.
- 4× MI50 32 GB (gfx906), one engine at a time (standing crash rule, see "driver budget").
- `amdgpu.no_system_mem_limit = N` (the guard rail is ON; do not flip it — the 2026-10-20
  reset is the exceed-behavior).

## How the ported engines take RAM (source, both engines)

Both `glm_generate` and `ds41_generate` route their host expert cache through
`strata::core::PinnedArena` (`tools/ds4/ds4_moe.cpp:953`, `tools/v41/engine/ds41_moe.cpp:1488`):
mmap the arena, then `cudaHostRegister(...Portable|Mapped)` — under ROCm that is
`hipHostRegister` → a **KFD userptr mapping**, which the driver charges against the same
"resident system memory limit" that logged `SVM mapping failed, exceeds resident system
memory limit` minutes before the 2026-10-20 watchdog reset (`/usr/src/amdgpu-.../kfd_svm.c:343`
calls `amdgpu_amdkfd_reserve_mem_limit(..., USERPTR, ...)` on exactly that path). Three
consequences, each with its guard:

1. **Engine self-refusal** (catchable): both tier codes refuse an arena that would leave less
   than `mem_floor_gib = 3` MemAvailable (`ds4_moe.cpp:945`, `ds41_moe.cpp:1480`) and cap the
   arena at `min(arena_gib, max_arena_gib)`; both generate mains set `max_arena_gib =
   arena_gib`, so the flag is the ceiling. The floor is a hygiene guard, not the P4 safety
   margin — the witness scripts enforce that.
2. **Registration refusal is SOFT**: if the driver declines the whole-arena
   `cudaHostRegister`, `PinnedArena` logs a note and degrades (sliced registration, then
   unregistered = ~2× slower PCIe share, still correct). So EVERY P4.2/P4.4 log must be
   grepped for `cudaHostRegister` (the smoke scripts now do):
   `PORTABLE ok` = full pin; `limited to N GiB` / `FAILED` = degraded tier — a PASS with a
   failed pin witnesses correctness, not the pinned path.
3. **The exact driver limit is not readable on this box** (partial `/usr/src` tree; sysfs
   exposes only the boolean). The witnessed envelope: 49 GiB of GPU-attached host memory on
   ONE engine booted fine (2026-10-19 native 512K boot); the crash happened with TWO engines
   mapping (49 + 40 GiB) at once. A single engine registering 60-64 GiB sits between those
   two data points — the first P4.2 boot EXTENDS the envelope, and it does so with the scope
   cap and script floors below already armed.

## Ledger — GLM-5.3-Flash smoke (P4.2: `--arena-gib 60 --ctx 32768`, no LoRA/prune/bias)

| component | GiB | where it lives | source |
| --- | --- | --- | --- |
| pinned expert arena | 60.0 | RAM, userptr-pinned | `--arena-gib 60` (config default, `glm_generate.cpp:48`) |
| dense half (attention + shared + head) | 7.3 | RAM (mmap), uploaded to VRAM | GGUF breakdown: 156.31 GB = routed 148.50 + **dense 7.81** + nextn 0.03 (computed from metadata, 2026-10-18) |
| KV @ 32768 tok, MLA F16 (hardcoded `glm_dense.cpp:842`) | 0.4 | VRAM (host: staging only) | 12 MLA layers × 13,056 B/tok = 0.398 GiB |
| KDA constant state (34 layers) | 0.14 | VRAM | 139 MB total, ctx-independent |
| HIP runtime + graph-capture host side | ~2 *estimate* | RAM | first boot log replaces it |
| **steady RSS** | **≈ 69** | | |

Scope cap (`MemoryMax=84G`, wrapper) leaves ~15 GiB headroom. Page-cache churn over the
156 GB routed file is reclaimable and needs no reservation.

## Ledger — DS4.1-Flash smoke (P4.4: `--arena-gib 64`, ctx auto = 4096 floor)

| component | GiB | where it lives | source |
| --- | --- | --- | --- |
| pinned expert arena | 64.0 | RAM, userptr-pinned | `strata-ds41-906.json` (authors' serve number) |
| dense half Q8_0 → Q6_K requant | 5.4 steady / 12.4 transient peak | RAM | wrapper notes: 7.59 → 5.86 GB at load (`STRATA_DENSE_REQUANT=q6_K`); peak holds both halves |
| engram host row cache | 0.5 | RAM | default 2 M rows × 264 B (geometry, 2026-10-18) |
| chunk-prestage region | 1.5 | RAM | `STRATA_PRESTAGE_GIB=1.5` (wrapper default) |
| KV @ 4096 tok (BF16 default; ctx auto = prompt+n+64, floor 4096, `ds41_generate.cpp:286`) | 0.01 | VRAM | 2,560 B/tok × 4096 |
| HIP runtime + misc host | ~2 *estimate* | RAM | first boot log replaces it |
| **steady** | **≈ 73.5** | | |
| **transient peak (during requant)** | **≈ 81** | | 3 GiB under the 84 GiB scope cap |

The **190 GiB engram tables never enter RAM**: the loader preads table rows on demand
(`ds41_engram.cpp`), which is also why the engram gate (`check_engram_tables.py`) is header +
sample-reads only. The authors run arena 64 under the same 84 GiB cap on their box — that is
their witness that the ledger closes; if P4.4 shows the scope OOM-killing the requant peak,
drop `ARENA_GIB=60` (an OOM-kill inside a scope is a catchable exit, not a driver wedge —
the arena is already resident at that point).

**Both engines at once is arithmetically impossible**: 60 + 64 GiB of pinned arenas > 125 GiB
total. One engine at a time is RAM-enforced here even before the crash rule.

## Smoke context: GLM 32768, DS4.1 auto-4096 — the choice is VRAM, not RAM

KV at these ctx is 0.4 / 0.01 GiB (VRAM side) — orders of magnitude from mattering. The reason
to smoke at reduced ctx is VRAM and geometry: expert-slot pressure and the 40-layer window
rings scale with ctx, and GLM's 4-shard split boundary arithmetic (Xolotl tuned on 3 shards)
is untested. DS4.1 takes the engine's own auto ctx (no `--ctx` passed); GLM pins 32768 as
drafted in its config. For reference at the P4b target (512K): GLM MLA = 6.5 GiB, DS4.1 =
1.25 GiB VRAM — still not the RAM problem; the VRAM fight is expert slots (per the 2026-10-18
geometry note).

## Refuse floors for the witness scripts (replace the earlier placeholders)

Floor = arena + dense + scratch + 10 GiB margin against reclaim thrash during arena fill
(arena fill reads ~60 GiB from NVMe while pinning):

| engine | floor arithmetic | script default |
| --- | --- | --- |
| GLM (arena 60) | 60 + 8 + 3 + 10 = 81 | `MIN_AVAIL_GIB = arena + 25 = 85` (script, ≥ floor ✓) |
| DS4.1 (arena 64) | 64 + 8 + 3 + 10 = 85 | `MIN_AVAIL_GIB = arena + 25 = 89` (script, ≥ floor ✓) |

The scripts (`tools/gfx906/run-p4.2-glm-smoke.sh`, `run-p4.4-ds41-smoke.sh`) enforce
`MemAvailable >= arena+25` AND zero live engines BEFORE any model touch, so the engine's own
3 GiB floor should never be the thing that fires. Override with `MIN_AVAIL_GIB=...` only
deliberately.

## What a boot can falsify here (falsifiable list)

1. `cudaHostRegister ... PORTABLE ok` absent / `FAILED` in the P4.2/P4.4 log ⇒ the 60-64 GiB
   single-engine pin exceeds the driver budget ⇒ lower `ARENA_GIB` until it registers, and
   this doc's "driver budget" section gets a measured ceiling.
2. Measured steady RSS materially above ≈69 (GLM) / ≈73.5 (DS4.1) GiB ⇒ the *estimate* rows
   were wrong in the log's favor of more reserve; update the floor.
3. Scope OOM-kill (exit code 137 / `memory` in the log tail) during DS4.1 load ⇒ the requant
   transient exceeds 84 GiB ⇒ `ARENA_GIB=60` or disable requant (`STRATA_DENSE_REQUANT=`).
