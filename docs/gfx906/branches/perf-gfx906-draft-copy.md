# `perf/gfx906-draft-copy`

Keeps the adaptive swap copies from crowding the MTP draft. Stacked on `main` (on top of `perf/gfx906-adapt-copy`).

## Why

The kernel-copy swaps run beside the drafter's round graph. With the default 64-block grid they inflated the draft by
~1.2 ms a window (draft 4.1 ms = clean 2.9 + 1.2). A smaller grid takes longer but leaves the compute units to the draft.

## Switches

| Switch | What | Production | Output |
| --- | --- | --- | --- |
| `STRATA_ADAPT_COPY_BLOCKS=N` | grid of the swap kernel copy (default 64, unchanged) | 16 | exact |
| `STRATA_ADAPT_COPY_BATCH=1` | a swap round's copies in one launch | off | exact |

Both switches only act with `STRATA_HIP_ADAPT_KERNEL_COPY=1` (`perf/gfx906-adapt-copy`); `ADAPT_COPY_BATCH` also only on
the serve path with one GPU.

Also compiled in: VlDubr's unroll hints (c3d0ca379c: `gdn_ab`, `mtp_select`, `argmax_rows`, `row_top_prob_split`, norm)
and a float4 copy in `copy_from_mapped_multi`; same loads, same sum order. And draft-head cases in
`mmvq_wave_rows_parity --bench` (`STRATA_BENCH_CASE=dhead` filters them).

## Measured

- Grid sweep: 32 blocks +0.8%, **16 +1.2..+2.2%** (two runs), 12 +1.7%, 8 +1.8%, 4 −0.5% (the copy no longer finishes
  before the next verify). 8–16 is a plateau. Draft 4.18 → 3.37 ms at 16 blocks; output hashes identical.
- One-launch batch: equal to per-swap launches (≤ +0.3%).
- Unroll hints: +0.3% (noise level), exact; the draft's select step 0.020 → 0.010 ms.
