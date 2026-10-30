# `perf/topk-router`

Two gfx906 router kernels and a split decode top-k, from other people's work. Stacked on `main`.

## What it does

- **Two-stage decode top-k** for the QSA layers' block selection (MarcoSon, upstream PR #3987).
- **1- and 2-token BF16 router kernels** for the 2560 × 512 router (0FL01/Strata 2fea9696). They need
  `STRATA_MMVF_ROWS=1`.

## Switches

| Switch | Default | Production | Output |
| --- | --- | --- | --- |
| `STRATA_GFX906_TOPK_SPLIT=1` | off | on | exact |
| `STRATA_GFX906_ROUTER_T1_ROWS=1`, `STRATA_GFX906_ROUTER_T2_EXACT=1` | off | set, but inert (see below) | exact |

## Measured

- Top-k split: about +3% decode at 169K context, nothing at short context. `qsa_select_bench`: 0.166 → 0.071 ms at 131K,
  0.255 → 0.095 ms at 258K.
- Router rows: +0.4–0.7% decode when they were added.

Since `perf/gfx906-router`, `STRATA_GFX906_ROUTER_GEMV` (set to `auto` in production) takes 1–4-token router calls
before these two kernels, so they only run with that switch off.
