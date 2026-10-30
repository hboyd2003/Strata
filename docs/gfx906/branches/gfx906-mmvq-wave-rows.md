# `gfx906/mmvq-wave-rows`

Two or four weight rows per wavefront in the fork's wave64 MMVQ kernel. Own branch on upstream.

## What it does

With several tokens in a verify window, the dense projections are bound by activation loads, not weights: each row's
wavefront re-reads every column's q8_1 activations (on gfx906, Q6_K at 4 columns is 21 vector loads per row, 16 of them
activations). R rows per wavefront share those loads once the per-row bounds check leaves the k loop
(`wave_rows_dot<GUARD>` runs unchecked when all R rows exist and keeps the checked loop for a matrix's tail).
Every row's sum is the R = 1 kernel's: same lanes, same block order, same butterfly.

## Switch

| Switch | Default | Production | Output |
| --- | --- | --- | --- |
| `STRATA_MMVQ_WAVE_ROWS=1\|2\|4` | 1 | 2 | exact |

## Measured

- `mmvq_wave_rows_parity` (gfx906 test builds): R = 2 and 4 bitwise equal to R = 1 on the model's projection shapes at
  1–5 columns; `--bench` times each.
- Kernel, R = 2 vs 1: faster in all 30 shape/column cases, +2.7% to +34.6%.
- Decode: 62.5 → 65.5 tok/s (+4.7%), output hashes identical. At 177K context: 48.3 → 52.6 tok/s, prefill unchanged.

Measured before `perf/gfx906-decode-kernels`. With its `STRATA_MMVQ_IACC=1` (production), Q6_K / Q4_K / Q5_K take the
IACC kernel and its own rows, so `WAVE_ROWS` now only affects IQ4_XS.
