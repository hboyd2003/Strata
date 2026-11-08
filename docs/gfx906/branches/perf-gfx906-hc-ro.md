# `perf/gfx906-hc-ro`

A rounding-only hyper-connection read for the verify window. Stacked on `main`.

## Switches

| Switch | What | Production | Output |
| --- | --- | --- | --- |
| `STRATA_GR_RO=1` | hc read: split-K `down` with the norm folded in, a last-block reduce per row group, then the default `up` | on | rounding |
| `STRATA_GR_RO=2` | the same with a 4-rows × 8-waves `down` (benched: no better) | off | rounding |

Each layer runs two hc reads per verify window, 96 in all. The default read on gfx906 launches a norm kernel, then a
`down` projection on 41 blocks (19 of the 60 CUs idle), then `up`. **GR_RO** (gfx906 only) replaces the norm and `down`
with one launch:
- `down` is split over 20 column chunks × 11 row groups (220 blocks of 4 waves). Each block stages its chunk of R' ×
  w_norm (pending write folded in) and keeps each lane's 8 columns in registers. An 8-row reduce-scatter over the wave
  takes 10 shuffles a token.
- The last of a row group's 20 chunk blocks (an atomic counter) reduces that group's rows in a fixed order.
- The default read's `up` kernel runs after it, unchanged.

The result is deterministic run to run but sums in another order than the default read, so it is not bitwise.

The counters are a small per-device table allocated when the verifier starts (before any graph capture). Each call site
gets its own slot, keyed by its `lo` workspace. Without a slot, or with Q8 hc weights, the read falls back to the
default.

Upstream's `STRATA_GR_V3` (a similar split read) can't run on gfx906: it needs 80 KB of LDS at 8 tokens, and its
two-half fallback is disabled (#375).

## Measured

- `hc_ro_parity` (T = 1..8; the verify read, a read without the pending write, and the final mixer): worst relative
  difference 3.3e-7 on mixed / inject / R / lo / rs, and two runs are bitwise equal.
- One read, weights cold, 96 reads replayed as a graph:

  | T | default | GR_RO |
  | --- | --- | --- |
  | 1 | 40.8 µs | 32.2 µs |
  | 3 | 58.2 µs | 48.2 µs |
  | 5 | 79.0 µs | 63.0 µs |

  A kernel trace showed a separate reduce launch cost ~7 µs of latency per read, which is why the reduce sits in
  `down`'s last blocks.
- Decode (A/B, four controls with identical answers): **85.66 tok/s vs 82.18–82.61 (+3.7 to +4.2%)**, ms per window
  −4.3 to −4.5%, GPU time per window −1.0 to −1.4 ms. hc norm stages go to ~0 and `down` drops 19–22%; `up` is ~4%
  slower. Long-context (32K) decode 72.4 vs 68.8–70.5 tok/s; prefill flat. First-pass answers change, as expected for
  a summation-order change.
