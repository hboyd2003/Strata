# `perf/gfx906-argmax`

Builds upstream's multi-block head pick (`argmax_rows`) and top-probability (`row_top_prob_split`) kernels for gfx906. Stacked on `main`. Opt-in with `STRATA_MULTI_BLOCK_ARGMAX=1`; with it unset, the one-block kernels run as before.

## What the branch changes

- Both kernels sat under `#if !defined(__HIPCC__)`. The gfx906 build compiles the CUDA tree as HIP, so `__HIPCC__` is defined there even though `STRATA_USE_HIP` is not. The MTP drafter and the verify pass's output head therefore fell back to one block per row over the whole vocabulary.
- The branch builds them for `STRATA_HIP_GFX906` too. The compat layer runs them as 32-lane logical warps (width-32 shuffles). `kTopBlocks` stays at 8.
- `verify_parity` checks both kernels bitwise against the one-block kernels.

## Switches

| Switch | What | Production | Output |
| --- | --- | --- | --- |
| `STRATA_MULTI_BLOCK_ARGMAX=1` | the multi-block head pick and top probability on gfx906 (`0` = off) | on | exact (bitwise) |

## Measured (MI50, Qwen3.8-Flash-Next IQ3_S, `terminal/bench/gfx906/ab.py` standard suite, 3 reps)

| | off | on |
| --- | ---: | ---: |
| ms per decode window | 27.99 | 27.72 (−0.9%) |
| verify output-head stage, ms per window | 1.27 | 1.18 (−7.6%) |
| drafter argmax step, ms per graph | 0.079 | 0.031 |
| drafter argmax + probability round, ms | 0.087 | 0.048 |

- Decode tok/s moved by less than the run-to-run noise (about +0.4% net of the control arm).
- At temperature 1 with thinking on, the head stage held at −7.5% and the draft at −5%, but tok/s could not be read at that noise.
- Output hashes were identical with the switch on and off.
