# `perf/gfx906-steddyman`

gfx906 dense GEMV kernels and an expert down kernel, hand-ported from [steddyman/Strata](https://github.com/steddyman/Strata) (Stephen Eddy, MI50), plus their `mmvq_bench` tool. Stacked on `main`. Both switches default to off, and with them off nothing changes.

## What the branch changes

- **Dense GEMVs (`STRATA_G906_DENSE`).** Q6_K, Q4_K, Q5_K and IQ4_XS matrix-vector kernels for one to four tokens. Each row is split across a group of 8 to 64 lanes, chosen from the row's width, and the groups are summed with DPP. They replace the wave-per-row kernels only where they measured faster, picked by shape alone:
  - rows of at most 4,096 inputs (`STRATA_G906_MAX_IN`), so the 2,560-input projections and the output head. The 6,144-input matrices stay on the wave kernels.
  - Q6_K and Q4_K also need at least 1,024 rows (`STRATA_G906_MIN_OUT`). The 640-row shared-expert gate/up was slower on the new kernels (8 → 13 µs).
- **Expert down (`STRATA_G906_DOWN=1`).** The routed experts' IQ4_NL down rows, with whole blocks per lane, on the mode 7/8 path ahead of `STRATA_DOWN_LPR`. `STRATA_G906_DOWN=all` also takes Q2_0, which benched 18-19% slower, so it is not used.
- Not ported: their mode 9 gate/up (slower than this fork's 16-lane gate/up), dense IQ4_NL / Q2_0 / Q3_K / IQ3_S (not these weights, or slower) and an IQ4_XS draft head.

## Switches

| Switch | What | Production | Output |
| --- | --- | --- | --- |
| `STRATA_G906_DENSE=1` | the dense GEMVs above (`all`, or a list of `q6k,q4k,q5k,iq4xs`) | on | rounding only |
| `STRATA_G906_DOWN=1` | the IQ4_NL expert down rows (`all` adds Q2_0) | on | rounding only |
| `STRATA_G906_MAX_IN`, `STRATA_G906_MIN_OUT` | the shape limits above (4096, 1024) | default | |
| `STRATA_G906_LG`, `STRATA_G906_LPR`, `STRATA_G906_LG_RPB`, `STRATA_G906_RPB`, `STRATA_G906_Q4K_RPB` | launch overrides for the bench | unset | |

## Measured (MI50, Qwen3.8-Flash-Next IQ3_S, `bench/gfx906/ab.py` standard suite, 3 reps)

| | off (two runs) | both on |
| --- | ---: | ---: |
| ms per decode window | 25.05 / 25.09 | 24.31 (−3.0%) |
| decode tok/s | 90.92 / 91.06 | 94.26 (+3.6%) |
| ms per window at ~32K | 26.69 / 26.50 | 25.46 (−4%) |
| board power, decode | 177.6 / 178.1 W | 170.9 W (−4%) |

- `STRATA_G906_DENSE` alone: −2.7% per window. `STRATA_G906_DOWN` alone: −0.6%.
- Per stage: the output head 1.16 → 0.98 ms, the q/k/v GEMVs 2.90 → 2.62 ms.
- Draft acceptance (0.68), prefill and time to first token are unchanged.
- Parity: on the model's shapes, the largest relative difference from the old kernels is 1.3-1.9e-7. Each column matches across token counts, and all outputs are finite. Output changes by rounding only.
