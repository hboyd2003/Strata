# `perf/gfx906-pr5033`

A hand port of the parts of upstream [#5033](https://github.com/Niko1221/Strata/pull/5033) (freapsy, two MI50s) that
reach this box's path, plus one extension of its idea. Stacked on `main`. #5033 reports +24% on another model and two
cards; here the result is about −2.7% ms per window, because production's K-quant GEMVs already run the `MMVQ_IACC`
layout rather than the wave kernel #5033 rewrites.

## Switches

| Switch | What | Production | Output |
| --- | --- | --- | --- |
| `STRATA_MMVQ_IACC_XLDS=q6k` | stages the activation columns in LDS in the iacc layout (own extension; also `1`/`all`, `q4k`, `q5k`) | `q6k` | bitwise iacc |
| `STRATA_MMVQ_XLDS=N` | the same staging in the wave kernel, calls of ≤ N columns | `4` | bitwise |
| `STRATA_MMVQ_Q6X4=N` | Q6_K calls of ≤ N columns in `q6k_wave4_kernel` (one lane does four wave lanes' work), ahead of iacc. #5033's `=1` means every T; here it means T = 1 | `1` | rounding only against iacc |
| `STRATA_GR_UP_OCC=1` | hc `up` per token count: T = 1 at 3 waves a SIMD, T 2–4 chunk-major at the compiler's occupancy, T ≥ 5 unchanged | on | bitwise |
| `STRATA_GFX906_EXP_GLOBAL` (compile time, default 1) | expert blobs read with `global_load`, not `flat_load`; the nibble mask as one `v_perm` | on | bitwise |

`Q6X4` is bitwise the wave kernel, not iacc, so with it T = 1 calls round as the wave kernel does and T ≥ 2 calls as iacc
does. No input loses precision.

## Measured

A/B on one MI50, before the unused modes were removed (same kernels), two arms per variant against two production arms (harness v3, 7 prompts):

| | ms/window | GPU-reach wait | draft ms | decode tok/s, 1st pass |
| --- | ---: | ---: | ---: | ---: |
| production | 26.29 | 18.05 | 3.24 | 85.7 |
| exact switches (all above but `Q6X4`) | 25.77 (−2.0%) | −2.7% | −0.5% | 87.6 (+2.3%) |
| with `Q6X4=1` | 25.59 (−2.7%) | −3.0% | −5.9% | 89.2 (+4.1%) |

Part of the last row's tok/s is draft acceptance on another set of answers (0.68 against 0.66), so ms per window is the
better figure. One repeat of the exact switches matched production's first-pass hashes on all 7 prompts.

Stages with `Q6X4=1`: hc `up` −5..−10%, `out_proj` −6%, `head` −6.5%, `qkv` −4%. The shared expert moved only −1.5%,
far less than the IQ4_XS bench below, so `XLDS` is the weakest piece. The global-load change alone was flat. The hc
switch costs 8 MiB of VRAM (804 to 796 free). 4K prompts read 1–1.5% slower in one A/B and flat in another with the same
switches; unresolved.

Kernel bench (`mmvq_wave_rows_parity --bench`, against production's kernel for that type):

- `q6k_wave4_kernel`: T = 1 −10..−18% (draft head 395 → 328 µs), T ≥ 2 +4..+21%, hence T = 1 only.
- `IACC_XLDS`: Q6_K −1..−12% at T 2–5; Q5_K +4% at T 1–2 and +54% at T = 6, so Q6_K only.
- `XLDS` on IQ4_XS: −5..−26% at T 1–4.

hc `up` on the `GR_RO` read (`hc_ro_parity`, graph, µs): T 1 31.3 → 29.1, T 3 44.9 → 43.9, T 4 51.2 → 49.5, T 5 58.6 → 63.6
(T 5 is why it stops at 4). #5033's form, 3 waves at every T, spills 216–1360 bytes a lane from T = 2 on ROCm 10.1 and runs
2–3× slower, so it was not kept. Neither were a four-lane Q4_K kernel (loses to iacc from T = 2) and 8 wavefronts a block
(worse than 4 nearly everywhere).

## Parity

`mmvq_wave_rows_parity` (10 model shapes, T 1–6, every config bitwise its reference), `fused_gr_bench` (same hashes),
`hc_ro_parity`, `native_grouped_parity`, then A/B first-pass hashes.

## Not ported

DPP shuffles (`perf/gfx906-dpp-shfl` has them), the mode-7 expert kernels and `IQ_MMVQ_*` (production is `EXP_MODE=8` and
its dense weights are not IQ), `GR_DOWN_PF` / `GR_DOWN_TG` (`GR_RO` has its own `down`), the pipeline fixes (no pipelining
here), the sampler's eight loads (not tested).
