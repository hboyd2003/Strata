# `perf/gfx906-gu16`

A 16-lane expert gate/up kernel, and an exact one-token hc up. Stacked on `main`.

## Switches

| Switch | What | Production | Output |
| --- | --- | --- | --- |
| `STRATA_GU16=1` | mode-8 fused gate/up with 16 lanes a row (`native_gu_fused16_kernel`) | on | rounding |
| `STRATA_GR_UP_EXACT_T=1` | exact-T hc up kernel for one-token reads (VlDubr c3d0ca379c) | on | exact |

**GU16** (gfx906 only) replaces mode 8's fused gate/up, so it needs `STRATA_EXP_MODE=8` (production has it). The 80 dot items of a 2560-wide IQ row go 5 a lane over 16 lanes (was 2.5 over 32), with a 4-step row
reduction and 2 passes of 32 rows. `amdgpu_waves_per_eu(3)` holds it to 81–84 VGPRs (3 waves per SIMD). Most calls are
bitwise equal; a few flip one q8_1 block.

## Measured

- Kernel, real rows, 10 groups × 3 tokens: IQ2_S 76.8 → 66.6 µs, IQ3_XXS 76.2 → 63.7, IQ3_S 71.7 → 70.0, IQ4_XS
  54.1 → 45.5. A whole expert step −9..−12%.
- Decode: GU16 **+2.0%**, ms per window −2.2%, VRAM expert hits −9.7%; acceptance unchanged; prefill flat.
  `GR_UP_EXACT_T` adds +0.1–0.4% (noise level, bitwise). Its T = 2–4 variants needed 142–225 VGPRs, so only T = 1 runs.
- `native_grouped_parity`: 0 failures; `gr_parity` and `fused_gr_bench` bitwise for `GR_UP_EXACT_T`.
