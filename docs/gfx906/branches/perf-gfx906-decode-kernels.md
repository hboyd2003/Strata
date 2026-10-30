# `perf/gfx906-decode-kernels`

gfx906 decode kernel changes, each behind its own default-off switch, plus two benches. Stacked on `main`.

## Switches

| Switch | What | Production | Output | Alone |
| --- | --- | --- | --- | --- |
| `STRATA_MMVQ_MUL24=1` | 24-bit scale products in the q4/q5/q6/IQ4_XS wave dots | on | exact | +1.6% |
| `STRATA_POST_COPY_WIDE=1` | plan copy one word a thread (up to 1024), CPU rows one float4 a thread | on | exact | +1.3% |
| `STRATA_DOWN_LPR=2\|4\|8\|16` | lanes per row in the expert down kernel (modes 7/8) | 4 | rounding | +0.5% (ms/window −2%) |
| `STRATA_MMVQ_IACC=1\|all\|q6k,q4k,q5k` | integer-accumulating wave64 MMVQ for Q6_K / Q4_K / Q5_K | on | rounding | +1.1% (noise) |
| `STRATA_MMVQ_Q6K_R4=1` | 4 rows a wave for big Q6_K matrices at 2+ columns | off | exact | neutral |
| `STRATA_NIB_SHIFT=1` | IQ3_S nibble mask without `v_mul_lo` | off | exact | neutral |

The "Alone" column is one A/B each, with noise of about ±1%. With `MMVQ_IACC` on, Q6_K / Q4_K / Q5_K take the IACC
kernel instead of the wave kernel, so in production `MMVQ_MUL24` only reaches IQ4_XS, and `MMVQ_Q6K_R4` and
`MMVQ_WAVE_ROWS` don't touch those three types. IACC's own tuning: `STRATA_MMVQ_IACC_ROWS=1|2|4` (default 4 for
`n_out ≥ 4096`, else 2) and `STRATA_MMVQ_IACC_K=2|4` (default 4; 2 was slower).

Production also turns on two upstream switches measured in the same batch: `STRATA_GR_NORM_FAST=1` (+1.7%, exact) and
`STRATA_VERIFY_QDEDUP=1` (small, exact).

**`MMVQ_IACC`:** a lane takes 8 or 16 consecutive elements of *one* Q8_1 block, so each column costs 2–4 dp4a, one
convert and 1–2 FMAs per run instead of per 4 elements. Per-row scales are folded once (exact in fp32), all loads are
issued before use, and the wave sum uses DPP instead of 6 `ds_bpermute` steps. 4 rows a wave for `n_out ≥ 4096`, else 2.
Against fp64 on real weights its error is 0.82–0.96× the old kernel's.

## Measured

- All six production switches together: 71.3 → 75.9 tok/s (**+6.4%**), ms per window −5.3%, GPU-reach wait −6.6%; qkv
  gemv −10%, head −9%, out_proj −12%, shared quant −12%, VRAM expert hits −10%. Prefill flat. The four exact switches alone
  (`MUL24`, `POST_COPY_WIDE`, `GR_NORM_FAST`, `VERIFY_QDEDUP`): +1.6%, same output; `IACC`, neutral alone, adds about 3 points on top of the other five.
- Parity: `mmvq_wave_rows_parity` bitwise with `MUL24` and `Q6K_R4`; `native_grouped_parity` 0 failures; `mmvq_iacc_parity`
  50/50 within its error bounds.

## Tools

- `stage_bench`: per-kernel graph timings of the verify window's router, shared expert and post stages (synthetic).
- `mmvq_iacc_parity`: the IACC kernel against the wave kernel and fp64, with `--bench`.
