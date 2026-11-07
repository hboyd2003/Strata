# `perf/gfx906-pf-rocblas-h16`

Two prompt-path changes for gfx906. Stacked on `main`. Both affect prefill only; the decode kernels aren't touched.

- **rocBLAS solution table.** `Gemm::f16_inplace` can run a tuned rocBLAS kernel per shape, but the reader was compiled only for the HIP backend. This widens the CMake gate (from logicalor, xxDoman's E5b) and adds table support, with token buckets up to 32768. This branch first shipped a table made on an MI50; [`perf/gfx906-rocblas-autotune`](perf-gfx906-rocblas-autotune.md) replaced it with one tuned on your card at server start. rocBLAS's own pick is badly off at T=32768: 170.6 ms for N=10240 against 45.2 ms at T=16384, and 90.9 ms tuned.
- **hc skip-widen** (from andrewprock). The up projection's FP16 rows stay packed and `gr_mix_r_h16` reads them directly, so the widen pass goes.

## Switches

| Switch | What | Production | Output |
| --- | --- | --- | --- |
| `STRATA_ROCBLAS_TUNING=<file or dir>` | the table; a directory resolves to `<arch>-rocblas-<version>.txt` | set by `perf/gfx906-rocblas-autotune` | rounding only |
| `STRATA_HC_MIX_H16=1` | hc skip-widen | on | exact |

A table is valid for one card type and one rocBLAS build. The server now re-tunes it when those change.

Tuner comparison with the default kernel: relative L2 about 2e-5 on the shapes read. `STRATA_HC_UPMIX=1 STRATA_HC_UPMIX_CHECK=4` logged a max difference of 0 on four reads with skip-widen on.

## Measured

One MI50, prefill passes only, against the same binary with both off:

| tok/s | rocBLAS | skip-widen | both |
| --- | ---: | ---: | ---: |
| ~200 tokens | +8.1% | -0.2% | +7.9% |
| 4K prompt | +4.8% | -0.3% | +6.3% |
| 16K prompt | +1.6% | +0.9% | +3.2% |
| 33K prompt | +5.7% | +0.6% | +7.4% |

rocBLAS alone also lowers prefill power about 13% and takes 32 MiB more VRAM.
