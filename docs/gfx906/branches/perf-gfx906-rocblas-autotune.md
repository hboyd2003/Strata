# `perf/gfx906-rocblas-autotune`

Tunes rocBLAS for the card at server start instead of shipping a table. Stacked on `main`. Prompt path only; the decode kernels aren't touched.

Before the engine loads, the server runs `tune_rocblas --ensure` on the table directory. It tunes when there is no table or the setup has changed, and otherwise does nothing. The startup log says what it found and why it tuned. A failed tune never stops the server.

## Switches

| Switch | What | Default |
| --- | --- | --- |
| `STRATA_ROCBLAS_AUTOTUNE=0` | turns the startup check and tuning off | on |
| `STRATA_ROCBLAS_TUNING=<dir>` | where tables live | `/cache/rocblas` in the image |
| `STRATA_ROCBLAS_TUNING=<file>` | uses that table as it is and never re-tunes it | |

Mount `/cache` on persistent storage, or every new container tunes again.

## What causes a re-tune

Card type, rocBLAS build, HIP runtime, compute unit count, card name, VBIOS, power table (`pp_table`) or the tuner's shape list. The table records these as `# meta` lines. Runtime clock settings outside `pp_table` aren't tracked.

The table being replaced is renamed to `<name>.bak-<UTC time>` once the new one is complete, so a failed run leaves it in place. Nothing deletes the backups.

## Cost and output

The first start takes about 16 minutes longer on an MI50 (120 shapes, 970 s), and the server doesn't serve during that time. An unchanged restart skips tuning. The container can show as unhealthy during a tune, because the compose health check's start period is shorter.

A new table can change rounding, so answers can differ slightly after a re-tune. Relative L2 against the default kernel was about 2e-5 on the shapes read for the earlier table.

## Measured

One MI50, prefill passes only, 3 reps, against the previous hand-made table:

| tok/s | old table | fresh tune 1 | old table again |
| --- | ---: | ---: | ---: |
| ~4K | 727.4 | 726.9 | 727.9 |
| ~16K | 822.7 | 823.5 | 824.9 |

Two full tunes on the same card agreed on 94 of 120 picks and performed the same, so many kernels are near-ties. Running without a table costs 2-7% prefill speed. Decode wasn't re-run with fresh tables.

The tuner also gained the one engine product it had missed, 48 x 2560 with stride 96: 1.14-3.83x faster alone, no measurable change end to end.
