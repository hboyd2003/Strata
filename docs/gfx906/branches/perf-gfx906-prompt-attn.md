# `perf/gfx906-prompt-attn`

A prompt attention kernel for gfx906, from [pfandl/strata-gfx906](https://github.com/pfandl/strata-gfx906) (Jasmin Fazlic, two MI50s), without their results directory. Stacked on `main`.

`qsa_prompt_attn_batch()` has no kernel for this card, so the prompt path loops the decode kernels over 32 queries at a time: each query splits its cells across blocks, writes partials to scratch, and a merge kernel reads them back. `qsa_prompt_attn_gfx906.cu` keeps the per-cell arithmetic and changes the traversal: one block per (query, KV head) walks all the query's selected cells with a running softmax, with no scratch, merge kernel or query batches.

## Switches

| Switch | What | Production | Output |
| --- | --- | --- | --- |
| `STRATA_PROMPT_ATTN_GFX906=1` | the kernel above. int8 KV with 24 heads, 2 KV heads and 256 dimensions only; anything else takes the old loop | on | rounding only |

The softmax is taken in a different order, so the bits differ from the decode kernel's: `max_abs` 5.96e-06 at a 32K context, 5.72e-06 at 131K. No input loses precision.

## Measured

A/B on one MI50 with production's other switches on, two arms per variant (harness v3):

| tok/s | production | with the kernel | change |
| --- | ---: | ---: | ---: |
| 16K prompt | 785.0, 783.4 | 800.5, 800.9 | +2.1% |
| 4K prompt | 673.6, 669.4 | 682.3, 687.0 | +1.9% |
| 32K read | 738.6, 738.7 | 751.6, 751.0 | +1.7% |
| ~80K read | 772.1, 771.9 | 784.8, 784.1 | +1.6% |
| follow-up turns after 80K (1K, 5K, 7K) | 458.5, 642.9, 669.0 (first arm) | 461.7, 650.1, 677.3 (first arm) | +0.7..+1.2% |
| ~200-token prompt | 230.0, 232.4 | 232.1, 230.6 | flat |
| decode | 90.56, 90.40 | 90.31, 90.25 | flat |

Draft acceptance is unchanged (0.68, 0.68 against 0.68, 0.69). Decode ms per window is flat. The first prompt's answer hash stays the same and the other six change.

Component probe (2048 queries, `bench/gfx906_qsa_prompt_attn_probe`): 35.04 to 31.39 ms at 32K (1.116x), 35.56 to 32.10 ms at 131K (1.108x). With `STRATA_ATTN_MERGE_V2` off it reads 1.146x and 1.136x, so production's faster merge already takes a little of the gain. pfandl's own record measures the attention phase at −10..−13%; the harness here has no prefill phase table.

The probe alone isn't enough: pfandl's first version passed it and failed in the engine (top-1 80%), because it read the selection width per call rather than per query. Compare engine output when changing this kernel.
