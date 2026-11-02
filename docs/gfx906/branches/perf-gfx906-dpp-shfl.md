# `perf/gfx906-dpp-shfl`

Exact xor shuffles without LDS round trips, in the gfx906 HIP compatibility layer. Stacked on `main`.

## Switches

Both are compile-time defines (`-DCMAKE_HIP_FLAGS=-D...=0` turns one off); there is nothing to set at run time.

| Switch | What | Production | Output |
| --- | --- | --- | --- |
| `STRATA_GFX906_DPP_SHFL` (default 1) | constant-mask xor shuffles go through DPP / `ds_swizzle` instead of `ds_bpermute` | on | bitwise |
| `STRATA_GFX906_MMVF_LB` (default 1) | `__launch_bounds__` on `bf16_f32_mmvf_rows_kernel` | on | bitwise |

On gfx906 a CUDA warp shuffle becomes HIP's `__shfl_xor`: one `ds_bpermute` through the LDS crossbar plus a wait for
it, per butterfly step. Every warp and wave sum in the engine is a dependent chain of five or six of these, and most
decode kernels are small and latency bound, so the chain shows up in nearly every stage.

**DPP_SHFL** (`include/strata/platform/hip_compat/strata_hip.h`) maps `__shfl_xor_sync` and the gfx906 kernels' own
`__shfl_xor` calls. When the lane mask is a compile-time constant below the width (the unrolled butterflies), each step
reads exactly lane ^ m, for any data:

| xor | instruction |
| --- | --- |
| 1, 2 | DPP `quad_perm` [1,0,3,2] / [2,3,0,1] |
| 4 | DPP `row_ror` 4 and `row_ror` 12, one picked per lane by bit 2 |
| 8 | DPP `row_ror` 8 ((i − 8) mod 16 = i ^ 8) |
| 16 | `ds_swizzle` bitmask mode, xor 0x10 within 32 lanes |

xor 32, masks that are not constants, and types other than 32 or 64 bits keep `ds_bpermute`. Since each step returns
the same lane's value as before, every reduction keeps its order and its bits. In the engine binary `ds_bpermute`
goes from 82,244 instructions to 4,545.

**MMVF_LB**: without `__launch_bounds__`, HIP compiles a kernel for 1024-thread blocks, which caps it at 64 VGPRs on
wave64 (CUDA has no such cap). `bf16_f32_mmvf_rows_kernel` is launched with 256 threads; its 8-token instances spilled
5 VGPRs to scratch. With the bound they use 73 VGPRs and no scratch.

## Measured

- Micro-bench, one sum per iteration in a dependent chain, one wave per SIMD: 32-lane sum 233 → 89 ns, 64-lane sum
  275 → 131 ns, bitwise equal on 61,440 random lanes. With 4 waves per SIMD and 4 sums in flight: 1150 → 383 ns.
- Decode (A/B, standard suite, 3 reps, two controls): **87.01 tok/s vs 85.37 (+1.9%)**, ms per window −1.8%, GPU time
  per window 25.52 → 24.94 ms (−2.3%). Almost every stage gains: hc down / up −2 to −6%, GDN recurrence −5%, shared
  expert −5%, QSA attention −3.5%, qkv GEMV −2%. Long 32K read +0.8%, 16K prefill +0.6 to +1.1%.
- Both switches together with an exact `gfx906:sramecc+:xnack-` build: first-pass answers identical to the controls.
- Parity (gfx906): mmvq_wave_rows, mmvq_multi, native_grouped, gr, gr_multi, mmvf_rows, router_w64, router_fused,
  sampler, gdn, shared_expert, elementwise, qsa, qsa_topk: 0 failures.

## Tried with it, not kept

- `MMVF_LB` alone: neutral (the spilling instances only serve 5–8 token windows). Kept because it is free and exact.
- An exact target ID (`gfx906:sramecc+:xnack-` instead of `gfx906`): 2 fewer SGPRs per kernel, neutral, and the image
  would refuse cards that report xnack or sramecc differently. Not used.
- HIP counterparts of CUDA's `--use_fast_math` on the `native_*` files (flush denormals, approximate divide / sqrt /
  transcendentals): neutral and every answer changes. Dropped.
- HIP runtime settings for graph launches (`HIP_FORCE_DEV_KERNARG`, `DEBUG_CLR_AQL_BARRIER_OPT`,
  `ROC_USE_FGS_KERNARG`, `DEBUG_HIP_GRAPH_SEGMENT_SCHEDULING` / `BATCH_SIZE`, `DEBUG_CLR_DIRECT_DOORBELL`,
  `GPU_MAX_HW_QUEUES`, `HSA_ENABLE_INTERRUPT=0`): a graph node costs 1.45 µs with each of them; `ROC_SKIP_KERNEL_ARG_COPY=1`
  and `DEBUG_HIP_GRAPH_CLASSIC_PATH=1` raise it to 1.96 µs. The defaults are the fast path.
