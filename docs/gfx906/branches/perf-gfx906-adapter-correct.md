# `perf/gfx906-adapter-correct`

Makes the CUDA-to-HIP adapter use HIP's semantics where the plain mapping was stronger or different than needed. Stacked on `main`. There are no switches.

- **One-sided fences.** HIP's `__threadfence` is sequentially consistent, so a fence that only publishes also invalidates the CU's L1. The hc ticket, the argmax and probability counters, the plan's skip flags and the doorbell publish now use a release to publish and an acquire to read. Other backends keep `__threadfence` (`strata/kernels/fence.cuh`).
- **Fork/join events.** They only order GPU streams, so they are recorded with `hipEventReleaseToDevice` instead of a system-scope release. Whether captured graphs honour the flag wasn't checked.
- **Unsigned `__dp4a`.** Uses `v_dot4_u32_u8`.
- **`__any_sync` and `__all_sync`.** They vote over the caller's 32-lane warp, as `__ballot_sync` does. Nothing calls them today.

## Measured

One MI50, standard A/B with 3 reps and a profile pass. This build also had the exact target ID `gfx906:sramecc+:xnack-`, which isn't part of the branch.

| | base | this | base again |
| --- | ---: | ---: | ---: |
| decode tok/s, first pass | 89.60 | 90.00 | 90.54 |
| ms/window | 25.49 | 25.50 | 25.38 |
| GPU total, ms/window | 23.66 | 23.92 | 23.81 |
| long read tok/s | 803.9 | 801.8 | 801.5 |

No measurable speed change. Parity programs showed no failures, and the first six output hashes match a set from a build without the change. A prefill-only re-run without the target ID also matched base (264-token prompts: 1146 ms against 1144 and 1151). The reused-context first token read 367 ms against about 287 in one pass, and that wasn't re-measured.
