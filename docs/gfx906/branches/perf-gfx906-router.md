# `perf/gfx906-router`

A faster router stage for gfx906: a 64-lane top-10 and a reworked router GEMV. Bitwise. Stacked on `main`.

## Switches

| Switch | What | Production | Output |
| --- | --- | --- | --- |
| `STRATA_GFX906_ROUTER_W64=1` | the 512-expert top-10 as one 64-lane wave per token (8 experts a lane, uint-key DPP max, ballot ties, serial sum order kept) | on | exact |
| `STRATA_GFX906_ROUTER_GEMV=1\|2\|4\|auto` | the 512 × 2560 router GEMV with hoisted loads and a swizzle/DPP butterfly (same operands as the xor tree), plus 64 early-exit padding blocks | auto | exact |
| `STRATA_GFX906_ROUTER_FUSED=1` | GEMV + top-10 + doorbell publish in one launch | off | exact |

`auto` picks rows per block by token count: 1, 2, 2, 4 for T = 1..4.

`ROUTER_GEMV` needs `STRATA_MMVF_ROWS=1`, is skipped under `STRATA_TSUM=1`, and covers windows of 1–4 tokens only (5–8
keep the older kernel). It is tried before `perf/topk-router`'s `GFX906_ROUTER_T1_ROWS` / `_T2_EXACT`, so with
`GEMV=auto` those two never run.

## Measured

- Kernels: top-10 11.1 → 6.1 µs a layer at any T. GEMV T1 11.6 → 5.9 µs, T2 10.2 → 7.9, T3 15.1 → 9.5, T4 16.5 → 11.0.
  The whole router chain a layer: T1 26.2 → 15.3 µs, T4 31.0 → 20.3.
- Decode: W64 alone +2.6%, W64 + GEMV auto **+3.6%** (75.8 → 78.5 tok/s), ms per window 28.67 → 27.66, verify −4.0%.
  Output hashes identical in every arm. The kernel bench under-predicted the gain about threefold.
- `ROUTER_FUSED`: within 1.3 µs of the three separate launches in the kernel bench (not A/B'd end to end); the top-10
  is no longer slow, so fusing saves nothing.
- Parity: `router_w64_parity` (0 of 280 launches differ, including ties, equal logits and underflow), `router_fused_parity`.

Credits: the padding blocks follow VlDubr's 57fa6d85 (the 512-block anomaly); their `route_wave64` is kept as a parity
and bench arm.
