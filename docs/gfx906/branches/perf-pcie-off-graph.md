# `perf/pcie-off-graph`

`--pcie-frac 0` drops the PCIe branch from the verify graph. Own branch on upstream; the idea is
ramanujbhattacharjee93/Strata 702af1f0 (`STRATA_FZ_NOPCIE`), tied to the existing flag instead of a new switch.

## What it does

With an explicit `--pcie-frac 0` on one GPU (no layer split, peer or remote cache) the session never stages a missed
expert over PCIe: the planner's share is off and a request's own `pcie_frac` is ignored. Each layer's verify graph is
then captured without the PCIe part: no wait on flag B and no empty grouped launch per layer and group. The auto share
and every multi-GPU path keep the branch, as does `STRATA_VERIFY_DEVICE_PLAN=1`.

## Measured

| Setting | Production | Output |
| --- | --- | --- |
| `--pcie-frac 0` | yes | rounding: a few missed experts run on the CPU instead of over PCIe |

Against the default auto share: decode +3.3% (68.0 → 70.3 tok/s, control +0.4%): GPU-reach wait −4.5%, GPU stages −4.2%, CPU experts +13%. Prefill flat.
`--pcie-frac 0` without this branch gives the same output but only +1.2%.
