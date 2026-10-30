# `perf/gfx906-adapt-copy`

Adaptive expert swaps copied by a kernel instead of SDMA, on gfx906. Stacked on `main`.

## Why

The adaptive expert cache swaps ~15 experts (~30 MB) into VRAM per decode window. On this MI50 the SDMA engine caps
host-to-device copies at 12.4–13.5 GB/s, while a kernel reading mapped host memory does 25–27 GB/s (the PCIe 4.0 x16 link
is healthy). Upstream's kernel copy existed, but only under `STRATA_USE_HIP`, which the gfx906 build doesn't define, so
it was compiled out here.

## Switches

| Switch | What | Default | Production | Output |
| --- | --- | --- | --- | --- |
| `STRATA_HIP_ADAPT_KERNEL_COPY=1` | swap copies as a mapped-read kernel; now built on gfx906 | off | on | exact |
| `STRATA_ADAPT_ASYNC=0` (upstream) | pins the adaptive tier to blocking | armed here | 0 | exact |
| `STRATA_MTP_HIP_PRIO=1`, `STRATA_COPY_HIP_PRIO=1` | priority streams for the drafter / copies; now built on gfx906 | off | off | exact |

`ADAPT_ASYNC=0` is a precaution: with `STRATA_USE_HIP` undefined, the self-deciding async tier is armed here. It never
fired on this box, but forced on it measured −3.4%.

The branch also counts kernel copies and their fallbacks by reason (size, source or destination alignment, unmapped
memory): the first of each is printed, totals every 100,000 copies. Grep the engine log for "adaptive kernel copy".

## Measured

- Kernel copy: **+1.5%** in two separate A/Bs, output hashes identical, 0 fallbacks in 62,000 copies; draft + adapt
  wait 3.92 → 3.45–3.61 ms per window.
- Priority streams: neutral (−2.1..+0.2%). Queue priority does not arbitrate compute units on gfx906; it only moves the
  wait between draft and adapt.
