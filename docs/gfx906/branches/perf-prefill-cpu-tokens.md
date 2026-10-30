# `perf/prefill-cpu-tokens`

Lets the prefill CPU share take experts routed by more than 8 tokens. Own branch on upstream; the commit is Baptiste
Rajaut's (`feat/cascade-vnni-027`, 6dacba36).

## What it does

Upstream's prefill CPU share (`STRATA_PREFILL_CPU_SHARE`) stages short prompt chunks after routing and lets the CPU pool
compute the non-resident experts that only a few tokens route to (≤8). On gfx906 the share can't switch itself on (that
code is compiled out), so it never ran here until set explicitly. `STRATA_PREFILL_CPU_TOKENS` raises the token limit,
running larger groups as jobs of 8 (clamped to 1–256).

## Switches

| Switch | Default | Production | Output |
| --- | --- | --- | --- |
| `STRATA_PREFILL_CPU_SHARE=auto` (upstream) | off on gfx906 | auto | rounding (CPU vs GPU) |
| `STRATA_PREFILL_CPU_TOKENS` (this branch) | 8 | 16 | rounding |
| `STRATA_PREFILL_CPU_SHARE_MAX` (upstream) | 3072 | 8192 | rounding |

## Measured

- Share + 16 tokens: ~260-token prompts −28% (1751 → 1263 ms), 2K follow-up turns −7..−8%; 4K/16K prefill and decode
  flat. Prefill power 149 → 173 W. The control (staging without the CPU work) made follow-ups 3.4% *slower*, so the gain
  is the CPU work.
- `SHARE_MAX` 8192 (one sweep, 2 reps): 4K prompt −4.5%, 5K follow-up turn +4.6% tok/s; 6144 gave the same 4K gain;
  4096 neutral; fresh 80K reads unaffected (their chunks are 32K).
- No output hash changed in any test, although the CPU's rounding differs from the GPU's by construction.
