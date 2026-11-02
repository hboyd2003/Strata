# `gfx906/ab`

The benchmark harness every change in this fork was measured with: `bench/gfx906/ab.py`. Own branch on upstream.

It talks to a running Strata server over its API (and, optionally, reads the engine log), so it works with any build
and config. It never restarts or reconfigures the server: to compare two configs, run it once on each and compare.

## The standard suite

```bash
python3 bench/gfx906/ab.py run results/mine
```

That is the **standard** suite, the one to share (its `results.json` says `"suite": "standard"`). Every request is
greedy with thinking off, so a build gives the same answers every time.

| Pass | Prompts | What it measures |
| --- | --- | --- |
| decode (× 2 reps) | 2 code (write an LRU cache with tests; find and fix the bugs in a function), 2 writing (an essay, an explanation), 3 specific-knowledge questions (the Haber-Bosch process, the War of the Spanish Succession, the Chandrasekhar limit); 350–800 tokens out | decode tok/s overall and per kind, draft acceptance, the answers' hashes |
| long context (× 2 reps) | a ~32K-token document (the first 70K characters of the engine's reference manual, then 40K of one of its source files) and a question that needs both; 350 tokens out | rep 1: reading 32K tokens; both: decode at ~32K context; rep 2: reusing the document from the conversation cache |
| prefill | three ~200-token prompts, a ~4K and a ~16K-token prompt, each with a nonce so nothing is reused | prompt tok/s at three sizes |
| follow-up turn | the 16K exchange plus ~1.8K new tokens | reading a new turn over a cached conversation |

It takes about 3.5 minutes on the MI50. The long prompts are built from the engine's docs and source at a fixed commit
(upstream v0.2.8), so every machine sends the same tokens; this needs a clone with the fork's history. The long-context
prompt carries a per-run nonce (so rep 1 really reads it, even if an earlier run left it in a disk session), which
means its answer can't match across runs; its hash is reported on its own.

**The MI50's standard result** (2026-10-30, production build and config, after a restart): decode 82.0 tok/s (code
93, writing 77, knowledge 78), draft acceptance 0.67, 27.9 ms per window; long context: 32,284 tokens read at
700 tok/s, decode at that depth 69.8 tok/s, rep 2 reused 32,277 tokens; prefill 221 / 633 / 741 tok/s at ~200 / 4K /
16K tokens, follow-up turn 537 tok/s.

**Before a run:** restart the server, and keep everything else off it. The adaptive expert cache carries over what the
server served before, which changes the speed and, by rounding, the answers: answer hashes only match between runs that
start from a fresh start.

## Options

| Option | |
| --- | --- |
| `--url URL`, `--model NAME`, `--api-key KEY` | the server (default `http://127.0.0.1:8080`, model `strata`; or `STRATA_API_KEY`) |
| `--log SRC` | the engine log, for the per-window split: a file path or `docker:CONTAINER:PATH` (in the gfx906 image: `docker:strata:/tmp/strata-<pack>.log`) |
| `--card N` | the AMD card whose sensors to sample (default: the one with the most VRAM) |
| `--label TEXT` | the run's name in comparisons |
| `--reps N`, `--pf-reps N`, `--no-prefill`, `--no-long` | decode reps (2), prefill reps (1), skip prefill, skip the long-context prompt |
| `--pf-long N[,N]` | extra fresh prefill prompts of N characters (350000 ≈ 105K tokens) |
| `--long N` | one N-character prompt of source text instead of the decode prompts |
| `--live` | the long-context agent shape: a fresh ~80K-token prompt, then turns of ~1.2K / 5K / 7K new tokens that each generate 400 (~2.5 min a rep; `--live-reps`, `--live-ctx`, `--live-turns`, `--live-interleave`) |
| `--source DIR` | build the prompts from DIR instead (not comparable with anyone else's) |

Any pass option makes the run "custom".

**What the engine log adds.** Without `--log`, a run records what the API returns: tok/s, token counts, draft
acceptance, hashes, plus the card's sensors and the host CPU. With `--log` and `STRATA_DECODE_TIMING=1` in the
server's environment, every request also gets the engine's own numbers: the ms-per-window split (verify, GPU-reach
wait, per-layer host time, draft, adapt wait), tokens per window, CPU experts / VRAM hits / PCIe per layer-window,
the expert cache hit rate, and session save / restore times. With `STRATA_VERIFY_PROFILE=1` as well, it records the
per-GPU-stage times of the GDN and QSA layers. The profiler slows decode, so don't compare that run's tok/s.

## Comparing

```bash
python3 bench/gfx906/ab.py compare results/base results/new results/base-again -o compare.md
```

One table of every metric with % deltas against the first run, and whether each run's first-pass answers match the
first's. It warns when the runs used different prompts or options.

How this fork reads an A/B: the base runs twice (first and last), and the spread between the two is the noise. On the
MI50 that is about ±1% decode tok/s, so a smaller difference is "neutral". A change is called **exact** when its first-pass answer hashes match the base's, and
**rounding only** when they differ only because the order of floating-point sums changed. Some answers flip now and
then even between two runs of the same build (the adaptive expert cache), which is why the base runs twice: a hash that
differs between the two base runs says nothing about the change.

## Outputs

- `results.json`: the system (CPU, RAM, GPU and its PCIe link), the server's engine block from `/metrics`, the
  options, every request's raw and parsed numbers with its engine log lines, and the per-pass means.
- `summary.md`: the key metrics of the run.
