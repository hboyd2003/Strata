# `perf/gfx906-kv-grow`

Makes `--kv-grow` (the elastic K/V, `STRATA_KV_GROW=1`) work on gfx906. Stacked on `main`.

With the elastic K/V, the K/V cache maps only the cells the current conversation needs. The expert cache keeps the rest of the VRAM. As a conversation grows, the expert cache gives up slots in 8,192-cell steps. A shorter conversation gives them back. Before giving up a slot, the engine moves a hotter expert into a colder expert's slot, so the smaller cache keeps the experts that matter most. All of this is upstream code. On gfx906 it was compiled out: the build used the virtual-memory stub, so the flag only logged "off".

## What the branch changes

- **Takes the HIP virtual-memory path on gfx906.** The MI50 supports it: 4 KiB granularity, used in 2 MiB chunks. Reads through these mappings are as fast as through `hipMalloc` (1.000-1.010x on random reads).
- **Flushes stale GPU translations after every remap.** On this box, after `hipMemUnmap` and `hipMemMap` at a reused address, the GPU keeps reading the chunk that was there before. Two addresses can end up on one chunk, and the data is silently wrong with no fault. AMD root-caused this in the amdgpu kernel driver ([ROCm/TheRock#8744](https://github.com/ROCm/TheRock/issues/8744)): with all VRAM visible through the BAR, the page tables are updated by the CPU and the TLB is never invalidated. A fix is under review upstream. Until it ships, the branch allocates and frees 64 MiB after each mapped run, which makes the driver flush. A 4 KiB buffer doesn't trigger the flush. Each flush costs about 2.4 ms and only runs while the K/V is resizing. A soak of the K/V's moves showed thousands of wrong chunks without the flush and none in 600 steps with it.

## Measured (MI50, Qwen3.8-Flash-Next IQ3_S, int8 K/V, 262,144 max context)

| | without | with `--kv-grow` |
| --- | ---: | ---: |
| expert slots, short context | 11,501 | 13,130 |
| decode ms per window, short prompts | 25.36 | 24.95 |
| fewest slots holding experts, conversation at 232K tokens | 11,501 | 11,639 |
| ms per window, 137K-232K context | 27.05 | 26.65 |

- Short decode is 1.8% faster per window, and long-context decode is unchanged. At the 262,144 maximum the expert cache ends up about where the fixed reservation had it.
- Integrity: codewords at the start, middle and end of each conversation were checked.
  - Six conversations of 18K-96K tokens, switched one at a time, all at once, and after a restart: 0 of 42 wrong.
  - One conversation grown from 110K to 232K tokens, with a short one asked in between: the same 1 of 12 wrong as without the flag (a model mix-up at 232K that returned another real codeword), and 0 of 4 wrong after a restart from disk.
  - No errors.
- Output: rounding only. Experts sit on different slots, so a few CPU-vs-GPU rounding differences change some answers.

`STRATA_KV_GROW_HOLD=1` keeps the K/V as large as the longest conversation parked in RAM. Because a conversation of about 232K tokens is too big for the RAM conversation cache, it is parked on disk, and the hold doesn't apply to it. The box doesn't set it.
