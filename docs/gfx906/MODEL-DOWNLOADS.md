# Model downloads: DeepSeek-V4.1-Flash and GLM-5.3-Flash (this box)

Verified against the Hugging Face API on 2026-10-18 (repo IDs, file names, sizes); all repos
carry an MIT card license. Destination layout follows this box's convention:
`/ai/models/hf.co/<org>/<repo>`. Free space at write time: **696 GiB** on `/ai` (2.0 TiB volume).

Run every step inside `tmux` (`hf download` resumes interrupted files; `reassemble.sh`
checksums every part before joining or deleting anything). Do **not** run the downloads in
parallel - the disk math below assumes one at a time.

## Which GLM quant, and why (peasantsmith/Maya vs neuralll/GSQ-RCO)

| | **Maya-S-v2 IQ2_XXS** (`peasantsmith`) | **GSQ-RCO 3.0bit q4kattn** (`neuralll`) |
| --- | --- | --- |
| quantized from | stock `zai-org/GLM-5.3-Flash` | `pfeifferj/GLM-5.3-Flash-GSQ-RCO-GGUF` (RCO uncensored pipeline); neuralll's variant re-stores the non-expert Q8_0 tensors as Q4_K (~8.3 GB of the file); all 126 routed-expert tensors bit-identical to pfeifferj |
| recipe | imatrix: IQ2_XXS gate/up + IQ2_S/IQ3_XXS down + Q6_K dense | GPTQ-style 3.0-bit experts, Q4_K attention |
| size | 96.5 GB, 3 true GGUF shards | 113.6 GB, single file |
| uncensored? | no - needs the abliteration LoRA (`GCSA-AiLab/GLM-5.3-Flash-Abliterated-LoRA-GGUF`, applied with `--lora ... --lora-exps`) | yes, baked (RCO) |
| Xolotl's own quality bench (same harness, `docs/glm/FINDINGS.md` s26) | chat ppl **5.337**, code ppl 3.720 | chat ppl 5.547, code ppl **3.668** |
| Xolotl's headline GLM config (`docs/xolotl/BENCHMARKS.md`) | **yes**: Maya-S-v2 + LoRA + soft prune + route bias, 512K ctx, **25.3 tok/s** decode (RTX 4090 + 90 GB DDR4) | no - the RCO file is the earlier engine-shaping file (`docs/glm/FAST_START.md`: 16-20 t/s, 796 s arena load on their box) |
| kernels it needs | standard IQ2/IQ3/Q6_K types - the kernel families this fork already compiles for gfx906 | GSQ/PTQ expert types + the q4k-attn path (PTQ1_0 CPU rows arrived with the merge) |
| MTP draft file | n/a (project-maya refuses MTP on one GPU; Xolotl did not port MTP to GLM either) | `neuralll/GLM-5.3-Flash-MTP-GGUF` exists (depth-1 acceptance 70-75%) but only splices onto the RCO file |

**Choice for this box: the Maya family** - it is what Xolotl's production bench and configs were
measured with, its quant types are ones this fork already builds for gfx906, and its chat
perplexity is better than RCO's. Maya-L is already downloaded here (see step 1); if the engine
loads its 4-shard split, serve that - the gated fallback is Maya-S-v2, the only Maya the
authors benchmarked. **RCO stays the optional second download** if code quality matters (better
code ppl, no LoRA dependency) or as the fallback for
`neuralll/llama.cpp`. `Maya-M` (116 GB IQ2_S, KL ~23 % better than S in the PEERS survey) is the
quality upgrade, but Xolotl gated only S-v2 - untested territory. `Maya-L` (156 GB, 99.2 % of
FP8) is already on this box - it runs off the file tier + VRAM cache with a smaller RAM arena,
at lower expert residency than the S files on the authors' 90 GB box.

## 1. GLM-5.3-Flash - ALREADY ON DISK (Maya-L), only 2 small downloads left

`Maya-L IQ3_S` (156.3 GB, all 4 shards byte-verified against the HF manifest on 2026-10-18) is
already at `/ai/models/hf.co/peasantsmith/GLM-5.3-Flash-Maya-GGUF/Maya-L/`:

```
GLM-5.3-Flash-Maya-L-IQ3_S-0000{1,2,3,4}-of-00004.gguf   47.7 + 48.0 + 47.3 + 13.3 GB
```

Maya-L is the *best* variant in the family (IQ3_S gate/up + IQ4_XS down + Q5_K sensitive; 99.2 %
of FP8, KL 0.188 per the PEERS survey) - it beats the Maya-S-v2 that Xolotl's own bench runs.
One caveat: Xolotl only gated **S-v2** through their serve harness (their split-aware tier note
says "shard 1 = blk.0-23 only" for the 3-shard layout); whether the engine's split-aware tier
handles L's 4-shard layout is untested here. If L refuses to load, fall back to S-v2 (command
kept below).

What still needs downloading:

```bash
# abliteration LoRA (the engine applies it with --lora + --lora-exps)
hf download GCSA-AiLab/GLM-5.3-Flash-Abliterated-LoRA-GGUF \
  --local-dir /ai/models/hf.co/GCSA-AiLab/GLM-5.3-Flash-Abliterated-LoRA-GGUF

# tokenizer source (small) - tools/glm/serve/export_tokenizer.py <this dir> <out>
hf download zai-org/GLM-5.3-Flash \
  --include "tokenizer*" "chat_template*" "special_tokens*" "generation_config.json" \
  --local-dir /ai/models/hf.co/zai-org/GLM-5.3-Flash

# vision mmproj for the strata-vision path (1.1 GB) - YES, wanted (operator 2026-10-18):
# vision/mmproj-GLM-5.3-Flash-F16.gguf; model-independent, serves whichever Maya quant boots.
# Caveat: GLM's --vision wiring in the Xolotl engine is unverified during the port; DeepSeek-V4.1
# vision is explicitly NOT wired in their engine yet (the tower ships unwired in V41-MTP-Vision).
hf download peasantsmith/GLM-5.3-Flash-Maya-GGUF --include "vision/*" \
  --local-dir /ai/models/hf.co/peasantsmith/GLM-5.3-Flash-Maya-GGUF

df -h /ai   # checkpoint: ~696 GiB free (these three downloads total ~0.5 GB + 1.1 GB vision)
```

Fallback only (if Maya-L does not load): the gated S-v2 set (~96.5 GB):

```bash
hf download peasantsmith/GLM-5.3-Flash-Maya-GGUF \
  --include "Maya-S-v2-IQ2_XXS/*" \
  --local-dir /ai/models/hf.co/peasantsmith/GLM-5.3-Flash-Maya-GGUF
```

The soft-prune list is already in the branch: `tools/glm/serve/prune-ezct-0.25.txt`.

### Optional instead: the RCO route (single file, uncensored baked)

```bash
hf download neuralll/GLM-5.3-Flash-GSQ-RCO-3.0bit-Q4Kattn-GGUF \
  --local-dir /ai/models/hf.co/neuralll/GLM-5.3-Flash-GSQ-RCO-3.0bit-Q4Kattn-GGUF
# MTP draft (4.6 GB) - RCO-file only; Xolotl's GLM engine does not use it yet
hf download neuralll/GLM-5.3-Flash-MTP-GGUF \
  --local-dir /ai/models/hf.co/neuralll/GLM-5.3-Flash-MTP-GGUF
```

## 2. DeepSeek-V4.1-Flash - 163 + 5.7 + 203 GB (start only after step 1)

```bash
hf download IndrasMirror/DeepSeek-V4.1-Flash-GSQ-Uncensored \
  --local-dir /ai/models/hf.co/IndrasMirror/DeepSeek-V4.1-Flash-GSQ-Uncensored

# join the 16 parts (plain BYTE ranges, not gguf-split shards - the GSQ expert types
# PTQ1_0/Q2_0 are unknown to gguf-split, which is why they were split this way).
# --delete-parts keeps the peak at ~163 GB instead of ~326 GB.
bash /ai/models/hf.co/IndrasMirror/DeepSeek-V4.1-Flash-GSQ-Uncensored/reassemble.sh \
  /ai/models/hf.co/IndrasMirror/DeepSeek-V4.1-Flash-GSQ-Uncensored --delete-parts

# Engram shards (203 GB): NOT in the GSQ repo - the two 101.5 GB shards are FP8 rows
# byte-identical to the original checkpoint, taken from the quant's source repo.
hf download dealignai/DeepSeek-V4.1-Flash-UNCENSORED-FP8 \
  model-00047-of-00048.safetensors model-00048-of-00048.safetensors \
  model.safetensors.index.json config.json \
  --local-dir /ai/models/hf.co/dealignai/DeepSeek-V4.1-Flash-UNCENSORED-FP8

df -h /ai   # checkpoint: 1.1 TiB free measured 2026-10-18 after GSQ + GLM bits (volume grew to 2.5 TiB)
```

The same GSQ download also brings `V41-MTP-Vision.Q2_K.gguf` (5.7 GB, the DSpark draft - the
card's recommended variant), `rank.bin` (expert hotness), `V41-GSQ.gguf.experts.json` (expert
offsets), `SHA256SUMS`, and `tokenizer/` (chat template + vocab). The whole-repo download also
brings the 8.4 GB `V41-MTP-Vision.gguf`; use the Q2_K file for DSpark. DeepSeek vision is not
wired in their engine, so nothing extra is downloaded for it.

**Not a download, but the follow-up:** turning the two FP8 shards into run-time Engram tables
is `python3 tools/v41/extract_engram.py <FP8 dir> --to layers.1=<dir-a> --to layers.14=<dir-b>`
(one target per fast drive is ideal; decode reads ~48 random 256-byte rows per token) and it
transiently wants shards + tables ~= +406 GB. On this box both sides of that fit on `/ai` (1.1 TiB
free at the 2026-10-18 checkpoint), and the tables live with the model that runs them:

```bash
python3 tools/v41/extract_engram.py \
  /ai/models/hf.co/dealignai/DeepSeek-V4.1-Flash-UNCENSORED-FP8 \
  --to layers.1=/ai/models/hf.co/IndrasMirror/DeepSeek-V4.1-Flash-GSQ-Uncensored/engram/l1 \
  --to layers.14=/ai/models/hf.co/IndrasMirror/DeepSeek-V4.1-Flash-GSQ-Uncensored/engram/l14
```

**DONE 2026-10-18 21:58.** Both tables landed, 2m49s, exit 0, 2000 random rows per tensor verified
against the FP8 source:
`engram/l1/engram-L1.safetensors` (101,377,628,584 B) and
`engram/l14/engram-L14.safetensors` (101,380,404,328 B), each with its own
`model.safetensors.index.json`. Note the script is on the merge branch, so run it from that tree
(`/ai/engines/gfx906/worktrees/xolotl-merge/tools/v41/extract_engram.py`) - the main checkout's
branch predates the Xolotl merge and has no `tools/v41`.
The `/ai` volume is now at 680 GB free.

The `dealignai` FP8 shards were **deleted 2026-10-18** (operator), so the extracted tables are the
only local copy; they were verified at extraction (all tensor sizes + 2000 random rows per tensor
against the source) but not byte-exhaustively - a byte-exhaustive checker is now
`tools/v41/verify_engram.py` for any future re-extraction. Rollback if ever doubted: re-download
the two shards from `dealignai/DeepSeek-V4.1-Flash-UNCENSORED-FP8` (203 GB) and re-run. `/ai` is
back to 869 GB free.

After the extraction reports ~203 GB of tables, the FP8 shards can be deleted. The engine then runs
with `--engram /ai/models/hf.co/IndrasMirror/DeepSeek-V4.1-Flash-GSQ-Uncensored/engram/l1:/ai/models/hf.co/IndrasMirror/DeepSeek-V4.1-Flash-GSQ-Uncensored/engram/l14`.
Without `--engram` the engine warns `the Engram add is skipped (not the real model)` - Engram is an
architectural n-gram memory added at layers 1 and 14, so a no-Engram boot tests plumbing only.

## Steady-state totals

| model | GB |
| --- | ---: |
| GLM Maya-L (already on disk) + LoRA + tokenizer | 156.3 + ~0.3 |
| DeepSeek GSQ (joined) + DSpark | ~169 |
| Engram FP8 shards (until extracted) | 203 |
| Engram tables after extraction (shards deleted) | ~203 |
| **to download for both models servable** | **~372** (Maya-L was already there) |
