# `serve/qwen-code-tail`

Lets Qwen Code conversations reuse the prompt cache. Stacked on `main`.

## The problem

Qwen Code appends its memory catalog to every request as its own short user turn (a `<system-reminder>`) and drops it
from the history on the next turn. Strata checkpoints at the last `<|im_start|>`, which then sits past the point where
consecutive prompts differ, so every turn re-read the whole prompt (restored sessions too). `--tail-role-token` can't fix
it: matching the user role would also drop real user and tool turns.

## The fix

| Setting | Default | Production |
| --- | --- | --- |
| engine `--tail-reminder-ids A,B,..` | none | set by the server |
| config `"qwen_code_tail": true` | off | on |

The engine leaves a short (≤4096 tokens) turn that opens with those token ids out of the checkpoint; the server option
computes the ids of `<system-reminder>`. Output is unaffected (only what is cached changes).

## Measured

A synthetic two-turn Qwen Code conversation: turn 2 reused 0 tokens and read 9,929 in 14.6 s without it, and reused
9,865 / read 64 in 0.8 s with it. Verified on that synthetic test only, not on restored sessions.
