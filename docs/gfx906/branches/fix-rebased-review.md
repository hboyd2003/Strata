# `fix/rebased-review`

Fixes for bugs a code review found in the code main takes from other people's branches: the cassettesgoboom and
janus-reith gfx906 forks and upstream PRs #3251 and #3565. The `rebased/*` copies only ever follow their source, so the
fixes live here instead. Stacked on `main`, merged after the fork's own branches.

None of these touch the MI50's production path except the auto-session ones (production runs `auto_sessions`).

## Auto sessions (PR #3565)

- **Conversation key.** It was the hash of the first user message. Two chats that open with the same message (a
  fixed preamble, a side agent's set prompt) shared one key and overwrote each other's file. When the client sends its
  own conversation id, it now goes into the key: Codex's `session-id` + `thread-id`, opencode's
  `x-opencode-session-id` / `X-Session-Id`, any client's `x-session-id`, or Qwen Code's Responses `prompt_cache_key`.
  Qwen Code's Chat wire sends none by default, so its keys don't change.
- With `system_to_user`, the moved date/time block counted as part of the first message, so the key changed between
  turns. It is left out now.
- The prune after saving the held conversation could delete the incoming conversation's file just before restoring it.
  That file is spared now.
- `/slots/0` refuses `auto-*` names while auto sessions are on, since the prune would delete them.
- A save or restore that kills the engine is still logged, and the engine now restarts before the request runs.
- Tests: `serve/test_auto_sessions.py`.

## Session restore (PR #3251)

- The streaming restore lost the check that each K/V part in the file is exactly the size it lands in. A short part
  left the previous conversation's bytes in the pool, and a long one turned a clean refusal into a FATAL exit. Restores
  check it again before any device write.
- A refusal before the first block set the session continuable (`live_ok`) even when it wasn't. It keeps its earlier
  value now.

## Correctness trace, split serving, tools (cassettesgoboom)

All but the first are only reached with the opt-in `STRATA_CORRECTNESS_TRACE=1`:

- `verify.cpp` used the x86-only `_mm_sfence()`, which broke the aarch64 CUDA build. It uses `strata_store_fence()` like
  every other site.
- With split serving, `SplitEngine.generate` takes `request_id` and passes it on, and the prefill-only GEN carries the
  trace ID.
- With `--batch`, the next leg of the same request is no longer refused as a duplicate while the slot it just stopped
  drains.
- If the switch was set only in the run config's `env` block, every request failed with "missing correctness trace
  request ID". The server now follows the engine's environment.
- `tools/gfx906.py bench-validate` reports malformed records instead of crashing. `ActiveRequests`, which nothing used,
  moved into its test.

## Prompt path (janus-reith)

A dangling `else` made the plain group gather run after the nibble gather too. This only affects builds with the 4-bit
nibble experts, not the gfx906 build.
