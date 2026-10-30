# `serve/auto-sessions-monitor`

An Auto sessions card in the Monitor. Own branch, on top of `rebased/pr-3565` (auto sessions).

`/metrics` gains `"auto_sessions"`: the limits, the conversation the engine holds (its tokens, whether it will be saved
on the next switch), the `auto-*.bin` files on disk with their total size, and the switches, saves, restores, prunes and
failures since the start. The Monitor shows it under the Conversation cache card: files against `auto_sessions_keep`,
their size against `auto_sessions_max_gb`, and the counts below.
