# `fix/session-restore-real-peak`

Session restores ask for the RAM the file really needs. Stacked on `main` (on top of PR #3251's restore code).

## The problem

The streaming restore's RAM check asked for the blocks plus the metadata the engine's *limits* could admit (502–856 MiB
here), although a real file's metadata is ~226 MiB (running state and one checkpoint, almost constant in tokens). On a
box with ~2.8 GB free beside a 48 GB expert arena, most restores were refused. The apply pass also re-parsed the
whole metadata with no admission check (a real peak of ~485 MiB).

## The fix

Ask for the blocks first, then for each metadata array as the parse reaches it, priced from its declared size before it
is allocated. The apply pass hashes and discards the metadata (`In::discard`), so a restore holds it once. A restore now
asks for ~258 MiB instead of 502–856. Tests: `conversation_file_test`.

Production also lowers the RAM floor (`--conversation-cache-min-free-mib 1536`, default 2560), which restores need too.
