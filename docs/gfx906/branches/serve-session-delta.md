# `serve/session-delta`

Delta session files: an auto-session save appends only a turn's new K/V instead of rewriting the conversation, and
restores read ahead. Own design. Stacked on `main`.

## Why

Before this branch, an auto-session save (on every conversation switch) rewrote the whole conversation. On this box's
agent traffic those v1 saves were 1.3 GB median and up to 2.9 GB, about 70 GB written an hour, although ~92% of a file
is K/V whose cells never change once written.

## What it does

| Switch | Default | Production | Output |
| --- | --- | --- | --- |
| `STRATA_SESSION_DELTA=1` | off (saves stay v1) | on | exact |

- **Format v2:** `<path>` is a head with the v1 metadata (running state, deepest checkpoint), each K/V layer's part sizes
  and a reference to its log (id, committed length, digest). `<path>.<16 hex>.kv` is an append-only log of
  (layer, part, offset, n) records; a later record of the same bytes wins; writes end on 4 KiB (O_DIRECT where the
  filesystem takes it, else buffered).
- **A save** appends only the bytes past each part's stable prefix, then replaces the head (temp + rename) once the log
  is flushed, so a crash leaves the old head and its committed prefix. A full write (no base, a mismatched log, or a log
  over 1.25× the K/V + 64 MiB) starts a new log and removes the old one.
- **Lineage:** the engine tracks the path the live K/V was last saved to or restored from and how many leading tokens
  are unchanged since, lowered at each request's re-read point and dropped when cells come from elsewhere.
- **Restores** read v1 and v2 through the same passes; the log is validated (no gaps, limits, coverage, digest) before
  any byte is applied. Small log records are merged per part before the device copy (each copy costs a stream sync).
- **Read-ahead** (also for v1): two threads keep four 16 MiB blocks in flight with `pread` while the caller parses,
  hashes and applies, and K/V is handed over in place instead of copied.
- **Server:** a session's `.kv` log counts toward its auto session's size (the Monitor and `auto_sessions_max_gb`) and
  is deleted with it when pruned.
- **Platforms:** POSIX only. On Windows `STRATA_SESSION_DELTA` is ignored (saves stay v1) and restores keep the
  synchronous reader.

## Measured

- Test, 22K tokens: 492–570 MB → 262 MB a save (a 225 MiB running-state head + ~25 MiB of K/V a turn).
- **Live traffic** (114 saves over 5.8 hours on 2026-10-30, conversations 20K–240K tokens): an append is **262 MiB
  median** (p90 489 MiB) whatever the conversation's length. The first save of a conversation the engine has no v2 file
  for is still a full write (696 MiB median, up to 2.7 GiB); 63 of the 114 saves were full writes, mostly first saves
  of new conversations. 74 GiB written in those 5.8 hours (~13 GiB an hour; the v1 figure above was a different day's
  traffic).
- `restore_bench` on a 601 MB session: v1 366–375 → 263–278 ms; a delta file with 20 appends 268–278 ms (sink calls
  1365 → 232).
- In the engine, a delta restore is still somewhat slower than a v1 restore: at 34K tokens v1 334–370 ms, delta
  371–408 ms (both faster than the 473–478 ms before the read-ahead).
- Restored conversations give identical greedy replies to v1 restores. `conversation_file_test`: 326 checks.

**Compatibility:** builds without this branch read v1 files but refuse v2 heads, so conversations saved as v2 are read
again from scratch after a rollback.
