"""The engine-fault ledger: every time the engine dies on its own, this records the death, the
restart that followed, and the fault rate - and says out loud when that rate has grown past the
point where a row-recovery engine would have been cheaper than restart-everything.

Why (issue #24 / P1.3): a batch window that hits an error after a CUDA/HIP submission cannot be
repaired in place - "any error after a CUDA/HIP submission is sticky uncertain" (docs/BATCHING.md,
"Failed windows"). The engine therefore exits and the server restarts it and re-admits every slot
from its checkpoint: a whole-engine blast radius, paid by every stream, not just the bad row. That
is the documented choice (arm A) because the alternative is not free: resuming the surviving slots
without a restart means snapshotting every slot's recurrent state before every window, which costs
about `SNAPSHOT_TAX` of decode, forever, healthy or not.

Below the crossover rate, restart-everything is cheaper; above it, recovery is. Both sides come
from measurements on the box the claim is made about - see
`benchmarks/gfx906/receipts/2026-10-21-p1.2-prefill-rows-nogo.json` for the window cost, the M5
figure for the state size, and `2026-10-21-p1.4-batch-mtp-acceptance.json` for a warm restart. The
ledger is what turns "infrequent" from an assumption into a counted number.
"""

import json
import os
import time

# --- the crossover, with where each number comes from -------------------------------------------
# The engine's recurrent state has to be restored to survive a failed window. On a GDN model a
# decode state snapshot is 118 MB per slot (docs/CONCURRENCY-RADIX.md, "Checkpoints, not pages").
STATE_BYTES_PER_SLOT = 118 << 20
# Restoring it costs a read and a write, and one MI50 moves about 0.7 TB/s of HBM in practice.
# 8 slots therefore pay ~2 x 118 MiB x 8 / 0.7 TB/s ~ 2.7 ms per window.
HBM_BYTES_PER_S = 0.7e12
# A full 8-row window measured 115.28 ms on this box (P0.5), so the snapshot tax is ~2.45 %.
WINDOW_MS_B8 = 115.28
SNAPSHOT_TAX = 2 * STATE_BYTES_PER_SLOT * 8 / HBM_BYTES_PER_S / (WINDOW_MS_B8 / 1000.0)

# Restart-everything costs what an event costs: the warm boot to READY measured ~30-35 s (the P1.4
# arms each did engine boot + 4 test runs inside 45-65 s; a cold, page-cache-flushed boot took
# 1-3 min), plus re-admitting every slot that was streaming (1.1-1.8 s for 2-4 slots, P0.2).
RESTART_S = 37.0

# One event costs RESTART_S of every stream's time; the tax costs SNAPSHOT_TAX of every window's.
# Setting the two hourly costs equal, the stream count cancels: crossover = 3600 * tax / restart.
CROSSOVER_FAULTS_PER_H = SNAPSHOT_TAX * 3600.0 / RESTART_S

REMINDER = ("if faults this frequent keep up, a row-recovery engine (resume the surviving slots from a "
            "per-window state snapshot) would pay for itself: at {rate:.1f} faults/h the restart cost is "
            "{restart_pct:.1f}% of aggregate throughput, above the {tax:.1f}% that snapshotting costs "
            "all the time. The crossover is {cross:.1f} faults/h. See docs/BATCHING.md, 'Failed windows'.")

# A restart is worth reporting even when the engine answers again at once: it is a whole-engine
# stall, and the operator should see it in the server log, not only in the ledger file.
REPORT = ("[strata] the engine died and was restarted: {reason} | {streams} stream(s) were live, "
          "back after {restart:.1f} s | {n} death(s) in the last hour ({rate:.1f}/h), "
          "{day} in the last 24 h | ledger: {path}")


def ledger_path(log_path: str | None, override: str | None = None) -> str:
    """Where the ledger lives: next to the engine log (the server's own log dir), unless configured."""
    if override:
        return override
    if log_path:
        return os.path.join(os.path.dirname(os.path.abspath(log_path)), "engine-faults.jsonl")
    return os.path.join(os.path.expanduser("~"), ".strata", "engine-faults.jsonl")


def note_death(path: str, *, reason: str = "", streams: int = 0, restart_s: float = 0.0,
               now: float | None = None) -> dict:
    """Append one death and its restart. Returns the line as a dict, plus the counted rate so the
    caller can report it. Never raises: a ledger that cannot be written must not take the server
    down on top of the engine it is recording."""
    now = time.time() if now is None else now
    row = {"t": round(now, 1), "reason": reason[:400], "streams": streams,
           "restart_s": round(restart_s, 1)}
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
    except OSError:
        pass
    row["deaths_last_hour"] = count_in(path, 3600.0, now)
    row["deaths_last_day"] = count_in(path, 86400.0, now)
    return row


def count_in(path: str, window_s: float, now: float | None = None) -> int:
    """Deaths recorded in the last window_s. A missing or unreadable ledger counts zero - the
    ledger reports faults, it never becomes one."""
    now = time.time() if now is None else now
    try:
        with open(path, encoding="utf-8") as f:
            lines = f.readlines()
    except OSError:
        return 0
    n = 0
    for line in lines:
        try:
            t = json.loads(line).get("t")
        except (ValueError, AttributeError):
            continue
        if isinstance(t, (int, float)) and now - t <= window_s:
            n += 1
    return n


def restart_cost_pct(deaths_per_h: float) -> float:
    """What restart-everything costs as a share of aggregate decode at this fault rate. Every
    stream loses RESTART_S per death, and every stream runs 3600 s per hour, so streams cancel."""
    return 100.0 * deaths_per_h * RESTART_S / 3600.0


def reminder(deaths_per_h: float) -> str:
    """The B-full reminder, or "" while the rate is below the crossover. The ledger never restarts
    or repairs anything by itself; past the crossover it says the quiet part out loud."""
    if deaths_per_h < CROSSOVER_FAULTS_PER_H:
        return ""
    return REMINDER.format(rate=deaths_per_h, restart_pct=restart_cost_pct(deaths_per_h),
                           tax=100.0 * SNAPSHOT_TAX, cross=CROSSOVER_FAULTS_PER_H)


def report(row: dict, path: str) -> str:
    """The server-log line for one death: what died, how long the stall was, the counted rate,
    and the reminder if the rate has crossed the crossover. The rate is the count itself: the
    window it was counted over is one hour."""
    n = row.get("deaths_last_hour", 0)
    out = REPORT.format(reason=row.get("reason") or "reason not in the engine log",
                        streams=row.get("streams", 0), restart=row.get("restart_s", 0.0),
                        n=n, rate=float(n),
                        day=row.get("deaths_last_day", 0), path=path)
    r = reminder(row.get("deaths_last_hour", 0))
    return out + ("\n[strata] " + r if r else "")
