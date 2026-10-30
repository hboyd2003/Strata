"""Session files on disk: the head the engine's SAVE names, and with STRATA_SESSION_DELTA=1 its K/V logs beside it
(<name>.<16 hex>.kv, each up to the whole conversation's K/V).  Removing a session means removing both."""
from __future__ import annotations

import os
import re

DELTA_LOG = re.compile(r"(.+)\.[0-9a-f]{16}\.kv")


def logs_of(path: str) -> list[str]:
    """The delta K/V logs of the head at `path` (whether or not the head exists)."""
    folder, name = os.path.split(path)
    try:
        with os.scandir(folder or ".") as it:
            return [e.path for e in it if e.name.startswith(name + ".") and (m := DELTA_LOG.fullmatch(e.name))
                    and m.group(1) == name]
    except OSError:
        return []


def remove(path: str, strict: bool = False) -> None:
    """Remove the head at `path`, then its logs (no head is left pointing at a removed log).  `strict`: a head that
    cannot be removed raises OSError and its logs stay; otherwise every failure is ignored (a throwaway save)."""
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass                                    # a failed save can leave its log without a head
    except OSError:
        if strict:
            raise
    for log in logs_of(path):
        try:
            os.unlink(log)
        except OSError:
            pass
