"""Before the engine loads: make sure the rocBLAS tuning table fits this setup, tuning it when it doesn't.

tune_rocblas --ensure does the comparing, tuning and backing up (tools/hip/tune_rocblas.cpp).  This needs the GPU to itself, so it
runs before anything else uses it.  STRATA_ROCBLAS_AUTOTUNE=0 turns it off.  A failure here never stops the server: the engine then
uses the old table, or rocBLAS's default kernels, and says so in its log.
"""
from __future__ import annotations

import shutil
import subprocess
import time
from pathlib import Path

TIMEOUT_S = 3600


def ensure_table(env: dict, backend: str | None, timeout_s: float = TIMEOUT_S, popen=subprocess.Popen) -> str:
    """Returns what it did: disabled, no-tool, no-dir, pinned, ran, timeout or failed."""
    tool = shutil.which("tune_rocblas", path=env.get("PATH"))
    if backend != "hip" or tool is None:
        return "no-tool"
    if str(env.get("STRATA_ROCBLAS_AUTOTUNE", "1")) == "0":
        print("[strata] rocBLAS tuning: off (STRATA_ROCBLAS_AUTOTUNE=0)", flush=True)
        return "disabled"
    where = env.get("STRATA_ROCBLAS_TUNING")
    if not where:
        return "no-dir"
    if Path(where).is_file():
        print(f"[strata] rocBLAS tuning: using the table {where} as set; it isn't re-tuned", flush=True)
        return "pinned"
    print(f"[strata] rocBLAS tuning: checking the table in {where} (STRATA_ROCBLAS_AUTOTUNE=0 turns this off) ...", flush=True)
    try:
        proc = popen([tool, "--ensure", where], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                errors="replace")
    except OSError as e:
        print(f"[strata] rocBLAS tuning: could not start the tuner: {e}", flush=True)
        return "failed"
    end = time.monotonic() + timeout_s
    timed_out = False
    assert proc.stdout is not None
    for line in proc.stdout:
        if line.startswith("tune: "):
            print("[strata] rocBLAS " + line.rstrip(), flush=True)
        if time.monotonic() > end:
            timed_out = True
            proc.kill()
            break
    code = proc.wait()
    if timed_out:
        print(f"[strata] rocBLAS tuning: stopped after {timeout_s:.0f} s; the engine keeps the table it finds", flush=True)
        return "timeout"
    if code != 0:
        print(f"[strata] rocBLAS tuning: the tuner failed (exit {code}); the engine keeps the table it finds", flush=True)
        return "failed"
    return "ran"
