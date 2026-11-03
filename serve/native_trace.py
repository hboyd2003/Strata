"""Small, opt-in bridge helpers for correlating native correctness traces."""
from __future__ import annotations

import inspect
import os
import re


REQUEST_ID = re.compile(r"^[0-9a-f]{32}$")
TRACE_ENV = "STRATA_CORRECTNESS_TRACE"


_engine_env = False   # the switch set only in the engine's environment (the run config's "env" block)


def enabled() -> bool:
    return os.environ.get(TRACE_ENV) == "1" or _engine_env


def follow_engine_env(env: dict) -> None:
    """The engine traces when its own environment says so, and then refuses a GEN without a trace ID: send them."""
    global _engine_env
    _engine_env = env.get(TRACE_ENV) == "1"


def field(request_id: str | None) -> str:
    """Return the optional native protocol field, rejecting absent or unsafe IDs."""
    if request_id is None or not REQUEST_ID.fullmatch(request_id):
        raise ValueError("correctness trace requires a 32-character lowercase hex request ID")
    return f"trace_id={request_id}"


def command(kind: str, max_new: int, sampling_keys: str = "", request_id: str | None = None,
            slot: int | None = None, embeddings: str | None = None) -> str:
    """Build a GEN/GENI/BGEN/BGENI head, preserving the legacy spelling when tracing is off."""
    if kind not in ("GEN", "GENI", "BGEN", "BGENI"):
        raise ValueError("unsupported native generation command")
    if kind.startswith("B") and slot is None:
        raise ValueError("batch command requires a slot")
    if kind.endswith("I") and not embeddings:
        raise ValueError("image command requires an embeddings path")
    trace_key = f" {field(request_id)}" if enabled() else ""
    prefix = f"{kind} {slot} {int(max_new)}" if kind.startswith("B") else f"{kind} {int(max_new)}"
    tail = f" {embeddings}" if kind.endswith("I") else ""
    return f"{prefix}{sampling_keys}{trace_key}{tail}"


def accepts_request_id(method) -> bool:
    """Whether an engine endpoint explicitly accepts the opt-in correlation keyword."""
    try:
        params = inspect.signature(method).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(p.name == "request_id" or p.kind is inspect.Parameter.VAR_KEYWORD for p in params)
