"""Helpers for human-readable benchmark run identifiers."""

from __future__ import annotations

import re
import threading
from datetime import datetime, timezone

_SAFE_COMPONENT_RE = re.compile(r"[^A-Za-z0-9_.-]+")
_TIMESTAMP_LOCK = threading.Lock()
_LAST_TIMESTAMP_BASE: str | None = None
_LAST_TIMESTAMP_COUNT = 0


def safe_run_id_component(value: str, *, default: str = "run") -> str:
    """Return a filesystem/glob-safe representation of a run identifier."""
    component = _SAFE_COMPONENT_RE.sub("_", str(value)).strip("._-")
    return component or default


def new_timestamp_run_id(*suffix_parts: str) -> str:
    """Return a readable timestamp-based run id.

    The timestamp is UTC and path-safe (colons are replaced with hyphens).  A
    small in-process counter is appended only when the clock returns the same
    microsecond twice, which keeps IDs stable-looking while avoiding collisions
    in tight loops.
    """
    global _LAST_TIMESTAMP_BASE, _LAST_TIMESTAMP_COUNT

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S.%fZ")
    with _TIMESTAMP_LOCK:
        if timestamp == _LAST_TIMESTAMP_BASE:
            _LAST_TIMESTAMP_COUNT += 1
        else:
            _LAST_TIMESTAMP_BASE = timestamp
            _LAST_TIMESTAMP_COUNT = 0
        collision_suffix = (
            f"_{_LAST_TIMESTAMP_COUNT:03d}" if _LAST_TIMESTAMP_COUNT else ""
        )

    suffix = "_".join(
        safe_run_id_component(part) for part in suffix_parts if str(part).strip()
    )
    run_id = f"{timestamp}{collision_suffix}"
    if suffix:
        run_id = f"{run_id}_{suffix}"
    return run_id
