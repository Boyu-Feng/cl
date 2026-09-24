"""Small logging setup helpers for CLI and benchmark runs."""

from __future__ import annotations

import contextlib
import contextvars
import json
import logging
import sys
import traceback
from collections.abc import Iterator, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .run_ids import safe_run_id_component

_LOG_CONTEXT: contextvars.ContextVar[dict[str, Any]] = contextvars.ContextVar(
    "clbench_log_context",
    default={},
)

_CAP_SUFFIX = "...[truncated]"
_MAX_FIELD_CHARS = 1000
RUN_LOG_LEVEL = 25

logging.addLevelName(RUN_LOG_LEVEL, "RUN")

_STANDARD_RECORD_KEYS = frozenset(
    set(logging.makeLogRecord({}).__dict__)
    | {
        "args",
        "asctime",
        "created",
        "exc_info",
        "exc_text",
        "filename",
        "funcName",
        "levelname",
        "levelno",
        "lineno",
        "module",
        "msecs",
        "message",
        "msg",
        "name",
        "pathname",
        "process",
        "processName",
        "relativeCreated",
        "stack_info",
        "thread",
        "threadName",
    }
)


def build_run_log_path(task_name: str, run_group_id: str) -> Path:
    """Return the per-run JSONL log path."""
    return (
        Path("results")
        / task_name
        / "live"
        / safe_run_id_component(run_group_id)
        / "run.log"
    )


def verbosity_to_level(verbosity: int) -> int:
    """Map CLI verbosity to a stdlib logging level."""
    if verbosity <= 0:
        return RUN_LOG_LEVEL
    if verbosity == 1:
        return logging.INFO
    return logging.DEBUG


@contextlib.contextmanager
def bind_logging_context(**fields: Any) -> Iterator[None]:
    """Temporarily bind common fields to all log records in this context."""
    current = dict(_LOG_CONTEXT.get())
    current.update({key: value for key, value in fields.items() if value is not None})
    token = _LOG_CONTEXT.set(current)
    try:
        yield
    finally:
        _LOG_CONTEXT.reset(token)


class _DynamicStderr:
    """Proxy writes to the current ``sys.stderr`` for redirect-friendly tests."""

    def write(self, text: str) -> int:
        return sys.stderr.write(text)

    def flush(self) -> None:
        sys.stderr.flush()


class ContextFilter(logging.Filter):
    """Attach contextvars fields to each record."""

    def filter(self, record: logging.LogRecord) -> bool:
        for key, value in _LOG_CONTEXT.get().items():
            if not hasattr(record, key):
                setattr(record, key, value)
        return True


class JsonLineFormatter(logging.Formatter):
    """Format log records as capped JSON objects, one per line."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
        }
        payload.update(_record_extra_fields(record))
        if record.exc_info is not None:
            payload["exception"] = _cap_value(
                "".join(traceback.format_exception(*record.exc_info))
            )
        if record.stack_info:
            payload["stack"] = _cap_value(record.stack_info)
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def configure_logging(
    *,
    verbosity: int,
    debug_console: bool,
    log_file_path: Path | None,
) -> None:
    """Configure root logging for JSONL file output and optional console output."""
    level = verbosity_to_level(verbosity)
    # File logs are the durable debugging record, so capture all app messages there
    # while keeping console output governed by the requested verbosity.
    file_level = logging.DEBUG
    root_level = min(level, file_level) if log_file_path is not None else level
    root_logger = logging.getLogger()
    for handler in root_logger.handlers:
        handler.close()
    root_logger.handlers.clear()
    root_logger.setLevel(root_level)
    for noisy_logger in ("LiteLLM", "litellm"):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)

    context_filter = ContextFilter()
    if log_file_path is not None:
        log_file_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
        file_handler.setLevel(file_level)
        file_handler.setFormatter(JsonLineFormatter())
        file_handler.addFilter(context_filter)
        root_logger.addHandler(file_handler)

    if debug_console:
        console_handler = logging.StreamHandler(_DynamicStderr())
        console_handler.setLevel(level)
        console_handler.setFormatter(
            logging.Formatter("clbench: %(levelname)s: %(name)s: %(message)s")
        )
        console_handler.addFilter(context_filter)
        root_logger.addHandler(console_handler)


def reset_logging_for_tests() -> None:
    """Reset root logging state after tests that configure handlers."""
    root_logger = logging.getLogger()
    for handler in root_logger.handlers:
        handler.close()
    root_logger.handlers.clear()
    root_logger.setLevel(logging.WARNING)


def _record_extra_fields(record: logging.LogRecord) -> dict[str, Any]:
    return {
        key: _cap_value(value)
        for key, value in record.__dict__.items()
        if key not in _STANDARD_RECORD_KEYS and not key.startswith("_")
    }


def _cap_value(value: Any) -> Any:
    if isinstance(value, str):
        if len(value) <= _MAX_FIELD_CHARS:
            return value
        return value[: _MAX_FIELD_CHARS - len(_CAP_SUFFIX)] + _CAP_SUFFIX
    if isinstance(value, Mapping):
        return {str(key): _cap_value(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_cap_value(item) for item in value]
    try:
        json.dumps(value)
    except (TypeError, ValueError):
        return _cap_value(repr(value))
    return value
