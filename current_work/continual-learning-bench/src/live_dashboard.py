"""Helpers for live benchmark snapshot files and the lightweight viewer server."""

from __future__ import annotations

from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from typing import Any
from urllib.parse import quote

from .run_ids import safe_run_id_component
from .trace_storage import _atomic_write_json


def build_live_output_dir(task_name: str, run_group_id: str) -> Path:
    """Return the directory where live benchmark artifacts should be written."""
    return Path("results") / task_name / "live" / safe_run_id_component(run_group_id)


def build_live_trace_path(
    task_name: str,
    run_group_id: str,
    label: str,
) -> Path:
    """Return the live snapshot path for a baseline or rollout trace."""
    safe_label = "".join(
        ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in label
    )
    return build_live_output_dir(task_name, run_group_id) / f"{safe_label}.json"


def build_live_manifest_path(task_name: str, run_group_id: str) -> Path:
    """Return the live manifest path for a benchmark run group."""
    return build_live_output_dir(task_name, run_group_id) / "manifest.json"


def write_live_manifest(path: Path, payload: dict[str, Any]) -> None:
    """Atomically update the live manifest."""
    _atomic_write_json(path, payload)


def _root_relative_path(path: Path, root_dir: Path) -> str:
    """Return a server-relative path for a file under the served root."""
    if path.is_absolute():
        return path.resolve().relative_to(root_dir.resolve()).as_posix()
    return path.as_posix()


class _NoCacheSimpleHTTPRequestHandler(SimpleHTTPRequestHandler):
    """Static file handler with caching disabled for live JSON polling."""

    def end_headers(self) -> None:  # pragma: no cover - exercised in integration
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()

    def log_message(self, format: str, *args) -> None:  # noqa: A002
        return


def start_live_dashboard_server(
    *,
    root_dir: Path,
    manifest_path: Path,
    viewer_path: str = "viewers/single_task_viewer.html",
    host: str = "127.0.0.1",
    port: int = 0,
) -> tuple[ThreadingHTTPServer, Thread, str]:
    """Start a lightweight static server for the live dashboard viewer."""
    handler = partial(_NoCacheSimpleHTTPRequestHandler, directory=str(root_dir))
    server = ThreadingHTTPServer((host, port), handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()

    manifest_rel = _root_relative_path(manifest_path, root_dir)
    viewer_url = (
        f"http://{host}:{server.server_address[1]}/{viewer_path}"
        f"?artifact={quote(manifest_rel)}"
    )
    return server, thread, viewer_url


def build_run_all_output_dir(system_name: str, run_group_id: str) -> Path:
    """Return the directory where run-all aggregate artifacts are written."""
    safe_system = "".join(
        ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in system_name
    )
    return (
        Path("results") / "run-all" / safe_system / safe_run_id_component(run_group_id)
    )


def build_run_all_manifest_path(system_name: str, run_group_id: str) -> Path:
    """Return the path to the run-all aggregate manifest."""
    return build_run_all_output_dir(system_name, run_group_id) / "manifest.json"


def write_run_all_manifest(path: Path, payload: dict[str, Any]) -> None:
    """Atomically update the run-all aggregate manifest."""
    _atomic_write_json(path, payload)
