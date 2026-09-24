"""ACE-specific artifact export helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from ...artifacts import register_artifact_exporter


def _build_ace_artifact_manifest(artifacts: dict[str, Any]) -> dict[str, Any]:
    snapshots = list(artifacts.get("playbook_snapshots", []))
    final_playbook = str(artifacts.get("final_playbook", ""))
    final_bullet_count = sum(
        1 for line in final_playbook.splitlines() if line.startswith("[")
    )
    return {
        "type": "ace",
        "playbook_snapshot_count": len(snapshots),
        "final_playbook_bullet_count": final_bullet_count,
    }


def _save_ace_artifacts(
    artifacts: dict[str, Any],
    trace_path: Path,
) -> Optional[Path]:
    artifact_dir = trace_path.parent / "artifacts" / trace_path.stem
    snapshots_dir = artifact_dir / "playbook_snapshots"
    snapshots_dir.mkdir(parents=True, exist_ok=True)

    initial_playbook = str(artifacts.get("initial_playbook", ""))
    final_playbook = str(artifacts.get("final_playbook", ""))
    snapshots = list(artifacts.get("playbook_snapshots", []))

    (artifact_dir / "initial_playbook.md").write_text(
        initial_playbook, encoding="utf-8"
    )
    (artifact_dir / "final_playbook.md").write_text(final_playbook, encoding="utf-8")

    manifest_snapshots = []
    for snapshot in snapshots:
        update_count = int(snapshot.get("update_count", 0))
        snapshot_filename = f"update_{update_count:04d}.md"
        (snapshots_dir / snapshot_filename).write_text(
            str(snapshot.get("playbook", "")),
            encoding="utf-8",
        )
        manifest_snapshot = dict(snapshot)
        manifest_snapshot.pop("playbook", None)
        manifest_snapshot["path"] = f"playbook_snapshots/{snapshot_filename}"
        manifest_snapshots.append(manifest_snapshot)

    manifest = {
        "artifact_type": "ace",
        "trace_file": str(trace_path),
        "initial_playbook_path": "initial_playbook.md",
        "final_playbook_path": "final_playbook.md",
        "playbook_snapshots": manifest_snapshots,
    }
    (artifact_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    return artifact_dir


def ensure_registered() -> None:
    register_artifact_exporter(
        "ace",
        build_manifest=_build_ace_artifact_manifest,
        save=_save_ace_artifacts,
    )
