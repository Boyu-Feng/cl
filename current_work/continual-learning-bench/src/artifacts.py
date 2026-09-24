"""Generic system artifact export helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional


ManifestBuilder = Callable[[dict[str, Any]], dict[str, Any]]
ArtifactSaver = Callable[[dict[str, Any], Path], Optional[Path]]
BaselineArtifactBuilder = Callable[[list[dict[str, Any]]], Optional[dict[str, Any]]]


@dataclass(frozen=True)
class ArtifactExporter:
    """Exporter callbacks for a specific system artifact type."""

    build_manifest: ManifestBuilder
    save: ArtifactSaver
    build_baseline: BaselineArtifactBuilder | None = None


_EXPORTERS: dict[str, ArtifactExporter] = {}


def register_artifact_exporter(
    artifact_type: str,
    *,
    build_manifest: ManifestBuilder,
    save: ArtifactSaver,
    build_baseline: BaselineArtifactBuilder | None = None,
) -> None:
    """Register callbacks for a system artifact type."""
    _EXPORTERS[artifact_type] = ArtifactExporter(
        build_manifest=build_manifest,
        save=save,
        build_baseline=build_baseline,
    )


def build_artifact_manifest(artifacts: Optional[dict[str, Any]]) -> dict[str, Any]:
    """Build a trace-friendly manifest summary for exported system artifacts."""
    if not artifacts:
        return {}

    artifact_type = str(artifacts.get("artifact_type") or "unknown")
    exporter = _EXPORTERS.get(artifact_type)
    if exporter is None:
        return {"type": artifact_type}

    manifest = dict(exporter.build_manifest(artifacts))
    manifest.setdefault("type", artifact_type)
    return manifest


def save_artifacts(
    artifacts: Optional[dict[str, Any]],
    trace_path: Path,
) -> Optional[Path]:
    """Save optional system artifacts alongside the trace."""
    if not artifacts:
        return None

    artifact_type = str(artifacts.get("artifact_type") or "unknown")
    exporter = _EXPORTERS.get(artifact_type)
    if exporter is None:
        return _save_generic_artifacts(artifacts, trace_path, artifact_type)

    return exporter.save(artifacts, trace_path)


def build_baseline_system_artifacts(
    trace_payloads: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Build merged baseline artifacts using the registered system exporter."""
    seen_artifact_types: set[str] = set()
    for trace in trace_payloads:
        system_artifacts = trace.get("system_artifacts")
        if not isinstance(system_artifacts, dict):
            continue
        artifact_type = str(system_artifacts.get("artifact_type") or "")
        if not artifact_type or artifact_type in seen_artifact_types:
            continue
        seen_artifact_types.add(artifact_type)

        exporter = _EXPORTERS.get(artifact_type)
        if exporter is None or exporter.build_baseline is None:
            continue
        baseline_artifacts = exporter.build_baseline(trace_payloads)
        if baseline_artifacts:
            return baseline_artifacts

    return None


def _save_generic_artifacts(
    artifacts: dict[str, Any],
    trace_path: Path,
    artifact_type: str,
) -> Path:
    artifact_dir = trace_path.parent / "artifacts" / trace_path.stem
    artifact_dir.mkdir(parents=True, exist_ok=True)

    payload_path = artifact_dir / "artifacts.json"
    payload_path.write_text(
        json.dumps(artifacts, indent=2, default=_json_default),
        encoding="utf-8",
    )
    manifest = {
        "artifact_type": artifact_type,
        "trace_file": str(trace_path),
        "artifact_payload_path": payload_path.name,
    }
    (artifact_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    return artifact_dir


def _json_default(value: Any) -> str:
    return repr(value)
