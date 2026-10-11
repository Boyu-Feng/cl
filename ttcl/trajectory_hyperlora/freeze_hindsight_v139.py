"""Freeze a new content-bound manifest for hindsight online learning.

The manifest carries no target actions, rewards, or source labels.  It binds
the exact game content and the changed update protocol so a later run cannot
silently reuse the older four-step or independent-source binding.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def digest(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_manifest(manifest: dict, arguments: dict, data_root: Path) -> None:
    """Reject changed settings, order, or game bytes before loading the model."""
    spec = manifest.get('protocol_spec')
    if not isinstance(spec, dict):
        raise ValueError('missing hindsight protocol binding')
    for key, value in spec.items():
        if key != 'name' and arguments.get(key) != value:
            raise ValueError(f'protocol mismatch: {key}')
    seen = set()
    for index, row in enumerate(manifest['targets']):
        content = {k: v for k, v in row.items()
                   if k not in ('input_content_sha256', 'reviewed_target', 'review_note')}
        if (row.get('reviewed_target') is not True or
                row.get('index') != index or row.get('protocol') != spec or
                row.get('source_manifest_sha256') != manifest.get('source_manifest_sha256') or
                digest(content) != row.get('input_content_sha256')):
            raise ValueError('changed or unreviewed input binding')
        relative = Path(row['game'])
        if relative.is_absolute() or '..' in relative.parts or row['game'] in seen:
            raise ValueError('invalid or repeated game path')
        seen.add(row['game'])
        if sha256(data_root / relative) != row['game_sha256']:
            raise ValueError('game content changed')
    if not seen or not 0 < arguments['limit'] <= len(seen):
        raise ValueError('invalid run length')


def freeze(source: Path, data_root: Path, output: Path, *,
           delta_fraction: float = 0.25,
           protocol_name: str = "hindsight_shared_source_full_trajectory_v139",
           settings: dict | None = None) -> dict:
    if output.exists():
        raise FileExistsError(output)
    old = json.loads(source.read_text())
    targets = old.get("targets")
    if not isinstance(targets, list) or not targets:
        raise ValueError("source manifest has no targets")
    protocol = {
        "name": protocol_name,
        "verified_source": "base",
        "shared_write_source": True,
        "success_only_updates": True,
        "direct_init_frozen": True,
        "credit_steps": 0,
        "credit_discount": 0.9,
        "delta_fraction": delta_fraction,
        "delta_initial_weight": 0.1,
        "max_steps": 20,
        "max_new_tokens": 48,
    }
    if settings:
        protocol.update(settings)
    frozen = []
    seen = set()
    for index, row in enumerate(targets):
        game = row["game"]
        path = data_root / game
        if not path.is_file():
            raise FileNotFoundError(path)
        game_hash = sha256(path)
        if game_hash != row.get("game_sha256"):
            raise ValueError(f"game content changed: {game}")
        content = {
            "index": index,
            "round": row.get("round"),
            "family": row["family"],
            "game": game,
            "game_sha256": game_hash,
            "protocol": protocol,
            "source_manifest_sha256": sha256(source),
        }
        key = digest(content)
        if key in seen:
            raise ValueError("duplicate input binding")
        seen.add(key)
        frozen.append({**content, "input_content_sha256": key,
                       "reviewed_target": True,
                       "review_note": "Protocol and game-content binding audited; no outcome labels used."})
    result = {
        "protocol": "New content-bound targets for shared-source full-trajectory hindsight diagnostic; no outcome-derived selection",
        "source_manifest_sha256": sha256(source),
        "protocol_spec": protocol,
        "targets": frozen,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("ttcl/data/alfworld_delta"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--delta-fraction", type=float, default=0.25)
    parser.add_argument("--protocol-name", default="hindsight_shared_source_full_trajectory_v139")
    parser.add_argument("--settings-json", default=None,
                        help="JSON object of all behavior-affecting protocol settings")
    args = parser.parse_args()
    settings = json.loads(args.settings_json) if args.settings_json else None
    result = freeze(args.source, args.data_root, args.output,
                    delta_fraction=args.delta_fraction,
                    protocol_name=args.protocol_name, settings=settings)
    print(json.dumps({"targets": len(result["targets"]),
                      "output": str(args.output),
                      "source_manifest_sha256": result["source_manifest_sha256"]}))
