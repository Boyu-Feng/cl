"""Interpolate a new ALFWorld hypernetwork with its frozen warm start.

The same scalar also interpolates the optional initial-observation context,
so alpha zero exactly recovers the old generator and alpha one the new one.
Only trainable hypernetwork/LoRA parameters are mixed; Qwen stays frozen.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ttcl.trajectory_hyperlora.alfworld_same_game_retry import file_hash


def mix(args):
    if args.output.exists() or args.report.exists():
        raise FileExistsError("Fresh mixed checkpoint/report paths required")
    old = torch.load(args.old, map_location="cpu", weights_only=True)
    new = torch.load(args.new, map_location="cpu", weights_only=True)
    for key in ("rank", "layers", "encoder_kind", "relation_bottleneck",
                "source_encoder"):
        if old[key] != new[key]:
            raise ValueError(f"Incompatible generator metadata: {key}")
    if (old.get("context_mode", "none") != "none" or
            new.get("context_mode") != "initial" or
            new.get("source_truncation") != "head_tail"):
        raise ValueError("Expected old context-free and new initial-context generators")
    before, after = old["trainable_state"], new["trainable_state"]
    if set(before) != set(after):
        raise ValueError("Incompatible trainable parameter names")
    mixed = {}
    for name in before:
        left, right = before[name], after[name]
        if left.shape != right.shape or left.dtype != right.dtype:
            raise ValueError(f"Incompatible trainable tensor: {name}")
        mixed[name] = torch.lerp(left.float(), right.float(), args.alpha).to(
            left.dtype)
    checkpoint = {key: old[key] for key in ("rank", "layers", "encoder_kind",
        "relation_bottleneck", "source_encoder")}
    checkpoint.update({"trainable_state": mixed,
        "training_domain": "alfworld_interpolated_generator",
        "context_mode": "initial", "context_strength": args.alpha,
        "source_max_tokens": new["source_max_tokens"],
        "source_truncation": new["source_truncation"],
        "repeat_initial_observation": new["repeat_initial_observation"],
        "old_checkpoint_sha256": file_hash(args.old),
        "new_checkpoint_sha256": file_hash(args.new),
        "interpolation_alpha": args.alpha})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output)
    report = {"protocol": "Linear interpolation of every trainable old/new hypernetwork and LoRA parameter plus initial-context strength; frozen Qwen; no evaluation rewards used",
              "old_checkpoint_sha256": file_hash(args.old),
              "new_checkpoint_sha256": file_hash(args.new),
              "mixed_checkpoint_sha256": file_hash(args.output),
              "alpha": args.alpha, "tensor_count": len(mixed)}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--old", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_next_task_warmstart_20261005/seed42_v2.pt"))
    parser.add_argument("--new", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_expert_all42_contrast1000_20261006.pt"))
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if not 0 <= args.alpha <= 1:
        parser.error("Interpolation alpha must be between zero and one")
    mix(args)


if __name__ == "__main__":
    main()
