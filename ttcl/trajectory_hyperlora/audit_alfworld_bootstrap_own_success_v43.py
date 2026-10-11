"""Audit own-success bootstrap inputs and the two equal-target model artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from ttcl.trajectory_hyperlora.alfworld_online_from_empty import file_hash
from ttcl.trajectory_hyperlora.alfworld_persistent_failure_delta_v36 import save
from ttcl.trajectory_hyperlora.alfworld_bootstrap_own_success_v43 import (
    checked, parser,
)


def audit(args):
    _, successes, suffix, fixed, sources, targets = checked(args)
    report = json.loads(args.train_report.read_text())
    hyper = torch.load(args.hyper_model, map_location="cpu", weights_only=False)
    static = torch.load(args.static_model, map_location="cpu", weights_only=False)
    trainer_hash = file_hash(Path(__file__).with_name(
        "alfworld_bootstrap_own_success_v43.py"))
    review_hash = file_hash(args.review)
    if (any(report.get(k) != v for k, v in fixed.items()) or
            report.get("review_sha256") != review_hash or
            report.get("hyper_model_sha256") != file_hash(args.hyper_model) or
            report.get("static_model_sha256") != file_hash(args.static_model) or
            report.get("own_successes") != 31 or
            report.get("hyper_cross_task_actions") !=
                report.get("static_actions") or
            len(successes) != 31 or len(suffix) != 120 or
            len(sources) != 31 or len(targets) != 120 or
            hyper.get("review_sha256") != review_hash or
            static.get("review_sha256") != review_hash or
            hyper.get("trainer_sha256") != trainer_hash or
            static.get("trainer_sha256") != trainer_hash or
            hyper.get("rank") != 4 or hyper.get("layers") != 2 or
            static.get("rank") != 4 or static.get("layers") != 2 or
            len(static.get("A", [])) != 2 or
            len(static.get("factors", [])) != 2):
        raise ValueError("Changed or incomplete v43 own-success training")
    for name, tensor in hyper["trainable_state"].items():
        if not torch.isfinite(tensor.float()).all():
            raise ValueError(f"Nonfinite hypernetwork parameter {name}")
    for tensor in static["A"] + static["factors"]:
        if not torch.isfinite(tensor.float()).all():
            raise ValueError("Nonfinite static LoRA factor")
    return {"audit": "passed", "own_successes_replayed": 31,
        "equal_action_targets": report["static_actions"],
        "bootstrap_prefix": 120, "future_suffix": 120,
        "review_sha256": review_hash,
        "train_report_sha256": file_hash(args.train_report),
        "hyper_model_sha256": file_hash(args.hyper_model),
        "static_model_sha256": file_hash(args.static_model)}


if __name__ == "__main__":
    p = parser()
    p.add_argument("--audit-output", type=Path, default=Path(
        "results/trajectory_hyperlora/alfworld_bootstrap_own_success_v43b_train_audit_20261010.json"))
    args = p.parse_args()
    if args.command != "train" or args.audit_output.exists():
        raise ValueError("Audit requires completed training and fresh output")
    result = audit(args)
    save(args.audit_output, result)
    print(json.dumps(result), flush=True)
