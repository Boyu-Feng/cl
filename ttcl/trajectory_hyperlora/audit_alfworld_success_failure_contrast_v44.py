"""Audit v44 source episodes, paired targets, and output checkpoint lineage."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from ttcl.trajectory_hyperlora.alfworld_online_from_empty import file_hash
from ttcl.trajectory_hyperlora.alfworld_persistent_failure_delta_v36 import save
from ttcl.trajectory_hyperlora.train_alfworld_success_failure_contrast_v44 import (
    checked, parser,
)


def audit(args):
    successes, failures, suffix, fixed, targets = checked(args)
    report = json.loads(args.contrast_report.read_text())
    model = torch.load(args.contrast_model, map_location="cpu", weights_only=True)
    if (len(successes) != 31 or len(failures) != 89 or len(suffix) != 120 or
            len(targets) != 366 or
            any(report.get(k) != v for k, v in fixed.items()) or
            report.get("review_sha256") != file_hash(args.contrast_review) or
            report.get("model_sha256") != file_hash(args.contrast_model) or
            report.get("action_targets") != 366 or
            model.get("contrast_review_sha256") != file_hash(args.contrast_review) or
            model.get("contrast_trainer_sha256") != file_hash(Path(__file__).with_name(
                "train_alfworld_success_failure_contrast_v44.py")) or
            model.get("parent_hyper_sha256") != file_hash(args.hyper_model)):
        raise ValueError("Changed or incomplete v44 contrast training")
    if any(not torch.isfinite(t.float()).all()
           for t in model["trainable_state"].values()):
        raise ValueError("Nonfinite contrast checkpoint")
    return {"audit": "passed", "own_successes_replayed": 31,
        "own_failures_replayed": 89, "action_targets": 366,
        "review_sha256": file_hash(args.contrast_review),
        "report_sha256": file_hash(args.contrast_report),
        "model_sha256": file_hash(args.contrast_model)}


if __name__ == "__main__":
    p = parser()
    p.add_argument("--audit-output", type=Path, default=Path(
        "results/trajectory_hyperlora/alfworld_success_failure_contrast_v44_train_audit_20261010.json"))
    args = p.parse_args()
    if args.command != "train" or args.audit_output.exists():
        raise ValueError("Audit requires completed training and fresh output")
    result = audit(args)
    save(args.audit_output, result)
    print(json.dumps(result), flush=True)
