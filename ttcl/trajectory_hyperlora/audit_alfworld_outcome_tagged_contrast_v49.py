"""Audit own-prefix tagged contrast targets and checkpoint lineage."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from ttcl.trajectory_hyperlora.alfworld_online_from_empty import file_hash
from ttcl.trajectory_hyperlora.alfworld_persistent_failure_delta_v36 import save
from ttcl.trajectory_hyperlora.train_alfworld_outcome_tagged_contrast_v49 import (
    checked, parser,
)


def audit(args):
    successes, failures, suffix, fixed, targets = checked(args)
    report = json.loads(args.tag_report.read_text())
    model = torch.load(args.tag_model, map_location="cpu", weights_only=True)
    if (len(successes) != 31 or len(failures) != 89 or len(suffix) != 120 or
            len(targets) != 366 or
            any(report.get(k) != v for k, v in fixed.items()) or
            report.get("review_sha256") != file_hash(args.tag_review) or
            report.get("model_sha256") != file_hash(args.tag_model) or
            report.get("action_targets") != 366 or
            model.get("tag_review_sha256") != file_hash(args.tag_review) or
            model.get("tag_trainer_sha256") != file_hash(Path(__file__).with_name(
                "train_alfworld_outcome_tagged_contrast_v49.py")) or
            model.get("parent_hyper_sha256") != file_hash(args.hyper_model) or
            any(not torch.isfinite(t.float()).all()
                for t in model["trainable_state"].values())):
        raise ValueError("Changed or incomplete v49 tagged contrast")
    return {"audit": "passed", "own_successes_replayed": 31,
        "own_failures_replayed": 89, "action_targets": 366,
        "review_sha256": file_hash(args.tag_review),
        "report_sha256": file_hash(args.tag_report),
        "model_sha256": file_hash(args.tag_model)}


if __name__ == "__main__":
    p = parser()
    paths = {"parent_audit": "results/trajectory_hyperlora/alfworld_success_failure_contrast_v44_train_audit_20261010.json",
        "tag_review": "data/annotations/alfworld_outcome_tagged_contrast_v49_train120_20261010.json",
        "tag_model": "results/trajectory_hyperlora/alfworld_outcome_tagged_contrast_v49_20261010.pt",
        "tag_report": "results/trajectory_hyperlora/alfworld_outcome_tagged_contrast_v49_20261010.json",
        "audit_output": "results/trajectory_hyperlora/alfworld_outcome_tagged_contrast_v49_train_audit_20261010.json"}
    for name, value in paths.items():
        p.add_argument("--" + name.replace("_", "-"), type=Path,
                       default=Path(value))
    p.add_argument("--tag-steps", type=int, default=300)
    p.add_argument("--tag-lr", type=float, default=1e-5)
    args = p.parse_args()
    if args.command != "train" or args.audit_output.exists():
        raise ValueError("Audit requires completed training and fresh output")
    result = audit(args)
    save(args.audit_output, result)
    print(json.dumps(result), flush=True)
