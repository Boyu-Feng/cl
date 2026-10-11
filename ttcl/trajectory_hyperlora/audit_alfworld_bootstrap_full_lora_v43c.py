"""Audit the matched fresh full A+B static LoRA control."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from ttcl.trajectory_hyperlora.alfworld_online_from_empty import file_hash
from ttcl.trajectory_hyperlora.alfworld_persistent_failure_delta_v36 import save
from ttcl.trajectory_hyperlora.alfworld_bootstrap_own_success_v43 import (
    checked, examples, parser,
)


def audit(args):
    _, successes, suffix, fixed, _, _ = checked(args)
    report = json.loads(args.full_static_report.read_text())
    model = torch.load(args.full_static_model, map_location="cpu",
                       weights_only=True)
    trainer_hash = file_hash(Path(__file__).with_name(
        "train_alfworld_bootstrap_full_lora_v43c.py"))
    if (len(successes) != 31 or len(suffix) != 120 or
            len(examples(successes)) != 366 or
            report.get("bootstrap_review_sha256") != file_hash(args.review) or
            report.get("bootstrap_trainer_sha256") != fixed["runner_sha256"] or
            report.get("trainer_sha256") != trainer_hash or
            report.get("model_sha256") != file_hash(args.full_static_model) or
            report.get("action_targets") != 366 or
            report.get("steps") != 600 or report.get("seed") != 42 or
            model.get("review_sha256") != file_hash(args.review) or
            model.get("trainer_sha256") != trainer_hash or
            model.get("bootstrap_trainer_sha256") != fixed["runner_sha256"] or
            model.get("rank") != 4 or model.get("layers") != 2 or
            model.get("steps") != 600 or model.get("seed") != 42 or
            len(model.get("A", [])) != 2 or
            len(model.get("factors", [])) != 2 or
            any(not torch.isfinite(t.float()).all()
                for t in model["A"] + model["factors"])):
        raise ValueError("Changed or incomplete matched full LoRA")
    return {"audit": "passed", "own_successes_replayed": 31,
        "action_targets": 366,
        "review_sha256": file_hash(args.review),
        "model_sha256": file_hash(args.full_static_model),
        "report_sha256": file_hash(args.full_static_report)}


if __name__ == "__main__":
    p = parser()
    p.add_argument("--full-static-model", type=Path, default=Path(
        "results/trajectory_hyperlora/alfworld_bootstrap_own_success_v43c_full_static_20261010.pt"))
    p.add_argument("--full-static-report", type=Path, default=Path(
        "results/trajectory_hyperlora/alfworld_bootstrap_own_success_v43c_full_static_20261010.json"))
    p.add_argument("--audit-output", type=Path, default=Path(
        "results/trajectory_hyperlora/alfworld_bootstrap_own_success_v43c_full_static_audit_20261010.json"))
    args = p.parse_args()
    if args.command != "train" or args.audit_output.exists():
        raise ValueError("Audit requires completed training and fresh output")
    result = audit(args)
    save(args.audit_output, result)
    print(json.dumps(result), flush=True)
