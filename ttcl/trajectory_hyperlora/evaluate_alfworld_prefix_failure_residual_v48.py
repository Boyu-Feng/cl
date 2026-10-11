"""Use prior own success/failure B difference as a bounded first-attempt residual."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from ttcl.trajectory_hyperlora.alfworld_online_from_empty import digest, file_hash
from ttcl.trajectory_hyperlora.alfworld_persistent_failure_delta_v36 import save
from ttcl.trajectory_hyperlora.alfworld_zero_shot_probe import load_agent, run_episode
from ttcl.trajectory_hyperlora.contextual_alf_source import (
    contextual_source_fields, contextual_text_fields,
)
from ttcl.trajectory_hyperlora.alfworld_success_gated_factors_v27 import bounded_source_text
from ttcl.trajectory_hyperlora.audit_my_online_update_v28_reward import replay
from ttcl.trajectory_hyperlora.train_alfworld_success_failure_contrast_v44 import parser
from ttcl.trajectory_hyperlora.evaluate_alfworld_success_failure_contrast_v44 import bound as contrast_bound


class ResidualSourceAgent:
    """Apply clipped H(success)-H(failure) to H(success) at each actor state."""

    def __init__(self, agent, positive_fields, negative_fields, cap_fraction):
        self.agent = agent
        self.positive_fields = positive_fields
        self.negative_fields = negative_fields
        self.cap_fraction = cap_fraction
        self.a_grams = [layer.a.detach().float() @ layer.a.detach().float().T
                        for layer in agent.adapters]

    def __getattr__(self, name):
        return getattr(self.agent, name)

    def set_source(self, fields, oracle_bits=None, target_fields=None):
        if fields is None:
            self.agent.set_source(None)
            return
        self.agent.set_source(self.positive_fields, target_fields=target_fields)
        positive = [layer.b.detach().clone() for layer in self.agent.adapters]
        self.agent.set_source(self.negative_fields, target_fields=target_fields)
        for layer, good, gram in zip(self.agent.adapters, positive,
                                     self.a_grams, strict=True):
            difference = good - layer.b
            good_flat = good.float().reshape(-1, good.shape[-1])
            difference_flat = difference.float().reshape(-1, difference.shape[-1])
            norm = torch.sqrt(((difference_flat @ gram) * difference_flat).sum().clamp_min(0))
            limit = self.cap_fraction * torch.sqrt(
                ((good_flat @ gram) * good_flat).sum().clamp_min(0))
            multiplier = torch.clamp(limit / norm.clamp_min(1e-12), max=1.)
            layer.b = good + difference * multiplier.to(difference.dtype)


def bound(args):
    successes, failures, suffix, parent_fixed, parent_bindings = contrast_bound(args)
    parent = json.loads(args.contrast_eval_review.read_text())
    if (parent.get("targets") != parent_bindings or
            parent.get("targets_sha256") != digest(parent_bindings) or
            parent_fixed["contrast_model_sha256"] != file_hash(args.residual_model)):
        raise ValueError("Changed reviewed v44 suffix sources")
    fixed = {"method": "v48 prior success minus prior failure dynamic B residual",
        "parent_eval_review_sha256": file_hash(args.parent_eval_review),
        "contrast_eval_review_sha256": file_hash(args.contrast_eval_review),
        "contrast_review_sha256": parent_fixed["contrast_review_sha256"],
        "hyper_model_sha256": file_hash(args.hyper_model),
        "residual_model_sha256": file_hash(args.residual_model),
        "runner_sha256": file_hash(Path(__file__)),
        "actor_sha256": file_hash(Path(__file__).with_name("alfworld_zero_shot_probe.py")),
        "cap_fraction": args.cap_fraction, "gain": 1.,
        "max_steps": 30, "max_new_tokens": 64,
        "constrain_actions": True, "actor_history_turns": 2,
        "loop_guard_max": 2}
    targets = []
    for i, (row, parent_binding) in enumerate(zip(
            suffix, parent_bindings, strict=True)):
        content = {"index": i, "game": row["game"],
            "game_sha256": parent_binding["game_sha256"],
            "parent_input_content_sha256": parent_binding["input_content_sha256"],
            "good_source_index": parent_binding["good_source_index"],
            "bad_source_index": parent_binding["bad_source_index"],
            "good_source_records_sha256": parent_binding["good_source_records_sha256"],
            "bad_source_records_sha256": parent_binding["bad_source_records_sha256"],
            "runner_sha256": fixed["runner_sha256"],
            "residual_model_sha256": fixed["residual_model_sha256"]}
        targets.append({**content, "input_content_sha256": digest(content),
                        "reviewed_target": True})
    return successes, failures, suffix, fixed, targets


def prepare(args):
    if args.eval_review.exists():
        raise FileExistsError(args.eval_review)
    _, _, _, fixed, targets = bound(args)
    save(args.eval_review, {"protocol": "Official train-stream suffix120, fresh H trained on own prefix successes then prefix success/failure contrast; prior own success and failed same-family sources form dynamic B_good + clip(B_good-B_bad,25%) at each actor state. One 30-step attempt. No current-task failure or suffix outcome in parameter rule.",
        **fixed, "targets": targets, "targets_sha256": digest(targets)})
    print(json.dumps({"reviewed_targets": len(targets)}), flush=True)


def evaluate(args):
    if args.eval_output.exists():
        raise FileExistsError(args.eval_output)
    successes, failures, suffix, fixed, targets = bound(args)
    review = json.loads(args.eval_review.read_text())
    if (review.get("targets") != targets or
            review.get("targets_sha256") != digest(targets) or
            any(review.get(k) != v for k, v in fixed.items())):
        raise ValueError("Changed reviewed v48 suffix targets")
    agent, tokenizer = load_agent(args.model, args.residual_model,
                                  args.device, args.gpu_fraction)
    report = {"protocol": review["protocol"],
        "review_sha256": file_hash(args.eval_review),
        **fixed, "rows": [], "failures": []}
    for i, (row, binding) in enumerate(zip(suffix, targets, strict=True)):
        good = successes[binding["good_source_index"]]
        bad = failures[binding["bad_source_index"]]
        good_fields = contextual_source_fields(agent, tokenizer,
            good["source_records"], args.device, 2048, pooling="both")
        bad_text, _, _ = bounded_source_text(tokenizer,
            bad["source_records"], 2048)
        bad_fields = contextual_text_fields(agent, tokenizer, bad_text,
            args.device, 2048, pooling="both")
        wrapped = ResidualSourceAgent(agent, good_fields, bad_fields,
                                      args.cap_fraction)
        episode = run_episode(wrapped, tokenizer, args.data_root / row["game"],
            good_fields, adapter=True, device=args.device,
            max_steps=30, max_new_tokens=64, constrain_actions=True,
            actor_history_turns=2, loop_guard_max=2)
        if (episode["status"] != "complete" or
                episode["initial_observation"] !=
                row["arms"]["base"]["initial_observation"]):
            report["failures"].append({"index": i})
        report["rows"].append({"index": i, "game": row["game"],
            "input_content_sha256": binding["input_content_sha256"],
            "episode": episode})
        report["summary"] = {"n": len(report["rows"]),
            "successes": sum(x["episode"]["reward"] for x in report["rows"]),
            "failures": len(report["failures"])}
        save(args.eval_output, report)
        print(json.dumps(report["summary"]), flush=True)
        if report["failures"]:
            raise RuntimeError("Incomplete v48 episode")


def audit(args):
    _, _, suffix, fixed, targets = bound(args)
    review = json.loads(args.eval_review.read_text())
    report = json.loads(args.eval_output.read_text())
    if (review.get("targets") != targets or
            review.get("targets_sha256") != digest(targets) or
            any(review.get(k) != v or report.get(k) != v for k, v in fixed.items()) or
            report.get("review_sha256") != file_hash(args.eval_review) or
            len(report.get("rows", [])) != 120 or report.get("failures")):
        raise ValueError("Changed or incomplete v48 suffix evaluation")
    wins = 0.
    for i, (source, target, row) in enumerate(zip(
            suffix, targets, report["rows"], strict=True)):
        if (row["index"] != i or row["game"] != source["game"] or
                row["input_content_sha256"] != target["input_content_sha256"] or
                row["episode"]["status"] != "complete" or
                row["episode"]["steps"] > 30):
            raise ValueError(f"Changed target or episode {i}")
        replay(args.data_root / row["game"], row["episode"])
        wins += row["episode"]["reward"]
    if report["summary"] != {"n": 120, "successes": wins, "failures": 0}:
        raise ValueError("Changed v48 score summary")
    return {"audit": "passed", "summary": report["summary"],
        "replayed_episodes": 120,
        "report_sha256": file_hash(args.eval_output),
        "review_sha256": file_hash(args.eval_review)}


if __name__ == "__main__":
    p = parser()
    p.add_argument("--parent-eval-review", type=Path, default=Path(
        "data/annotations/alfworld_bootstrap_own_success_v43c_suffix120_20261010.json"))
    p.add_argument("--contrast-eval-review", type=Path, default=Path(
        "data/annotations/alfworld_success_failure_contrast_v44_suffix120_20261010.json"))
    p.add_argument("--residual-model", type=Path, default=Path(
        "results/trajectory_hyperlora/alfworld_success_failure_contrast_v44_20261010.pt"))
    p.add_argument("--eval-review", type=Path, default=Path(
        "data/annotations/alfworld_prefix_failure_residual_v48_suffix120_20261010.json"))
    p.add_argument("--eval-output", type=Path, default=Path(
        "results/trajectory_hyperlora/alfworld_prefix_failure_residual_v48_suffix120_20261010.json"))
    p.add_argument("--audit-output", type=Path, default=Path(
        "results/trajectory_hyperlora/alfworld_prefix_failure_residual_v48_suffix120_audit_20261010.json"))
    p.add_argument("--cap-fraction", type=float, default=.25)
    p.add_argument("--audit-only", action="store_true")
    args = p.parse_args()
    if not 0 <= args.cap_fraction <= 1:
        raise ValueError("Cap fraction must lie in [0,1]")
    if args.audit_only:
        if args.audit_output.exists():
            raise FileExistsError(args.audit_output)
        result = audit(args)
        save(args.audit_output, result)
        print(json.dumps(result), flush=True)
    else:
        (prepare if args.command == "prepare" else evaluate)(args)
