"""Distill same-task failed-episode B corrections into a pre-attempt head.

Only audited official train trajectories are used. The frozen hypernetwork's
pre-attempt latent is the input. A ridge head predicts the bounded B residual
seen after failure; a regularized gate predicts whether the old first attempt
would fail. Neither head sees the target's failed trajectory at inference.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from ttcl.trajectory_hyperlora.alfworld_online_from_empty import digest, file_hash
from ttcl.trajectory_hyperlora.alfworld_online_accumulate_lora_v1 import factors_hash
from ttcl.trajectory_hyperlora.alfworld_failure_retry_residual_v31b import bounded_residual
from ttcl.trajectory_hyperlora.alfworld_success_gated_factors_v27 import episode_factor
from ttcl.trajectory_hyperlora.alfworld_zero_shot_probe import load_agent
from ttcl.trajectory_hyperlora.contextual_alf_source import (
    contextual_source_fields, contextual_text_fields, task_context_text,
)
from ttcl.trajectory_hyperlora.alfworld_persistent_failure_delta_v37 import (
    bindings as v37_bindings, ordered_pairs,
)
from ttcl.trajectory_hyperlora.alfworld_persistent_failure_delta_v36 import save


def reviewed_train(args):
    pairs = ordered_pairs(args)
    fixed, targets = v37_bindings(args, pairs)
    review = json.loads(args.v37_review.read_text())
    report = json.loads(args.v37_report.read_text())
    audit = json.loads(args.v37_audit.read_text())
    if (review.get("targets") != targets or review.get("targets_sha256") != digest(targets)
            or any(review.get(k) != v or report.get(k) != v for k, v in fixed.items())
            or report.get("review_sha256") != file_hash(args.v37_review)
            or audit.get("audit") != "passed"
            or audit.get("report_sha256") != file_hash(args.v37_report)
            or len(report.get("rows", [])) != 48 or report.get("failures")):
        raise ValueError("Changed audited 48 train episodes")
    episodes = [x["baseline"] for x in report["rows"]]
    for pair, ep, row in zip(pairs, episodes, report["rows"], strict=True):
        if (pair["target_game"] != row["game"] or ep["status"] != "complete"
                or ep["steps"] > 50 or
                file_hash(args.data_root / row["game"]) != pair["target_game_sha256"]):
            raise ValueError("Changed train target or episode")
    return pairs, episodes


def bindings(args, pairs, episodes):
    fixed = {"method": "v40 train-only first-attempt B correction distillation",
        "pairs_sha256": file_hash(args.pairs),
        "v37_review_sha256": file_hash(args.v37_review),
        "v37_report_sha256": file_hash(args.v37_report),
        "v37_audit_sha256": file_hash(args.v37_audit),
        "checkpoint_sha256": file_hash(args.checkpoint),
        "model_config_sha256": file_hash(args.model / "config.json"),
        "trainer_sha256": file_hash(Path(__file__)),
        "actor_sha256": file_hash(Path(__file__).with_name("alfworld_zero_shot_probe.py")),
        "teacher_residual_sha256": file_hash(Path(__file__).with_name("alfworld_failure_retry_residual_v31b.py")),
        "max_relative": .25, "training_split": "official train only",
        "ridge_lambda": args.ridge_lambda, "gate_l2": args.gate_l2,
        "gate_steps": args.gate_steps, "seed": args.seed}
    targets = []
    for i, (pair, ep) in enumerate(zip(pairs, episodes, strict=True)):
        content = {"index": i, "game": pair["target_game"],
            "game_sha256": pair["target_game_sha256"],
            "pair_input_content_sha256": pair["input_content_sha256"],
            "expert_source_records_sha256": pair["source_records_sha256"],
            "first_episode_sha256": digest(ep),
            "trainer_sha256": fixed["trainer_sha256"],
            "checkpoint_sha256": fixed["checkpoint_sha256"]}
        targets.append({**content, "input_content_sha256": digest(content),
                        "reviewed_target": True})
    return fixed, targets


def preattempt(agent, tokenizer, records, initial, device):
    fields = contextual_source_fields(agent, tokenizer, records, device, 2048,
                                      pooling="both")
    target = contextual_text_fields(agent, tokenizer,
        task_context_text(initial, initial), device, 2048, pooling="both")
    with torch.no_grad():
        source = F.layer_norm(fields["pair_contextual"].float(),
                              (fields["pair_contextual"].shape[-1],))
        goal = F.layer_norm(target["pair_contextual"].float(),
                            (target["pair_contextual"].shape[-1],))
        pair = torch.cat((source * goal, (source - goal).abs()), dim=-1)
        latent = agent.encode(fields) + agent.task_pair_latent(pair)
        agent.set_source(fields, target_fields=target)
        expert = [layer.b.detach().cpu().float().clone() for layer in agent.adapters]
        agent.set_source(None)
    return latent.detach().cpu().float().flatten(), expert, fields


def flatten(factors):
    return torch.cat([x.detach().cpu().float().flatten() for x in factors])


def unflatten(vector, shapes):
    result, offset = [], 0
    for shape in shapes:
        size = int(torch.tensor(shape).prod())
        result.append(vector[offset:offset + size].reshape(shape))
        offset += size
    if offset != vector.numel():
        raise ValueError("Changed correction factor width")
    return result


def normalize(x, center):
    return F.normalize(x.float() - center.float(), dim=-1, eps=1e-8)


def predict(model, latent):
    z = normalize(latent, model["feature_center"])
    gate = torch.sigmoid(z @ model["gate_weight"] + model["gate_bias"])
    kernel = z @ model["failure_features"].T
    flat = model["mean_delta"] + kernel @ model["ridge_alpha"]
    return float(gate), unflatten(flat, model["factor_shapes"])


def prepare(args):
    if args.review.exists():
        raise FileExistsError(args.review)
    pairs, episodes = reviewed_train(args)
    fixed, targets = bindings(args, pairs, episodes)
    save(args.review, {"protocol": "48 audited, content-bound official train first attempts; fail-only teacher Delta B; success episodes teach gate off; no dev or unseen labels in training",
        **fixed, "targets": targets, "targets_sha256": digest(targets)})
    print(json.dumps({"train_targets": len(targets),
        "old_successes": sum(x["reward"] for x in episodes)}), flush=True)


def train(args):
    if args.model_output.exists() or args.report_output.exists():
        raise FileExistsError("Use new v40 output paths")
    pairs, episodes = reviewed_train(args)
    fixed, targets = bindings(args, pairs, episodes)
    review = json.loads(args.review.read_text())
    if (review.get("targets") != targets or review.get("targets_sha256") != digest(targets)
            or any(review.get(k) != v for k, v in fixed.items())):
        raise ValueError("Changed reviewed training targets")
    agent, tokenizer = load_agent(args.model, args.checkpoint,
                                  args.device, args.gpu_fraction)
    xs, ys, labels, rows = [], [], [], []
    shapes = None
    for i, (pair, ep, binding) in enumerate(zip(pairs, episodes, targets, strict=True)):
        latent, expert, _ = preattempt(agent, tokenizer, pair["source_records"],
                                      ep["initial_observation"], args.device)
        if shapes is None:
            shapes = [list(x.shape) for x in expert]
        elif shapes != [list(x.shape) for x in expert]:
            raise ValueError("Changed factor shapes")
        row = {"index": i, "game": binding["game"],
            "input_content_sha256": binding["input_content_sha256"],
            "first_reward": ep["reward"],
            "expert_factor_sha256": factors_hash(expert),
            "prelatent_sha256": digest(latent.tolist())}
        xs.append(latent)
        labels.append(float(not ep["reward"]))
        if not ep["reward"]:
            failed, _, original, retained = episode_factor(agent, tokenizer, ep,
                argparse.Namespace(device=args.device, context_tokens=2048))
            toward, info = bounded_residual(agent, expert, failed,
                                            max_relative=.25)
            delta = [-x for x in toward]
            ys.append(flatten(delta))
            row.update(teacher_delta_sha256=factors_hash(delta),
                failed_factor_sha256=factors_hash(failed),
                teacher_delta_info=info,
                failed_source_tokens_original=original,
                failed_source_tokens_retained=retained)
        rows.append(row)
        print(json.dumps({"encoded": i + 1, "failures": len(ys)}), flush=True)
    x = torch.stack(xs)
    y = torch.stack(ys)
    label = torch.tensor(labels)
    center = x.mean(dim=0)
    z = normalize(x, center)
    failure_z = z[label.bool()]
    mean_delta = y.mean(dim=0)
    kernel = failure_z @ failure_z.T
    ridge_alpha = torch.linalg.solve(kernel + args.ridge_lambda * torch.eye(len(y)),
                                     y - mean_delta)
    torch.manual_seed(args.seed)
    gate_weight = torch.zeros(z.shape[1], requires_grad=True)
    gate_bias = torch.zeros((), requires_grad=True)
    optimizer = torch.optim.Adam([gate_weight, gate_bias], lr=.05)
    for _ in range(args.gate_steps):
        logits = z @ gate_weight + gate_bias
        loss = F.binary_cross_entropy_with_logits(logits, label) + \
            args.gate_l2 * gate_weight.square().sum()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        probs = torch.sigmoid(z @ gate_weight + gate_bias).tolist()
    model = {"feature_center": center, "failure_features": failure_z,
        "mean_delta": mean_delta, "ridge_alpha": ridge_alpha,
        "gate_weight": gate_weight.detach(), "gate_bias": gate_bias.detach(),
        "factor_shapes": shapes, "train_review_sha256": file_hash(args.review),
        "checkpoint_sha256": fixed["checkpoint_sha256"],
        "trainer_sha256": fixed["trainer_sha256"],
        "ridge_lambda": args.ridge_lambda, "gate_l2": args.gate_l2,
        "gate_steps": args.gate_steps, "seed": args.seed}
    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model, args.model_output)
    result = {"protocol": review["protocol"], **fixed,
        "train_review_sha256": file_hash(args.review),
        "model_sha256": file_hash(args.model_output),
        "n": len(rows), "old_successes": sum(x["reward"] for x in episodes),
        "failure_teachers": len(ys),
        "gate_probability_range": [min(probs), max(probs)],
        "gate_probability_mean": sum(probs) / len(probs),
        "rows": [{**row, "gate_probability_train": probs[i]}
                 for i, row in enumerate(rows)]}
    save(args.report_output, result)
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}), flush=True)


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=("prepare", "train"))
    defaults = {
        "pairs": "data/annotations/alf_reward_router_train72_reviewed_20261007.json",
        "v36_report": "results/trajectory_hyperlora/alfworld_persistent_failure_delta_v36_dev24_20261009.json",
        "v36_audit": "results/trajectory_hyperlora/alfworld_persistent_failure_delta_v36_dev24_audit_20261009.json",
        "v37_review": "data/annotations/alfworld_persistent_failure_delta_v37_interleave_20261009.json",
        "v37_report": "results/trajectory_hyperlora/alfworld_persistent_failure_delta_v37_interleave_20261009.json",
        "v37_audit": "results/trajectory_hyperlora/alfworld_persistent_failure_delta_v37_interleave_audit_20261009.json",
        "checkpoint": "results/trajectory_hyperlora/alf_sibling_simple40_onpolicy600_20261006.pt",
        "model": "current_work/delta-Mem/model/Qwen3-4B-Instruct-2507",
        "data_root": "ttcl/data/alfworld_delta",
        "review": "data/annotations/alfworld_failure_distill_v40_train48_20261010.json",
        "model_output": "results/trajectory_hyperlora/alfworld_failure_distill_v40_train48_20261010.pt",
        "report_output": "results/trajectory_hyperlora/alfworld_failure_distill_v40_train48_20261010.json"}
    for name, value in defaults.items():
        p.add_argument("--" + name.replace("_", "-"), type=Path, default=Path(value))
    p.add_argument("--order", default="interleave", choices=("interleave",))
    p.add_argument("--ridge-lambda", type=float, default=.5)
    p.add_argument("--gate-l2", type=float, default=.1)
    p.add_argument("--gate-steps", type=int, default=500)
    p.add_argument("--seed", type=int, default=20261010)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--gpu-fraction", type=float, default=.65)
    return p


if __name__ == "__main__":
    args = parser().parse_args()
    (prepare if args.command == "prepare" else train)(args)
