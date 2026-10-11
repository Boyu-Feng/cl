"""Train a matrix-state updater on reviewed ALFWorld train trajectories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ttcl.trajectory_hyperlora.alfworld_online_from_empty import digest, file_hash, records_from_episode
from ttcl.trajectory_hyperlora.alfworld_success_gated_factors_v27 import bounded_source_text, episode_factor
from ttcl.trajectory_hyperlora.alfworld_zero_shot_probe import load_agent
from ttcl.trajectory_hyperlora.contextual_alf_source import contextual_text_fields, task_context_text
from ttcl.trajectory_hyperlora.joint_b_update_v33 import JointBUpdater, features


def checked_sources(args):
    result = []
    for name, report_path, audit_path, review_path in zip(
            ("train_seq0_6", "train_seq6_6"), args.reports, args.audits,
            args.source_reviews, strict=True):
        report = json.loads(report_path.read_text())
        audit = json.loads(audit_path.read_text())
        review = json.loads(review_path.read_text())
        if (report["sequence"] != name or len(report["games"]) != 18 or
                report["failures"] or audit["audit"] != "passed" or
                audit["report_sha256"] != file_hash(report_path) or
                report["review_sha256"] != file_hash(review_path) or
                report["checkpoint_sha256"] != file_hash(args.checkpoint) or
                len(review["targets"]) != 18):
            raise ValueError(f"Changed reviewed train source: {name}")
        for row, target in zip(report["games"], review["targets"], strict=True):
            episode = row["online"]
            if (row["game"] != target["game"] or
                    "/train/" not in "/" + row["game"] or
                    target["game_sha256"] != file_hash(args.data_root / row["game"]) or
                    episode["status"] != "complete"):
                raise ValueError("Changed reviewed train trajectory")
        result.append(report)
    return result


def bindings(args):
    reports = checked_sources(args)
    return {"reports_sha256": [file_hash(x) for x in args.reports],
        "audits_sha256": [file_hash(x) for x in args.audits],
        "source_reviews_sha256": [file_hash(x) for x in args.source_reviews],
        "checkpoint_sha256": file_hash(args.checkpoint),
        "model_config_sha256": file_hash(args.model / "config.json"),
        "trainer_sha256": file_hash(Path(__file__)),
        "updater_sha256": file_hash(Path(__file__).with_name("joint_b_update_v33.py")),
        "episode_sha256": [[digest(row["online"]) for row in report["games"]]
                           for report in reports],
        "training_split": "official train only", "context_tokens": 2048}


def prepare(args):
    if args.review.exists():
        raise FileExistsError(args.review)
    value = bindings(args)
    args.review.parent.mkdir(parents=True, exist_ok=True)
    args.review.write_text(json.dumps({"protocol": "Reviewed own train trajectories to teach U(B_old, new trajectory) a full-prefix factor; no valid_unseen data", **value}, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"sequences": 2, "episodes": 36}))


def pooled_factor(agent, tokenizer, records, initial, device):
    text, _, _ = bounded_source_text(tokenizer, records, 2048)
    source = contextual_text_fields(agent, tokenizer, text, device, 2048,
                                    pooling="both")
    target = contextual_text_fields(agent, tokenizer,
        task_context_text(initial, initial), device, 2048, pooling="both")
    with torch.no_grad():
        agent.set_source(source, target_fields=target)
        value = [layer.b.detach().cpu().float().clone() for layer in agent.adapters]
        agent.set_source(None)
    return value


def train(args):
    if args.model_output.exists() or args.report_output.exists():
        raise FileExistsError("Use new joint updater output paths")
    review = json.loads(args.review.read_text())
    if any(review[key] != val for key, val in bindings(args).items()):
        raise ValueError("Changed joint updater training review")
    reports = checked_sources(args)
    agent, tokenizer = load_agent(args.model, args.checkpoint,
                                  args.device, args.gpu_fraction)
    data = []
    for sequence, report in enumerate(reports):
        prefix = []
        previous = None
        for index, row in enumerate(report["games"]):
            episode = row["online"]
            current_records = records_from_episode(episode)
            new, _, _, _ = episode_factor(agent, tokenizer, episode,
                argparse.Namespace(device=args.device, context_tokens=2048))
            prefix.extend(current_records)
            teacher = pooled_factor(agent, tokenizer, prefix,
                episode["initial_observation"], args.device)
            if previous is not None:
                x = features(previous, new, episode["reward"],
                             episode["steps"], index)
                projected = []
                target_error = []
                for old_layer, new_layer, teacher_layer in zip(
                        previous, new, teacher, strict=True):
                    direction = new_layer - old_layer
                    numerator = ((teacher_layer - old_layer) * direction).sum()
                    denominator = direction.square().sum().clamp_min(1e-12)
                    gate = (numerator / denominator).clamp(0, 1)
                    projected.append(gate)
                    fitted = old_layer + gate * direction
                    target_error.append(float((fitted - teacher_layer).norm() /
                                              teacher_layer.norm().clamp_min(1e-8)))
                data.append({"sequence": sequence, "index": index,
                    "episode_sha256": digest(episode), "x": x,
                    "target": torch.stack(projected),
                    "projection_error": target_error})
            previous = teacher
    xs = torch.stack([x["x"] for x in data])
    ys = torch.stack([x["target"] for x in data])
    torch.manual_seed(args.seed)
    model = JointBUpdater(xs.shape[1], ys.shape[1], hidden=args.hidden)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=.01)
    train_ids = [i for i, row in enumerate(data) if row["sequence"] == 0]
    dev_ids = [i for i, row in enumerate(data) if row["sequence"] == 1]
    for _ in range(args.epochs):
        loss = (model(xs[train_ids]) - ys[train_ids]).square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        train_loss = float((model(xs[train_ids]) - ys[train_ids]).square().mean())
        dev_loss = float((model(xs[dev_ids]) - ys[dev_ids]).square().mean())
        mean_gate = torch.tensor([1 / (row["index"] + 1) for row in data])
        mean_dev_loss = float((mean_gate[dev_ids, None] - ys[dev_ids]).square().mean())
    # Freeze the same recipe on all reviewed training-domain trajectories.
    torch.manual_seed(args.seed)
    final = JointBUpdater(xs.shape[1], ys.shape[1], hidden=args.hidden)
    optimizer = torch.optim.AdamW(final.parameters(), lr=args.lr,
                                  weight_decay=.01)
    for _ in range(args.epochs):
        loss = (final(xs) - ys).square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": final.state_dict(), "feature_dim": xs.shape[1],
        "layers": ys.shape[1], "hidden": args.hidden,
        "review_sha256": file_hash(args.review),
        "seed": args.seed, "epochs": args.epochs, "lr": args.lr,
        "objective": "full-prefix factor imitation, projected to old/new segment"},
        args.model_output)
    result = {"protocol": review["protocol"],
        "review_sha256": file_hash(args.review),
        "model_sha256": file_hash(args.model_output),
        "train_pairs": len(train_ids), "dev_pairs": len(dev_ids),
        "train_gate_mse": train_loss, "dev_gate_mse": dev_loss,
        "fixed_mean_dev_gate_mse": mean_dev_loss,
        "mean_projection_relative_error": sum(sum(row["projection_error"])/len(row["projection_error"]) for row in data)/len(data),
        "rows": [{k: v for k, v in row.items() if k not in ("x", "target")}
                 for row in data]}
    args.report_output.write_text(json.dumps(result, ensure_ascii=False,
                                              indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "train"))
    root = Path("results/trajectory_hyperlora")
    notes = Path("data/annotations")
    parser.add_argument("--reports", type=Path, nargs=2, default=[root /
        f"alfworld_success_gated_factors_v27_{name}_20261009.json"
        for name in ("train_seq0_6", "train_seq6_6")])
    parser.add_argument("--audits", type=Path, nargs=2, default=[root /
        f"alfworld_success_gated_factors_v27_{name}_audit_20261009.json"
        for name in ("train_seq0_6", "train_seq6_6")])
    parser.add_argument("--source-reviews", type=Path, nargs=2, default=[notes /
        f"alfworld_success_gated_factors_v27_{name}_20261009.json"
        for name in ("train_seq0_6", "train_seq6_6")])
    parser.add_argument("--review", type=Path, default=notes /
        "alfworld_joint_b_update_v33_train_reviewed_20261009.json")
    parser.add_argument("--model-output", type=Path, default=root /
        "alfworld_joint_b_update_v33_train_20261009.pt")
    parser.add_argument("--report-output", type=Path, default=root /
        "alfworld_joint_b_update_v33_train_20261009.json")
    parser.add_argument("--model", type=Path, default=Path(
        "current_work/delta-Mem/model/Qwen3-4B-Instruct-2507"))
    parser.add_argument("--checkpoint", type=Path, default=root /
        "alf_sibling_simple40_onpolicy600_20261006.pt")
    parser.add_argument("--data-root", type=Path, default=Path(
        "ttcl/data/alfworld_delta"))
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--gpu-fraction", type=float, default=.65)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20261009)
    options = parser.parse_args()
    (prepare if options.command == "prepare" else train)(options)
