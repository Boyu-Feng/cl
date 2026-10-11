"""Fit trainable ordered-event utility on reviewed future-reward pairs.

The LoRA residual head is initialized here and trained later by actual
environment rollouts. This first stage tests whether source utility is
predictable at all when whole target sequences are left out.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ttcl.trajectory_hyperlora.alfworld_online_from_empty import digest, file_hash
from ttcl.trajectory_hyperlora.my_online_update_v28_model import OnlineUpdateModel
from ttcl.trajectory_hyperlora.train_alf_future_utility_selector_v1 import (
    best_static, target_reward_matrix,
)


def checked_dataset(args):
    features = [torch.load(path, map_location="cpu", weights_only=True)
                for path in args.features]
    reports = [json.loads(path.read_text()) for path in args.matrices]
    audits = [json.loads(path.read_text()) for path in args.audits]
    reviews = [json.loads(path.read_text()) for path in args.reviews]
    if len(features) != len(reports) or len(reports) != 2:
        raise ValueError("Expected two reviewed training matrices")
    checkpoint = audits[0]["checkpoint_sha256"]
    for i, (encoded, report, audit, review) in enumerate(zip(
            features, reports, audits, reviews, strict=True)):
        if (audit["review_sha256"] != file_hash(args.reviews[i]) or
                audit["raw_report_sha256"] != file_hash(args.matrices[i]) or
                audit["checkpoint_sha256"] != checkpoint or
                encoded["review_sha256"] != audit["review_sha256"] or
                encoded["checkpoint_sha256"] != checkpoint or
                report["review_sha256"] != audit["review_sha256"] or
                report["checkpoint_sha256"] != checkpoint or
                report["failures"] or len(report["base"]) != 18 or
                len(report["pairs"]) != 144 or
                len(review["targets"]) != 18 or
                digest([row["episode"]["initial_observation"]
                        for row in report["base"]]) !=
                encoded["base_observations_sha256"]):
            raise ValueError(f"Changed training lineage at matrix {i}")
    for key in ("global", "events", "deltas", "event_counts"):
        if not torch.equal(features[0][key], features[1][key]):
            raise ValueError("Source event representations changed")
    source = {key: features[0][key] for key in
              ("global", "events", "deltas", "event_counts")}
    queries = torch.cat([encoded["targets"] for encoded in features])
    rewards = torch.cat([target_reward_matrix(report, 8, 18)
                         for report in reports]).float()
    base = torch.tensor([row["episode"]["reward"] for report in reports
                         for row in report["base"]], dtype=torch.float32)
    groups = [row["sequence_index"] for review in reviews
              for row in review["targets"]]
    if len(groups) != 36 or len(set(groups)) != 19:
        raise ValueError("Changed sequence-grouped training targets")
    bindings = {"checkpoint_sha256": checkpoint,
        "feature_sha256": [file_hash(path) for path in args.features],
        "matrix_sha256": [file_hash(path) for path in args.matrices],
        "audit_sha256": [file_hash(path) for path in args.audits],
        "review_sha256": [file_hash(path) for path in args.reviews],
        "target_group_sha256": digest(groups)}
    return source, queries, rewards, base, groups, bindings


def fit(source, queries, rewards, base, indices, *, seed, epochs, lr):
    torch.manual_seed(seed)
    model = OnlineUpdateModel(queries.shape[-1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                  weight_decay=.01)
    label = rewards[indices] - base[indices, None]
    weights = 1 + 2 * label.abs()
    for _ in range(epochs):
        predicted, _ = model(source["global"], source["events"],
                             source["deltas"], source["event_counts"],
                             queries[indices])
        loss = (weights * (predicted - label).square()).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    return model, float(loss.detach())


def train(args):
    if args.output.exists() or args.save_model.exists():
        raise FileExistsError("Use fresh report and model paths")
    torch.set_num_threads(2)
    source, queries, rewards, base, groups, bindings = checked_dataset(args)
    unique = sorted(set(groups))
    rows = []
    for fold in range(5):
        held = [i for i, group in enumerate(groups)
                if unique.index(group) % 5 == fold]
        train_ids = [i for i in range(len(groups)) if i not in held]
        model, loss = fit(source, queries, rewards, base, train_ids,
            seed=args.seed + fold, epochs=args.epochs, lr=args.lr)
        with torch.no_grad():
            scores, _ = model(source["global"], source["events"],
                source["deltas"], source["event_counts"], queries[held])
        static = best_static(rewards[train_ids], base[train_ids])
        for position, target_id in enumerate(held):
            score = scores[position]
            selected = int(score.argmax()) if score.max() > 0 else -1
            rows.append({"target_id": target_id, "fold": fold,
                "sequence_id": groups[target_id],
                "selected_source": selected,
                "selected_reward": float(base[target_id] if selected < 0
                                         else rewards[target_id, selected]),
                "static_source": static,
                "static_reward": float(base[target_id] if static < 0
                                       else rewards[target_id, static]),
                "base_reward": float(base[target_id]),
                "oracle_reward": float(max(base[target_id],
                                           rewards[target_id].max())),
                "scores": score.tolist(), "fit_loss": loss})
    rows.sort(key=lambda x: x["target_id"])
    model, final_loss = fit(source, queries, rewards, base,
        list(range(len(groups))), seed=args.seed, epochs=args.epochs,
        lr=args.lr)
    args.save_model.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "width": queries.shape[-1],
        "hidden": model.hidden, "code_dim": model.code_dim,
        "epochs": args.epochs, "seed": args.seed, "lr": args.lr,
        "bindings": bindings, "stage": "utility_pretrain_only"},
        args.save_model)
    summary = {"n": len(rows),
        "selected": sum(row["selected_reward"] for row in rows),
        "static_fold": sum(row["static_reward"] for row in rows),
        "base": sum(row["base_reward"] for row in rows),
        "oracle": sum(row["oracle_reward"] for row in rows),
        "abstentions": sum(row["selected_source"] < 0 for row in rows)}
    result = {"protocol": "我的在线更新 v28: trainable ordered-feedback encoder and shared future-return utility; five sequence-group folds, fixed hyperparameters; LoRA residual not yet reward-trained",
        "bindings": bindings, "epochs": args.epochs, "seed": args.seed,
        "lr": args.lr, "cross_validation_unit": "plan_sequence",
        "summary": summary, "final_fit_loss": final_loss,
        "rows": rows, "model_sha256": file_hash(args.save_model)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False,
                                        indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, nargs=2, default=[Path(
        "results/trajectory_hyperlora/alf_future_reward_train8x18_features_20261007.pt"),
        Path("results/trajectory_hyperlora/alf_future_reward_additional8x18_features_v3_20261007.pt")])
    parser.add_argument("--matrices", type=Path, nargs=2, default=[Path(
        "results/trajectory_hyperlora/alf_own_success_future_reward_8x18_20261007.json"),
        Path("results/trajectory_hyperlora/alf_own_success_future_reward_additional8x18_v3_merged_20261007.json")])
    parser.add_argument("--audits", type=Path, nargs=2, default=[Path(
        "results/trajectory_hyperlora/alf_own_success_future_reward_8x18_audited_20261007.json"),
        Path("results/trajectory_hyperlora/alf_own_success_future_reward_additional8x18_v3_merged_audited_20261007.json")])
    parser.add_argument("--reviews", type=Path, nargs=2, default=[Path(
        "data/annotations/alf_own_success_future_reward_8x18_reviewed_20261007.json"),
        Path("data/annotations/alf_own_success_future_reward_additional8x18_v3_reviewed_20261007.json")])
    parser.add_argument("--output", type=Path, default=Path(
        "results/trajectory_hyperlora/my_online_update_v28_utility_train_20261009.json"))
    parser.add_argument("--save-model", type=Path, default=Path(
        "results/trajectory_hyperlora/my_online_update_v28_utility_train_20261009.pt"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=311)
    train(parser.parse_args())
