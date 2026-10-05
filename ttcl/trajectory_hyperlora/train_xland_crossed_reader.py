"""Diagnostic query-conditioned relation reader on two-object histories."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from ttcl.trajectory_hyperlora.train_xland_relation_reader import (
    ACTION_IDS, SharedReadout, relation_scores,
)


def examples(items: list[dict]) -> tuple[torch.Tensor, torch.Tensor]:
    features, labels = [], []
    for item in items:
        for query_index in (0, 1):
            for variant in ("hold_near", "near_hold"):
                query = item["arms"][variant]["queries"][query_index]
                model_input = query["model_input"]
                content_sha = hashlib.sha256(json.dumps(
                    model_input, sort_keys=True,
                    separators=(",", ":")).encode()).hexdigest()
                if not query["reviewed_target"] or content_sha != query["input_sha256"]:
                    raise ValueError("Unreviewed or mismatched target query")
                feature_input = {
                    "source_trials": {str(i): episode["steps"] for i, episode in
                                      enumerate(model_input["source_episodes"])},
                    "goal": model_input["goal"],
                }
                features.append(relation_scores(feature_input))
                labels.append(ACTION_IDS.index(query["target_action"]))
    return torch.tensor(features, dtype=torch.float32), torch.tensor(labels)


def score(model, features, targets) -> dict:
    with torch.no_grad():
        own = model(features).argmax(-1)
        no_source = model(torch.zeros_like(features)).argmax(-1)
        wrong = model(features.reshape(-1, 2, 2).flip(1).reshape(-1, 2)).argmax(-1)
        grouped = own.reshape(-1, 4)
    return {"n": len(targets),
            "correct_source": int((own == targets).sum()),
            "no_source": int((no_source == targets).sum()),
            "wrong_source": int((wrong == targets).sum()),
            "source_swap_action_changes": int((own != wrong).sum()),
            "within_history_distinct_action_count": int(
                (grouped[:, 0] != grouped[:, 2]).sum() +
                (grouped[:, 1] != grouped[:, 3]).sum()),
            "within_history_count": 2 * len(grouped)}


def run(args: argparse.Namespace) -> dict:
    if args.output.exists() or args.checkpoint.exists():
        raise FileExistsError("Fresh outputs required")
    raw = args.annotations.read_bytes()
    data = json.loads(raw)
    split = {name: examples(data["split"][name])
             for name in ("train", "dev", "test")}
    torch.manual_seed(args.seed)
    model = SharedReadout()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    for _ in range(args.steps):
        features, targets = split["train"]
        loss = F.cross_entropy(model(features), targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    result = {"protocol": "Two-object history; exact tile-delta/action relation with learned scalar reader; target has new distractor; train-only future-action labels; no LoRA or RL",
              "annotations_sha256": hashlib.sha256(raw).hexdigest(),
              "seed": args.seed, "steps": args.steps, "lr": args.lr,
              "learned_scale": float(model.scale.detach()),
              "final_train_loss": float(loss.detach()),
              "feature_distributions": {name: sorted({tuple(row) for row in
                                                     features.tolist()})
                                        for name, (features, _) in split.items()},
              **{name: score(model, *split[name])
                 for name in ("train", "dev", "test")}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    torch.save({"state_dict": model.state_dict(),
                "annotations_sha256": result["annotations_sha256"],
                "seed": args.seed}, args.checkpoint)
    print(json.dumps({name: result[name] for name in ("train", "dev", "test")}),
          flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, default=Path(
        "data/annotations/xland_crossed_reviewed_v1_20261005.json"))
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=.05)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    if args.steps < 1 or args.lr <= 0:
        parser.error("Positive step budget and learning rate required")
    run(args)


if __name__ == "__main__":
    main()
