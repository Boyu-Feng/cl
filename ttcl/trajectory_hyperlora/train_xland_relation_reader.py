"""Train a tiny future-action reader over source transition relations.

This is a diagnostic upper bound for representation design, not HyperLoRA:
tile identity is compared exactly, while only the shared action readout is
learned. No named object/rule slots are assigned to model parameters.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


ACTION_IDS = (0, 3)


def counts(state: dict) -> Counter:
    observed = [tuple(tile) for row in state["observation"] for tile in row]
    pocket = tuple(state["pocket"])
    if pocket[0] != 0:
        observed.append(pocket)
    return Counter(observed)


def relation_scores(model_input: dict) -> tuple[float, float]:
    """Query a variable-size observation-change/action relation by goal tile."""
    goal = tuple(model_input["goal"])
    evidence = Counter()
    for trial in model_input["source_trials"].values():
        for step in trial:
            before = counts(step["state"])
            after = counts(step["next_state"])
            evidence[step["action"]] += after[goal] - before[goal]
    return tuple(float(evidence[action]) for action in ACTION_IDS)


class SharedReadout(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.1))
        self.bias = nn.Parameter(torch.zeros(2))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.scale * features + self.bias


def examples(rows: list[dict]) -> tuple[torch.Tensor, torch.Tensor]:
    features = []
    labels = []
    for item in rows:
        for name in ("hold", "near"):
            arm = item["arms"][name]
            model_input = arm["model_input"]
            expected = {"source_trials": model_input["source_trials"],
                        "target_initial_state": model_input["target_initial_state"],
                        "goal": model_input["goal"]}
            digest = hashlib.sha256(json.dumps(
                expected, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            if not arm["reviewed_target"] or digest != arm["input_sha256"]:
                raise ValueError("Unreviewed or mismatched trajectory input")
            features.append(relation_scores(model_input))
            labels.append(ACTION_IDS.index(arm["target_action"]))
    return torch.tensor(features, dtype=torch.float32), torch.tensor(labels)


def score(model: SharedReadout, feature: torch.Tensor,
          label: torch.Tensor) -> dict:
    with torch.no_grad():
        own = model(feature).argmax(-1)
        no_source = model(torch.zeros_like(feature)).argmax(-1)
        # Adjacent rows are the two hidden-rule arms of the same public target.
        wrong = model(feature.reshape(-1, 2, 2).flip(1).reshape(-1, 2)).argmax(-1)
    return {"n": len(label),
            "correct_source": int((own == label).sum()),
            "no_source": int((no_source == label).sum()),
            "wrong_source": int((wrong == label).sum()),
            "cue_sensitive_pairs": int((own.reshape(-1, 2)[:, 0] !=
                                         own.reshape(-1, 2)[:, 1]).sum())}


def run(args: argparse.Namespace) -> dict:
    if args.output.exists() or args.checkpoint.exists():
        raise FileExistsError("Fresh result and checkpoint paths required")
    raw = args.annotations.read_bytes()
    data = json.loads(raw)
    train_x, train_y = examples(data["split"]["train"])
    dev_x, dev_y = examples(data["split"]["dev"])
    test_x, test_y = examples(data["split"]["test"])
    torch.manual_seed(args.seed)
    model = SharedReadout()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    losses = []
    for _ in range(args.steps):
        logits = model(train_x)
        loss = F.cross_entropy(logits, train_y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    result = {"protocol": "Exact tile-change/action relation with learned shared scalar readout; train-only future-action labels; no LoRA, no step labels or RL",
              "annotations_sha256": hashlib.sha256(raw).hexdigest(),
              "seed": args.seed, "steps": args.steps, "lr": args.lr,
              "learned_scale": float(model.scale.detach()),
              "learned_bias": model.bias.detach().tolist(),
              "final_train_loss": losses[-1],
              "feature_distributions": {split: {
                  "unique": sorted({tuple(row) for row in features.tolist()}),
                  "n": len(features)}
                  for split, features in (("train", train_x), ("dev", dev_x),
                                          ("test", test_x))},
              "train": score(model, train_x, train_y),
              "dev": score(model, dev_x, dev_y),
              "test": score(model, test_x, test_y)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    torch.save({"state_dict": model.state_dict(),
                "annotations_sha256": result["annotations_sha256"],
                "seed": args.seed}, args.checkpoint)
    print(json.dumps({split: result[split] for split in ("train", "dev", "test")}),
          flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, default=Path(
        "data/annotations/xland_rule_pairs_reviewed_v1_20261005.json"))
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=.05)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    if args.steps < 1 or args.lr <= 0:
        parser.error("Positive training budget and learning rate required")
    run(args)


if __name__ == "__main__":
    main()
