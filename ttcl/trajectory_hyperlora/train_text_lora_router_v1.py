"""Text-conditioned risk gate for a frozen trajectory-generated LoRA.

Uses paired environment rewards from reviewed train games. The feature
extractor parses neither ALFWorld family labels nor its action vocabulary;
it can consume task/trajectory text from another environment unchanged.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

import torch

from ttcl.trajectory_hyperlora.alfworld_same_game_retry import file_hash


WIDTH = 1024
TOKEN = re.compile(r"[a-z0-9]+")
LAMBDAS = (0.1, 1.0, 10.0, 100.0)
THRESHOLDS = (0.0, 0.05, 0.1)


def visible_task(observation: str) -> str:
    marker = "Your task is to:"
    if marker in observation:
        return observation.split(marker, 1)[1].split("\n", 1)[0].strip()
    return observation[:512]


def words(text: str) -> list[str]:
    return TOKEN.findall(text.lower())


def text_features(target_observation: str,
                  source_records: list[dict]) -> torch.Tensor:
    """Generic hashed word and word-pair features; no named task slots."""
    if not source_records:
        raise ValueError("Missing trajectory source")
    target = words(visible_task(target_observation))
    source = words(visible_task(source_records[0]["observation"]))
    if not target or not source:
        raise ValueError("Empty task or source text")
    values = torch.zeros(WIDTH + 2, dtype=torch.float64)
    for prefix, tokens, weight in (("target", target, 1.0),
                                   ("source", source, .5)):
        for term in set(tokens):
            key = (prefix + ":" + term).encode()
            index = int.from_bytes(hashlib.sha256(key).digest()[:4], "big") % WIDTH
            values[index] += weight
        for left, right in set(zip(tokens, tokens[1:])):
            key = (prefix + ":" + left + "_" + right).encode()
            index = int.from_bytes(hashlib.sha256(key).digest()[:4], "big") % WIDTH
            values[index] += weight
    values[:WIDTH] /= values[:WIDTH].norm().clamp_min(1.0)
    values[WIDTH] = len(set(target) & set(source)) / len(set(target) | set(source))
    values[WIDTH + 1] = 1.0
    return values


def fit(x: torch.Tensor, y: torch.Tensor, ridge: float) -> torch.Tensor:
    if x.ndim != 2 or y.shape != (x.shape[0],) or ridge <= 0:
        raise ValueError("Invalid ridge training matrix")
    gram = x @ x.T
    return x.T @ torch.linalg.solve(gram + ridge * torch.eye(len(y),
                                                              dtype=x.dtype), y)


def score(rows: list[dict], predictions: torch.Tensor,
          threshold: float) -> dict:
    if len(rows) != len(predictions):
        raise ValueError("Prediction/row count mismatch")
    use = (predictions > threshold).tolist()
    return {"n": len(rows), "mounted": sum(use),
            "base": sum(row["arms"]["base"]["reward"] for row in rows),
            "lora": sum(row["arms"]["lora"]["reward"] for row in rows),
            "gated": sum(row["arms"]["lora" if enabled else "base"]["reward"]
                         for row, enabled in zip(rows, use, strict=True)),
            "decisions": [{"game": row["game"], "mount": enabled,
                           "prediction": float(predictions[index]),
                           "base": row["arms"]["base"]["reward"],
                           "lora": row["arms"]["lora"]["reward"]}
                          for index, (row, enabled) in enumerate(
                              zip(rows, use, strict=True))]}


def select_hyperparameters(x, y, rows):
    # Four prespecified folds on ordered reviewed train rows. Each held-out
    # game's reward is used only to compare candidate hyperparameters; the
    # separate development split is never touched by selection.
    trials = []
    for ridge in LAMBDAS:
        heldout = torch.zeros(len(rows), dtype=torch.float64)
        for fold in range(4):
            training = [i for i in range(len(rows)) if i % 4 != fold]
            validation = [i for i in range(len(rows)) if i % 4 == fold]
            weights = fit(x[training], y[training], ridge)
            heldout[validation] = x[validation] @ weights
        for threshold in THRESHOLDS:
            result = score(rows, heldout, threshold)
            trials.append({"ridge": ridge, "threshold": threshold,
                           "gated": result["gated"],
                           "mounted": result["mounted"]})
    selected = max(trials, key=lambda row: (row["gated"],
        -row["mounted"], row["ridge"], row["threshold"]))
    return selected, trials


def reviewed_text(args, pairs):
    review = json.loads(args.reviewed_pairs.read_text())
    if len(review["pairs"]) != 72:
        raise ValueError("Router reviewed target count changed")
    lookup = {row["target_game"]: row for row in review["pairs"]}
    if len(lookup) != 72:
        raise ValueError("Duplicate router target")
    for row in pairs:
        source = lookup.get(row["game"])
        if (source is None or source["input_content_sha256"] !=
                row["input_content_sha256"] or
                source["source_records_sha256"] !=
                row["source_records_sha256"] or
                source["split"] != row["split"]):
            raise ValueError("Paired reward/source review binding mismatch")
        yield source


def fit_router(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    review_sha = file_hash(args.reviewed_pairs)
    expected = json.loads(args.reviewed_pairs.read_text())
    checkpoint_sha = expected["checkpoint_sha256"]
    by_game = {}
    part_sha = {}
    for part in args.parts:
        raw = json.loads(part.read_text())
        if (raw["reviewed_pairs_sha256"] != review_sha or
                raw["checkpoint_sha256"] != checkpoint_sha or
                raw["max_steps"] != 50 or raw["max_new_tokens"] != 64 or
                raw["actor_history_turns"] != 2 or raw["failures"]):
            raise ValueError("Collection failure or protocol mismatch")
        part_sha[str(part)] = file_hash(part)
        for row in raw["games"]:
            if row["game"] in by_game or row["arms"]["base"]["status"] != "complete" or \
                    row["arms"]["lora"]["status"] != "complete" or \
                    row["arms"]["base"]["initial_observation"] != \
                    row["arms"]["lora"]["initial_observation"]:
                raise ValueError("Duplicate or invalid collected reward pair")
            by_game[row["game"]] = row
    if set(by_game) != {row["target_game"] for row in expected["pairs"]}:
        raise ValueError("Reward collection incomplete or has extra games")
    ordered = [by_game[row["target_game"]] for row in expected["pairs"]]
    source_rows = list(reviewed_text(args, ordered))
    x = torch.stack([text_features(row["arms"]["base"]["initial_observation"],
                                   source["source_records"])
                     for row, source in zip(ordered, source_rows, strict=True)])
    train_index = [i for i, row in enumerate(ordered) if row["split"] == "train"]
    dev_index = [i for i, row in enumerate(ordered) if row["split"] == "dev"]
    if len(train_index) != 48 or len(dev_index) != 24:
        raise ValueError("Expected 48/24 train/development separation")
    y = torch.tensor([row["arms"]["lora"]["reward"] -
                      row["arms"]["base"]["reward"] for row in ordered],
                     dtype=torch.float64)
    choice, trials = select_hyperparameters(x[train_index], y[train_index],
                                           [ordered[i] for i in train_index])
    weights = fit(x[train_index], y[train_index], choice["ridge"])
    train_score = score([ordered[i] for i in train_index],
                        x[train_index] @ weights, choice["threshold"])
    dev_score = score([ordered[i] for i in dev_index],
                      x[dev_index] @ weights, choice["threshold"])
    result = {"protocol": "Hashed raw task/source text ridge uplift gate; 48 reviewed train reward pairs; four-fold train-only selection; separate 24 reviewed development pairs; same frozen LoRA checkpoint and 50-step actor as official pilot; no ALFWorld family feature",
        "reviewed_pairs_sha256": review_sha,
        "checkpoint_sha256": checkpoint_sha,
        "part_sha256": part_sha,
        "feature_width": WIDTH + 2, "selected": choice,
        "train_cv": trials,
        "weights": weights.tolist(),
        "train": train_score, "dev": dev_score}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"selected": choice,
        "train": {key: train_score[key] for key in ("n","mounted","base","lora","gated")},
        "dev": {key: dev_score[key] for key in ("n","mounted","base","lora","gated")}}), flush=True)
    return result


def evaluate_official(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    model = json.loads(args.router.read_text())
    official = json.loads(args.official_result.read_text())
    review = json.loads(args.official_review.read_text())
    if (official["target_review_sha256"] != file_hash(args.official_review) or
            official["checkpoint_sha256"] != model["checkpoint_sha256"] or
            len(official["games"]) != 134 or
            official["summary"]["overall"]["n"] != 134 or
            any(row["arms"][arm]["status"] != "complete"
                for row in official["games"] for arm in ("base", "lora"))):
        raise ValueError("Official result/review/checkpoint mismatch")
    sources = {row["game"]: row for row in review["targets"]}
    if len(sources) != 134:
        raise ValueError("Official source review incomplete")
    rows = []
    features = []
    for row in official["games"]:
        source = sources.get(row["game"])
        if (source is None or source["input_content_sha256"] !=
                row["target_input_content_sha256"] or
                source["source_records_sha256"] !=
                row["source_records_sha256"]):
            raise ValueError("Official source-target binding mismatch")
        rows.append(row)
        features.append(text_features(row["arms"]["base"]["initial_observation"],
                                      source["source_records"]))
    predictions = torch.stack(features) @ torch.tensor(model["weights"],
                                                    dtype=torch.float64)
    summary = score(rows, predictions, model["selected"]["threshold"])
    result = {"protocol": "Read-only recomposition of existing 134 paired ALFWorld rollouts with frozen text-trained LoRA-use gate; no target reward used in fitting; same actor rollouts and official won; historical MemRL has different actor/protocol",
        "router_sha256": file_hash(args.router),
        "official_result_sha256": file_hash(args.official_result),
        "official_review_sha256": file_hash(args.official_review),
        "summary": summary}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: summary[k] for k in ("n","mounted","base","lora","gated")}),
          flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("fit", "evaluate"))
    parser.add_argument("--reviewed-pairs", type=Path, default=Path(
        "data/annotations/alf_reward_router_train72_reviewed_20261007.json"))
    parser.add_argument("--parts", nargs="*", type=Path)
    parser.add_argument("--router", type=Path)
    parser.add_argument("--official-result", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_memrl134_simple40_reward_lora_audited_20261006.json"))
    parser.add_argument("--official-review", type=Path, default=Path(
        "data/annotations/alf_memrl134_hyperlora_train_sources_reviewed_20261006.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "fit":
        if not args.parts:
            parser.error("Fit requires collection parts")
        fit_router(args)
    else:
        if args.router is None:
            parser.error("Evaluate requires a frozen router")
        evaluate_official(args)


if __name__ == "__main__":
    main()
