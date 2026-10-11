"""Retrieve prior own-success trajectories by task text before acting."""

from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path
import re

from ttcl.trajectory_hyperlora.alfworld_online_from_empty import digest, file_hash
from ttcl.trajectory_hyperlora.alfworld_persistent_failure_delta_v36 import save
from ttcl.trajectory_hyperlora.alfworld_zero_shot_probe import load_agent, run_episode
from ttcl.trajectory_hyperlora.contextual_alf_source import contextual_source_fields
from ttcl.trajectory_hyperlora.audit_my_online_update_v28_reward import replay
from ttcl.trajectory_hyperlora.alfworld_bootstrap_own_success_v43 import (
    checked as bootstrap_checked, parser,
)
from ttcl.trajectory_hyperlora.evaluate_alfworld_bootstrap_own_success_v43 import source_for


STOP = {"the", "a", "an", "to", "in", "on", "with", "and", "of", "at", "is",
        "your", "task", "you", "have", "put", "place", "find", "some"}


def goal_tokens(observation):
    goal = observation.split("Your task is to:", 1)[-1]
    return Counter(word for word in re.findall(r"[a-z]+", goal.lower())
                   if word not in STOP)


def choose_source(target, successes):
    candidates = [(i, item) for i, item in enumerate(successes)
                  if item["family"] == target["family"]]
    if not candidates:
        raise ValueError("No success source in family")
    target_tokens = goal_tokens(target["arms"]["base"]["initial_observation"])
    source_tokens = [(i, goal_tokens(item["actions"][0]["current_observation"]))
                     for i, item in candidates]
    documents = [target_tokens] + [tokens for _, tokens in source_tokens]
    df = Counter(word for doc in documents for word in doc)
    idf = {word: math.log((len(documents) + 1) / (count + 1)) + 1
           for word, count in df.items()}
    def weighted(doc):
        return {word: count * idf[word] for word, count in doc.items()}
    def cosine(a, b):
        left, right = weighted(a), weighted(b)
        dot = sum(value * right.get(word, 0.) for word, value in left.items())
        left_norm = math.sqrt(sum(value * value for value in left.values()))
        right_norm = math.sqrt(sum(value * value for value in right.values()))
        return dot / (left_norm * right_norm) if left_norm and right_norm else 0.
    return max(source_tokens, key=lambda pair: (
        cosine(target_tokens, pair[1]),
        digest(["v50_tie", target["game"], successes[pair[0]]["game"]])))[0]


def bound(args):
    _, successes, suffix, train_fixed, _, _ = bootstrap_checked(args)
    train = json.loads(args.train_report.read_text())
    parent = json.loads(args.parent_eval_review.read_text())
    if (train.get("review_sha256") != file_hash(args.review) or
            train.get("hyper_model_sha256") != file_hash(args.hyper_model) or
            any(train.get(k) != v for k, v in train_fixed.items()) or
            len(parent.get("targets", [])) != 120):
        raise ValueError("Changed bootstrap training or suffix source")
    fixed = {"method": "v50 prior own-success task-goal TF-IDF retrieval",
        "bootstrap_review_sha256": file_hash(args.review),
        "train_report_sha256": file_hash(args.train_report),
        "hyper_model_sha256": file_hash(args.hyper_model),
        "parent_eval_review_sha256": file_hash(args.parent_eval_review),
        "runner_sha256": file_hash(Path(__file__)),
        "actor_sha256": file_hash(Path(__file__).with_name("alfworld_zero_shot_probe.py")),
        "max_steps": 30, "max_new_tokens": 64,
        "constrain_actions": True, "actor_history_turns": 2,
        "loop_guard_max": 2}
    targets = []
    for i, (row, old) in enumerate(zip(suffix, parent["targets"], strict=True)):
        if (row["game"] != old["game"] or
                file_hash(args.data_root / row["game"]) != old["game_sha256"]):
            raise ValueError(f"Changed suffix task {i}")
        selected = choose_source(row, successes)
        original = source_for(row, successes)
        content = {"index": i, "game": row["game"],
            "game_sha256": old["game_sha256"],
            "parent_input_content_sha256": old["input_content_sha256"],
            "goal_sha256": digest(row["arms"]["base"]["initial_observation"]),
            "retrieved_source_index": selected,
            "retrieved_source_records_sha256": digest(successes[selected]["source_records"]),
            "original_source_index": original,
            "runner_sha256": fixed["runner_sha256"],
            "hyper_model_sha256": fixed["hyper_model_sha256"]}
        targets.append({**content, "input_content_sha256": digest(content),
                        "reviewed_target": True})
    return successes, suffix, fixed, targets


def prepare(args):
    if args.eval_review.exists():
        raise FileExistsError(args.eval_review)
    _, _, fixed, targets = bound(args)
    save(args.eval_review, {"protocol": "Official train-stream suffix120. Frozen fresh H trained only on own prefix successes. Before each one-attempt evaluation, select same-family prior own-success trajectory by TF-IDF cosine similarity of observed goal text; no suffix outcome used.",
        **fixed, "targets": targets, "targets_sha256": digest(targets)})
    print(json.dumps({"targets": len(targets),
        "source_changes": sum(x["retrieved_source_index"] !=
                              x["original_source_index"] for x in targets)}),
        flush=True)


def evaluate(args):
    if args.eval_output.exists():
        raise FileExistsError(args.eval_output)
    successes, suffix, fixed, targets = bound(args)
    review = json.loads(args.eval_review.read_text())
    if (review.get("targets") != targets or
            review.get("targets_sha256") != digest(targets) or
            any(review.get(k) != v for k, v in fixed.items())):
        raise ValueError("Changed reviewed goal retrieval")
    agent, tokenizer = load_agent(args.model, args.hyper_model,
                                  args.device, args.gpu_fraction)
    report = {"protocol": review["protocol"],
        "review_sha256": file_hash(args.eval_review),
        **fixed, "rows": [], "failures": []}
    for i, (row, target) in enumerate(zip(suffix, targets, strict=True)):
        source = successes[target["retrieved_source_index"]]
        fields = contextual_source_fields(agent, tokenizer,
            source["source_records"], args.device, 2048, pooling="both")
        episode = run_episode(agent, tokenizer, args.data_root / row["game"],
            fields, adapter=True, device=args.device,
            max_steps=30, max_new_tokens=64, constrain_actions=True,
            actor_history_turns=2, loop_guard_max=2)
        if (episode["status"] != "complete" or
                episode["initial_observation"] !=
                row["arms"]["base"]["initial_observation"]):
            report["failures"].append({"index": i})
        report["rows"].append({"index": i, "game": row["game"],
            "input_content_sha256": target["input_content_sha256"],
            "retrieved_source_index": target["retrieved_source_index"],
            "episode": episode})
        report["summary"] = {"n": len(report["rows"]),
            "successes": sum(x["episode"]["reward"] for x in report["rows"]),
            "failures": len(report["failures"])}
        save(args.eval_output, report)
        print(json.dumps(report["summary"]), flush=True)
        if report["failures"]:
            raise RuntimeError("Incomplete retrieval episode")


def audit(args):
    _, suffix, fixed, targets = bound(args)
    review = json.loads(args.eval_review.read_text())
    report = json.loads(args.eval_output.read_text())
    if (review.get("targets") != targets or
            review.get("targets_sha256") != digest(targets) or
            any(review.get(k) != v or report.get(k) != v for k, v in fixed.items()) or
            report.get("review_sha256") != file_hash(args.eval_review) or
            len(report.get("rows", [])) != 120 or report.get("failures")):
        raise ValueError("Changed or incomplete retrieval evaluation")
    wins = 0.
    for i, (source, target, row) in enumerate(zip(
            suffix, targets, report["rows"], strict=True)):
        if (row["index"] != i or row["game"] != source["game"] or
                row["input_content_sha256"] != target["input_content_sha256"] or
                row["retrieved_source_index"] != target["retrieved_source_index"] or
                row["episode"]["status"] != "complete" or
                row["episode"]["steps"] > 30):
            raise ValueError(f"Changed target or episode {i}")
        replay(args.data_root / row["game"], row["episode"])
        wins += row["episode"]["reward"]
    if report["summary"] != {"n": 120, "successes": wins, "failures": 0}:
        raise ValueError("Changed retrieval score summary")
    return {"audit": "passed", "summary": report["summary"],
        "replayed_episodes": 120,
        "report_sha256": file_hash(args.eval_output),
        "review_sha256": file_hash(args.eval_review)}


if __name__ == "__main__":
    p = parser()
    p.add_argument("--parent-eval-review", type=Path, default=Path(
        "data/annotations/alfworld_bootstrap_own_success_v43c_suffix120_20261010.json"))
    p.add_argument("--eval-review", type=Path, default=Path(
        "data/annotations/alfworld_goal_retrieval_v50_suffix120_20261010.json"))
    p.add_argument("--eval-output", type=Path, default=Path(
        "results/trajectory_hyperlora/alfworld_goal_retrieval_v50_suffix120_20261010.json"))
    p.add_argument("--audit-output", type=Path, default=Path(
        "results/trajectory_hyperlora/alfworld_goal_retrieval_v50_suffix120_audit_20261010.json"))
    p.add_argument("--audit-only", action="store_true")
    args = p.parse_args()
    if args.audit_only:
        if args.audit_output.exists():
            raise FileExistsError(args.audit_output)
        result = audit(args)
        save(args.audit_output, result)
        print(json.dumps(result), flush=True)
    else:
        (prepare if args.command == "prepare" else evaluate)(args)
