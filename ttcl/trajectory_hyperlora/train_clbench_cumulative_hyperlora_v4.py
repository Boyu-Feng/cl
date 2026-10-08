"""Train trajectory hyper-LoRA on rewarded cumulative-history action targets.

Only train/development indices 1-7 are used. A generic JSON-array merge
constructs candidates from public past and current actions; the official task
reward verifies whether the candidate is better. The actor at evaluation has
no reward oracle and receives no past text while acting.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ttcl.trajectory_hyperlora.alfworld_online_from_empty import digest, file_hash
from ttcl.trajectory_hyperlora.clbench_online_parameter_memory_v2 import (
    ROOT, compact_actor_messages, target_text,
)
from ttcl.trajectory_hyperlora.train_clbench_hyperlora_v1 import (
    clean_records, load_episode, train,
)


DOMAIN = "blind_spectrum_monitoring"


def merge_object_arrays(current: dict, history: list[dict]) -> dict:
    """Merge same-named arrays of objects without interpreting their fields."""
    result = json.loads(json.dumps(current))
    for prior in history:
        for key, value in result.items():
            old = prior.get(key)
            if (isinstance(value, list) and isinstance(old, list) and
                    all(isinstance(item, dict) for item in value + old)):
                seen = {json.dumps(item, ensure_ascii=False, sort_keys=True)
                        for item in value}
                for item in old:
                    serial = json.dumps(item, ensure_ascii=False, sort_keys=True)
                    if serial not in seen:
                        value.append(item)
                        seen.add(serial)
    return result


def official_score(index: int, action: dict) -> float:
    from ttcl.trajectory_hyperlora.clbench_online_parameter_memory_v2 import benchmark
    from src.tasks.blind_spectrum_monitoring.task import ScanReport, _score_report
    task = benchmark().make_task(DOMAIN, 42, independent=True)
    task.reset_baseline_instance(index)
    result = _score_report(ScanReport.model_validate(action),
        task._get_all_latent_channel_defs(), task.W, task.G, task.band_width)
    return float(result["score"])


def build_candidates():
    labels = []
    provenance = []
    for index in range(1, 8):
        target_row, target_episode, events, target_dir = load_episode(DOMAIN, index)
        if len(target_episode["steps"]) != 1:
            raise ValueError("Expected one official scan action")
        source = []
        source_hashes = []
        history_actions = []
        for prior in range(index):
            row, episode, _, directory = load_episode(DOMAIN, prior)
            if row["status"] != "complete" or len(episode["steps"]) != 1:
                raise ValueError("Incomplete public source")
            source.extend(clean_records(episode))
            history_actions.append(episode["steps"][0]["action"])
            source_hashes.append(file_hash(directory / "trajectory.json"))
        baseline_action = target_episode["steps"][0]["action"]
        candidate = merge_object_arrays(baseline_action, history_actions)
        base_score = official_score(index, baseline_action)
        candidate_score = official_score(index, candidate)
        if abs(base_score - float(target_row["reward"])) > 1e-6:
            raise ValueError("Recomputed official baseline reward changed")
        if candidate_score <= base_score:
            raise ValueError("Cumulative candidate did not improve official reward")
        event = next(x for x in events if x.get("action") == baseline_action and
                     x.get("parse_error") is None)
        messages = compact_actor_messages(event["messages"], 2)
        content = {"domain": DOMAIN,
            "source_index": list(range(index)), "target_index": index,
            "source_trajectory_sha256": source_hashes,
            "target_trajectory_sha256": file_hash(target_dir / "trajectory.json"),
            "target_responses_sha256": file_hash(target_dir / "responses.jsonl"),
            "source_records": source,
            "source_context": target_text(load_episode(DOMAIN, index - 1)[1]),
            "target_context": target_text(target_episode),
            "target_messages": messages,
            "target_action": json.dumps(candidate, ensure_ascii=False, sort_keys=True),
            "target_reward": candidate_score,
            "reward_delta": candidate_score - base_score,
            "split": "train" if index <= 5 else "dev"}
        labels.append({**content, "input_content_sha256": digest(content)})
        provenance.append({"index": index, "base_score": base_score,
            "candidate_score": candidate_score,
            "source_hashes": source_hashes,
            "target_hash": content["target_trajectory_sha256"]})
    return {"protocol": "Cumulative public JSON-array candidates with official train/dev reward; 8+ held out",
            "provenance": provenance, "labels": labels,
            "input_content_sha256": digest({"provenance": provenance, "labels": labels})}


def prepare(args):
    if args.candidates.exists() or args.review.exists():
        raise FileExistsError("Fresh candidate/review paths required")
    value = build_candidates()
    args.candidates.parent.mkdir(parents=True, exist_ok=True)
    args.candidates.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    review = {"candidates_sha256": file_hash(args.candidates),
        "annotations": [{"input_content_sha256": x["input_content_sha256"],
                         "approved": False, "review_basis": "pending"}
                        for x in value["labels"]]}
    args.review.parent.mkdir(parents=True, exist_ok=True)
    args.review.write_text(json.dumps(review, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"train": 5, "dev": 2,
                      "deltas": [x["reward_delta"] for x in value["labels"]]}), flush=True)


def checked(args):
    candidate = json.loads(args.candidates.read_text())
    review = json.loads(args.review.read_text())
    if (candidate != build_candidates() or
            review["candidates_sha256"] != file_hash(args.candidates) or
            [x["input_content_sha256"] for x in candidate["labels"]] !=
            [x["input_content_sha256"] for x in review["annotations"]] or
            not all(x["approved"] and x["review_basis"] != "pending"
                    for x in review["annotations"])):
        raise ValueError("Unreviewed cumulative training target")
    return candidate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "train"))
    parser.add_argument("--candidates", type=Path,
        default=ROOT / "data/annotations/clbench_cumulative_hyperlora_v4_candidates_20261008.json")
    parser.add_argument("--review", type=Path,
        default=ROOT / "data/annotations/clbench_cumulative_hyperlora_v4_reviewed_20261008.json")
    parser.add_argument("--checkpoint", type=Path,
        default=ROOT / "results/trajectory_hyperlora/clbench_hyperlora_v2_20261008.pt")
    parser.add_argument("--checkpoint-out", type=Path,
        default=ROOT / "results/trajectory_hyperlora/clbench_cumulative_hyperlora_v4_20261008.pt")
    parser.add_argument("--output", type=Path,
        default=ROOT / "results/trajectory_hyperlora/clbench_cumulative_hyperlora_v4_train_20261008.json")
    parser.add_argument("--model", type=Path,
        default=ROOT / "current_work/delta-Mem/model/Qwen3-4B-Instruct-2507")
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--gpu-fraction", type=float, default=.85)
    parser.add_argument("--source-tokens", type=int, default=8192)
    parser.add_argument("--context-limit", type=int, default=16384)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=10)
    args = parser.parse_args()
    (prepare(args) if args.command == "prepare" else
     train(args, include_thinking=True, reviewed_data=checked(args)))


if __name__ == "__main__":
    main()
