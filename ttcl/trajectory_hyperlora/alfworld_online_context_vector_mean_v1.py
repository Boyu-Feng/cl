"""Online ALFWorld control: persist source vectors, regenerate target LoRA."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from ttcl.trajectory_hyperlora.alfworld_online_from_empty import (
    digest, file_hash, records_from_episode,
)
from ttcl.trajectory_hyperlora.alfworld_online_persistent_context_lora_v1 import (
    bounded_source_text, checked_targets as checked_parent_targets,
)
from ttcl.trajectory_hyperlora.alfworld_zero_shot_probe import load_agent, run_episode
from ttcl.trajectory_hyperlora.contextual_alf_source import contextual_text_fields


def vector_hash(fields):
    value = hashlib.sha256()
    if fields is None:
        value.update(b"empty")
    else:
        for key in sorted(fields):
            x = fields[key].detach().cpu().contiguous().float()
            value.update(key.encode())
            value.update(str(tuple(x.shape)).encode())
            value.update(x.numpy().tobytes())
    return value.hexdigest()


def checked_targets(args):
    args.review = args.parent_persistent_review
    parent = checked_parent_targets(args)
    args.review = args.vector_review
    value = json.loads(args.vector_review.read_text())
    if (value["parent_review_sha256"] != file_hash(args.parent_persistent_review) or
            value["checkpoint_sha256"] != file_hash(args.checkpoint) or
            len(value["targets"]) != len(parent)):
        raise ValueError("Source-vector target review lineage changed")
    for i, (prior, row) in enumerate(zip(parent, value["targets"], strict=True)):
        content = {"index": i, "game": prior["game"],
            "parent_target_sha256": prior["input_content_sha256"],
            "parent_review_sha256": file_hash(args.parent_persistent_review),
            "method": "online_source_vector_mean"}
        if (not row["reviewed_target"] or
                row["input_content_sha256"] != digest(content) or
                any(row[key] != val for key, val in content.items())):
            raise ValueError("Changed source-vector target")
    return value["targets"]


def prepare(args):
    if args.vector_review.exists():
        raise FileExistsError(args.vector_review)
    args.review = args.parent_persistent_review
    parent = checked_parent_targets(args)
    targets = []
    for i, row in enumerate(parent):
        content = {"index": i, "game": row["game"],
            "parent_target_sha256": row["input_content_sha256"],
            "parent_review_sha256": file_hash(args.parent_persistent_review),
            "method": "online_source_vector_mean"}
        targets.append({**content, "input_content_sha256": digest(content),
            "reviewed_target": True,
            "review_basis": "Fresh frozen train target inherited from checked parent; only completed own live transitions enter source vector"})
    value = {"protocol": "Fresh content-bound source-vector online target review",
        "parent_review_sha256": file_hash(args.parent_persistent_review),
        "checkpoint_sha256": file_hash(args.checkpoint),
        "targets": targets}
    args.vector_review.parent.mkdir(parents=True, exist_ok=True)
    args.vector_review.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"reviewed_targets": len(targets)}), flush=True)


def update_fields(previous, fresh, count):
    if previous is None:
        if count != 0:
            raise ValueError("Missing previous source vector")
        return {key: x.detach().float().clone() for key, x in fresh.items()}
    if set(previous) != set(fresh) or count < 1:
        raise ValueError("Source vector keys or count changed")
    return {key: old + (fresh[key].detach().float() - old) / (count + 1)
            for key, old in previous.items()}


def evaluate(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    targets = checked_targets(args)
    agent, tokenizer = load_agent(args.model, args.checkpoint,
                                  args.device, args.gpu_fraction)
    if (agent.encoder_kind != "contextual" or
            not agent.task_conditioned or
            agent.task_context_scope != "current" or
            agent.task_pair_pooling != "mean"):
        raise ValueError("Expected current-context ALFWorld hypernetwork")
    fields = None
    updates = 0
    prior = []
    report = {"protocol": "From-empty ALFWorld control with frozen contextual hypernetwork; encode each own trajectory once, update running-mean source vector and generate task-conditioned LoRA from current target; no historical source text reread; 50 steps/64 tokens/2-turn history, constrained greedy, two-repeat loop guard",
        "vector_review_sha256": file_hash(args.vector_review),
        "checkpoint_sha256": file_hash(args.checkpoint),
        "context_tokens": args.context_tokens,
        "max_steps": 50, "max_new_tokens": 64,
        "actor_history_turns": 2, "loop_guard_max": 2,
        "games": [], "failures": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for row in targets:
        before = vector_hash(fields)
        episode = run_episode(agent, tokenizer,
            args.data_root / row["game"], fields or {},
            adapter=fields is not None, device=args.device,
            max_steps=50, max_new_tokens=64,
            constrain_actions=True, actor_history_turns=2,
            loop_guard_max=2)
        entry = {"game": row["game"],
            "input_content_sha256": row["input_content_sha256"],
            "prior_episode_count": len(prior),
            "prior_episodes_sha256": digest(prior),
            "source_vector_sha256_before": before,
            "episode": episode}
        if episode["status"] == "complete":
            try:
                records = records_from_episode(episode)
                text, original, retained = bounded_source_text(
                    tokenizer, records, args.context_tokens)
                fresh = contextual_text_fields(agent, tokenizer, text,
                    args.device, args.context_tokens, pooling="both")
                fields = update_fields(fields, fresh, updates)
                updates += 1
                entry.update({"new_records_sha256": digest(records),
                    "source_tokens_original": original,
                    "source_tokens_retained": retained})
                prior.append({"game": row["game"],
                    "reward": episode["reward"], "records": records})
            except Exception as exc:
                report["failures"].append({"game": row["game"],
                    "error": f"{type(exc).__name__}: {exc}"})
        else:
            report["failures"].append({"game": row["game"],
                "error": episode.get("error", "incomplete rollout")})
        entry["source_vector_sha256_after"] = vector_hash(fields)
        entry["update_count_after"] = updates
        report["games"].append(entry)
        report["summary"] = {"n": sum(x["episode"]["status"] == "complete"
                                        for x in report["games"]),
            "success": sum(x["episode"].get("reward", 0)
                           for x in report["games"]),
            "failures": len(report["failures"])}
        tmp = args.output.with_suffix(args.output.suffix + ".tmp")
        tmp.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        tmp.replace(args.output)
        print(json.dumps({"n": len(report["games"]),
            "summary": report["summary"]}), flush=True)
        if report["failures"]:
            raise RuntimeError("Source-vector online failure recorded")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "evaluate"))
    parser.add_argument("--model", type=Path, default=Path(
        "current_work/delta-Mem/model/Qwen3-4B-Instruct-2507"))
    parser.add_argument("--checkpoint", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_sibling_simple40_onpolicy600_20261006.pt"))
    parser.add_argument("--plan", type=Path, default=Path(
        "ttcl/results/experience_evolution/alfworld_delta_20260922/plan.json"))
    parser.add_argument("--data-root", type=Path, default=Path(
        "ttcl/data/alfworld_delta"))
    parser.add_argument("--parent-review", type=Path, default=Path(
        "data/annotations/alfworld_online_from_empty_train_seq6_6_reviewed_20261007.json"))
    parser.add_argument("--parent-persistent-review", type=Path, default=Path(
        "data/annotations/alfworld_online_context_persistent_train_seq6_6_reviewed_20261007.json"))
    parser.add_argument("--vector-review", type=Path, default=Path(
        "data/annotations/alfworld_online_context_vector_mean_train_seq6_6_reviewed_20261007.json"))
    parser.add_argument("--output", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_online_context_vector_mean_train_seq6_6_20261007.json"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpu-fraction", type=float, default=.65)
    parser.add_argument("--context-tokens", type=int, default=2048)
    args = parser.parse_args()
    if args.context_tokens < 2:
        parser.error("Invalid source token budget")
    (prepare if args.command == "prepare" else evaluate)(args)


if __name__ == "__main__":
    main()
