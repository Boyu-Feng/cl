"""Collect official train/dev cumulative text-history teacher rollouts.

The teacher sees all preceding completed public trajectories in the domain.
Paired baseline outcomes are bound to the same official task instances. This
is a training-data experiment; indices 8+ remain untouched.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from ttcl.trajectory_hyperlora.alfworld_online_from_empty import digest, file_hash
from ttcl.trajectory_hyperlora.clbench_online_parameter_memory_v2 import (
    Actor, BENCH, ROOT, benchmark, bounded_text, run_episode,
)
from ttcl.trajectory_hyperlora.contextual_alf_source import source_text
from ttcl.trajectory_hyperlora.train_clbench_hyperlora_v1 import clean_records, load_episode


class TextTeacher(Actor):
    def generate(self, messages, random_seed):
        messages = copy.deepcopy(messages)
        messages[0]["content"] += (
            "\n\nPrevious completed public task trajectory. Use it only when relevant "
            "to the current task:\n" + self.source)
        return super().generate(messages, random_seed)


def targets(args):
    base = benchmark()
    result = []
    for domain in args.domains:
        for index in range(1, 8):
            source_hashes = []
            for prior in range(index):
                source_row, _, _, source_dir = load_episode(domain, prior)
                if source_row["status"] != "complete":
                    raise ValueError("Incomplete source trajectory")
                source_hashes.append(file_hash(source_dir / "trajectory.json"))
            target_row, _, _, target_dir = load_episode(domain, index)
            task = base.make_task(domain, args.seed, independent=True)
            query = task.reset_baseline_instance(index)
            connection = getattr(task, "_conn", None)
            if connection is not None:
                connection.close()
            if (target_row["status"] != "complete" or
                    target_row["instance_id"] != query.instance_id or
                    target_row["initial_query_sha256"] != digest(query.prompt)):
                raise ValueError("Official source or target changed")
            result.append({"domain": domain, "index": index,
                "instance_id": query.instance_id,
                "initial_query_sha256": digest(query.prompt),
                "source_trajectory_sha256": source_hashes,
                "base_trajectory_sha256": file_hash(target_dir / "trajectory.json"),
                "base_reward": target_row["reward"],
                "split": "train" if index <= 5 else "dev"})
    return result


def binding(args):
    return {"domains": args.domains, "seed": args.seed,
            "model_config_sha256": file_hash(args.model / "config.json"),
            "checkpoint_sha256": file_hash(args.checkpoint),
            "context_limit": args.context_limit,
            "context_tokens": args.context_tokens,
            "action_tokens": args.action_tokens,
            "history_turns": args.history_turns,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_turns_per_instance": 64, "action_retries": 2}


def prepare(args):
    if args.review.exists():
        raise FileExistsError(args.review)
    import os
    os.chdir(BENCH)
    value = {"protocol": "Official CLBench train/dev cumulative-public-trajectory text teacher",
             "binding": binding(args), "targets": targets(args)}
    value["input_content_sha256"] = digest({"binding": value["binding"],
                                           "targets": value["targets"]})
    args.review.parent.mkdir(parents=True, exist_ok=True)
    args.review.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"targets": len(value["targets"])}), flush=True)


def evaluate(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    import os
    os.chdir(BENCH)
    review = json.loads(args.review.read_text())
    if (review["binding"] != binding(args) or
            review["targets"] != targets(args) or
            review["input_content_sha256"] != digest({
                "binding": review["binding"], "targets": review["targets"]})):
        raise ValueError("Text teacher input or lineage changed")
    actor = TextTeacher(args)
    actor.arm = "base"
    result = {"protocol": review["protocol"], "review_sha256": file_hash(args.review),
              "checkpoint_sha256": file_hash(args.checkpoint), "rows": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for target in review["targets"]:
        blocks = []
        for prior in range(target["index"]):
            _, source, _, _ = load_episode(target["domain"], prior)
            blocks.append(f"Completed prior task {prior+1}:\n" +
                          source_text(clean_records(source)))
        actor.source, original, kept = bounded_text(actor.tokenizer,
            "\n\n".join(blocks), args.context_tokens)
        directory = (args.output.parent / (args.output.stem + "_episodes") /
                     target["domain"] / f"episode_{target['index']+1:03}")
        row, _ = run_episode(args, actor, target["domain"], target["index"],
                             "base", directory)
        if (row["instance_id"] != target["instance_id"] or
                row["initial_query_sha256"] != target["initial_query_sha256"]):
            raise ValueError("Text teacher target changed")
        item = {"target": target, "text": row,
                "source_tokens_original": original,
                "source_tokens_retained": kept,
                "reward_delta": (float(row["reward"]) - float(target["base_reward"])
                                 if row["status"] == "complete" else None)}
        result["rows"].append(item)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps({"domain": target["domain"], "index": target["index"],
                          "base": target["base_reward"], "text": row["reward"],
                          "delta": item["reward_delta"]}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "evaluate"))
    parser.add_argument("--domains", nargs="+", choices=(
        "blind_spectrum_monitoring", "database_exploration"),
        default=["blind_spectrum_monitoring"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", type=Path,
        default=ROOT / "current_work/delta-Mem/model/Qwen3-4B-Instruct-2507")
    parser.add_argument("--checkpoint", type=Path,
        default=ROOT / "results/trajectory_hyperlora/clbench_hyperlora_v2_20261008.pt")
    parser.add_argument("--review", type=Path,
        default=ROOT / "data/annotations/clbench_text_teacher_v2_20261008.json")
    parser.add_argument("--output", type=Path,
        default=ROOT / "results/trajectory_hyperlora/clbench_text_teacher_v2_20261008.json")
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--gpu-fraction", type=float, default=.85)
    parser.add_argument("--context-limit", type=int, default=16384)
    parser.add_argument("--context-tokens", type=int, default=2048)
    parser.add_argument("--action-tokens", type=int, default=512)
    parser.add_argument("--history-turns", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=.7)
    parser.add_argument("--top-p", type=float, default=.9)
    args = parser.parse_args()
    args.episodes = 8
    (prepare if args.command == "prepare" else evaluate)(args)


if __name__ == "__main__":
    main()
