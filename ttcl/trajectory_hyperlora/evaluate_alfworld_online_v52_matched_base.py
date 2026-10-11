"""Matched base-only snapshot for currently adapter-active v52 stream tasks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ttcl.trajectory_hyperlora.alfworld_online_from_empty import digest, file_hash
from ttcl.trajectory_hyperlora.alfworld_bootstrap_own_success_v43 import new_agent
from ttcl.trajectory_hyperlora.alfworld_zero_shot_probe import run_episode
from ttcl.trajectory_hyperlora.alfworld_persistent_failure_delta_v36 import save
from ttcl.trajectory_hyperlora.alfworld_online_own_trajectory_v52 import (
    bound, parser as v52_parser,
)
from ttcl.trajectory_hyperlora.audit_my_online_update_v28_reward import replay


def main():
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=("prepare", "evaluate"))
    p.add_argument("--v52-report", type=Path, default=Path(
        "results/trajectory_hyperlora/alfworld_online_own_trajectory_v52_stream240_20261010.json"))
    p.add_argument("--review", type=Path, default=Path(
        "data/annotations/alfworld_online_v52_matched_base_snapshot_20261010.json"))
    p.add_argument("--output", type=Path, default=Path(
        "results/trajectory_hyperlora/alfworld_online_v52_matched_base_snapshot_20261010.json"))
    p.add_argument("--device", default="cuda:2")
    args = p.parse_args()
    v52_args = v52_parser().parse_args(["evaluate"])
    rows, fixed, targets = bound(v52_args)
    report = json.loads(args.v52_report.read_text())
    if report["review_sha256"] != file_hash(v52_args.review) or any(
            report.get(k) != value for k, value in fixed.items()):
        raise ValueError("Changed v52 report")
    selected = []
    for result in report["rows"]:
        if not result["adapter_active"]:
            continue
        i = result["index"]
        target = targets[i]
        if result["game"] != target["game"] or result["target_input_content_sha256"] != target["input_content_sha256"]:
            raise ValueError("Changed v52 task")
        content = {"index": i, "game": target["game"],
                   "game_sha256": target["game_sha256"],
                   "v52_target_input_content_sha256": target["input_content_sha256"],
                   "online_reward": result["online"]["reward"],
                   "frozen_reward": result["frozen_same_sources"]["reward"],
                   "online_episode_sha256": digest(result["online"]),
                   "frozen_episode_sha256": digest(result["frozen_same_sources"])}
        selected.append({**content, "input_content_sha256": digest(content),
                         "reviewed_target": True})
    if args.command == "prepare":
        if args.review.exists():
            raise FileExistsError(args.review)
        save(args.review, {"protocol": "Snapshot of v52 adapter-active tasks; same base actor, prompt history, legal action constraint, repeat guard and 30-step budget; no LoRA",
            "runner_sha256": file_hash(Path(__file__)),
            "v52_review_sha256": file_hash(v52_args.review),
            "v52_report_prefix_length": len(report["rows"]),
            "targets": selected, "targets_sha256": digest(selected)})
        print(json.dumps({"selected": len(selected), "prefix": len(report["rows"])}))
        return
    review = json.loads(args.review.read_text())
    if args.output.exists():
        raise FileExistsError(args.output)
    if (review["runner_sha256"] != file_hash(Path(__file__)) or
            review["v52_review_sha256"] != file_hash(v52_args.review) or
            review["targets"] != selected[:len(review["targets"])] or
            review["targets_sha256"] != digest(review["targets"])):
        raise ValueError("Changed matched-base review")
    torch.cuda.set_per_process_memory_fraction(.8, device=args.device)
    v52_args.device = args.device
    agent, tokenizer = new_agent(v52_args)
    agent.eval()
    output = {"protocol": review["protocol"],
              "review_sha256": file_hash(args.review), "rows": []}
    for target in review["targets"]:
        game = v52_args.data_root / target["game"]
        episode = run_episode(agent, tokenizer, game, {}, adapter=False,
            device=args.device, max_steps=30, max_new_tokens=64,
            constrain_actions=True, actor_history_turns=2, loop_guard_max=2)
        if episode["status"] != "complete" or episode["initial_observation"] != rows[target["index"]]["arms"]["base"]["initial_observation"]:
            raise RuntimeError(f"Incomplete matched base at {target['index']}")
        replay(game, episode)
        output["rows"].append({"index": target["index"],
            "input_content_sha256": target["input_content_sha256"],
            "base": episode, "online_reward": target["online_reward"],
            "frozen_reward": target["frozen_reward"]})
        output["summary"] = {"n": len(output["rows"]),
            "base": sum(x["base"]["reward"] for x in output["rows"]),
            "online": sum(x["online_reward"] for x in output["rows"]),
            "frozen": sum(x["frozen_reward"] for x in output["rows"])}
        save(args.output, output)
        print(json.dumps(output["summary"]), flush=True)


if __name__ == "__main__":
    main()
