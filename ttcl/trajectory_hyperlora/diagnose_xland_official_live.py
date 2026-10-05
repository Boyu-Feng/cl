"""Compare online, mismatched, and reviewed histories on identical observations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ttcl.trajectory_hyperlora.evaluate_xland_official_live import (
    digest, load_actor, select, sha256,
)
from ttcl.trajectory_hyperlora.train_xland_official_history_hyperlora import (
    source_tensor,
)
from ttcl.trajectory_hyperlora.train_xland_raw_hyperlora import checked_query


def run(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    records = [json.loads(path.read_text()) for path in args.inputs]
    rows = [row for record in records for row in record["rows"]]
    if len(rows) < 2 or len({row["ruleset_id"] for row in rows}) != len(rows):
        raise ValueError("Need distinct official rulesets")
    for record in records:
        if (record["checkpoint_sha256"] != sha256(args.checkpoint) or
                record["annotations_sha256"] != sha256(args.annotations)):
            raise ValueError("Rollout provenance mismatch")
    torch.cuda.set_per_process_memory_fraction(args.gpu_fraction,
                                               device=args.device)
    agent, tokenizer, choices = load_actor(args)
    annotation_items = json.loads(args.annotations.read_text())["split"]["test"]
    reviewed = {item["ruleset_id"]: checked_query(item["queries"][0])[0][
        "source_episodes"] for item in annotation_items}
    result = {"protocol": "Frozen model same initial official target observation; compare own online source, next disjoint ruleset online source, reviewed positive source, and none; action diagnostic only, not reward evaluation",
              "input_sha256": {str(path): sha256(path) for path in args.inputs},
              "checkpoint_sha256": sha256(args.checkpoint), "rows": []}
    for index, row in enumerate(rows):
        target_observation = row["with_memory"]["steps"][0]["state"][
            "observation"]
        output = {}
        for kind, source in (("own", row),
                             ("wrong", rows[(index + 1) % len(rows)]),
                             ("reviewed_positive", row),
                             ("none", None)):
            factors = None
            source_hash = None
            if source is not None:
                if kind == "reviewed_positive":
                    episodes = reviewed[row["ruleset_id"]]
                else:
                    steps = [{key: step[key] for key in
                        ("state", "action", "next_state", "reward", "done")}
                        for step in source["source"]["steps"]]
                    episodes = [{"goal": [0, 0], "steps": steps}]
                content = {"source_episodes": episodes,
                    "target_initial_state": {"observation":
                        target_observation, "pocket": [0, 0]},
                    "goal": [0, 0]}
                factors = agent.compile_adapters(source_tensor(content,
                                                                 args.device))
                source_hash = digest(content)
            action, prompt_hash, logits = select(agent, tokenizer, choices,
                target_observation, factors, args.device)
            output[kind] = {"action": action,
                            "source_input_sha256": source_hash,
                            "prompt_sha256": prompt_hash,
                            "choice_logits": logits}
        result["rows"].append({"task_id": row["task_id"],
            "ruleset_id": row["ruleset_id"],
            "wrong_ruleset_id": rows[(index + 1) % len(rows)]["ruleset_id"],
            "output": output})
    result["summary"] = {"n": len(rows),
        "own_vs_wrong_action_changes": sum(row["output"]["own"]["action"]
            != row["output"]["wrong"]["action"] for row in result["rows"]),
        "own_vs_none_action_changes": sum(row["output"]["own"]["action"]
            != row["output"]["none"]["action"] for row in result["rows"]),
        "wrong_vs_none_action_changes": sum(row["output"]["wrong"]["action"]
            != row["output"]["none"]["action"] for row in result["rows"]),
        "reviewed_positive_vs_own_action_changes": sum(row["output"][
            "reviewed_positive"]["action"] != row["output"]["own"][
            "action"] for row in result["rows"]),
        "reviewed_positive_vs_none_action_changes": sum(row["output"][
            "reviewed_positive"]["action"] != row["output"]["none"][
            "action"] for row in result["rows"])}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["summary"]), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--training-result", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, default=Path(
        "data/annotations/xland_official_history_reviewed_64_v1_20261005.json"))
    parser.add_argument("--model", type=Path, default=Path(
        "current_work/delta-Mem/model/Qwen3-4B-Instruct-2507"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpu-fraction", type=float, default=.6)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
