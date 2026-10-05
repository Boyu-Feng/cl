"""Exploratory source ablations for an already completed online sequence.

Each replacement history is a subset of the same actor's own earlier episodes.
No current or future task trajectory is allowed into the adapter input.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from ttcl.trajectory_hyperlora.alfworld_online_from_empty import (
    digest, records_from_episode,
)
from ttcl.trajectory_hyperlora.alfworld_zero_shot_probe import load_agent, run_episode
from ttcl.trajectory_hyperlora.relational_router_pilot import tokenize_records


def run(args: argparse.Namespace) -> dict:
    if args.output.exists():
        raise FileExistsError(args.output)
    raw = args.online_report.read_bytes()
    report = json.loads(raw)
    games = report["games"]
    if (report["checkpoint_sha256"] != hashlib.sha256(
            args.checkpoint.read_bytes()).hexdigest() or
            report["max_steps"] != args.max_steps or
            report["max_new_tokens"] != args.max_new_tokens or
            report["source_field_token_limit"] != args.field_tokens):
        raise ValueError("Counterfactual must preserve actor and budgets")
    if any(index <= 0 or index >= len(games) for index in args.indices):
        raise ValueError("Need target indices with prior online experience")
    agent, tokenizer = load_agent(args.model, args.checkpoint,
                                  args.device, args.gpu_fraction)
    result = {"protocol": "Exploratory subset-of-prior-online-history source control; same target, checkpoint and environment budget; selected after viewing full-sequence results",
              "online_report_sha256": hashlib.sha256(raw).hexdigest(),
              "indices": args.indices, "rows": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for index in args.indices:
        target = games[index]
        previous = games[:index]
        if any(item["online"]["status"] != "complete" for item in previous):
            raise ValueError("Incomplete prior online episode")
        memory = [records_from_episode(item["online"]) for item in previous]
        if sum(map(len, memory)) != target["memory_step_count_before"]:
            raise ValueError("Prior memory step binding changed")
        selections = {"first_episode": memory[0],
                      "last_episode": memory[-1],
                      "all_history": [record for episode in memory
                                      for record in episode]}
        row = {"index": index, "game": target["game"],
               "original_base_reward": target["base"]["reward"],
               "original_online_reward": target["online"]["reward"],
               "controls": {}}
        for name, records in selections.items():
            fields = tokenize_records(tokenizer, records, args.device,
                                      max_tokens=args.field_tokens)
            outcome = run_episode(agent, tokenizer,
                args.data_root / target["game"], fields,
                adapter=True, device=args.device, max_steps=args.max_steps,
                max_new_tokens=args.max_new_tokens, constrain_actions=True)
            row["controls"][name] = {"source_sha256": digest(records),
                                      "source_steps": len(records),
                                      "episode": outcome}
            if outcome["status"] != "complete" or \
                    outcome["initial_observation"] != \
                    target["online"]["initial_observation"]:
                raise RuntimeError("Counterfactual environment failure")
        result["rows"].append(row)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(json.dumps(result, ensure_ascii=False,
                                        indent=2) + "\n")
        temporary.replace(args.output)
        print(json.dumps({"index": index, "game": target["game"],
                          "rewards": {name: value["episode"]["reward"]
                                      for name, value in row["controls"].items()}}),
              flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--online-report", type=Path, required=True)
    parser.add_argument("--indices", type=int, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path(
        "current_work/delta-Mem/model/Qwen3-4B-Instruct-2507"))
    parser.add_argument("--checkpoint", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_next_task_warmstart_20261005/seed42_v2.pt"))
    parser.add_argument("--data-root", type=Path, default=Path(
        "ttcl/data/alfworld_delta"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpu-fraction", type=float, default=.40)
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--field-tokens", type=int, default=40)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
