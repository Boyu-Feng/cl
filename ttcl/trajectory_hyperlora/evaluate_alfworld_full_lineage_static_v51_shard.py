"""Run a disjoint frozen subset of v51 official ALFWorld evaluation targets."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from ttcl.trajectory_hyperlora.alfworld_online_from_empty import file_hash
from ttcl.trajectory_hyperlora.alfworld_persistent_failure_delta_v36 import save
from ttcl.trajectory_hyperlora.alfworld_zero_shot_probe import load_agent, run_episode
from ttcl.trajectory_hyperlora.evaluate_alfworld_full_lineage_static_v51 import checked, parser


def main():
    p = parser()
    p.add_argument("--start", type=int, required=True)
    p.add_argument("--end", type=int, required=True)
    p.add_argument("--shard-output", type=Path, required=True)
    args = p.parse_args()
    if args.command != "evaluate" or not 0 <= args.start < args.end <= 134:
        raise ValueError("Invalid frozen shard interval")
    if args.shard_output.exists():
        raise FileExistsError(args.shard_output)
    old, fixed, targets = checked(args)
    static = torch.load(args.static_model, map_location="cpu", weights_only=False)
    if static.get("review_sha256") != file_hash(args.train_review):
        raise ValueError("Changed static factors")
    agent, tokenizer = load_agent(args.model, args.old_checkpoint,
                                  args.device, args.gpu_fraction)
    if len(agent.adapters) != 2 or len(static["A"]) != 2 or len(static["factors"]) != 2:
        raise ValueError("Changed adapter geometry")
    with torch.no_grad():
        for layer, a in zip(agent.adapters, static["A"], strict=True):
            if layer.a.shape != a.shape:
                raise ValueError("Changed static A shape")
            layer.a.copy_(a.to(layer.a.device, layer.a.dtype))
    report = {"protocol": "v51 frozen disjoint shard, same actor and static LoRA",
              "review_sha256": file_hash(args.review), **fixed,
              "shard_runner_sha256": file_hash(Path(__file__)),
              "start": args.start, "end": args.end, "rows": [], "failures": []}
    for i in range(args.start, args.end):
        target, old_row = targets[i], old["rows"][i]
        episode = run_episode(agent, tokenizer, args.data_root / target["game"], {},
            adapter=False, fixed_adapter=static["factors"], device=args.device,
            max_steps=50, max_new_tokens=64, constrain_actions=True,
            actor_history_turns=2, loop_guard_max=2)
        if (episode["status"] != "complete" or
                episode["initial_observation"] != old_row["episode"]["initial_observation"]):
            report["failures"].append({"index": i, "status": episode["status"]})
        report["rows"].append({"index": i, "game": target["game"],
            "input_content_sha256": target["input_content_sha256"],
            "old_hyper_reward": old_row["old_hyper_reward"], "static": episode})
        report["summary"] = {"n": len(report["rows"]),
             "old_hyper": sum(r["old_hyper_reward"] for r in report["rows"]),
             "static": sum(r["static"]["reward"] for r in report["rows"]),
             "failures": len(report["failures"])}
        save(args.shard_output, report)
        print(json.dumps({"start": args.start, "end": args.end,
                          **report["summary"]}), flush=True)
        if report["failures"]:
            raise RuntimeError("Incomplete v51 shard episode")


if __name__ == "__main__":
    main()
