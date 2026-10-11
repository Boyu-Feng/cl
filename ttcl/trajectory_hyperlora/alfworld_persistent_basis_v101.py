"""ALFWorld online persistent-basis pilot.

The hypernetwork is frozen. Successful own trajectories are converted once to
LoRA factors and stored in a bounded key-addressed convex basis memory. Failed
episodes do not write. The base arm is run on the same reviewed game order.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from ttcl.trajectory_hyperlora.alfworld_online_from_empty import (
    digest, file_hash, records_from_episode,
)
from ttcl.trajectory_hyperlora.alfworld_zero_shot_probe import load_agent, run_episode
from ttcl.trajectory_hyperlora.contextual_alf_source import (
    contextual_text_fields, contextual_source_fields, task_context_text,
)
from ttcl.trajectory_hyperlora.online_parameter_memory_v2 import ParameterMemory


def run(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    review = json.loads(args.review.read_text())
    agent, tokenizer = load_agent(args.model, args.checkpoint, args.device,
                                  args.gpu_fraction)
    memory = ParameterMemory(capacity=args.capacity, temperature=args.temperature,
                             attention_mix=args.attention_mix)
    prior = []
    report = {"protocol": "Frozen hypernetwork; own-success-only persistent factor basis; same reviewed ALFWorld order",
              "review_sha256": file_hash(args.review),
              "checkpoint_sha256": file_hash(args.checkpoint), "rows": []}
    for target in review["targets"]:
        game = args.data_root / target["game"]
        base = run_episode(agent, tokenizer, game, {}, adapter=False,
                           device=args.device, max_steps=args.max_steps,
                           max_new_tokens=args.max_new_tokens,
                           constrain_actions=True)
        initial = base["initial_observation"]
        key_fields = contextual_text_fields(
            agent, tokenizer, task_context_text(initial, initial), args.device,
            args.context_tokens, pooling="both")
        factors, read_audit = memory.read(key_fields["pair_contextual"])
        online = run_episode(agent, tokenizer, game, {}, adapter=False,
                             fixed_adapter=factors, device=args.device,
                             max_steps=args.max_steps,
                             max_new_tokens=args.max_new_tokens,
                             constrain_actions=True)
        write = {"written": False, "reason": "failure_or_incomplete"}
        if online["status"] == "complete" and online["reward"] > 0:
            records = records_from_episode(online)
            source_fields = contextual_source_fields(
                agent, tokenizer, records, args.device,
                max_source_tokens=args.context_tokens, pooling="both")
            target_fields = contextual_text_fields(
                agent, tokenizer, task_context_text(initial, initial), args.device,
                args.context_tokens, pooling="both")
            with __import__("torch").no_grad():
                agent.set_source(source_fields, target_fields=target_fields)
                fresh = [layer.b.detach().cpu().float().clone()
                         for layer in agent.adapters]
                agent.set_source(None)
            slot = memory.write(key_fields["pair_contextual"], fresh)
            write = {"written": True, "slot": slot,
                     "source_sha256": digest(records),
                     "factor_norms": [float(x.norm()) for x in fresh]}
            prior.append(target["game"])
        report["rows"].append({"game": target["game"],
                               "game_sha256": target["game_sha256"],
                               "base": base, "persistent_basis": online,
                               "read": read_audit, "write": write,
                               "memory_digest": memory.digest(),
                               "prior_success_count": len(prior)})
        report["summary"] = {
            "completed": len(report["rows"]),
            "base_success": sum(x["base"]["reward"] for x in report["rows"]),
            "persistent_basis_success": sum(x["persistent_basis"]["reward"]
                                             for x in report["rows"]),
            "writes": sum(x["write"]["written"] for x in report["rows"]),
            "persistent_only": sum(x["persistent_basis"]["reward"] == 1 and
                                    x["base"]["reward"] == 0 for x in report["rows"]),
            "base_only": sum(x["base"]["reward"] == 1 and
                              x["persistent_basis"]["reward"] == 0 for x in report["rows"]),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps({"game": target["game"], "base": base["reward"],
                          "persistent_basis": online["reward"],
                          "write": write["written"],
                          "summary": report["summary"]}), flush=True)
        if base["status"] != "complete" or online["status"] != "complete":
            raise RuntimeError("infrastructure failure recorded")
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, default=Path("current_work/delta-Mem/model/Qwen3-4B-Instruct-2507"))
    p.add_argument("--checkpoint", type=Path, default=Path("results/trajectory_hyperlora/alfworld_online_own_trajectory_v52b_20261010.pt"))
    p.add_argument("--review", type=Path, default=Path("data/annotations/alfworld_online_from_empty_train_seq0_6_reviewed_20261005.json"))
    p.add_argument("--data-root", type=Path, default=Path("ttcl/data/alfworld_delta"))
    p.add_argument("--output", type=Path, default=Path("results/trajectory_hyperlora/alfworld_persistent_basis_v101_20261010.json"))
    p.add_argument("--device", default="cuda:0"); p.add_argument("--gpu-fraction", type=float, default=.35)
    p.add_argument("--max-steps", type=int, default=20); p.add_argument("--max-new-tokens", type=int, default=48)
    p.add_argument("--context-tokens", type=int, default=2048); p.add_argument("--capacity", type=int, default=8)
    p.add_argument("--temperature", type=float, default=.05); p.add_argument("--attention-mix", type=float, default=.75)
    run(p.parse_args())
