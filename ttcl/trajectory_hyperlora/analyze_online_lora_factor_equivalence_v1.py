"""Compare history-reencoded and incremental-mean LoRA on identical episodes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ttcl.trajectory_hyperlora.alfworld_online_accumulate_lora_v1 import (
    mean_factors,
)
from ttcl.trajectory_hyperlora.alfworld_online_from_empty import (
    file_hash, records_from_episode,
)
from ttcl.trajectory_hyperlora.alfworld_zero_shot_probe import load_agent
from ttcl.trajectory_hyperlora.relational_router_pilot import tokenize_records


def factors(agent, tokenizer, records, args):
    fields = tokenize_records(tokenizer, records, args.device,
                              max_tokens=args.field_tokens)
    with torch.no_grad():
        agent.set_source(fields)
        result = [layer.b.detach().cpu().float().clone()
                  for layer in agent.adapters]
        agent.set_source(None)
    return result


def compare(left, right):
    a = torch.cat([x.flatten() for x in left]).double()
    b = torch.cat([x.flatten() for x in right]).double()
    return {"mean_l2": float(a.norm()), "all_l2": float(b.norm()),
            "relative_l2_difference": float((a - b).norm() / b.norm().clamp_min(1e-12)),
            "cosine": float(torch.dot(a, b) /
                            (a.norm() * b.norm()).clamp_min(1e-12))}


def run(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    agent, tokenizer = load_agent(args.model, args.checkpoint,
                                  args.device, args.gpu_fraction)
    report = {"protocol": "Read-only same-history factor comparison: persistent mean of per-episode generated B versus B generated from concatenated identical own online episodes; source encoder/head frozen",
        "checkpoint_sha256": file_hash(args.checkpoint),
        "sequences": [], "summary": {}}
    for path in args.control:
        data = json.loads(path.read_text())
        if (data["checkpoint_sha256"] != file_hash(args.checkpoint) or
                data["max_steps"] != 30 or data["max_new_tokens"] != 64 or
                data["source_field_token_limit"] != args.field_tokens or
                len(data["games"]) != 18):
            raise ValueError("Changed online control protocol")
        historical = []
        running = None
        increments = 0
        rows = []
        for index, game in enumerate(data["games"]):
            episode = game["online"]
            if episode["status"] != "complete":
                raise ValueError("Incomplete historical online episode")
            records = records_from_episode(episode)
            historical.extend(records)
            if len(records) >= 2:
                current = factors(agent, tokenizer, records, args)
                running = mean_factors(running, current, increments)
                increments += 1
            if running is not None and len(historical) >= 2:
                all_factors = factors(agent, tokenizer, historical, args)
                rows.append({"index": index, "game": game["game"],
                             "history_steps": len(historical),
                             **compare(running, all_factors)})
        report["sequences"].append({"control_sha256": file_hash(path),
                                    "rows": rows})
    all_rows = [row for sequence in report["sequences"]
                for row in sequence["rows"]]
    report["summary"] = {"prefixes": len(all_rows),
        "mean_cosine": sum(row["cosine"] for row in all_rows) / len(all_rows),
        "min_cosine": min(row["cosine"] for row in all_rows),
        "mean_relative_l2_difference": sum(row["relative_l2_difference"]
                                                for row in all_rows) / len(all_rows),
        "max_relative_l2_difference": max(row["relative_l2_difference"]
                                            for row in all_rows)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report["summary"]), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--control", type=Path, nargs="+", default=[
        Path("results/trajectory_hyperlora/alf_online_from_empty_seq0_6_statusfirst_20261005.json"),
        Path("results/trajectory_hyperlora/alf_online_all_history_train_seq6_6_20261007.json")])
    parser.add_argument("--checkpoint", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_next_task_warmstart_20261005/seed42_v2.pt"))
    parser.add_argument("--model", type=Path, default=Path(
        "current_work/delta-Mem/model/Qwen3-4B-Instruct-2507"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpu-fraction", type=float, default=.6)
    parser.add_argument("--field-tokens", type=int, default=40)
    parser.add_argument("--output", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_online_factor_equivalence_20261007.json"))
    run(parser.parse_args())


if __name__ == "__main__":
    main()
