"""Validate a generated LoRA on a paired probe before using it later.

This is an online memory-use controller, not weight training. It reads only
source trajectories and official probe rewards, then freezes a choice before
the independent final reset is executed. It never uses test reward to select.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ttcl.trajectory_hyperlora.evaluate_xland_official_live import (
    Worker, digest, load_actor, sha256, state,
)
from ttcl.trajectory_hyperlora.evaluate_xland_official_cross_episode_v4 import (
    compile_history, rollout,
)


ARMS = ("none", "all_history", "reward_gated")


def evaluate_reset(worker, agent, tokenizer, choices, sources, rule_id,
                   reset_seed, action_seed, args):
    initial = worker.send({"command": "reset", "ruleset_id": rule_id,
                           "seed": reset_seed})
    target_state = state(initial["observation"])
    all_adapter, all_hash = compile_history(agent, sources, target_state,
                                             args.device)
    gated_adapter, gated_hash = compile_history(agent, sources, target_state,
                                                 args.device, positive_only=True)
    adapter = {"none": None, "all_history": all_adapter,
               "reward_gated": gated_adapter}
    arms = {}
    for name in ARMS:
        # If no positive source exists, the gated arm is exactly no adapter.
        # Reuse its paired rollout rather than issuing an identical episode.
        if name == "reward_gated" and gated_adapter is None:
            arms[name] = arms["none"]
            continue
        if name == "reward_gated" and gated_hash == all_hash:
            arms[name] = arms["all_history"]
            continue
        arms[name] = rollout(worker, agent, tokenizer, choices, rule_id,
            reset_seed, args.budget, adapter[name], args.device,
            args.epsilon, action_seed, args.feedback_window,
            initial["observation"], args.action_novelty)
    return {"target_initial_sha256": digest(target_state),
            "all_history_input_sha256": all_hash,
            "reward_gated_input_sha256": gated_hash,
            "reset_seed": reset_seed, "action_seed": action_seed,
            "arms": arms}


def summarize(rows):
    totals = {arm: {"positive": 0, "return": 0.0} for arm in
              (*ARMS, "selected")}
    for row in rows:
        for arm in ARMS:
            target = row["final"]["arms"][arm]
            totals[arm]["positive"] += int(target["positive"])
            totals[arm]["return"] += target["return"]
        chosen = row["selected_arm"]
        target = row["final"]["arms"][chosen]
        totals["selected"]["positive"] += int(target["positive"])
        totals["selected"]["return"] += target["return"]
    return {"n": len(rows), "final": totals,
            "choices": {arm: sum(row["selected_arm"] == arm for row in rows)
                        for arm in ARMS}}


def run(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    source = json.loads(args.source_result.read_text())
    audit = json.loads(args.source_audit.read_text())
    if (audit["result_sha256"] != sha256(args.source_result) or
            not audit["passed"] or
            len(source["rows"]) != len(source["rule_ids"]) or
            any(row["source_prefix"] != source["source_episodes"]
                for row in source["rows"])):
        raise ValueError("Source prefixes are not a complete audited final set")
    if (source["checkpoint_sha256"] != sha256(args.checkpoint) or
            source["annotations_sha256"] != sha256(args.annotations) or
            source["benchmark_sha256"] != sha256(args.benchmark_path)):
        raise ValueError("Checkpoint, annotation or benchmark lineage changed")
    if args.device.startswith("cuda"):
        torch.cuda.set_per_process_memory_fraction(args.gpu_fraction,
                                               device=args.device)
    agent, tokenizer, _ = load_actor(args)
    tokens = [tokenizer(str(x), add_special_tokens=False).input_ids
              for x in range(6)]
    if any(len(ids) != 1 for ids in tokens):
        raise ValueError("Six action labels must each be atomic")
    choices = [ids[0] for ids in tokens]
    worker = Worker(args.xland_python, args.data_dir, args.benchmark,
                    args.environment, args.benchmark_path)
    output = {"protocol": "v5 reward-validated generated LoRA; audit-bound frozen source; matched no/all/reward-gated probe, conservative argmax on official probe return, then independent final reset; no test reward in selection",
              "source_result_sha256": sha256(args.source_result),
              "source_audit_sha256": sha256(args.source_audit),
              "checkpoint_sha256": sha256(args.checkpoint),
              "benchmark_sha256": sha256(args.benchmark_path),
              "annotations_sha256": sha256(args.annotations),
              "model_config_sha256": sha256(args.model / "config.json"),
              "rule_ids": source["rule_ids"],
              "budget": args.budget, "epsilon": args.epsilon,
              "action_novelty": args.action_novelty,
              "feedback_window": args.feedback_window,
              "rows": [], "failures": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        for source_row in source["rows"]:
            rid = source_row["ruleset_id"]
            try:
                sources = source_row["source_episodes"]
                if digest(sources) != source_row["source_sha256"]:
                    raise ValueError("Source trajectory content changed")
                probe = evaluate_reset(worker, agent, tokenizer, choices,
                    sources, rid, args.probe_seed + 1009 * rid,
                    args.probe_action_seed + 1009 * rid, args)
                # Order gives the unadapted actor precedence on ties.
                selected = max(ARMS, key=lambda arm: probe["arms"][arm]["return"])
                final = evaluate_reset(worker, agent, tokenizer, choices,
                    sources, rid, args.final_seed + 1009 * rid,
                    args.final_action_seed + 1009 * rid, args)
                output["rows"].append({"ruleset_id": rid,
                    "source_sha256": source_row["source_sha256"],
                    "source_positive": source_row["source_positive_so_far"],
                    "selected_arm": selected, "probe": probe,
                    "final": final})
            except Exception as exc:
                output["failures"].append({"ruleset_id": rid,
                    "error": f"{type(exc).__name__}: {exc}"})
            output["summary"] = summarize(output["rows"])
            temporary = args.output.with_suffix(args.output.suffix + ".tmp")
            temporary.write_text(json.dumps(output, indent=2) + "\n")
            temporary.replace(args.output)
            print(json.dumps({"rows": len(output["rows"]),
                "failures": len(output["failures"]),
                "summary": output["summary"]}), flush=True)
    finally:
        worker.close()
        agent.mount(None)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-result", type=Path, required=True)
    parser.add_argument("--source-audit", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--training-result", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--benchmark-path", type=Path, required=True)
    parser.add_argument("--environment", required=True)
    parser.add_argument("--model", type=Path, default=Path(
        "current_work/delta-Mem/model/Qwen3-4B-Instruct-2507"))
    parser.add_argument("--xland-python", type=Path, default=Path(
        "ttcl/.runtime/xland_env/bin/python"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/xland_minigrid"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpu-fraction", type=float, default=.6)
    parser.add_argument("--budget", type=int, default=200)
    parser.add_argument("--epsilon", type=float, default=.2)
    parser.add_argument("--feedback-window", type=int, default=4)
    parser.add_argument("--action-novelty", action="store_true")
    parser.add_argument("--probe-seed", type=int, default=50261007)
    parser.add_argument("--final-seed", type=int, default=60261007)
    parser.add_argument("--probe-action-seed", type=int, default=70261007)
    parser.add_argument("--final-action-seed", type=int, default=80261007)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
