"""Cross-domain real-reward pilot for the generic progress-sensitive guard.

Rerun the eight already-exposed XLand resets from the historical v9 online
report with the same frozen actor and no LoRA. Only the action controller
changes; this is development evidence, not a fresh benchmark test.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch

from ttcl.trajectory_hyperlora.action_feedback_guard_v29 import ActionFeedbackGuard
from ttcl.trajectory_hyperlora.evaluate_xland_official_live import (
    Worker, digest, load_actor, select, sha256, state,
)


def inputs(args):
    old = json.loads(args.old_report.read_text())
    if (old["failures"] or len(old["rows"]) != 8 or
            old["benchmark"] != "trivial_1m_v2" or
            old["checkpoint_sha256"] != sha256(args.checkpoint) or
            old["annotations_sha256"] != sha256(args.annotations) or
            old["training_result_sha256"] != sha256(args.training_result) or
            old["benchmark_sha256"] != sha256(args.benchmark_path) or
            old["budget"] != 200 or old["epsilon"] != .2 or
            old["feedback_window"] != 4 or old["action_novelty"]):
        raise ValueError("Changed historical XLand paired reset lineage")
    fixed = {"method": "我的在线更新 v29 generic action guard XLand pilot",
        "old_report_sha256": sha256(args.old_report),
        "checkpoint_sha256": sha256(args.checkpoint),
        "annotations_sha256": sha256(args.annotations),
        "training_result_sha256": sha256(args.training_result),
        "benchmark_sha256": sha256(args.benchmark_path),
        "runner_sha256": sha256(Path(__file__)),
        "guard_sha256": sha256(Path(__file__).with_name(
            "action_feedback_guard_v29.py")),
        "model_config_sha256": sha256(args.model / "config.json"),
        "budget": 200, "epsilon": .2, "feedback_window": 4,
        "repeat_limit": 2, "stagnation_limit": 2,
        "target_seed": 30261007, "action_seed": 40261007}
    rows = []
    for i, row in enumerate(old["rows"]):
        rule_id, prefix = row["ruleset_id"], row["source_prefix"]
        if (prefix != 2 or rule_id != 1011 + i or
                row["targets"]["none"]["steps"][0]["state"] !=
                row["targets"]["all_history"]["steps"][0]["state"]):
            raise ValueError("Changed historical XLand target order")
        content = {"index": i, "rule_id": rule_id,
            "source_prefix": prefix,
            "environment_seed": fixed["target_seed"] + 1009 * rule_id + prefix,
            "action_seed": fixed["action_seed"] + 1009 * rule_id + prefix,
            "initial_state_sha256": row["target_initial_sha256"],
            "fixed_sha256": digest(fixed)}
        rows.append({**content, "input_content_sha256": digest(content),
                     "reviewed_target": True})
    return old, fixed, rows


def prepare(args):
    if args.review.exists():
        raise FileExistsError(args.review)
    _, fixed, rows = inputs(args)
    args.review.parent.mkdir(parents=True, exist_ok=True)
    args.review.write_text(json.dumps({"protocol":
        "Fresh content-bound XLand reset review for generic action guard real reward pilot; exposed historical rules only",
        **fixed, "targets": rows, "targets_sha256": digest(rows)},
        indent=2)+"\n")
    print(json.dumps({"reviewed_targets": len(rows)}), flush=True)


def evaluate(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    old, fixed, targets = inputs(args)
    review = json.loads(args.review.read_text())
    if (review["targets"] != targets or
            review["targets_sha256"] != digest(targets) or
            any(review[key] != value for key, value in fixed.items())):
        raise ValueError("Changed reviewed XLand target content")
    if args.device.startswith("cuda"):
        torch.cuda.set_per_process_memory_fraction(args.gpu_fraction,
                                                   device=args.device)
    agent, tokenizer, _ = load_actor(args)
    tokens = [tokenizer(str(x), add_special_tokens=False).input_ids
              for x in range(6)]
    if any(len(ids) != 1 for ids in tokens):
        raise ValueError("Changed six-action tokenizer")
    choices = [ids[0] for ids in tokens]
    worker = Worker(args.xland_python, args.data_dir, "trivial_1m_v2",
                    args.environment, args.benchmark_path)
    result = {"protocol": "我的在线更新 v29: generic action-feedback guard on exposed official XLand resets, no LoRA, paired with historical v9 no-guard actor",
        "review_sha256": sha256(args.review),
        "old_report_sha256": sha256(args.old_report),
        "checkpoint_sha256": sha256(args.checkpoint),
        "guard_sha256": fixed["guard_sha256"],
        "rows": [], "failures": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        for target in targets:
            rule_id = target["rule_id"]
            current = worker.send({"command": "reset", "ruleset_id": rule_id,
                                   "seed": target["environment_seed"]})
            if (current["num_actions"] != 6 or
                    digest(state(current["observation"])) !=
                    target["initial_state_sha256"]):
                raise ValueError("Changed XLand reset")
            guard = ActionFeedbackGuard(2, 2)
            rng = random.Random(target["action_seed"])
            steps = []
            for _ in range(min(200, current["max_steps"])):
                before = state(current["observation"])
                model_action, prompt_hash, logits = select(agent, tokenizer,
                    choices, current["observation"], None, args.device,
                    steps[-4:])
                allowed = guard.allowed(before, range(6))
                action = max(allowed, key=lambda a: logits[a])
                if rng.random() < .2:
                    action = rng.choice(allowed)
                current = worker.send({"command": "step", "action": action})
                after = state(current["observation"])
                guard.observe(before, action, after, reward=current["reward"])
                steps.append({"state": before, "action": action,
                    "model_action": model_action, "next_state": after,
                    "reward": current["reward"], "done": current["done"],
                    "prompt_sha256": prompt_hash,
                    "choice_logits": logits})
                if current["done"]:
                    break
            episode = {"steps": steps,
                "return": sum(x["reward"] for x in steps),
                "positive": any(x["reward"] > 0 for x in steps),
                "ended": bool(steps and steps[-1]["done"]),
                "action_guard": guard.summary()}
            result["rows"].append({"index": target["index"],
                "rule_id": rule_id,
                "input_content_sha256": target["input_content_sha256"],
                "episode": episode,
                "old_no_guard_positive": old["rows"][target["index"]]
                    ["targets"]["none"]["positive"]})
            result["summary"] = {"n": len(result["rows"]),
                "guard_positive": sum(x["episode"]["positive"]
                                      for x in result["rows"]),
                "old_no_guard_positive": sum(x["old_no_guard_positive"]
                                              for x in result["rows"]),
                "guard_return": sum(x["episode"]["return"]
                                    for x in result["rows"])}
            tmp = args.output.with_suffix(args.output.suffix+".tmp")
            tmp.write_text(json.dumps(result, indent=2)+"\n")
            tmp.replace(args.output)
            print(json.dumps(result["summary"]), flush=True)
    except Exception as exc:
        result["failures"].append({"error": f"{type(exc).__name__}: {exc}",
                                   "completed_rows": len(result["rows"])})
        args.output.write_text(json.dumps(result, indent=2)+"\n")
        raise
    finally:
        worker.close()
        agent.mount(None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "evaluate"))
    parser.add_argument("--old-report", type=Path, default=Path(
        "results/trajectory_hyperlora/xland_mixed_v9_new_rules_online_seed42_20261007.json"))
    parser.add_argument("--checkpoint", type=Path, default=Path(
        "results/trajectory_hyperlora/xland_mixed_six_action_v9_seed42_20261007.pt"))
    parser.add_argument("--training-result", type=Path, default=Path(
        "results/trajectory_hyperlora/xland_mixed_six_action_v9_seed42_20261007.json"))
    parser.add_argument("--annotations", type=Path, default=Path(
        "data/annotations/xland_mixed_trivial_medium_reviewed_v9_20261007.json"))
    parser.add_argument("--benchmark-path", type=Path, default=Path(
        "data/xland_minigrid/trivial_1m_v2"))
    parser.add_argument("--model", type=Path, default=Path(
        "current_work/delta-Mem/model/Qwen3-4B-Instruct-2507"))
    parser.add_argument("--xland-python", type=Path, default=Path(
        "ttcl/.runtime/xland_env/bin/python"))
    parser.add_argument("--data-dir", type=Path, default=Path(
        "data/xland_minigrid"))
    parser.add_argument("--environment", default="XLand-MiniGrid-R1-9x9")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpu-fraction", type=float, default=.65)
    parser.add_argument("--review", type=Path, default=Path(
        "data/annotations/my_online_update_v29_xland_guard_20261009.json"))
    parser.add_argument("--output", type=Path, default=Path(
        "results/trajectory_hyperlora/my_online_update_v29_xland_guard_20261009.json"))
    args = parser.parse_args()
    (prepare if args.command == "prepare" else evaluate)(args)
