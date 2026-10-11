"""Independent official-environment replay of the XLand generic guard pilot."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from ttcl.trajectory_hyperlora.action_feedback_guard_v29 import ActionFeedbackGuard
from ttcl.trajectory_hyperlora.evaluate_xland_official_live import (
    Worker, digest, sha256, state,
)


def audit(args):
    review = json.loads(args.review.read_text())
    report = json.loads(args.report.read_text())
    if (review["targets_sha256"] != digest(review["targets"]) or
            review["runner_sha256"] != sha256(args.runner) or
            review["guard_sha256"] != sha256(args.guard) or
            review["benchmark_sha256"] != sha256(args.benchmark_path) or
            review["checkpoint_sha256"] != sha256(args.checkpoint) or
            report["review_sha256"] != sha256(args.review) or
            report["guard_sha256"] != review["guard_sha256"] or
            report["checkpoint_sha256"] != review["checkpoint_sha256"] or
            report["failures"] or
            len(report["rows"]) > len(review["targets"])):
        raise ValueError("Changed XLand guard lineage or failure record")
    if not args.partial and len(report["rows"]) != len(review["targets"]):
        raise ValueError("Incomplete XLand result")
    worker = Worker(args.xland_python, args.data_dir, "trivial_1m_v2",
                    args.environment, args.benchmark_path)
    try:
        for row, target in zip(report["rows"], review["targets"]):
            content = {key: val for key, val in target.items()
                       if key not in ("input_content_sha256", "reviewed_target")}
            if (not target["reviewed_target"] or
                    target["input_content_sha256"] != digest(content) or
                    row["index"] != target["index"] or
                    row["rule_id"] != target["rule_id"] or
                    row["input_content_sha256"] !=
                    target["input_content_sha256"]):
                raise ValueError("Changed XLand target binding")
            current = worker.send({"command": "reset",
                "ruleset_id": target["rule_id"],
                "seed": target["environment_seed"]})
            if digest(state(current["observation"])) != \
                    target["initial_state_sha256"]:
                raise ValueError("Changed official reset")
            guard = ActionFeedbackGuard(2, 2)
            rng = random.Random(target["action_seed"])
            steps = row["episode"]["steps"]
            if len(steps) > 200:
                raise ValueError("Exceeded declared action budget")
            for step in steps:
                before = state(current["observation"])
                if before != step["state"] or current["done"]:
                    raise ValueError("Changed recorded state or extra action")
                logits = step["choice_logits"]
                if (len(logits) != 6 or
                        step["model_action"] != max(range(6),
                            key=lambda a: logits[a])):
                    raise ValueError("Changed action logits")
                allowed = guard.allowed(before, range(6))
                selected = max(allowed, key=lambda a: logits[a])
                if rng.random() < .2:
                    selected = rng.choice(allowed)
                if selected != step["action"]:
                    raise ValueError("Changed action guard or sampling")
                current = worker.send({"command": "step",
                                       "action": selected})
                after = state(current["observation"])
                if (after != step["next_state"] or
                        current["reward"] != step["reward"] or
                        current["done"] != step["done"]):
                    raise ValueError("Changed official environment feedback")
                guard.observe(before, selected, after,
                              reward=current["reward"])
            ep = row["episode"]
            if (ep["action_guard"] != guard.summary() or
                    ep["return"] != sum(x["reward"] for x in steps) or
                    ep["positive"] != any(x["reward"] > 0 for x in steps) or
                    ep["ended"] != bool(steps and steps[-1]["done"])):
                raise ValueError("Changed episode summary")
    finally:
        worker.close()
    positive = sum(row["episode"]["positive"] for row in report["rows"])
    if not args.partial and (report["summary"]["n"] != len(report["rows"])
            or report["summary"]["guard_positive"] != positive):
        raise ValueError("Changed final XLand summary")
    return {"audit": "passed", "episodes": len(report["rows"]),
        "positive": positive, "report_sha256": sha256(args.report),
        "review_sha256": sha256(args.review)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--review", type=Path, default=Path(
        "data/annotations/my_online_update_v29_xland_guard_20261009.json"))
    parser.add_argument("--report", type=Path, default=Path(
        "results/trajectory_hyperlora/my_online_update_v29_xland_guard_20261009.json"))
    parser.add_argument("--runner", type=Path, default=Path(
        "ttcl/trajectory_hyperlora/evaluate_xland_generic_guard_v29.py"))
    parser.add_argument("--guard", type=Path, default=Path(
        "ttcl/trajectory_hyperlora/action_feedback_guard_v29.py"))
    parser.add_argument("--checkpoint", type=Path, default=Path(
        "results/trajectory_hyperlora/xland_mixed_six_action_v9_seed42_20261007.pt"))
    parser.add_argument("--benchmark-path", type=Path, default=Path(
        "data/xland_minigrid/trivial_1m_v2"))
    parser.add_argument("--xland-python", type=Path, default=Path(
        "ttcl/.runtime/xland_env/bin/python"))
    parser.add_argument("--data-dir", type=Path, default=Path(
        "data/xland_minigrid"))
    parser.add_argument("--environment", default="XLand-MiniGrid-R1-9x9")
    parser.add_argument("--partial", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = audit(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result), flush=True)
