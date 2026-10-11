"""Count raw evaluation episodes without mixing milestones and full runs."""
import argparse
import hashlib
import json
from pathlib import Path


def summarize(report):
    result = {"complete": report.get("complete", False), "stages": {}}
    for stage in ("shared", "onpolicy"):
        rows = report.get(stage, [])
        keys = [(r["task_id"], r["arm"], r["attempt_number"]) for r in rows]
        if len(keys) != len(set(keys)):
            raise ValueError(f"duplicate evaluation episodes in {stage}")
        arms = {}
        for arm in sorted({r["arm"] for r in rows}):
            selected = [r for r in rows if r["arm"] == arm]
            def count(items):
                return {"episodes": len(items), "positive_reward_episodes":
                        sum(bool(r["episode"]["positive"]) for r in items)}
            arms[arm] = {"all_recorded": count(selected), "milestones_only":
                         count([r for r in selected if r["attempt_number"] in (1, 2, 4, 8)])}
        result["stages"][stage] = arms
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("evaluation", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    raw = args.evaluation.read_bytes()
    result = summarize(json.loads(raw))
    result["source_sha256"] = hashlib.sha256(raw).hexdigest()
    text = json.dumps(result, indent=2) + "\n"
    if args.output:
        with args.output.open("x") as handle:
            handle.write(text)
    else:
        print(text, end="")
