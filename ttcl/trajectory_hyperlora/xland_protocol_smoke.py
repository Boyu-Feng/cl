"""CPU-only XLand-MiniGrid protocol smoke; no training or benchmark claims.

Run with an isolated environment containing xminigrid. Keep downloaded
rulesets and episode records in ignored data/results directories.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True,
                                    separators=(",", ":")).encode()).hexdigest()


def episode(env, params, key, actions: list[int]) -> dict:
    timestep = env.reset(params, key)
    rows = []
    for action in actions:
        if int(timestep.step_type) == 2:
            break
        before = timestep.observation.tolist()
        next_step = env.step(params, timestep, action)
        rows.append({"observation": before, "action": action,
                     "reward": float(next_step.reward),
                     "done": int(next_step.step_type) == 2,
                     "next_observation": next_step.observation.tolist()})
        timestep = next_step
    return {"rows": rows, "digest": digest(rows),
            "return": sum(row["reward"] for row in rows)}


def run(args: argparse.Namespace) -> dict:
    if args.output.exists():
        raise FileExistsError(args.output)
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ["XLAND_MINIGRID_DATA"] = str(args.data_dir.resolve())
    import jax  # optional, isolated dependency
    import xminigrid

    benchmark = xminigrid.load_benchmark(args.benchmark)
    env, default_params = xminigrid.make("XLand-MiniGrid-R1-9x9")
    cases = []
    for rule_id in args.rule_ids:
        ruleset = benchmark.get_ruleset(rule_id)
        payload = {"goal": ruleset.goal.tolist(),
                   "rules": ruleset.rules.tolist(),
                   "init_tiles": ruleset.init_tiles.tolist()}
        params = default_params.replace(ruleset=ruleset)
        actions = [random.Random(1000 + rule_id + i).randrange(
            env.num_actions(params)) for i in range(args.steps)]
        source = episode(env, params, jax.random.key(100 + rule_id), actions)
        replay = episode(env, params, jax.random.key(100 + rule_id), actions)
        target = episode(env, params, jax.random.key(200 + rule_id), actions)
        if source["digest"] != replay["digest"]:
            raise AssertionError("Identical ruleset, reset seed and actions diverged")
        cases.append({"ruleset_id": rule_id, "ruleset_sha256": digest(payload),
                      "source_reset_seed": 100 + rule_id,
                      "target_reset_seed": 200 + rule_id,
                      "source": source, "target": target,
                      "replay_identical": True})
    result = {"protocol": "XLand-MiniGrid CPU smoke only; fixed ruleset across source/target episodes; no learned policy or success evaluation",
              "xminigrid_version": xminigrid.__version__,
              "benchmark": args.benchmark, "num_rulesets": benchmark.num_rulesets(),
              "environment": "XLand-MiniGrid-R1-9x9",
              "num_actions": env.num_actions(default_params),
              "steps_requested": args.steps, "cases": cases}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"rulesets": result["num_rulesets"],
                      "actions": result["num_actions"],
                      "cases": [{"id": c["ruleset_id"],
                                 "source_rows": len(c["source"]["rows"]),
                                 "target_rows": len(c["target"]["rows"]),
                                 "replay_identical": c["replay_identical"]}
                                for c in cases]}), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path,
                        default=Path("data/xland_minigrid"))
    parser.add_argument("--rule-ids", type=int, nargs="+", default=(0, 1))
    parser.add_argument("--benchmark", choices=("trivial-21k", "small-1m"),
                        default="trivial-21k")
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.steps < 1 or any(index < 0 for index in args.rule_ids):
        parser.error("Positive steps and nonnegative ruleset IDs required")
    run(args)


if __name__ == "__main__":
    main()
