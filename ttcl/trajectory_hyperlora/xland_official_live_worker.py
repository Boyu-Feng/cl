"""JSONL bridge to a version-bound official XLand-MiniGrid environment.

Run only in the isolated xminigrid/JAX environment. No expert actions or
ruleset internals are sent to the actor.
"""

from __future__ import annotations

import json
import os
import sys
import argparse

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import xminigrid


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", default="medium-1m")
    parser.add_argument("--benchmark-path")
    parser.add_argument("--environment", default="XLand-MiniGrid-R1-13x13")
    args = parser.parse_args()
    benchmark = (xminigrid.benchmarks.load_benchmark_from_path(
        args.benchmark_path) if args.benchmark_path else
        xminigrid.load_benchmark(args.benchmark))
    env, defaults = xminigrid.make(args.environment)
    reset_fn = jax.jit(env.reset)
    step_fn = jax.jit(env.step)
    timestep = None
    params = None
    for line in sys.stdin:
        try:
            request = json.loads(line)
            if request["command"] == "reset":
                rule_id = int(request["ruleset_id"])
                seed = int(request["seed"])
                params = defaults.replace(ruleset=benchmark.get_ruleset(rule_id))
                timestep = reset_fn(params, jax.random.key(seed))
                response = {"observation": timestep.observation.tolist(),
                            "reward": float(timestep.reward),
                            "done": int(timestep.step_type) == 2,
                            "num_actions": int(env.num_actions(params)),
                            "max_steps": int(params.max_steps)}
            elif request["command"] == "step":
                if timestep is None or int(timestep.step_type) == 2:
                    raise ValueError("Reset required before stepping")
                action = int(request["action"])
                if not 0 <= action < env.num_actions(params):
                    raise ValueError("Action outside official environment")
                timestep = step_fn(params, timestep,
                                   jax.numpy.asarray(action, dtype=jax.numpy.int32))
                response = {"observation": timestep.observation.tolist(),
                            "reward": float(timestep.reward),
                            "done": int(timestep.step_type) == 2}
            else:
                raise ValueError("Unknown command")
            print(json.dumps(response, separators=(",", ":")), flush=True)
        except Exception as exc:
            print(json.dumps({"error": f"{type(exc).__name__}: {exc}"}),
                  flush=True)


if __name__ == "__main__":
    main()
