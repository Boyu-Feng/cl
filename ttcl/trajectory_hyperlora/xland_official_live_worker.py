"""JSONL bridge to the official XLand-MiniGrid medium-1m environment.

Run only in the isolated xminigrid/JAX environment. No expert actions or
ruleset internals are sent to the actor.
"""

from __future__ import annotations

import json
import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import xminigrid


def main() -> None:
    benchmark = xminigrid.load_benchmark("medium-1m")
    env, defaults = xminigrid.make("XLand-MiniGrid-R1-13x13")
    timestep = None
    params = None
    for line in sys.stdin:
        try:
            request = json.loads(line)
            if request["command"] == "reset":
                rule_id = int(request["ruleset_id"])
                seed = int(request["seed"])
                params = defaults.replace(ruleset=benchmark.get_ruleset(rule_id))
                timestep = env.reset(params, jax.random.key(seed))
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
                timestep = env.step(params, timestep, action)
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
