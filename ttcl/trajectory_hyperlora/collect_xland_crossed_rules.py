"""Collect two-object XLand histories requiring different actions within one source.

The two hidden-rule arms have the same objects, goals, source probe actions,
action histogram, and total number of successful probes. They swap which
object transforms on approach versus pickup. Every target is a fresh public
scene and is scored from an actual environment transition.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


PLANS = {"approach_then_pick": (0, 3), "pick_now": (3,)}


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True,
                                    separators=(",", ":")).encode()).hexdigest()


def compatible(left: dict, right: dict) -> bool:
    tiles = (tuple(left["source_tile"]), tuple(left["product_tile"]),
             tuple(right["source_tile"]), tuple(right["product_tile"]))
    return len(set(tiles)) == 4


def pair_items(items: list[dict]) -> list[tuple[dict, dict]]:
    remaining = list(items)
    paired = []
    while remaining:
        left = remaining.pop(0)
        match = next((i for i, right in enumerate(remaining)
                      if compatible(left, right)), None)
        if match is not None:
            paired.append((left, remaining.pop(match)))
    return paired


def run(args: argparse.Namespace) -> dict:
    if args.output.exists():
        raise FileExistsError(args.output)
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ["XLAND_MINIGRID_DATA"] = str(args.data_dir.resolve())
    import jax
    import jax.numpy as jnp
    import numpy as np
    import xminigrid
    from xminigrid.core.constants import TILES_REGISTRY, Colors, Tiles
    from xminigrid.core.goals import AgentHoldGoal
    from xminigrid.core.grid import room
    from xminigrid.core.observation import transparent_field_of_view
    from xminigrid.core.rules import AgentHoldRule, AgentNearRule
    from xminigrid.types import AgentState, RuleSet

    raw = args.single_candidates.read_bytes()
    singles = json.loads(raw)
    if singles["failed_candidates"]:
        raise ValueError("The source candidate collection has failed items")
    env, default_params = xminigrid.make("XLand-MiniGrid-R1-9x9")
    step = jax.jit(env.step)

    def scene(params, source_tile, *, target: bool):
        object_pos, agent_pos, direction = (
            ((3, 4), (4, 4), 0) if target else ((4, 5), (4, 4), 1))
        timestep = env.reset(params, jax.random.key(20261005))
        grid = room(9, 9).at[object_pos].set(source_tile)
        if target:
            grid = grid.at[3, 3].set(TILES_REGISTRY[Tiles.GOAL, Colors.ORANGE])
        agent = AgentState(position=jnp.asarray(agent_pos),
                           direction=jnp.asarray(direction),
                           pocket=TILES_REGISTRY[Tiles.EMPTY, Colors.EMPTY])
        return timestep.replace(
            state=timestep.state.replace(grid=grid, agent=agent,
                                         step_num=jnp.asarray(0)),
            observation=transparent_field_of_view(grid, agent, 5, 5))

    def public(timestep):
        return {"observation": np.asarray(timestep.observation).tolist(),
                "pocket": np.asarray(timestep.state.agent.pocket).tolist()}

    def execute(params, tile, *, target: bool, actions: tuple[int, ...]):
        timestep = scene(params, tile, target=target)
        trace = []
        for action in actions:
            before = public(timestep)
            after = step(params, timestep, action)
            trace.append({"state": before, "action": action,
                          "next_state": public(after),
                          "reward": float(after.reward),
                          "done": bool(after.last())})
            timestep = after
            if bool(after.last()):
                break
        return trace

    def has_product(state, product):
        obs = np.asarray(state["observation"])
        return bool(np.any(np.all(obs == product, axis=-1)) or
                    np.array_equal(np.asarray(state["pocket"]), product))

    output = {"protocol": "Two-object crossed-rule XLand counterfactuals from real transitions; no reviewed training targets",
              "single_candidates_sha256": hashlib.sha256(raw).hexdigest(),
              "split_plan_sha256": singles["split_plan_sha256"],
              "asset_sha256": singles["asset_sha256"],
              "benchmark": "small-1m", "split": {}, "failed_candidates": []}
    for split in ("train", "dev", "test"):
        output["split"][split] = []
        for left, right in pair_items(singles["split"][split]):
            objects = (left, right)
            pair_id = digest([(item["ruleset_id"], item["source_tile"],
                               item["product_tile"]) for item in objects])
            arms = {}
            for variant, kinds in (("hold_near", ("hold", "near")),
                                   ("near_hold", ("near", "hold"))):
                tiles = [jnp.asarray(item["source_tile"], dtype=jnp.uint8)
                         for item in objects]
                products = [jnp.asarray(item["product_tile"], dtype=jnp.uint8)
                            for item in objects]
                rules = []
                for kind, tile, product in zip(kinds, tiles, products, strict=True):
                    rule = (AgentHoldRule(tile, product) if kind == "hold" else
                            AgentNearRule(tile, product))
                    rules.append(rule.encode())
                encoded_rules = jnp.stack(rules)
                source_episodes = []
                queries = []
                for index, (tile, product) in enumerate(zip(tiles, products,
                                                            strict=True)):
                    ruleset = RuleSet(goal=AgentHoldGoal(product).encode(),
                                      rules=encoded_rules,
                                      init_tiles=jnp.stack(tiles))
                    params = default_params.replace(ruleset=ruleset)
                    for plan, actions in PLANS.items():
                        source_episodes.append({
                            "goal": objects[index]["product_tile"],
                            "plan_actions": actions,
                            "steps": execute(params, tile, target=False,
                                             actions=actions)})
                    outcomes = {plan: execute(params, tile, target=True,
                                              actions=actions)
                                for plan, actions in PLANS.items()}
                    effects = {plan: has_product(rows[0]["next_state"],
                                                 product)
                               for plan, rows in outcomes.items()}
                    target_state = outcomes["pick_now"][0]["state"]
                    label = (0 if effects["approach_then_pick"] else
                             3 if effects["pick_now"] else None)
                    queries.append({"goal": objects[index]["product_tile"],
                                    "target_initial_state": target_state,
                                    "target_action": label,
                                    "first_step_effects": effects,
                                    "target_plan_rewards": {
                                        plan: sum(row["reward"] for row in rows)
                                        for plan, rows in outcomes.items()},
                                    "target_plan_transitions": outcomes})
                arms[variant] = {
                    "model_history": source_episodes,
                    "queries": queries,
                    "history_sha256": digest(source_episodes),
                    "rules_sha256": digest(np.asarray(encoded_rules).tolist()),
                    "reviewed_target": False,
                }
            first, second = arms["hold_near"], arms["near_hold"]
            same_source_actions = ([s["action"] for ep in first["model_history"]
                                    for s in ep["steps"]] ==
                                   [s["action"] for ep in second["model_history"]
                                    for s in ep["steps"]])
            same_target_states = all(
                a["target_initial_state"] == b["target_initial_state"]
                for a, b in zip(first["queries"], second["queries"], strict=True))
            opposite_per_object = ([q["target_action"] for q in first["queries"]]
                                   == [3, 0] and
                                   [q["target_action"] for q in second["queries"]]
                                   == [0, 3])
            source_reward_totals = [sum(
                step["reward"] for ep in arm["model_history"]
                for step in ep["steps"]) for arm in (first, second)]
            valid = (same_source_actions and same_target_states and
                     opposite_per_object and
                     abs(source_reward_totals[0] - source_reward_totals[1]) < 1e-6 and
                     all(q["target_initial_state"] !=
                         arm["model_history"][2 * index]["steps"][0]["state"]
                         for arm in (first, second)
                         for index, q in enumerate(arm["queries"])))
            row = {"pair_id": pair_id,
                   "objects": [{"ruleset_id": item["ruleset_id"],
                                "rule_sha256": item["ruleset_rule_sha256"],
                                "source_tile": item["source_tile"],
                                "product_tile": item["product_tile"]}
                               for item in objects],
                   "arms": arms}
            (output["split"][split] if valid else
             output["failed_candidates"]).append(row)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"paired": {name: len(items) for name, items in
                                  output["split"].items()},
                      "failed": len(output["failed_candidates"])}), flush=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path,
                        default=Path("data/xland_minigrid"))
    parser.add_argument("--single-candidates", type=Path, default=Path(
        "results/trajectory_hyperlora/xland_rule_pairs_candidates_v3_20261005.json"))
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
