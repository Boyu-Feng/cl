"""Collect matched, informative XLand trajectories from real transitions.

Each pair shares the object, goal, layouts and probe action sequences; only
the hidden production rule changes. The source contains both candidate plans
executed in separate trials. A target label is recorded only after the real
environment shows a unique better plan under each rule, with opposite labels.
The output contains *candidates* and is not an approved training annotation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


PICKABLE = frozenset((3, 4, 5, 7, 11, 12))
PLANS = {"approach_then_pick": (0, 3), "pick_now": (3,)}


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True,
                                    separators=(",", ":")).encode()).hexdigest()


def eligible_pairs(benchmark, split_plan: dict) -> dict[str, list[dict]]:
    """Use only approved ruleset IDs; deduplicate source/product across splits."""
    import numpy as np

    used: set[tuple[int, ...]] = set()
    selected = {}
    for split in ("train", "dev", "test"):
        selected[split] = []
        for item in split_plan["split"][split]:
            ruleset_id = item["ruleset_id"]
            rules = np.asarray(benchmark.get_ruleset(ruleset_id).rules)
            for row_index, rule in enumerate(rules):
                pair = tuple(int(x) for x in rule[1:5])
                if (int(rule[0]) not in (1, 2) or pair[0] not in PICKABLE or
                        pair[2] not in PICKABLE or pair[:2] == pair[2:] or
                        pair in used):
                    continue
                used.add(pair)
                selected[split].append({"ruleset_id": ruleset_id,
                                        "ruleset_rule_sha256": item["rule_sha256"],
                                        "rule_row": row_index,
                                        "source_tile": pair[:2],
                                        "product_tile": pair[2:]})
    return selected


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

    split_bytes = args.split_plan.read_bytes()
    split_plan = json.loads(split_bytes)
    if split_plan["benchmark"] != "small-1m":
        raise ValueError("Expected the frozen small-1m rule split")
    asset = args.data_dir / "small_1m_v3"
    if hashlib.sha256(asset.read_bytes()).hexdigest() != split_plan["asset_sha256"]:
        raise ValueError("Ruleset asset differs from frozen split")
    benchmark = xminigrid.load_benchmark("small-1m")
    selected = eligible_pairs(benchmark, split_plan)
    env, default_params = xminigrid.make("XLand-MiniGrid-R1-9x9")
    step = jax.jit(env.step)

    def scene(params, source_tile, *, target: bool):
        # The same public room/object/agent state is used under both rule arms.
        if target:
            object_position, agent_position, direction = (3, 4), (4, 4), 0
        else:
            object_position, agent_position, direction = (4, 5), (4, 4), 1
        timestep = env.reset(params, jax.random.key(20261005))
        grid = room(9, 9).at[object_position].set(source_tile)
        if target:
            # An inert visible distractor makes the target observation novel;
            # the source/goal production rule never matches a GOAL tile.
            grid = grid.at[3, 3].set(TILES_REGISTRY[Tiles.GOAL, Colors.ORANGE])
        agent = AgentState(position=jnp.asarray(agent_position),
                           direction=jnp.asarray(direction),
                           pocket=TILES_REGISTRY[Tiles.EMPTY, Colors.EMPTY])
        state = timestep.state.replace(grid=grid, agent=agent, step_num=jnp.asarray(0))
        return timestep.replace(
            state=state,
            observation=transparent_field_of_view(grid, agent, 5, 5))

    def public(timestep):
        return {"observation": np.asarray(timestep.observation).tolist(),
                "pocket": np.asarray(timestep.state.agent.pocket).tolist()}

    def contains_product(public_state, product_tile):
        observed = np.asarray(public_state["observation"])
        pocket = np.asarray(public_state["pocket"])
        product = np.asarray(product_tile)
        return bool(np.any(np.all(observed == product, axis=-1)) or
                    np.array_equal(pocket, product))

    def execute(params, tile, *, target: bool, actions: tuple[int, ...]):
        timestep = scene(params, tile, target=target)
        rows = []
        for action in actions:
            before = public(timestep)
            next_step = step(params, timestep, action)
            rows.append({"state": before, "action": action,
                         "next_state": public(next_step),
                         "reward": float(next_step.reward),
                         "done": bool(next_step.last())})
            timestep = next_step
            if bool(timestep.last()):
                break
        return rows

    output = {"protocol": "Paired XLand real-transition probe; same source/target public states and same source probe actions under hold-vs-near hidden rules; no reviewed training targets",
              "benchmark": "small-1m", "xminigrid_version": xminigrid.__version__,
              "split_plan_sha256": hashlib.sha256(split_bytes).hexdigest(),
              "asset_sha256": split_plan["asset_sha256"],
              "plans": PLANS, "split": {}, "failed_candidates": []}
    for split, items in selected.items():
        output["split"][split] = []
        for item in items:
            tile = jnp.asarray(item["source_tile"], dtype=jnp.uint8)
            product = jnp.asarray(item["product_tile"], dtype=jnp.uint8)
            arms = {}
            for name, rule in (("hold", AgentHoldRule(tile, product)),
                               ("near", AgentNearRule(tile, product))):
                ruleset = RuleSet(goal=AgentHoldGoal(product).encode(),
                                  rules=rule.encode()[None, :],
                                  init_tiles=tile[None, :])
                params = default_params.replace(ruleset=ruleset)
                source = {plan: execute(params, tile, target=False, actions=actions)
                          for plan, actions in PLANS.items()}
                outcomes = {plan: execute(params, tile, target=True, actions=actions)
                            for plan, actions in PLANS.items()}
                rewards = {plan: sum(row["reward"] for row in rows)
                           for plan, rows in outcomes.items()}
                first_step_effects = {
                    plan: contains_product(rows[0]["next_state"], product)
                    for plan, rows in outcomes.items()}
                best = max(rewards, key=rewards.get)
                effect_action = (0 if first_step_effects["approach_then_pick"]
                                 else 3 if first_step_effects["pick_now"] else None)
                public_target = outcomes["pick_now"][0]["state"]
                model_input = {"source_trials": source,
                               "target_initial_state": public_target,
                               "goal": item["product_tile"]}
                arms[name] = {"source_trials": source,
                              "target_initial_state": public_target,
                              "target_plan_rewards": rewards,
                              "target_first_step_effects": first_step_effects,
                              "target_best_action": effect_action,
                              "target_plan_transitions": outcomes,
                              "target_best_plan": best,
                              "input_sha256": digest(model_input),
                              "reviewed_target": False,
                              "rule_encoding_sha256": digest(
                                  np.asarray(rule.encode()).tolist())}
            valid = (arms["hold"]["target_best_action"] == 3 and
                     arms["near"]["target_best_action"] == 0 and
                     sum(arms["hold"]["target_first_step_effects"].values()) == 1 and
                     sum(arms["near"]["target_first_step_effects"].values()) == 1 and
                     arms["hold"]["target_best_plan"] == "pick_now" and
                     arms["near"]["target_best_plan"] == "approach_then_pick" and
                     arms["hold"]["target_initial_state"] ==
                     arms["near"]["target_initial_state"] and
                     arms["hold"]["source_trials"]["pick_now"][0]["state"] !=
                     arms["hold"]["target_initial_state"] and
                     [row["action"] for trial in arms["hold"]["source_trials"].values()
                      for row in trial] ==
                     [row["action"] for trial in arms["near"]["source_trials"].values()
                      for row in trial])
            if not valid:
                output["failed_candidates"].append({"split": split, **item,
                                                      "arms": arms})
                continue
            output["split"][split].append({**item, "arms": arms})
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
    parser.add_argument("--split-plan", type=Path, default=Path(
        "results/trajectory_hyperlora/xland_small_rule_split_20261005.json"))
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
