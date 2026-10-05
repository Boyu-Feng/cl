"""Collect three-mechanism XLand counterfactuals with varied layouts.

Environment rules are used solely to generate and verify candidate data. The
model input contains public observations, actions, rewards, and goals only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from pathlib import Path

from ttcl.trajectory_hyperlora.collect_xland_crossed_rules import digest


ACTIONS = (0, 3, 4)
KINDS = ("hold", "near", "tile_near")


def layouts(seed: int) -> tuple[tuple, tuple]:
    rng = random.Random(seed)
    choices = []
    directions = ((-1, 0), (0, 1), (1, 0), (0, -1))
    for y in range(2, 7):
        for x in range(2, 7):
            for direction, (dy, dx) in enumerate(directions):
                for side in (-1, 1):
                    by, bx = y + dy - side * dx, x + dx + side * dy
                    ay, ax = y + 2 * dy, x + 2 * dx
                    if all(1 <= value <= 7 for value in (ay, ax, by, bx)):
                        choices.append((y, x, direction, side))
    source = rng.choice(choices)
    target = rng.choice([choice for choice in choices
                         if choice[:3] != source[:3] and choice[2] != source[2]])
    return source, target


def eligible_items(single_candidates: dict) -> dict[str, list[tuple[dict, dict]]]:
    output = {}
    for split, items in single_candidates["split"].items():
        candidates = []
        for index, item in enumerate(items):
            partner = next((other for offset in range(1, len(items))
                            if (other := items[(index + offset) % len(items)]) and
                            tuple(other["source_tile"]) not in {
                                tuple(item["source_tile"]), tuple(item["product_tile"])}), None)
            if partner is not None:
                candidates.append((item, partner))
        output[split] = candidates
    return output


def run(args: argparse.Namespace) -> dict:
    if args.output.exists():
        raise FileExistsError(args.output)
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ["XLAND_MINIGRID_DATA"] = str(args.data_dir.resolve())
    import jax
    import jax.numpy as jnp
    import numpy as np
    import xminigrid
    from xminigrid.core.constants import DIRECTIONS
    from xminigrid.core.goals import AgentHoldGoal
    from xminigrid.core.grid import room
    from xminigrid.core.observation import transparent_field_of_view
    from xminigrid.core.rules import AgentHoldRule, AgentNearRule, TileNearRule
    from xminigrid.types import AgentState, RuleSet

    raw = args.single_candidates.read_bytes()
    singles = json.loads(raw)
    if singles["failed_candidates"]:
        raise ValueError("Failed source candidates must stay visible")
    env, default_params = xminigrid.make("XLand-MiniGrid-R1-9x9")
    step = jax.jit(env.step)
    selected = eligible_items(singles)

    def scene(params, source, partner, layout):
        y, x, direction, side = layout
        dy, dx = map(int, np.asarray(DIRECTIONS)[direction])
        front = (y + dy, x + dx)
        ahead = (y + 2 * dy, x + 2 * dx)
        beside = (front[0] - side * dx, front[1] + side * dy)
        timestep = env.reset(params, jax.random.key(20261005))
        grid = room(9, 9).at[ahead].set(source).at[beside].set(partner)
        agent = AgentState(position=jnp.asarray((y, x)),
                           direction=jnp.asarray(direction), pocket=source)
        return timestep.replace(
            state=timestep.state.replace(grid=grid, agent=agent,
                                         step_num=jnp.asarray(0)),
            observation=transparent_field_of_view(grid, agent, 5, 5))

    def public(timestep):
        return {"observation": np.asarray(timestep.observation).tolist(),
                "pocket": np.asarray(timestep.state.agent.pocket).tolist()}

    def execute(params, source, partner, layout, action):
        before = scene(params, source, partner, layout)
        after = step(params, before, action)
        return {"state": public(before), "action": action,
                "next_state": public(after), "reward": float(after.reward),
                "done": bool(after.last())}

    def has_product(state, product):
        obs = np.asarray(state["observation"])
        return bool(np.any(np.all(obs == product, axis=-1)) or
                    np.array_equal(np.asarray(state["pocket"]), product))

    output = {"protocol": "Real XLand first-step transitions for hold, agent-near and tile-near rules; random public layouts; labels unreviewed",
              "source_candidates_sha256": hashlib.sha256(raw).hexdigest(),
              "split_plan_sha256": singles["split_plan_sha256"],
              "asset_sha256": singles["asset_sha256"],
              "seed": args.seed, "split": {}, "failed_candidates": []}
    for split, items in selected.items():
        output["split"][split] = []
        for item, partner_item in items[:args.max_per_split]:
            source = jnp.asarray(item["source_tile"], dtype=jnp.uint8)
            product = jnp.asarray(item["product_tile"], dtype=jnp.uint8)
            partner = jnp.asarray(partner_item["source_tile"], dtype=jnp.uint8)
            item_id = digest([item["ruleset_id"], item["source_tile"],
                              item["product_tile"], partner_item["source_tile"]])
            source_layout, target_layout = layouts(
                args.seed ^ int(item_id[:8], 16))
            arms = {}
            for kind, rule in (("hold", AgentHoldRule(source, product)),
                               ("near", AgentNearRule(source, product)),
                               ("tile_near", TileNearRule(source, partner, product))):
                ruleset = RuleSet(goal=AgentHoldGoal(product).encode(),
                                  rules=rule.encode()[None, :],
                                  init_tiles=jnp.stack((source, partner)))
                params = default_params.replace(ruleset=ruleset)
                history = [{"goal": item["product_tile"], "plan_actions": [action],
                            "steps": [execute(params, source, partner,
                                              source_layout, action)]}
                           for action in ACTIONS]
                target = [execute(params, source, partner, target_layout, action)
                          for action in ACTIONS]
                effects = {str(action): has_product(row["next_state"], product)
                           for action, row in zip(ACTIONS, target, strict=True)}
                valid_actions = [action for action in ACTIONS if effects[str(action)]]
                arms[kind] = {"source_episodes": history,
                              "target_initial_state": target[0]["state"],
                              "target_transitions": target,
                              "first_step_effects": effects,
                              "target_action": valid_actions[0] if len(valid_actions) == 1 else None,
                              "source_sha256": digest(history),
                              "rules_sha256": digest(np.asarray(rule.encode()).tolist())}
            row = {"item_id": item_id, "ruleset_id": item["ruleset_id"],
                   "rule_sha256": item["ruleset_rule_sha256"],
                   "source_tile": item["source_tile"],
                   "product_tile": item["product_tile"],
                   "partner_tile": partner_item["source_tile"],
                   "source_layout": source_layout, "target_layout": target_layout,
                   "arms": arms}
            expected = dict(zip(KINDS, (3, 0, 4), strict=True))
            shared_target = len({digest(arm["target_initial_state"])
                                 for arm in arms.values()}) == 1
            shared_actions = len({tuple(step["steps"][0]["action"] for step in
                                        arm["source_episodes"])
                                  for arm in arms.values()}) == 1
            source_changes = [digest(arm["source_episodes"][0]["steps"][0]["state"])
                              for arm in arms.values()]
            valid = (shared_target and shared_actions and len(set(source_changes)) == 1 and
                     all(arms[kind]["target_action"] == action
                         for kind, action in expected.items()) and
                     all(arm["source_episodes"][0]["steps"][0]["state"] !=
                         arm["target_initial_state"] for arm in arms.values()))
            (output["split"][split] if valid else
             output["failed_candidates"]).append(row)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"items": {name: len(items) for name, items in
                                output["split"].items()},
                      "failed": len(output["failed_candidates"])}), flush=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path,
                        default=Path("data/xland_minigrid"))
    parser.add_argument("--single-candidates", type=Path, default=Path(
        "results/trajectory_hyperlora/xland_rule_pairs_candidates_large_20261005.json"))
    parser.add_argument("--seed", type=int, default=202610052)
    parser.add_argument("--max-per-split", type=int, default=1000000)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
