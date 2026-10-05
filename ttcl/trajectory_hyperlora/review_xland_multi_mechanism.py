"""Independently bind multi-mechanism targets to recorded public transitions."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from ttcl.trajectory_hyperlora.collect_xland_crossed_rules import digest
from ttcl.trajectory_hyperlora.collect_xland_multi_mechanism import ACTIONS, KINDS
from ttcl.trajectory_hyperlora.review_xland_rule_pairs import has_product


EXPECTED = {"hold": 3, "near": 0, "tile_near": 4}


def validate(candidate: dict) -> dict:
    used_ids, used_rule_hashes = set(), {}
    reviewed = {}
    for split in ("train", "dev", "test"):
        reviewed[split] = []
        for item in candidate["split"][split]:
            if item["item_id"] in used_ids:
                raise ValueError("Repeated item ID")
            used_ids.add(item["item_id"])
            rule_hash = item["rule_sha256"]
            if rule_hash in used_rule_hashes and used_rule_hashes[rule_hash] != split:
                raise ValueError("Rule content crosses split")
            used_rule_hashes[rule_hash] = split
            if (len({tuple(item[key]) for key in (
                    "source_tile", "product_tile", "partner_tile")}) != 3 or
                    set(item["arms"]) != set(KINDS) or
                    item["source_layout"] == item["target_layout"]):
                raise ValueError("Invalid three-mechanism item")
            target_states, source_states, source_actions = [], [], []
            arms = {}
            for kind in KINDS:
                arm = item["arms"][kind]
                history = arm["source_episodes"]
                target = arm["target_transitions"]
                if (digest(history) != arm["source_sha256"] or
                        len(history) != 3 or len(target) != 3):
                    raise ValueError("Source content or probe count changed")
                product = item["product_tile"]
                effects = []
                for index, action in enumerate(ACTIONS):
                    episode = history[index]
                    source_step = episode["steps"]
                    target_step = target[index]
                    if (episode["goal"] != product or
                            tuple(episode["plan_actions"]) != (action,) or
                            len(source_step) != 1 or
                            source_step[0]["action"] != action or
                            target_step["action"] != action):
                        raise ValueError("Probe action or goal mismatch")
                    for step in (source_step[0], target_step):
                        if not math.isfinite(step["reward"]):
                            raise ValueError("Nonfinite environment reward")
                    source_effect = has_product(source_step[0]["next_state"], product)
                    target_effect = has_product(target_step["next_state"], product)
                    if source_effect != target_effect:
                        raise ValueError("Source effect fails on target layout")
                    if bool(arm["first_step_effects"][str(action)]) != target_effect:
                        raise ValueError("Recorded effect is not public effect")
                    effects.append(target_effect)
                if has_product(target[0]["state"], product):
                    raise ValueError("Target product already present")
                if target[0]["state"] != target[1]["state"] or \
                        target[0]["state"] != target[2]["state"]:
                    raise ValueError("Target actions do not share initial state")
                if sum(effects) != 1 or ACTIONS[effects.index(True)] != EXPECTED[kind]:
                    raise ValueError("Target not exclusive or expected action changed")
                if arm["target_action"] != EXPECTED[kind]:
                    raise ValueError("Candidate label differs from audited transition")
                source_states.append(history[0]["steps"][0]["state"])
                source_actions.append(tuple(ep["steps"][0]["action"] for ep in history))
                target_states.append(target[0]["state"])
                model_input = {"source_episodes": history,
                               "target_initial_state": target[0]["state"],
                               "goal": product}
                arms[kind] = {"queries": [{"model_input": model_input,
                                            "input_sha256": digest(model_input),
                                            "target_action": EXPECTED[kind],
                                            "reviewed_target": True,
                                            "review_basis": "Independent action, effect, reward, split and content audit"}],
                              "rules_sha256": arm["rules_sha256"]}
            if (len({digest(value) for value in target_states}) != 1 or
                    len({digest(value) for value in source_states}) != 1 or
                    len(set(source_actions)) != 1 or
                    source_states[0] == target_states[0]):
                raise ValueError("Counterfactual public states or action budgets differ")
            reviewed[split].append({"item_id": item["item_id"],
                                    "rule_sha256": rule_hash,
                                    "source_layout": item["source_layout"],
                                    "target_layout": item["target_layout"],
                                    "arms": arms})
    return reviewed


def run(args: argparse.Namespace) -> dict:
    if args.output.exists():
        raise FileExistsError(args.output)
    raw = args.candidates.read_bytes()
    candidate = json.loads(raw)
    reviewed = validate(candidate)
    result = {"protocol": "Reviewed XLand three-mechanism action targets with content-bound public inputs",
              "candidates_sha256": hashlib.sha256(raw).hexdigest(),
              "split_plan_sha256": candidate["split_plan_sha256"],
              "asset_sha256": candidate["asset_sha256"],
              "failed_candidate_count": len(candidate["failed_candidates"]),
              "failed_candidate_ids": [item["item_id"] for item in candidate["failed_candidates"]],
              "split": reviewed}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"reviewed": {name: len(items) for name, items in
                                    reviewed.items()},
                      "failed": result["failed_candidate_count"]}), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
