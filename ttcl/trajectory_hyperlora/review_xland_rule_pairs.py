"""Independently audit every synthetic XLand label and bind it to input bytes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True,
                                    separators=(",", ":")).encode()).hexdigest()


def has_product(state: dict, product: list[int]) -> bool:
    if state["pocket"] == product:
        return True
    return any(tile == product for row in state["observation"] for tile in row)


def validate(candidate: dict) -> dict:
    if candidate["failed_candidates"]:
        raise ValueError("Candidate collection has failed items")
    seen_rules: set[str] = set()
    seen_pairs: set[tuple[int, ...]] = set()
    reviewed = {}
    for split in ("train", "dev", "test"):
        reviewed[split] = []
        for item in candidate["split"][split]:
            pair = tuple(item["source_tile"] + item["product_tile"])
            if pair in seen_pairs:
                raise ValueError("Source/product pair crosses the split")
            seen_pairs.add(pair)
            # Different transformation rows may originate from one ruleset;
            # only a split boundary must be disjoint.
            for other_split in ("train", "dev", "test"):
                if other_split == split:
                    continue
                if any(other["ruleset_rule_sha256"] == item["ruleset_rule_sha256"]
                       for other in candidate["split"][other_split]):
                    raise ValueError("Production rule crosses the split")
            seen_rules.add(item["ruleset_rule_sha256"])
            arms = item["arms"]
            if set(arms) != {"hold", "near"}:
                raise ValueError("Counterfactual rule arm missing")
            inputs = {}
            labels = {}
            for name, expected_action in (("hold", 3), ("near", 0)):
                arm = arms[name]
                if arm["reviewed_target"]:
                    raise ValueError("Candidate was already marked reviewed")
                model_input = {
                    "source_trials": arm["source_trials"],
                    "target_initial_state": arm["target_initial_state"],
                    "goal": item["product_tile"],
                }
                if digest(model_input) != arm["input_sha256"]:
                    raise ValueError("Input content binding mismatch")
                for plan, actions in (("approach_then_pick", (0, 3)),
                                      ("pick_now", (3,))):
                    source = arm["source_trials"][plan]
                    target = arm["target_plan_transitions"][plan]
                    if tuple(step["action"] for step in source) != actions:
                        raise ValueError("Source probe action sequence changed")
                    if tuple(step["action"] for step in target) != actions[:len(target)]:
                        raise ValueError("Target action sequence changed")
                    if abs(sum(step["reward"] for step in target) -
                           arm["target_plan_rewards"][plan]) > 1e-6:
                        raise ValueError("Target return mismatch")
                    if target[0]["state"] != arm["target_initial_state"]:
                        raise ValueError("Target reset mismatch")
                    actual = has_product(target[0]["next_state"],
                                         item["product_tile"])
                    if actual != arm["target_first_step_effects"][plan]:
                        raise ValueError("Target effect label contradicts transition")
                    if (has_product(source[0]["next_state"],
                                    item["product_tile"]) != actual):
                        raise ValueError("Source evidence does not transfer")
                effects = arm["target_first_step_effects"]
                if (sum(effects.values()) != 1 or
                        (effects["approach_then_pick"] and expected_action != 0) or
                        (effects["pick_now"] and expected_action != 3) or
                        arm["target_best_action"] != expected_action):
                    raise ValueError("Target action is ambiguous or wrong")
                if arm["target_plan_rewards"][arm["target_best_plan"]] <= (
                        arm["target_plan_rewards"]["approach_then_pick"]
                        if arm["target_best_plan"] == "pick_now" else
                        arm["target_plan_rewards"]["pick_now"]):
                    raise ValueError("Best plan does not win on environment return")
                inputs[name] = model_input
                labels[name] = expected_action
            if (inputs["hold"]["target_initial_state"] !=
                    inputs["near"]["target_initial_state"] or
                    inputs["hold"]["source_trials"]["pick_now"][0]["state"] !=
                    inputs["near"]["source_trials"]["pick_now"][0]["state"] or
                    inputs["hold"]["source_trials"]["pick_now"][0]["state"] ==
                    inputs["hold"]["target_initial_state"]):
                raise ValueError("Public state pairing or novel target failed")
            if (arms["hold"]["rule_encoding_sha256"] ==
                    arms["near"]["rule_encoding_sha256"]):
                raise ValueError("Counterfactual hidden rules are identical")
            reviewed[split].append({
                "ruleset_id": item["ruleset_id"],
                "ruleset_rule_sha256": item["ruleset_rule_sha256"],
                "source_tile": item["source_tile"],
                "product_tile": item["product_tile"],
                "arms": {name: {"model_input": inputs[name],
                                "input_sha256": arms[name]["input_sha256"],
                                "target_action": labels[name],
                                "reviewed_target": True,
                                "review_basis": "Independent full-item input-hash, transition-effect, action-sequence and environment-return audit"}
                         for name in ("hold", "near")},
            })
    return reviewed


def run(args: argparse.Namespace) -> dict:
    if args.output.exists():
        raise FileExistsError(args.output)
    raw = args.candidates.read_bytes()
    candidate = json.loads(raw)
    reviewed = validate(candidate)
    result = {"protocol": "Reviewed synthetic XLand target-action annotations; only train split may be used for optimization",
              "candidates_sha256": hashlib.sha256(raw).hexdigest(),
              "split_plan_sha256": candidate["split_plan_sha256"],
              "asset_sha256": candidate["asset_sha256"],
              "split": reviewed}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({name: len(rows) for name, rows in reviewed.items()}),
          flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, default=Path(
        "results/trajectory_hyperlora/xland_rule_pairs_candidates_v3_20261005.json"))
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
