"""Audit crossed-rule XLand annotations with content-bound target queries."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from ttcl.trajectory_hyperlora.review_xland_rule_pairs import digest, has_product


def validate(candidate: dict) -> dict:
    if candidate["failed_candidates"]:
        raise ValueError("Failed pair candidates must stay outside review")
    used_ids: set[str] = set()
    used_rules: dict[str, str] = {}
    output = {}
    for split in ("train", "dev", "test"):
        output[split] = []
        for item in candidate["split"][split]:
            if item["pair_id"] in used_ids:
                raise ValueError("Duplicate pair ID")
            used_ids.add(item["pair_id"])
            for obj in item["objects"]:
                rule_sha = obj["rule_sha256"]
                if rule_sha in used_rules and used_rules[rule_sha] != split:
                    raise ValueError("Production rule crosses train/dev/test")
                used_rules[rule_sha] = split
            if len(item["objects"]) != 2 or set(item["arms"]) != {
                    "hold_near", "near_hold"}:
                raise ValueError("Pair lacks two objects or both hidden-rule arms")
            arms = item["arms"]
            if arms["hold_near"]["rules_sha256"] == arms["near_hold"]["rules_sha256"]:
                raise ValueError("Hidden rules were not swapped")
            reviewed_arms = {}
            for variant, expected in (("hold_near", (3, 0)),
                                      ("near_hold", (0, 3))):
                arm = arms[variant]
                if arm["reviewed_target"]:
                    raise ValueError("Candidate already claims review")
                history = arm["model_history"]
                if digest(history) != arm["history_sha256"] or len(history) != 4:
                    raise ValueError("Source history content mismatch")
                if [tuple(ep["plan_actions"]) for ep in history] != [
                        (0, 3), (3,), (0, 3), (3,)]:
                    raise ValueError("Source probe budgets are not matched")
                source_rewards = sum(step["reward"] for ep in history
                                     for step in ep["steps"])
                queries = []
                for index, (query, correct_action) in enumerate(zip(
                        arm["queries"], expected, strict=True)):
                    product = item["objects"][index]["product_tile"]
                    if query["goal"] != product or query["target_action"] != correct_action:
                        raise ValueError("Goal or target action mismatch")
                    if query["target_initial_state"] == history[2 * index]["steps"][0]["state"]:
                        raise ValueError("Target reuses source observation")
                    effects = query["first_step_effects"]
                    for plan, actions in (("approach_then_pick", (0, 3)),
                                          ("pick_now", (3,))):
                        source_ep = history[2 * index + (plan == "pick_now")]
                        if source_ep["goal"] != product:
                            raise ValueError("Source goal not bound to query")
                        if tuple(step["action"] for step in source_ep["steps"]) != actions:
                            raise ValueError("Source action sequence changed")
                        target = query["target_plan_transitions"][plan]
                        if tuple(step["action"] for step in target) != actions[:len(target)]:
                            raise ValueError("Target action sequence changed")
                        if target[0]["state"] != query["target_initial_state"]:
                            raise ValueError("Target public state differs by plan")
                        actual = has_product(target[0]["next_state"], product)
                        source_actual = has_product(source_ep["steps"][0]["next_state"],
                                                    product)
                        if actual != effects[plan] or source_actual != actual:
                            raise ValueError("Source-to-target effect did not transfer")
                        if abs(sum(step["reward"] for step in target) -
                               query["target_plan_rewards"][plan]) > 1e-6:
                            raise ValueError("Target reward sum mismatch")
                    if sum(effects.values()) != 1 or not effects[
                            "approach_then_pick" if correct_action == 0 else "pick_now"]:
                        raise ValueError("Target effect is not exclusive")
                    model_input = {"source_episodes": history,
                                   "target_initial_state": query["target_initial_state"],
                                   "goal": product}
                    queries.append({"model_input": model_input,
                                    "input_sha256": digest(model_input),
                                    "target_action": correct_action,
                                    "reviewed_target": True,
                                    "review_basis": "Independent full-item action, effect, reward, split and content-hash audit"})
                reviewed_arms[variant] = {"source_reward_total": source_rewards,
                                          "history_sha256": arm["history_sha256"],
                                          "queries": queries}
            first = arms["hold_near"]
            second = arms["near_hold"]
            if ([step["action"] for ep in first["model_history"] for step in ep["steps"]] !=
                    [step["action"] for ep in second["model_history"] for step in ep["steps"]] or
                    abs(reviewed_arms["hold_near"]["source_reward_total"] -
                        reviewed_arms["near_hold"]["source_reward_total"]) > 1e-6 or
                    any(a["target_initial_state"] != b["target_initial_state"]
                        for a, b in zip(first["queries"], second["queries"], strict=True))):
                raise ValueError("Global source statistics or target public states differ")
            output[split].append({"pair_id": item["pair_id"],
                                  "objects": item["objects"],
                                  "arms": reviewed_arms})
    return output


def run(args: argparse.Namespace) -> dict:
    if args.output.exists():
        raise FileExistsError(args.output)
    raw = args.candidates.read_bytes()
    candidate = json.loads(raw)
    reviewed = validate(candidate)
    result = {"protocol": "Reviewed two-object crossed-rule XLand annotations; only train split may optimize parameters",
              "candidates_sha256": hashlib.sha256(raw).hexdigest(),
              "split_plan_sha256": candidate["split_plan_sha256"],
              "asset_sha256": candidate["asset_sha256"],
              "split": reviewed}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({name: len(items) for name, items in reviewed.items()}),
          flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, default=Path(
        "results/trajectory_hyperlora/xland_crossed_candidates_v1_20261005.json"))
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
