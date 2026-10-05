"""Create newly reviewed bindings for trial-order robustness evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from ttcl.trajectory_hyperlora.collect_xland_crossed_rules import digest
from ttcl.trajectory_hyperlora.review_xland_multi_mechanism import (
    EXPECTED, validate,
)
from ttcl.trajectory_hyperlora.review_xland_rule_pairs import has_product


PERMUTATION = (2, 0, 1)


def run(args: argparse.Namespace) -> dict:
    if args.output.exists():
        raise FileExistsError(args.output)
    candidate_bytes = args.candidates.read_bytes()
    candidate = json.loads(candidate_bytes)
    validate(candidate)  # independent complete source/target action audit
    output = {"protocol": "New content-bound review for cyclically permuted source trials; target actions independently rederived from original environment transitions",
              "candidates_sha256": hashlib.sha256(candidate_bytes).hexdigest(),
              "permutation": PERMUTATION, "split": {}}
    for split, items in candidate["split"].items():
        output["split"][split] = []
        for item in items:
            arms = {}
            for kind, arm in item["arms"].items():
                history = [arm["source_episodes"][index]
                           for index in PERMUTATION]
                target = arm["target_transitions"]
                product = item["product_tile"]
                effects = [has_product(step["next_state"], product)
                           for step in target]
                if sum(effects) != 1 or target[effects.index(True)]["action"] != \
                        EXPECTED[kind]:
                    raise ValueError("New target audit failed")
                model_input = {"source_episodes": history,
                               "target_initial_state": arm["target_initial_state"],
                               "goal": product}
                arms[kind] = {"queries": [{"model_input": model_input,
                    "input_sha256": digest(model_input),
                    "target_action": EXPECTED[kind], "reviewed_target": True,
                    "review_basis": "Original real transitions rechecked; new ordered input content bound"}]}
            output["split"][split].append({"item_id": item["item_id"],
                                            "arms": arms})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({name: len(items) for name, items in
                      output["split"].items()}), flush=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
