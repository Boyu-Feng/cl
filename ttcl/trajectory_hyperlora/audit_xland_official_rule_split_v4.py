"""Audit official benchmark rule-content isolation for live v4 probes.

Rule arrays are used for split verification only and never passed to the
actor, source encoder, or hypernetwork.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def run(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    import jax
    import numpy as np
    import xminigrid

    benchmark = xminigrid.benchmarks.load_benchmark_from_path(args.benchmark_path)
    annotation = json.loads(args.annotations.read_text())
    old_ids = {split: {int(row["ruleset_id"]) for row in rows}
               for split, rows in annotation["split"].items()}
    new_ids = set(args.rule_ids)
    if len(new_ids) != len(args.rule_ids) or new_ids.intersection(
            set().union(*old_ids.values())):
        raise ValueError("Repeated or reused ruleset ID")

    def content_hash(rule_id):
        rule = benchmark.get_ruleset(rule_id)
        digest = hashlib.sha256()
        for leaf in jax.tree_util.tree_leaves(rule):
            array = np.asarray(leaf)
            digest.update(str(array.dtype).encode())
            digest.update(str(array.shape).encode())
            digest.update(array.tobytes())
        return digest.hexdigest()

    old_hashes = {split: {content_hash(rule_id) for rule_id in ids}
                  for split, ids in old_ids.items()}
    new_hashes = {rule_id: content_hash(rule_id) for rule_id in args.rule_ids}
    collisions = {split: sorted(set(new_hashes.values()) & hashes)
                  for split, hashes in old_hashes.items()}
    if len(set(new_hashes.values())) != len(new_hashes) or any(collisions.values()):
        raise ValueError("New ruleset content overlaps prior splits")
    result = {"protocol": "Official benchmark rule-content split audit; hidden rules used only by auditor",
              "benchmark_path_sha256": hashlib.sha256(
                  args.benchmark_path.read_bytes()).hexdigest(),
              "annotations_sha256": hashlib.sha256(
                  args.annotations.read_bytes()).hexdigest(),
              "prior_split_rule_count": {name: len(ids) for name, ids in old_ids.items()},
              "new_rule_hashes": {str(k): value for k, value in new_hashes.items()},
              "collisions": collisions, "passed": True}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"passed": True, "new_rules": len(new_hashes),
        "prior_rules": sum(map(len, old_ids.values()))}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-path", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--rule-ids", nargs="+", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
