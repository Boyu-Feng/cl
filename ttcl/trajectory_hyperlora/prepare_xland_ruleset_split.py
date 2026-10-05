"""Freeze exact-content-disjoint XLand ruleset IDs before trajectory collection."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def select_split(digests: list[str], seed: int, sizes: tuple[int, int, int]) -> dict:
    if len(digests) != len(set(digests)):
        raise ValueError("Duplicate ruleset contents require grouped splitting")
    if any(size < 1 for size in sizes) or sum(sizes) > len(digests):
        raise ValueError("Invalid split sizes")
    ranked = sorted(range(len(digests)), key=lambda index: hashlib.sha256(
        f"{seed}:{digests[index]}".encode()).digest())
    start = 0
    result = {}
    for name, size in zip(("train", "dev", "test"), sizes, strict=True):
        result[name] = [{"ruleset_id": index, "content_sha256": digests[index]}
                        for index in ranked[start:start + size]]
        start += size
    return result


BENCHMARK_FILES = {"trivial-21k": "trivial_21k",
                   "small-1m": "small_1m_v3"}


def run(args: argparse.Namespace) -> dict:
    if args.output.exists():
        raise FileExistsError(args.output)
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ["XLAND_MINIGRID_DATA"] = str(args.data_dir.resolve())
    import numpy as np
    import xminigrid

    benchmark = xminigrid.load_benchmark(args.benchmark)
    goals = np.asarray(benchmark.goals)
    rules = np.asarray(benchmark.rules)
    tiles = np.asarray(benchmark.init_tiles)
    # Split by production-rule encoding, not full ruleset identity: changing
    # only goal/initial tiles must never leak the same hidden rule into test.
    representative: dict[str, int] = {}
    for index in range(len(rules)):
        rule_digest = hashlib.sha256(rules[index].tobytes()).hexdigest()
        representative.setdefault(rule_digest, index)
    unique_rules = list(representative)
    selected = select_split(unique_rules, args.seed,
                            (args.train, args.dev, args.test))
    split = {}
    for name, rows in selected.items():
        split[name] = []
        for row in rows:
            rule_sha = row["content_sha256"]
            index = representative[rule_sha]
            content_sha = hashlib.sha256(
                goals[index].tobytes() + rules[index].tobytes() +
                tiles[index].tobytes()).hexdigest()
            split[name].append({"ruleset_id": index,
                                "rule_sha256": rule_sha,
                                "content_sha256": content_sha})
    asset = args.data_dir / BENCHMARK_FILES[args.benchmark]
    if not asset.is_file():
        raise FileNotFoundError(asset)
    result = {"protocol": "Production-rule-disjoint ruleset split selected by salted SHA-256; source and target episodes must stay within the same ruleset and split; no trajectories or reward labels included",
              "benchmark": args.benchmark, "xminigrid_version": xminigrid.__version__,
              "asset_sha256": hashlib.sha256(asset.read_bytes()).hexdigest(),
              "seed": args.seed, "total_rulesets": len(rules),
              "unique_rule_digests": len(unique_rules),
              "split": split}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"total": len(rules), "unique_rules": len(unique_rules),
                      "selected": {name: len(items) for name, items in split.items()}}),
          flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path,
                        default=Path("data/xland_minigrid"))
    parser.add_argument("--benchmark", choices=tuple(BENCHMARK_FILES),
                        default="small-1m")
    parser.add_argument("--seed", type=int, default=20261005)
    parser.add_argument("--train", type=int, default=64)
    parser.add_argument("--dev", type=int, default=16)
    parser.add_argument("--test", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
