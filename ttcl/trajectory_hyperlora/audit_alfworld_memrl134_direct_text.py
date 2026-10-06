"""Merge and audit disjoint same-source ALFWorld 134 direct-text segments."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path

from ttcl.trajectory_hyperlora.alfworld_memrl134_transfer import checked_targets
from ttcl.trajectory_hyperlora.alfworld_same_game_retry import file_hash
from ttcl.trajectory_hyperlora.alfworld_sibling_transfer import sibling_memory_text


def exact_sign_p(gains: int, losses: int) -> float:
    n = gains + losses
    return min(1.0, 2.0 * sum(math.comb(n, index)
        for index in range(max(gains, losses), n + 1)) / 2 ** n) if n else 1.0


def run(args: argparse.Namespace) -> dict:
    if args.output.exists():
        raise FileExistsError(args.output)
    targets = checked_targets(args)
    prior = json.loads(args.lora_result.read_text())
    if (len(targets) != 134 or len(prior["games"]) != 134 or
            prior["target_review_sha256"] != file_hash(args.target_review) or
            prior["summary"]["overall"]["base"] != 25 or
            prior["summary"]["overall"]["lora"] != 47):
        raise ValueError("Frozen LoRA target lineage changed")
    old = {item["game"]: item for item in prior["games"]}
    files = [args.prefix, *args.chunks]
    expected_offsets = [0, 55, 82, 108]
    expected_counts = [55, 27, 26, 26]
    if len(files) != 4:
        raise ValueError("Expected interrupted prefix and three disjoint chunks")
    merged = []
    for path, offset, count in zip(files, expected_offsets,
                                   expected_counts, strict=True):
        piece = json.loads(path.read_text())
        if (piece["target_review_sha256"] != file_hash(args.target_review) or
                piece["checkpoint_sha256"] != prior["checkpoint_sha256"] or
                piece["lora_result_sha256"] != file_hash(args.lora_result) or
                piece["failures"] or len(piece["games"]) != count or
                piece["summary"]["n"] != count or
                piece.get("offset", offset) != offset or
                piece.get("limit", count) != count):
            raise ValueError(f"Incomplete or changed text segment: {path}")
        expected = targets[offset:offset + count]
        if [item["game"] for item in piece["games"]] != \
                [row["game"] for row in expected]:
            raise ValueError(f"Text segment target order changed: {path}")
        merged.extend(piece["games"])
    if len(merged) != 134 or len({item["game"] for item in merged}) != 134:
        raise ValueError("Merged text targets are incomplete or repeated")
    families = defaultdict(lambda: {"n": 0, "base": 0, "lora": 0,
                                    "text": 0})
    discordance = {"lora_only": 0, "text_only": 0, "both": 0, "neither": 0}
    for row, item in zip(targets, merged, strict=True):
        previous = old[row["game"]]
        episode = item["text"]
        memory = sibling_memory_text(row["source_records"])
        expected_hash = hashlib.sha256(memory.encode()).hexdigest()
        if (item["target_input_content_sha256"] !=
                row["input_content_sha256"] or
                item["source_records_sha256"] != row["source_records_sha256"] or
                item["family"] != row["family"] or
                episode["status"] != "complete" or
                episode["reward"] not in (0., 1.) or
                episode["memory_sha256"] != expected_hash or
                episode["initial_observation"] !=
                previous["arms"]["base"]["initial_observation"] or
                episode["initial_commands_sha256"] !=
                previous["arms"]["base"]["initial_commands_sha256"]):
            raise ValueError(f"Text source, reset, or result changed: {row['game']}")
        value = families[row["family"]]
        value["n"] += 1
        value["base"] += previous["arms"]["base"]["reward"]
        value["lora"] += previous["arms"]["lora"]["reward"]
        value["text"] += episode["reward"]
        lora, text = previous["arms"]["lora"]["reward"], episode["reward"]
        key = ("both" if lora and text else "lora_only" if lora else
               "text_only" if text else "neither")
        discordance[key] += 1
    summary = {
        "n": 134, "failures": 0,
        "base": sum(value["base"] for value in families.values()),
        "lora": sum(value["lora"] for value in families.values()),
        "text": sum(value["text"] for value in families.values()),
        "by_family": dict(sorted(families.items())),
        "paired_lora_vs_text": {**discordance,
            "exact_sign_p_two_sided": exact_sign_p(
                discordance["lora_only"], discordance["text_only"])},
    }
    result = {
        "protocol": "Merged interrupted prefix (exit 130 after last completed game) plus three disjoint same-protocol text chunks; 134/134 official valid_unseen targets content-bound; no duplicated/omitted targets, no failed episodes; paired with existing frozen LoRA and base outcomes",
        "target_review_sha256": file_hash(args.target_review),
        "lora_result_sha256": file_hash(args.lora_result),
        "segments": [{"path": str(path), "sha256": file_hash(path),
                      "offset": offset, "count": count}
                     for path, offset, count in zip(files, expected_offsets,
                                                    expected_counts, strict=True)],
        "summary": summary,
        "games": merged,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--memrl-plan", type=Path, default=Path(
        "results/memrl_comparison/20260928_budgeted/plan.json"))
    parser.add_argument("--source-candidates", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_sibling_train600_candidates_20261006.json"))
    parser.add_argument("--source-review", type=Path, default=Path(
        "data/annotations/alf_sibling_train600_reviewed_20261006.json"))
    parser.add_argument("--retry-candidates", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_same_game_retry_candidates_20261006.json"))
    parser.add_argument("--retry-review", type=Path, default=Path(
        "data/annotations/alf_same_game_retry_reviewed_20261006.json"))
    parser.add_argument("--all-source-review", type=Path, default=Path(
        "data/annotations/alf_expert_first_attempts_reviewed_20261006.json"))
    parser.add_argument("--target-review", type=Path, default=Path(
        "data/annotations/alf_memrl134_hyperlora_train_sources_reviewed_20261006.json"))
    parser.add_argument("--data-root", type=Path, default=Path(
        "ttcl/data/alfworld_delta"))
    parser.add_argument("--lora-result", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_memrl134_simple40_reward_lora_audited_20261006.json"))
    parser.add_argument("--prefix", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_memrl134_direct_text_20261007.json"))
    parser.add_argument("--chunks", type=Path, nargs=3, default=[Path(
        f"results/trajectory_hyperlora/alf_memrl134_direct_text_chunk_{offset}_{count}_20261007.json")
        for offset, count in ((55, 27), (82, 26), (108, 26))])
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
