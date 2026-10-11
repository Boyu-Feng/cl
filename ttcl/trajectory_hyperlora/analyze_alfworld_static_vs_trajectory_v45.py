"""Paired 134-task analysis after environment audits of v40-v42 arms."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from math import comb
from pathlib import Path

from ttcl.trajectory_hyperlora.alfworld_online_from_empty import file_hash
from ttcl.trajectory_hyperlora.alfworld_persistent_failure_delta_v36 import save


def audited(report_path, audit_path):
    report = json.loads(report_path.read_text())
    audit = json.loads(audit_path.read_text())
    if (audit.get("audit") != "passed" or
            audit.get("report_sha256") != file_hash(report_path)):
        raise ValueError(f"Unaudited or changed report: {report_path}")
    return report


def pair(a, b):
    left = sum(bool(x) and not bool(y) for x, y in zip(a, b, strict=True))
    right = sum(not bool(x) and bool(y) for x, y in zip(a, b, strict=True))
    n = left + right
    p = min(1., 2 * sum(comb(n, i) for i in range(min(left, right) + 1)) / 2**n) if n else 1.
    return {"first_only": left, "second_only": right,
            "discordant": n, "exact_two_sided_p": p}


def analyze(args):
    corrected = audited(args.corrected, args.corrected_audit)["rows"]
    sft = audited(args.sft, args.sft_audit)["rows"]
    grpo = audited(args.grpo, args.grpo_audit)["rows"]
    rotated = audited(args.rotated, args.rotated_audit)["rows"]
    old = json.loads(args.retry.read_text())
    retry_audit = json.loads(args.retry_audit.read_text())
    if (retry_audit.get("audit") != "passed" or
            retry_audit.get("report_sha256") != file_hash(args.retry)):
        raise ValueError("Changed failure-retry report")
    if any(len(rows) != 134 for rows in (corrected, sft, grpo, rotated)):
        raise ValueError("Changed sample count")
    if any((corrected[i]["game"] != sft[i]["game"] or
            corrected[i]["game"] != grpo[i]["game"] or
            corrected[i]["game"] != rotated[i]["game"] or
            corrected[i]["game"] != old["games"][i]["game"])
           for i in range(134)):
        raise ValueError("Arm task order changed")
    values = {
        "base": [x["base_reward"] for x in sft],
        "old_hyper": [x["old_hyper_reward"] for x in sft],
        "predicted_correction_all_on": [x["episode"]["reward"]
            for x in corrected],
        "static_sft_B_only": [x["episode"]["reward"] for x in sft],
        "static_grpo_B_only": [x["episode"]["reward"] for x in grpo],
        "rotated_success_source": [x["episode"]["reward"]
            for x in rotated],
    }
    if (sum(values["base"]) != 41 or sum(values["old_hyper"]) != 68 or
            sum(values["predicted_correction_all_on"]) != 64 or
            sum(values["static_sft_B_only"]) != 62 or
            sum(values["static_grpo_B_only"]) != 47 or
            sum(values["rotated_success_source"]) != 74):
        raise ValueError("Changed reported official successes")
    if any((corrected[i]["old_first_reward"] != values["old_hyper"][i] or
            grpo[i]["old_hyper_reward"] != values["old_hyper"][i] or
            rotated[i]["old_hyper_reward"] != values["old_hyper"][i])
           for i in range(134)):
        raise ValueError("Changed paired old hypernetwork reward")
    retry_wins = [x["first_reward"] or x["retries"]["away_from_failure"]["reward"]
                  for x in old["games"]]
    if sum(retry_wins) != 83:
        raise ValueError("Changed same-task retry result")
    families = defaultdict(list)
    for i, row in enumerate(sft):
        families[row["game"].split("/")[2].split("-")[0]].append(i)
    family_summary = {name: {arm: sum(values[arm][i] for i in indices)
                              for arm in values}
        for name, indices in sorted(families.items())}
    return {"protocol": "Paired descriptive analysis on repeatedly exposed official valid_unseen 134, equal first-attempt environment budget except historical failed-only retry",
        "report_sha256": {name: file_hash(path) for name, path in {
            "corrected": args.corrected, "sft": args.sft,
            "grpo": args.grpo, "rotated": args.rotated,
            "retry": args.retry}.items()},
        "scores": {name: sum(rewards) for name, rewards in values.items()},
        "historical_two_attempt_failure_retry": sum(retry_wins),
        "paired": {
            "old_hyper_vs_static_sft": pair(values["old_hyper"], values["static_sft_B_only"]),
            "old_hyper_vs_static_grpo": pair(values["old_hyper"], values["static_grpo_B_only"]),
            "old_hyper_vs_rotated_source": pair(values["old_hyper"], values["rotated_success_source"]),
            "old_hyper_vs_predicted_correction": pair(values["old_hyper"], values["predicted_correction_all_on"]),
            "static_sft_vs_static_grpo": pair(values["static_sft_B_only"], values["static_grpo_B_only"]),
        },
        "oracle_gate_upper_bound_for_predicted_head": sum(max(a, b) for a, b in zip(
            values["old_hyper"], values["predicted_correction_all_on"], strict=True)),
        "family_scores": family_summary,
        "n": 134}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    names = {"corrected": "alfworld_failure_distill_v40_all_on_134_20261010.json",
        "corrected_audit": "alfworld_failure_distill_v40_all_on_134_audit_20261010.json",
        "sft": "alfworld_static_lora_sft_v41_unseen134_20261010.json",
        "sft_audit": "alfworld_static_lora_sft_v41_unseen134_audit_20261010.json",
        "grpo": "alfworld_static_lora_grpo_v41b_unseen134_20261010.json",
        "grpo_audit": "alfworld_static_lora_grpo_v41b_unseen134_audit_20261010.json",
        "rotated": "alfworld_source_permutation_v42_134_20261010.json",
        "rotated_audit": "alfworld_source_permutation_v42_134_audit_20261010.json",
        "retry": "alfworld_failure_retry_v32_134_20261009.json",
        "retry_audit": "alfworld_failure_retry_v32_134_audit_20261009.json"}
    for name, value in names.items():
        p.add_argument("--" + name.replace("_", "-"), type=Path,
                       default=Path("results/trajectory_hyperlora") / value)
    p.add_argument("--output", type=Path, default=Path(
        "results/trajectory_hyperlora/alfworld_static_vs_trajectory_v45_analysis_20261010.json"))
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = analyze(args)
    save(args.output, result)
    print(json.dumps({k: v for k, v in result.items() if k != "family_scores"}),
          flush=True)
