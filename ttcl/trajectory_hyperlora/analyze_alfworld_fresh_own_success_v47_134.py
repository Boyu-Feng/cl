"""Paired official134 analysis of fresh own-success and direct LoRA arms."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from math import comb
from pathlib import Path

from ttcl.trajectory_hyperlora.alfworld_online_from_empty import file_hash
from ttcl.trajectory_hyperlora.alfworld_persistent_failure_delta_v36 import save


def paired(a, b):
    first = sum(bool(x) and not bool(y) for x, y in zip(a, b, strict=True))
    second = sum(bool(y) and not bool(x) for x, y in zip(a, b, strict=True))
    n = first + second
    p = min(1., 2 * sum(comb(n, i) for i in range(min(first, second) + 1)) /
            2**n) if n else 1.
    return {"first_only": first, "second_only": second,
            "exact_two_sided_p": p}


def analyze(args):
    report = json.loads(args.report.read_text())
    audit = json.loads(args.audit.read_text())
    if (audit.get("audit") != "passed" or
            audit.get("report_sha256") != file_hash(args.report) or
            len(report.get("rows", [])) != 134 or report.get("failures")):
        raise ValueError("Changed or unaudited v47 evaluation")
    names = ("old_hyper", "fresh_hyper", "contrast_hyper", "full_static")
    values = {name: [] for name in names}
    families = defaultdict(list)
    changed_paths = {"fresh_vs_contrast": 0, "fresh_vs_full_static": 0}
    for i, row in enumerate(report["rows"]):
        if row["index"] != i:
            raise ValueError("Changed official target order")
        values["old_hyper"].append(row["old_hyper_reward"])
        for name in names[1:]:
            values[name].append(row[name]["reward"])
        commands = lambda episode: [step["command"] for step in episode["trajectory"]]
        changed_paths["fresh_vs_contrast"] += (
            commands(row["fresh_hyper"]) != commands(row["contrast_hyper"]))
        changed_paths["fresh_vs_full_static"] += (
            commands(row["fresh_hyper"]) != commands(row["full_static"]))
        families[row["game"].split("/")[2].split("-")[0]].append(i)
    if (sum(values["old_hyper"]) != 68 or
            any(sum(values[name]) != report["summary"][name]
                for name in names)):
        raise ValueError("Changed score summary")
    return {"protocol": "Paired description on repeatedly exposed official valid_unseen134; one 50-step attempt per arm with common legal-command and repetition guard",
        "n": 134, "report_sha256": file_hash(args.report),
        "scores": {name: sum(values[name]) for name in names},
        "changed_action_sequences": changed_paths,
        "paired": {
            "fresh_hyper_vs_full_static": paired(
                values["fresh_hyper"], values["full_static"]),
            "contrast_hyper_vs_full_static": paired(
                values["contrast_hyper"], values["full_static"]),
            "contrast_hyper_vs_fresh_hyper": paired(
                values["contrast_hyper"], values["fresh_hyper"]),
            "old_hyper_vs_fresh_hyper": paired(
                values["old_hyper"], values["fresh_hyper"]),
            "old_hyper_vs_contrast_hyper": paired(
                values["old_hyper"], values["contrast_hyper"]),
        },
        "family_scores": {family: {name: sum(values[name][i]
            for i in indices) for name in names} for family, indices in
            sorted(families.items())}}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    root = Path("results/trajectory_hyperlora")
    p.add_argument("--report", type=Path, default=root /
        "alfworld_fresh_own_success_v47_unseen134_20261010.json")
    p.add_argument("--audit", type=Path, default=root /
        "alfworld_fresh_own_success_v47_unseen134_audit_20261010.json")
    p.add_argument("--output", type=Path, default=root /
        "alfworld_fresh_own_success_v47_unseen134_analysis_20261010.json")
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = analyze(args)
    save(args.output, result)
    print(json.dumps({k: v for k, v in result.items() if k != "family_scores"}),
          flush=True)
