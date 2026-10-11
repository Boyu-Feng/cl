"""Paired descriptive analysis of v39 cross-task memory and audited v32 retry."""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path


def main():
    root = Path("results/trajectory_hyperlora")
    persistent = json.loads((root / "alfworld_persistent_failure_delta_v39_134_20261010.json").read_text())
    retry = json.loads((root / "alfworld_failure_retry_v32_134_20261009.json").read_text())
    audit = json.loads((root / "alfworld_persistent_failure_delta_v39_134_audit_20261010.json").read_text())
    if (audit["audit"] != "passed" or persistent["summary"]["n"] != 134 or
            retry["summary"]["n"] != 134):
        raise ValueError("Incomplete experiment")
    rows = persistent["rows"]
    old = retry["games"]
    if any(a["game"] != b["game"] or a["old_first_reward"] != b["first_reward"]
           for a, b in zip(rows, old, strict=True)):
        raise ValueError("Unpaired task sequence")
    by_family = defaultdict(lambda: {"n": 0, "old": 0, "persistent": 0,
                                     "same_task_two_attempt": 0})
    quartiles = []
    for i, (a, b) in enumerate(zip(rows, old, strict=True)):
        family = a["game"].split("/")[2].split("-")[0]
        s = by_family[family]
        s["n"] += 1
        s["old"] += b["first_reward"]
        s["persistent"] += a["episode"]["reward"]
        s["same_task_two_attempt"] += bool(b["first_reward"] or
            b["retries"].get("away_from_failure", {}).get("reward", 0))
    for start, end in ((0, 34), (34, 68), (68, 101), (101, 134)):
        part = rows[start:end]
        quartiles.append({"indices": [start, end], "old": sum(x["old_first_reward"] for x in part),
            "persistent": sum(x["episode"]["reward"] for x in part),
            "updates_before_start": part[0]["failure_updates_before"],
            "updates_after_end": part[-1]["failure_updates_after"]})
    ratios = [x["read_info"]["applied_effective_norm"] /
              x["read_info"]["reference_effective_norm"]
              for x in rows if x["read_info"]["reference_effective_norm"]]
    scales = [x["read_info"]["read_scale"] for x in rows[1:]]
    persistent_rescues = {i for i, x in enumerate(rows)
                          if not x["old_first_reward"] and x["episode"]["reward"]}
    same_task_rescues = {i for i, x in enumerate(old)
                         if not x["first_reward"] and
                         x["retries"].get("away_from_failure", {}).get("reward", 0)}
    summary = {"old_first": sum(x["old_first_reward"] for x in rows),
        "persistent_one_attempt": sum(x["episode"]["reward"] for x in rows),
        "same_task_two_attempt": retry["summary"]["two_attempt_wins"]["away_from_failure"],
        "static_two_attempt": retry["summary"]["two_attempt_wins"]["static_retry"],
        "random_two_attempt": retry["summary"]["two_attempt_wins"]["matched_random"],
        "persistent_only_wins": sum(not a["old_first_reward"] and a["episode"]["reward"] for a in rows),
        "persistent_only_losses": sum(a["old_first_reward"] and not a["episode"]["reward"] for a in rows),
        "same_task_rescues": sum(not x["first_reward"] and
            x["retries"].get("away_from_failure", {}).get("reward", 0) for x in old),
        "rescue_overlap": len(persistent_rescues & same_task_rescues),
        "persistent_only_rescues": len(persistent_rescues - same_task_rescues),
        "same_task_only_rescues": len(same_task_rescues - persistent_rescues),
        "read_norm_ratio_mean": statistics.mean(ratios),
        "read_norm_ratio_median": statistics.median(ratios),
        "read_scale_mean": statistics.mean(scales),
        "read_scale_median": statistics.median(scales),
        "read_scale_below_one": sum(x < 1 for x in scales),
        "read_zero": sum(x["read_info"]["applied_effective_norm"] == 0 for x in rows),
        "quartiles": quartiles, "families": dict(sorted(by_family.items())),
        "diagnostic_only_old_failed_local_delta_alignment":
            audit["diagnostic_only_old_failed_local_delta_alignment"]}
    out = root / "alfworld_persistent_failure_delta_v39_134_analysis_20261010.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
