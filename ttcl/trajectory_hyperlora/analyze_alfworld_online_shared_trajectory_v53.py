"""Paired outcomes and time blocks for the preregistered v53 contrasts."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


def validate_rows(report):
    rows = report["rows"]
    if len(rows) != 240 or report["failures"]:
        raise ValueError("Complete run required; do not select stopping time from scores")
    if [r["index"] for r in rows] != list(range(240)):
        raise ValueError("Expected each chronological task exactly once")
    suffix = rows[120:]
    for row in suffix:
        for arm in ("hyper_online", "online_static_lora", "hyper_frozen",
                    "hyper_shifted_actions", "hyper_rotated"):
            ep = row.get(arm)
            required = arm in ("hyper_online", "online_static_lora", "hyper_frozen") or row["adapter_active"]
            if required:
                if ep is None or ep.get("status") != "complete" or ep.get("reward") not in (0, 1):
                    raise ValueError(f"Missing or invalid episode: task {row['index']}, {arm}")
            elif ep is not None:
                raise ValueError(f"Unexpected source intervention on inactive task {row['index']}")
    return suffix


def paired(rows, left, right):
    rows = [r for r in rows if r.get(left) is not None and r.get(right) is not None]
    a = sum(r[left]["reward"] == 1 and r[right]["reward"] == 0 for r in rows)
    b = sum(r[left]["reward"] == 0 and r[right]["reward"] == 1 for r in rows)
    discordant = a + b
    p = min(1., 2 * sum(math.comb(discordant, k) for k in range(min(a, b)+1))
            / 2**discordant) if discordant else 1.
    trajectories = [r for r in rows if "trajectory" in r[left] and "trajectory" in r[right]]
    changed = sum([s["command"] for s in r[left]["trajectory"]] !=
                  [s["command"] for s in r[right]["trajectory"]] for r in trajectories)
    return {"n": len(rows), "left": left, "right": right,
            "trajectory_pairs_available": len(trajectories), "action_sequences_changed": changed,
            "left_success": sum(r[left]["reward"] for r in rows),
            "right_success": sum(r[right]["reward"] for r in rows),
            "left_only": a, "right_only": b, "net_success": a-b,
            "mcnemar_exact_two_sided_p": p}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("report", type=Path)
    args = p.parse_args()
    report = json.loads(args.report.read_text())
    rows = validate_rows(report)
    comparisons = [paired(rows, "hyper_online", arm) for arm in (
        "online_static_lora", "hyper_frozen", "hyper_shifted_actions")]
    ordered = sorted(comparisons, key=lambda x: x["mcnemar_exact_two_sided_p"])
    running = 0.
    for i, comparison in enumerate(ordered):
        running = max(running, min(1., (len(ordered)-i)*comparison["mcnemar_exact_two_sided_p"]))
        comparison["holm_adjusted_p_three_primary_contrasts"] = running
    result = {"report_sha256": hashlib.sha256(args.report.read_bytes()).hexdigest(),
        "scored_tasks": len(rows),
        "adapter_active_tasks": sum(r["adapter_active"] for r in rows),
        "content_comparison_scope": "Only tasks with an active adapter have source interventions; missing required episodes are rejected",
        "primary": comparisons,
        "same_family_source_substitution_exploratory": paired(rows, "hyper_online", "hyper_rotated"),
        "static_comparison_adapter_active_subset": paired(
            [r for r in rows if r["adapter_active"]], "hyper_online", "online_static_lora"),
        "blocks": [{"start_index": start, "end_index_exclusive": start+30,
            "comparisons": [paired([r for r in rows if start <= r["index"] < start+30],
                "hyper_online", arm) for arm in (
                    "online_static_lora", "hyper_frozen", "hyper_shifted_actions")]}
            for start in range(120, 240, 30)],
        "interpretation_limit": "Previously exposed train tasks and one seed; content corruption can introduce distribution shift; this is mechanism evidence, not independent generalization proof"}
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
