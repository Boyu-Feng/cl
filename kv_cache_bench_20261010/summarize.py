"""Produce a coverage-aware table from immutable per-question JSONL records."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

EXPECTED = {
    "locomo": {"multi_hop": 282, "temporal": 321, "open_domain": 96, "single_hop": 841},
    "mab": {"accurate_retrieval": 1100, "test_time_learning": 500,
            "long_range_understanding": 71, "selective_forgetting": 600},
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("input", type=Path)
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    records = [json.loads(x) for x in args.input.read_text().splitlines() if x.strip()]
    if not records:
        raise SystemExit("No records")
    suite = records[0]["suite"]
    categories = list(EXPECTED[suite])
    by_cell = defaultdict(dict)
    for item in records:
        if item["suite"] != suite:
            raise ValueError("Mixed suites")
        by_cell[(item["method"], item["eviction_fraction"], item["category"])][item["id"]] = item
    lines = [f"# {suite} KV benchmark status", "",
             f"Source: `{args.input}`. Seed 42. Values are means over successfully completed questions.",
             "Avg. is the unweighted mean of the four category means when all four have data.",
             "Each cell includes `completed/target` coverage; no partial cell is an official category score.",
             "Both methods physically evict the stated fraction of prefill KV pairs.", "",
             "| Method | KV evicted | Avg. | " + " | ".join(categories) + " |",
             "|---|---:|---:|" + "|".join("---:" for _ in categories) + "|"]
    for method in ("snapkv", "kvzip"):
        for ratio in (0.75, 0.5, 0.25):
            values = []
            means = []
            total_n = 0
            for category in categories:
                cell = list(by_cell[(method, ratio, category)].values())
                scores = [float(x["score"]) for x in cell if x["status"] == "complete"]
                n = len(scores)
                total_n += n
                expected = EXPECTED[suite][category]
                if n:
                    means.append(sum(scores) / n)
                values.append(f"{sum(scores)/n:.3f} ({n}/{expected})" if n else f"— (0/{expected})")
            total_expected = sum(EXPECTED[suite].values())
            avg = (f"{sum(means)/len(means):.3f} ({total_n}/{total_expected})"
                   if len(means) == len(categories) else f"— ({total_n}/{total_expected})")
            lines.append(f"| {method} | {ratio:.0%} | {avg} | " + " | ".join(values) + " |")
    failures = [x for x in records if x["status"] != "complete"]
    lines += ["", f"Recorded rows: {len(records)}; failures or skips: {len(failures)}."]
    if failures:
        lines += ["", "## Failure records", ""]
        for x in failures[:30]:
            lines.append(f"- `{x['id']}` {x['method']} {x['eviction_fraction']}: "
                         f"{x['status']} {x.get('error', '')}")
        if len(failures) > 30:
            lines.append(f"- … {len(failures)-30} more failures in the JSONL")
    text = "\n".join(lines) + "\n"
    if args.output:
        args.output.write_text(text)
    else:
        print(text)


if __name__ == "__main__":
    main()
