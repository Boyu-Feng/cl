"""Separate DetectiveQA JSON-answer diagnostic from official strict scoring."""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path


def normalize(text):
    text = str(text).lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def extracted_answer(prediction):
    try:
        value = json.loads(prediction)
        if isinstance(value, dict) and isinstance(value.get("answer"), str):
            return value["answer"]
    except (json.JSONDecodeError, TypeError):
        pass
    found = re.search(r'"answer"\s*:\s*"([^"\\]*(?:\\.[^"\\]*)*)"', prediction)
    if found:
        return found.group(1)
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("input", type=Path)
    args = p.parse_args()
    cells = defaultdict(list)
    for line in args.input.read_text().splitlines():
        row = json.loads(line)
        if row.get("category") != "long_range_understanding" or row.get("status") != "complete":
            continue
        answer = extracted_answer(row["prediction"])
        references = row["answer"] if isinstance(row["answer"], list) else [row["answer"]]
        diagnostic = None if answer is None else int(normalize(answer) in {normalize(x) for x in references})
        cells[(row["method"], row["eviction_fraction"])].append((row["score"], diagnostic))
    print("# DetectiveQA JSON answer diagnostic")
    print("\nThe first score is strict exact match on raw output. The second extracts the JSON `answer` field; it is a local diagnostic and must not replace the benchmark score.\n")
    print("| Method | Eviction | N | Strict | Extracted-answer diagnostic |")
    print("|---|---:|---:|---:|---:|")
    for (method, ratio), values in sorted(cells.items()):
        parsed = [x[1] for x in values if x[1] is not None]
        print(f"| {method} | {ratio:.0%} | {len(values)} | "
              f"{sum(x[0] for x in values)/len(values):.3f} | "
              f"{sum(parsed)/len(parsed):.3f} ({len(parsed)}/{len(values)} parsed) |" if parsed else
              f"| {method} | {ratio:.0%} | {len(values)} | "
              f"{sum(x[0] for x in values)/len(values):.3f} | — (0 parsed) |")


if __name__ == "__main__":
    main()
