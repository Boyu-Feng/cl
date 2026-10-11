"""Combine fixed-seed one-question pilots for the active two-method comparison."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "results/kv_cache_bench_20261010"
FILES = {
    "locomo": ("locomo_pilot_seed42.jsonl", "locomo_echo_pilot_seed42.jsonl"),
    "mab": ("mab_pilot_v2_seed42.jsonl", "mab_echo_pilot_seed42.jsonl"),
}


def main():
    for suite, (snap_file, echo_file) in FILES.items():
        snap = [json.loads(line) for line in (ROOT / snap_file).read_text().splitlines()]
        echo = [json.loads(line) for line in (ROOT / echo_file).read_text().splitlines()]
        records = [dict(row, reused_from=snap_file) for row in snap
                   if row["method"] == "snapkv" and row["status"] == "complete"]
        records += [row for row in echo if row["method"] == "echopress"]
        out = ROOT / f"{suite}_snap_echo_pilot_seed42.jsonl"
        out.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records))
        print(out, len(records))


if __name__ == "__main__":
    main()
