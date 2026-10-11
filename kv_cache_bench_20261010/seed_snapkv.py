"""Carry completed SnapKV rows into the two-method run without recomputing them."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/kv_cache_bench_20261010"
SOURCES = {
    "locomo_full_seed42.jsonl": "locomo_snap_echo_seed42.jsonl",
    "mab_native_seed42.jsonl": "mab_snap_echo_seed42.jsonl",
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    old = ROOT / "data/kv_cache_bench_20261010/upstream/kvpress/kvpress/presses/snapkv_press.py"
    new = ROOT / "data/kv_cache_bench_20261010/upstream/echopress_repo/kvpress/presses/snapkv_press.py"
    if digest(old) != digest(new):
        raise RuntimeError("SnapKV source changed; rerun instead of reusing rows")
    for source, target in SOURCES.items():
        src, dst = OUT / source, OUT / target
        existing = set()
        if dst.exists():
            for line in dst.read_text().splitlines():
                row = json.loads(line)
                existing.add((row["id"], row["method"], row["eviction_fraction"]))
        copied = 0
        with dst.open("a") as file:
            for line in src.read_text().splitlines():
                row = json.loads(line)
                key = (row["id"], row["method"], row["eviction_fraction"])
                if row["method"] != "snapkv" or row["status"] != "complete" or key in existing:
                    continue
                row["reused_from"] = source
                file.write(json.dumps(row, ensure_ascii=False) + "\n")
                existing.add(key)
                copied += 1
        print(target, copied, "SnapKV rows reused")


if __name__ == "__main__":
    main()
