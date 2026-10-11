"""Prepare a domain-blind general trajectory corpus for v99 transfer.

This is a content-preserving derivative of the audited v93 training rows.  The
domain and benchmark identifiers remain in the manifest for audit only; they
are removed from every encoder input and from the training schedule records.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "results/trajectory_hyperlora/joint_domains_v93_20261010/dataset.json"
OUT = ROOT / "results/trajectory_hyperlora/general_corpus_v99_20261010"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def filehash(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    tmp.replace(path)


def prepare():
    if (OUT / "dataset.json").exists():
        raise RuntimeError(f"refusing to overwrite {OUT / 'dataset.json'}")
    source = json.loads(SOURCE.read_text())
    rows = []
    for row in source["rows"]:
        # Only these fields are allowed to reach the generic encoder.  In
        # particular, no dataset name, benchmark id, target id, or answer.
        generic = {
            "messages": row["messages"],
            "query": row["query"],
            "source": row["source"],
            "answer": row["answer"],
            "input_content_sha256": row["input_content_sha256"],
        }
        generic["generic_row_sha256"] = digest(generic)
        rows.append(generic)
    rng = random.Random(9901)
    order = list(range(len(rows)))
    rng.shuffle(order)
    schedule = [order[i % len(order)] for i in range(600)]
    manifest = {
        "protocol": "general_corpus_v99",
        "source_dataset": str(SOURCE.relative_to(ROOT)),
        "source_dataset_sha256": filehash(SOURCE),
        "rows": len(rows),
        "schedule_updates": len(schedule),
        "schedule_sha256": digest(schedule),
        "domain_counts_audit_only": dict(Counter(source["rows"][i]["domain"] for i in range(len(rows)))),
        "encoder_fields": ["source", "query"],
        "forbidden_encoder_fields": ["domain", "id", "source_id", "parent", "answer", "benchmark", "task_id"],
        "split_rule": "inherited reviewed train rows; transfer targets remain outside this source dataset",
        "seed": 9901,
    }
    dataset = {"rows": rows, "schedule": schedule}
    manifest["dataset_sha256"] = digest(dataset)
    write(OUT / "dataset.json", dataset)
    write(OUT / "manifest.json", manifest)
    print(json.dumps({"output": str(OUT), "rows": len(rows), "updates": len(schedule), "manifest": manifest}, ensure_ascii=False))


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    prepare()
