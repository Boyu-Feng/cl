"""Build a target-disjoint trajectory corpus from CLBench only.

XLand, ALFWorld and HLE files are never read.  Each example is a causal
prefix followed by its next observed action; task names are kept in the audit
manifest but are not supplied as metadata to the encoder.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = ROOT / "results/memrl_grounded_evidence"
OUT = ROOT / "results/trajectory_hyperlora/external_clbench_corpus_v100_20261010"
SOURCE_RUNS = ("20261002_v15_spectrum_303_full90", "20261002_v15_poker_303_full120",
               "20261002_v15_database_303_full30", "20261002_v15_cohort_303_full20")


def digest(x):
    return hashlib.sha256(json.dumps(x, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def filehash(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(p, x):
    p.parent.mkdir(parents=True, exist_ok=True)
    q = p.with_suffix(p.suffix + ".tmp"); q.write_text(json.dumps(x, ensure_ascii=False, indent=2)); q.replace(p)


def main():
    if (OUT / "dataset.json").exists():
        raise RuntimeError("refusing to overwrite frozen corpus")
    rows = []; manifests = []; counts = Counter()
    for dirname in SOURCE_RUNS:
        root = SOURCE_ROOT / dirname / "vanilla"
        for epdir in sorted(root.glob("episode_*")):
            trajp = epdir / "public_trajectory.json"
            if not trajp.exists():
                continue
            traj = json.loads(trajp.read_text())
            manifests.append({"path": str(trajp.relative_to(ROOT)), "sha256": filehash(trajp), "steps": len(traj)})
            prefix = []
            for step in traj:
                query = str(step["query"])
                action = json.dumps(step["action"], ensure_ascii=False, sort_keys=True)
                source = "Causal trajectory prefix:\n" + "\n".join(
                    f"step={x['step']}\nobservation={x['query']}\naction={json.dumps(x['action'], ensure_ascii=False, sort_keys=True)}\nfeedback={x.get('public_feedback','')}"
                    for x in prefix
                )
                if not prefix:
                    source = "Causal trajectory prefix is empty."
                row = {"messages": [{"role": "user", "content": query}], "query": query,
                       "source": source, "answer": action}
                row["input_content_sha256"] = digest(row)
                row["generic_row_sha256"] = digest(row)
                rows.append(row); counts[dirname] += 1; prefix.append(step)
    if len(rows) < 20:
        raise RuntimeError(f"too few external rows: {len(rows)}")
    rng = random.Random(10001); order = list(range(len(rows))); rng.shuffle(order)
    schedule = [order[i % len(order)] for i in range(600)]
    dataset = {"rows": rows, "schedule": schedule}
    manifest = {"protocol": "external_clbench_corpus_v100", "source_root": str(SOURCE_ROOT.relative_to(ROOT)),
                "source_files": manifests, "source_counts_audit_only": dict(counts), "rows": len(rows),
                "schedule_updates": len(schedule), "schedule_sha256": digest(schedule),
                "dataset_sha256": digest(dataset), "target_datasets_read": [],
                "target_datasets_forbidden": ["xland", "alfworld", "hle"],
                "encoder_fields": ["source", "query"], "forbidden_fields": ["domain", "benchmark", "task_id", "answer"]}
    write(OUT / "dataset.json", dataset); write(OUT / "manifest.json", manifest)
    print(json.dumps({"rows": len(rows), "counts": counts, "manifest": manifest}, ensure_ascii=False))


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args(); main()
