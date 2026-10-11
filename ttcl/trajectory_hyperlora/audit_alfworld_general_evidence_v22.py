"""Independent environment replay and evidence-read audit for v22."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from ttcl.experience_evolution.environment import make_env
from ttcl.trajectory_hyperlora.alfworld_same_game_retry import file_hash
from ttcl.trajectory_hyperlora.general_evidence_v22 import (
    digest, extract_records, text_provider, verify,
)


def run(args):
    report = json.loads(args.report.read_text())
    review = json.loads(args.review.read_text())
    source_review = json.loads(args.source_review.read_text())
    if (report["review_sha256"] != file_hash(args.review) or
            review["parent_source_review_sha256"] != file_hash(args.source_review) or
            report["failures"] or len(report["games"]) != len(review["targets"])):
        raise ValueError("Incomplete or changed report lineage")
    sources = {x["input_content_sha256"]: x
               for x in source_review["annotations"]}
    rows = []
    for row, target in zip(report["games"], review["targets"], strict=True):
        if (row["game"] != target["target_game"] or
                row["input_content_sha256"] != target["input_content_sha256"] or
                file_hash(args.data_root / row["game"]) !=
                target["target_game_sha256"]):
            raise ValueError("Changed target binding")
        source = sources[target["parent_input_content_sha256"]]
        records = source["source_records"]
        if digest(records) != target["source_records_sha256"]:
            raise ValueError("Changed source content")
        evidence = extract_records(records, source_index=0,
            source_sha256=target["source_records_sha256"], reward=1.)
        if not all(verify(item, records, target["source_records_sha256"])
                   for item in evidence):
            raise ValueError("Unverified evidence")
        provider = text_provider(evidence, target_index=1)
        replayed = {}
        for arm in ("guarded_lora", "guarded_lora_evidence"):
            episode = row[arm]
            env = make_env(args.data_root / row["game"])
            try:
                state = env.reset()
                initial = str(state["feedback"])
                if initial != episode["initial_observation"]:
                    raise ValueError("Changed initial observation")
                for step, trace in enumerate(episode["trajectory"]):
                    before = str(state["feedback"])
                    command = trace["command"]
                    if command not in state["admissible_commands"]:
                        raise ValueError("Invalid command in replay")
                    if arm == "guarded_lora_evidence":
                        hint = provider(initial, before,
                                        list(state["admissible_commands"]))
                        expected = hashlib.sha256(hint.encode()).hexdigest()
                        if expected != episode["evidence_read_sha256"][step]:
                            raise ValueError("Evidence read changed")
                    state, _, done = env.step(command)
                    if (str(state["feedback"]) != trace["observation"] or
                            bool(state["won"]) != trace["won"] or
                            (done or state["won"]) and
                            step != len(episode["trajectory"]) - 1):
                        raise ValueError("Environment replay diverged")
                won = float(bool(state["won"]))
                if won != episode["reward"]:
                    raise ValueError("Official reward changed")
                replayed[arm] = won
            finally:
                env.close()
        rows.append({"game": row["game"], **replayed})
    return {"report_sha256": file_hash(args.report),
        "review_sha256": file_hash(args.review),
        "source_review_sha256": file_hash(args.source_review),
        "summary": {"n": len(rows), "failed_replays": 0,
                    "guarded_lora": sum(x["guarded_lora"] for x in rows),
                    "guarded_lora_evidence": sum(x["guarded_lora_evidence"]
                                                  for x in rows)},
        "rows": rows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--source-review", type=Path, default=Path(
        "data/annotations/alf_sibling_simple_replica30_reviewed_20261006.json"))
    parser.add_argument("--data-root", type=Path,
                        default=Path("ttcl/data/alfworld_delta"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result["summary"]))


if __name__ == "__main__":
    main()
