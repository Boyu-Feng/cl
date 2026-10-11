"""Independent source, writer, prompt, and outcome audit for v24."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from ttcl.trajectory_hyperlora.alfworld_online_from_empty import digest, file_hash
from ttcl.trajectory_hyperlora.structured_summary_evidence_v24 import (
    build_summary, digest as evidence_digest, render, select, writer_input,
)
from ttcl.trajectory_hyperlora.clbench_online_typed_gate_v21 import reward_gate
from ttcl.trajectory_hyperlora.train_clbench_hyperlora_v1 import clean_records


def audit(args):
    report = json.loads(args.report.read_text())
    review = json.loads(args.review.read_text())
    binding = review["binding"]
    if (review["input_content_sha256"] != digest({"binding": binding,
            "targets": review["targets"]}) or
            report["review_sha256"] != file_hash(args.review) or
            report["binding"] != binding or
            binding["evidence_module_sha256"] != file_hash(args.evidence_module) or
            binding["runner_sha256"] != file_hash(args.runner)):
        raise ValueError("Changed review, model, or runner")
    if [r["index"] for r in report["rows"]] != list(range(len(report["rows"]))) or \
            (len(report["rows"]) != len(review["targets"]) and not args.partial):
        raise ValueError("Missing or reordered official targets")
    history = []
    rewards = []
    selected = {}
    failures = 0
    paired = []
    for row, target in zip(report["rows"], review["targets"]):
        index = row["index"]
        if (row["target"] != target or
                row["source_evidence_before"] != len(history)):
            raise ValueError("Changed source chronology")
        if index == 0 and row["lora"]["first_prompt_sha256"] != \
                row["lora_evidence"]["first_prompt_sha256"]:
            raise ValueError("Empty-memory first prompts differ")
        episodes = {}
        for arm in ("lora", "lora_evidence"):
            folder = (args.report.parent / (args.report.stem + "_episodes") /
                      arm / f"episode_{index+1:03}")
            outcome = json.loads((folder / "row.json").read_text())
            episode = json.loads((folder / "trajectory.json").read_text())
            if (outcome != row[arm] or outcome["instance_id"] !=
                    target["instance_id"] or outcome["initial_query_sha256"] !=
                    target["initial_query_sha256"]):
                raise ValueError("Changed official outcome row or task")
            if outcome["status"] == "complete":
                if (not episode["completed"] or
                        float(episode["reward"]) != float(outcome["reward"])):
                    raise ValueError("Changed official reward")
            elif outcome["reward"] is not None or episode["completed"]:
                raise ValueError("Failed episode was scored")
            episodes[arm] = episode
            if arm == "lora_evidence":
                response_path = folder / "responses.jsonl"
                if not response_path.exists() and outcome["actor_calls"]:
                    raise ValueError("Missing actor response log")
                lines = (response_path.read_text().splitlines()
                         if response_path.exists() else [])
                if len(lines) != outcome["actor_calls"]:
                    raise ValueError("Actor response count changed")
                for line in lines:
                    event = json.loads(line)
                    marker = "Return only JSON. Action schema:\n"
                    text = next((item["content"] for item in
                        reversed(event["messages"])
                        if item["role"] == "user" and marker in item["content"]), "")
                    query = text.split(marker, 1)[0]
                    chosen = select(history, target_index=index,
                                    target_query=query)
                    evidence_text = render(chosen) if chosen else ""
                    expected = ({"source_indices": [item["source_index"]
                                                   for item in chosen],
                        "evidence_sha256": evidence_digest(evidence_text),
                        "rendered_chars": len(evidence_text)}
                        if evidence_text else None)
                    if event.get("typed_evidence_prompt") != expected:
                        raise ValueError("Typed prompt source selection changed")
                    if expected:
                        selected["prompted"] = selected.get("prompted", 0) + 1
                    elif event.get("cached_from_lora"):
                        selected["cached_control"] = selected.get("cached_control", 0) + 1
        lora = row["lora"]
        expected_write = (lora["status"] == "complete" and
            index + 1 < len(review["targets"]) and
            reward_gate(float(lora["reward"]), rewards))
        if bool(row["write"]["written"]) != expected_write:
            raise ValueError("Parameter write chronology changed")
        if lora["status"] == "complete":
            reward = float(lora["reward"])
            records = [{"observation": step.get("query", ""),
                        "action": step["action"],
                        "feedback": step.get("public_feedback", "")}
                       for step in episodes["lora"]["steps"]]
            if records:
                value = writer_input(records, source_index=index,
                    reward=reward,
                    initial_query=episodes["lora"]["initial_public_query"])
                writer = row["writer"]
                if writer.get("writer_input") != value:
                    raise ValueError("Writer source content changed")
                attempts = writer["attempts"]
                if not 1 <= len(attempts) <= 2:
                    raise ValueError("Writer attempts changed")
                valid = []
                for item in attempts:
                    if item["status"] == "generation_error":
                        continue
                    raw = item["raw_response"]
                    if item["writer_response_sha256"] != evidence_digest(raw):
                        raise ValueError("Writer raw response changed")
                    try:
                        summary = build_summary(raw, value,
                            writer_prompt_sha256=item["rendered_prompt_sha256"],
                            writer_response_sha256=evidence_digest(raw))
                    except (ValueError, TypeError):
                        if item["status"] != "invalid":
                            raise ValueError("Writer validation status changed")
                    else:
                        if item["status"] != "valid":
                            raise ValueError("Writer validation status changed")
                        valid.append(summary)
                if writer["status"] == "valid":
                    if (len(valid) != 1 or valid[0] != row["source_summary"] or
                            attempts[-1]["status"] != "valid"):
                        raise ValueError("Structured summary changed")
                    history.append(valid[0])
                elif row["source_summary"] is not None or valid:
                    raise ValueError("Invalid writer created evidence")
            elif row["writer"]["status"] != "no_public_actions" or \
                    row["source_summary"] is not None:
                raise ValueError("Empty source produced evidence")
            if expected_write and row["write"]["source_records_sha256"] != \
                    digest(clean_records(episodes["lora"])):
                raise ValueError("Parameter source is not own public trajectory")
            rewards.append(reward)
        elif row["writer"]["status"] != "source_incomplete" or \
                row["source_summary"] is not None:
            raise ValueError("Incomplete source produced evidence")
        if row["source_evidence_after"] != len(history):
            raise ValueError("Missing or extra source evidence")
        if all(row[x]["status"] == "complete" for x in ("lora", "lora_evidence")):
            paired.append(row)
        else:
            failures += 1
    summary = report["summary"]
    means = {arm: (sum(x[arm]["reward"] for x in paired) / len(paired)
                   if paired else None)
             for arm in ("lora", "lora_evidence")}
    if (summary["completed_targets"] != len(report["rows"]) or
            summary["paired"] != len(paired) or
            any(not math.isclose(summary[f"{arm}_mean"], value,
                                 abs_tol=1e-12, rel_tol=0)
                for arm, value in means.items() if value is not None) or
            failures != len(report["failures"]) or
            [r["index"] for r in report["rows"]
             if r["writer"]["status"] in ("invalid", "generation_error")] !=
            [r["index"] for r in report["summary_failures"]]):
        raise ValueError("Summary or failure count changed")
    return {"audit": "passed", "report_sha256": file_hash(args.report),
        "review_sha256": file_hash(args.review), "paired": len(paired),
        "failures": failures, "evidence_decisions": selected}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--evidence-module", type=Path, default=Path(
        "ttcl/trajectory_hyperlora/structured_summary_evidence_v24.py"))
    parser.add_argument("--runner", type=Path, default=Path(
        "ttcl/trajectory_hyperlora/clbench_structured_summary_v24.py"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--partial", action="store_true",
                        help="Audit a still-running chronological prefix")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = audit(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
