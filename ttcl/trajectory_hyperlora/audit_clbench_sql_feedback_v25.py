"""Independent chronology, provenance, composition, and outcome audit."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from ttcl.trajectory_hyperlora.alfworld_online_from_empty import digest, file_hash
from ttcl.trajectory_hyperlora.general_evidence_v22 import (
    compose_structured, digest as evidence_digest, extract_records, verify,
)
from ttcl.trajectory_hyperlora.clbench_online_typed_gate_v21 import reward_gate
from ttcl.trajectory_hyperlora.train_clbench_hyperlora_v1 import clean_records
from ttcl.trajectory_hyperlora.sql_feedback_evidence_v25 import (
    apply as apply_sql_feedback, missing_identifiers,
)


def audit(args):
    report = json.loads(args.report.read_text())
    review = json.loads(args.review.read_text())
    gate = json.loads(args.gate_model.read_text())
    sql_gate = json.loads(args.sql_gate_model.read_text())
    binding = review["binding"]
    if (review["input_content_sha256"] != digest({"binding": binding,
            "targets": review["targets"]}) or
            report["review_sha256"] != file_hash(args.review) or
            report["binding"] != binding or
            binding["gate_model_sha256"] != file_hash(args.gate_model) or
            binding["sql_gate_model_sha256"] != file_hash(args.sql_gate_model) or
            binding["sql_operator_sha256"] != file_hash(args.sql_operator) or
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
                sql_used = False
                for line in (folder / "responses.jsonl").read_text().splitlines():
                    event = json.loads(line)
                    decision = event.get("composed_evidence")
                    if decision is None:
                        continue
                    marker = "Return only JSON. Action schema:\n"
                    text = next(item["content"] for item in
                        reversed(event["messages"])
                        if item["role"] == "user" and marker in item["content"])
                    query, schema_text = text.split(marker, 1)
                    pre_sql = json.loads(event.get("pre_sql_raw_response",
                                                    event["raw_response"]))
                    original = json.loads(event.get("generated_raw_response",
                        event.get("pre_sql_raw_response", event["raw_response"])))
                    composed, expected = compose_structured(original, history,
                        schema=json.loads(schema_text), gate_model=gate,
                        target_index=index, target_query=query)
                    if (expected != decision or
                            composed != pre_sql):
                        raise ValueError("Evidence edit or gate is not reproducible")
                    operator = decision["decision"]
                    selected[operator] = selected.get(operator, 0) + 1
                    bad = set()
                    for item in history:
                        bad.update(missing_identifiers(item.feedback))
                    if sql_used:
                        sql_action, sql_expected = composed, {
                            "decision": "unchanged", "reason":
                            "one_sql_evidence_use_per_episode"}
                    else:
                        sql_action, sql_expected = apply_sql_feedback(composed,
                            history, index, bad, sql_gate)
                        if sql_action != composed and any(
                                sql_action["content"] in str(message.get("content", ""))
                                for message in event["messages"][2:]
                                if message.get("role") == "assistant"):
                            sql_action, sql_expected = composed, {
                                "decision": "unchanged", "reason":
                                "query_already_used_this_episode"}
                    if (event.get("sql_feedback_evidence") != sql_expected or
                            sql_action != json.loads(event["raw_response"])):
                        raise ValueError("SQL feedback decision is not reproducible")
                    if sql_action != composed:
                        sql_used = True
                    selected["sql:"+sql_expected["decision"]] = (
                        selected.get("sql:"+sql_expected["decision"], 0) + 1)
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
                source_hash = evidence_digest(records)
                evidence_reward = (min(1.0, max(0.0, reward))
                    if binding["method"].endswith("bounded_feedback")
                    else reward)
                added = extract_records(records, source_index=index,
                    source_sha256=source_hash, reward=evidence_reward)
                if not all(verify(item, records, source_hash) for item in added):
                    raise ValueError("Source evidence content changed")
                history.extend(added)
            if expected_write and row["write"]["source_records_sha256"] != \
                    digest(clean_records(episodes["lora"])):
                raise ValueError("Parameter source is not own public trajectory")
            rewards.append(reward)
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
            failures != len(report["failures"])):
        raise ValueError("Summary or failure count changed")
    return {"audit": "passed", "report_sha256": file_hash(args.report),
        "review_sha256": file_hash(args.review), "paired": len(paired),
        "failures": failures, "evidence_decisions": selected}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--gate-model", type=Path, default=Path(
        "results/trajectory_hyperlora/clbench_typed_gate_v19_20261008.json"))
    parser.add_argument("--runner", type=Path, default=Path(
        "ttcl/trajectory_hyperlora/clbench_sql_feedback_v25.py"))
    parser.add_argument("--sql-gate-model", type=Path, default=Path(
        "results/trajectory_hyperlora/clbench_sql_feedback_gate_v25_20261009.json"))
    parser.add_argument("--sql-operator", type=Path, default=Path(
        "ttcl/trajectory_hyperlora/sql_feedback_evidence_v25.py"))
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
