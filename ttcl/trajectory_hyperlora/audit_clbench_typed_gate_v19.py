"""Independent BSM reward and provenance audit for typed evidence reads."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics

from jsonschema import Draft202012Validator

from ttcl.trajectory_hyperlora.alfworld_online_from_empty import digest, file_hash
from ttcl.trajectory_hyperlora.clbench_online_parameter_memory_v2 import ROOT, benchmark
from ttcl.trajectory_hyperlora.typed_evidence_v17 import extract, proposals, route


def audit(report_path: Path, review_path: Path, gate_path: Path):
    from src.tasks.blind_spectrum_monitoring.task import ScanReport, _score_report

    report = json.loads(report_path.read_text())
    review = json.loads(review_path.read_text())
    gate = json.loads(gate_path.read_text())
    binding = review["binding"]
    if ("blind_spectrum_monitoring" not in binding["domains"] or
            review["input_content_sha256"] != digest({
                "binding": binding, "targets": review["targets"]}) or
            report["review_sha256"] != file_hash(review_path) or
            report["checkpoint_sha256"] != binding["checkpoint_sha256"] or
            binding["gate_model_sha256"] != file_hash(gate_path) or
            binding["actor_script_sha256"] != file_hash(
                ROOT / "ttcl/trajectory_hyperlora/clbench_online_typed_gate_v19.py")):
        raise ValueError("Run, model, or script binding changed")
    rows = [x for x in report["rows"] if x["domain"] == "blind_spectrum_monitoring"]
    targets = [x for x in review["targets"] if x["domain"] == "blind_spectrum_monitoring"]
    if [x["index"] for x in rows] != list(range(binding["start"], binding["stop"])):
        raise ValueError("Missing or reordered target")
    history = []
    confirmed = failed = 0
    bench = benchmark()
    for row, target in zip(rows, targets, strict=True):
        if row["target"] != target:
            raise ValueError("Target changed")
        task = bench.make_task("blind_spectrum_monitoring", binding["seed"], independent=True)
        query = task.reset_baseline_instance(row["index"])
        if target["instance_id"] != query.instance_id or \
                target["initial_query_sha256"] != digest(query.prompt):
            raise ValueError("Official reset changed")
        latent = task._get_all_latent_channel_defs()
        for arm in ("base", "online"):
            directory = (report_path.parent /
                (report_path.stem + "_episodes") / "blind_spectrum_monitoring" /
                arm / f"episode_{row['index']+1:03}")
            outcome = row[arm]
            episode = json.loads((directory / "trajectory.json").read_text())
            if json.loads((directory / "row.json").read_text()) != outcome:
                raise ValueError("Episode row changed")
            if (outcome["instance_id"] != query.instance_id or
                    outcome["initial_query_sha256"] != digest(query.prompt)):
                raise ValueError("Episode target changed")
            if arm == "online":
                for line in (directory / "responses.jsonl").read_text().splitlines():
                    event = json.loads(line)
                    if "generated_raw_response" not in event:
                        continue
                    original = json.loads(event["generated_raw_response"])
                    marker = "Return only JSON. Action schema:\n"
                    text = next(item["content"] for item in reversed(event["messages"])
                        if item["role"] == "user" and marker in item["content"])
                    target_query, schema_text = text.split(marker, 1)
                    schema = json.loads(schema_text)
                    choices = []
                    for evidence in history:
                        if evidence.source_index < row["index"]:
                            choices.extend(item for item in proposals(original,
                                evidence, schema)
                                if item.operator in gate["operators"])
                    operator, selected, lower = route(choices, original, gate,
                        target_index=row["index"], target_query=target_query)
                    if selected is None or selected.action is None:
                        raise ValueError("Unjustified typed evidence edit")
                    expected = selected.action
                    actual = json.loads(event["raw_response"])
                    if actual != expected or not Draft202012Validator(schema).is_valid(actual):
                        raise ValueError("Evidence action or schema changed")
                    references = event.get("selected_evidence", [])
                    if len(references) != 1:
                        raise ValueError("Evidence selection count changed")
                    reference = references[0]
                    evidence = selected.evidence
                    if (reference["source_index"] != evidence.source_index or
                            reference["source_trajectory_sha256"] != evidence.trajectory_sha256 or
                            reference["source_action_sha256"] != evidence.action_sha256 or
                            reference["schema_path"] != evidence.path or
                            reference["operator"] != operator or
                            event["typed_gate"]["decision"] != operator or
                            abs(event["typed_gate"]["lower_utility"] - lower) > 1e-9):
                        raise ValueError("Selected evidence has no prior public source")
            if outcome["status"] == "complete":
                if len(episode["steps"]) != 1:
                    raise ValueError("Expected one official action")
                score = _score_report(ScanReport.model_validate(episode["steps"][0]["action"]),
                    latent, task.W, task.G, task.band_width)["score"]
                if float(score) != float(outcome["reward"]) or \
                        float(score) != float(episode["reward"]):
                    raise ValueError("Official score changed")
                confirmed += 1
                if arm == "online":
                    step = episode["steps"][0]
                    history.extend(extract(step["action"], source_index=row["index"],
                        trajectory_sha256=file_hash(directory / "trajectory.json"),
                        feedback=step.get("public_feedback"),
                        reward=float(outcome["reward"]), query=step.get("query", "")))
            elif outcome["reward"] is not None:
                raise ValueError("Failed episode counted as official reward")
            else:
                failed += 1
        if row["index"] == binding["start"] and \
                row["base"]["first_prompt_sha256"] != row["online"]["first_prompt_sha256"]:
            raise ValueError("Arms differ before first write")
    paired = [x for x in rows if x["base"]["status"] ==
              x["online"]["status"] == "complete"]
    summary = report["summary"]["blind_spectrum_monitoring"]
    if (summary["paired"] != len(paired) or
            summary["writes"] != sum(x["write"]["written"] for x in paired) or
            summary["base_mean"] != (statistics.mean(x["base"]["reward"]
                for x in paired) if paired else None) or
            summary["online_mean"] != (statistics.mean(x["online"]["reward"]
                for x in paired) if paired else None)):
        raise ValueError("Report summary changed")
    return {"audit": "passed", "report_sha256": file_hash(report_path),
            "review_sha256": file_hash(review_path),
            "gate_model_sha256": file_hash(gate_path),
            "completed_arms": confirmed, "failed_arms": failed, "paired": len(paired)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--gate-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = audit(args.report, args.review, args.gate_model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
