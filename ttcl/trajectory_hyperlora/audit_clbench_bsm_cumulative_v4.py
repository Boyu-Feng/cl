"""Read-only official-score and lineage audit for BSM cumulative experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics

from ttcl.trajectory_hyperlora.alfworld_online_from_empty import digest, file_hash
from ttcl.trajectory_hyperlora.clbench_online_parameter_memory_v2 import ROOT, benchmark
from ttcl.trajectory_hyperlora.train_clbench_cumulative_hyperlora_v4 import merge_object_arrays


def audit(report_path: Path, review_path: Path):
    from src.tasks.blind_spectrum_monitoring.task import ScanReport, _score_report

    report = json.loads(report_path.read_text())
    review = json.loads(review_path.read_text())
    binding = review["binding"]
    if ("blind_spectrum_monitoring" not in binding["domains"] or
            binding["start"] < 8 or
            review["input_content_sha256"] != digest({
                "binding": binding, "targets": review["targets"]}) or
            report["review_sha256"] != file_hash(review_path) or
            report["checkpoint_sha256"] != binding["checkpoint_sha256"]):
        raise ValueError("Review or checkpoint lineage mismatch")
    rows = [x for x in report["rows"] if x["domain"] == "blind_spectrum_monitoring"]
    targets = [x for x in review["targets"] if x["domain"] == "blind_spectrum_monitoring"]
    if [x["index"] for x in rows] != list(range(binding["start"], binding["stop"])):
        raise ValueError("Missing or reordered official targets")
    confirmed = 0
    failed = 0
    public_action_history = []
    bench = benchmark()
    for row, target in zip(rows, targets, strict=True):
        if row["target"] != target or row["domain"] != target["domain"]:
            raise ValueError("Target identity changed")
        task = bench.make_task("blind_spectrum_monitoring", binding["seed"],
                               independent=True)
        query = task.reset_baseline_instance(row["index"])
        if (target["instance_id"] != query.instance_id or
                target["initial_query_sha256"] != digest(query.prompt)):
            raise ValueError("Official reset changed")
        latent = task._get_all_latent_channel_defs()
        for arm in ("base", "online"):
            outcome = row[arm]
            if (outcome["instance_id"] != query.instance_id or
                    outcome["initial_query_sha256"] != digest(query.prompt)):
                raise ValueError("Episode target mismatch")
            directory = (report_path.parent /
                         (report_path.stem + "_episodes") /
                         "blind_spectrum_monitoring" / arm /
                         f"episode_{row['index']+1:03}")
            if arm == "online":
                for line in (directory / "responses.jsonl").read_text().splitlines():
                    event = json.loads(line)
                    if "generated_raw_response" in event:
                        original = json.loads(event["generated_raw_response"])
                        expected = merge_object_arrays(original, public_action_history)
                        if json.loads(event["raw_response"]) != expected:
                            raise ValueError("Event action used unbound history")
            episode = json.loads((directory / "trajectory.json").read_text())
            if outcome["status"] == "complete":
                if len(episode["steps"]) != 1:
                    raise ValueError("Expected one executed BSM report")
                action = episode["steps"][0]["action"]
                score = _score_report(ScanReport.model_validate(action), latent,
                                      task.W, task.G, task.band_width)["score"]
                if (float(outcome["reward"]) != float(score) or
                        float(episode["reward"]) != float(score)):
                    raise ValueError("Official BSM reward mismatch")
                confirmed += 1
                if arm == "online":
                    public_action_history.append(action)
            else:
                if outcome["reward"] is not None:
                    raise ValueError("Failed episode has fabricated reward")
                failed += 1
        if row["index"] == binding["start"] and \
                row["base"]["first_prompt_sha256"] != row["online"]["first_prompt_sha256"]:
            raise ValueError("Arms differ before first online write")
    paired = [x for x in rows if
              x["base"]["status"] == x["online"]["status"] == "complete"]
    summary = report["summary"]["blind_spectrum_monitoring"]
    if (summary["paired"] != len(paired) or
            summary["writes"] != sum(bool(x["write"]["written"]) for x in rows) or
            summary["base_mean"] != (statistics.mean(x["base"]["reward"] for x in paired)
                                     if paired else None) or
            summary["online_mean"] != (statistics.mean(x["online"]["reward"] for x in paired)
                                       if paired else None)):
        raise ValueError("Paired summary mismatch")
    return {"audit": "passed", "report_sha256": file_hash(report_path),
            "review_sha256": file_hash(review_path),
            "completed_arms": confirmed, "failed_arms": failed,
            "paired": len(paired)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = audit(args.report, args.review)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
