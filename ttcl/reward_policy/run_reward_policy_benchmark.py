"""Policy-gradient reward-feedback BSM experiment; one response per scan."""

from __future__ import annotations

import argparse
from collections import deque
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from ttcl.common.bsm import BENCH, parse_report, parse_scan_observation  # noqa: E402


def make_prompt(query, history):
    prompt = query.prompt
    if history:
        prompt += "\n\nPast public observations (may include noise):\n" + json.dumps(
            list(history)
        )
    return prompt + (
        '\n\nReturn ONLY one JSON object with a short "decision_summary" string '
        "(at most one sentence describing the evidence for your decision) and the "
        'required "transmitters" array. Use observations to assess persistent occupancy '
        "and uncertainty. Do not invent past measurements.\nReport schema: "
        + json.dumps(query.response_schema.model_json_schema())
    )


def run(args, policy_factory=None):
    from src.interface import Response
    from src.tasks.blind_spectrum_monitoring.task import BlindSpectrumMonitoringTask

    if policy_factory is None:
        from ttcl.reward_policy.reward_lora import RewardLoRA

        policy_factory = RewardLoRA
    output = Path(args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Choose a new output directory: {output}")
    data = Path(args.data_path).resolve()
    with data.open() as handle:
        available = sum(bool(line.strip()) for line in handle)
    count = min(args.num_scans, available)
    if count < 1:
        raise ValueError("Dataset must contain at least one scan")
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps(vars(args), indent=2))
    task = BlindSpectrumMonitoringTask(
        dataset_path=str(data),
        num_instances=count,
        seed=args.seed,
        repeat_instructions=True,
    )
    policy = policy_factory(args)
    history = deque(maxlen=args.history_scans)
    records = []
    feedback = []
    query = task.reset()
    start = time.monotonic()
    for index in range(count):
        prompt = make_prompt(query, history)
        sample = policy.sample(prompt)
        error = None
        try:
            report = parse_report(sample.text, query.response_schema)
        except ValueError as exc:
            error = str(exc)
            report = None
        # Exactly one task transition. Never score rejected candidate answers.
        if report is None:
            step = task.step(
                Response(
                    action=query.response_schema(transmitters=[]),
                    metadata={"latency_timeout": True},
                )
            )
        else:
            step = task.step(Response(action=report))
        reward = float(step.instance_outcome.reward)
        has_next = index + 1 < count and not step.done and step.next_query is not None
        record = {
            "scan": index + 1,
            "instance_id": query.instance_id,
            "prompt": prompt,
            "response": sample.text,
            "report": report.model_dump() if report is not None else None,
            "parse_error": error,
            "truncated": sample.truncated,
            "reward": reward,
            "updates_before_answer": policy.updates,
        }
        records.append(record)
        with (output / "responses.jsonl").open("a") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        # This is the ONLY scorer information supplied to the learner.
        update = policy.observe(
            sample, reward, update=args.mode == "online" and has_next
        )
        feedback.append(update)
        with (output / "updates.jsonl").open("a") as handle:
            handle.write(json.dumps({"scan": index + 1, **update}) + "\n")
        observation = parse_scan_observation(query.prompt, query.instance_id)
        history.append(
            {
                key: observation[key]
                for key in (
                    "scan_number",
                    "noise_floor_dbm",
                    "band_mhz",
                    "detected_peaks",
                )
            }
        )
        progress = {
            "completed": len(records),
            "total": count,
            "num_updates": policy.updates,
            "mean_score": sum(row["reward"] for row in records) / len(records),
            "invalid_reports": sum(row["parse_error"] is not None for row in records),
            "elapsed_seconds": time.monotonic() - start,
        }
        (output / "progress.json").write_text(json.dumps(progress, indent=2))
        print(
            f"{args.mode} scan={index + 1}/{count} reward={reward:.4f} "
            f"A={update['advantage']:+.3f} update={update['reason']} "
            f"mean={progress['mean_score']:.4f}",
            flush=True,
        )
        if not has_next:
            break
        query = step.next_query
    if policy.updates:
        policy.save(output / "latest_adapter")
    metrics = {
        **progress,
        "mode": args.mode,
        "model": args.model,
        "score_curve": [row["reward"] for row in records],
        "rejected_updates": sum(
            row["reason"] == "kl_or_numerical_rejection" for row in feedback
        ),
        "protocol": "scalar reward feedback after submission; one action per scan; no counterfactual scoring",
        "history_scans": args.history_scans,
        "summary": task.evaluate().summary,
    }
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default=str(ROOT / "current_work/delta-Mem/model/Qwen3-4B-Instruct-2507"),
    )
    parser.add_argument(
        "--data-path",
        default=str(
            BENCH / "data/blind_spectrum_monitoring/mixed_grid_lifecycle.jsonl"
        ),
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mode", choices=["frozen", "online"], default="online")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-scans", type=int, default=12)
    parser.add_argument("--history-scans", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--update-steps", type=int, default=2)
    parser.add_argument("--kl-beta", type=float, default=0.05)
    parser.add_argument("--max-update-kl", type=float, default=0.02)
    parser.add_argument("--max-input-tokens", type=int, default=8192)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    for name in (
        "num_scans",
        "warmup",
        "update_steps",
        "max_input_tokens",
        "max_new_tokens",
    ):
        if getattr(args, name) < 1:
            parser.error(f"{name} must be positive")
    if args.history_scans < 0:
        parser.error("history-scans must be nonnegative")
    for name in ("learning_rate", "kl_beta", "max_update_kl"):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0 or (name != "kl_beta" and value == 0):
            parser.error(f"Invalid {name}")
    return args


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2))
