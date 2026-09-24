"""Full-history ICL and independent-query controls on CLBench BSM."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from ttcl.common.bsm import BENCH, parse_report  # noqa: E402
from ttcl.common.local_qwen import ContextLimitError, LocalQwen, check_context  # noqa: E402,F401


def query_seed(seed, identity):
    payload = json.dumps([seed, identity, 0], ensure_ascii=False)
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], "big") % (2**63)


def append(path, value):
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")


class Conversation:
    """Keep every original turn; only the independent control resets."""
    def __init__(self, mode):
        self.mode = mode
        self.messages = []

    def begin(self, query):
        if self.mode == "independent":
            self.messages.clear()
        prompt = query.prompt + ("\n\nReturn ONLY one valid JSON object, with no explanation. Schema:\n"
                                 + json.dumps(query.response_schema.model_json_schema()))
        self.messages.append({"role": "user", "content": prompt})
        return copy.deepcopy(self.messages)

    def finish(self, raw_response, public_feedback):
        self.messages.append({"role": "assistant", "content": raw_response})
        if public_feedback:
            self.messages.append({"role": "user", "content": "FEEDBACK: " + public_feedback})


def run_mode(args, mode, model, task_factory=None):
    from src.interface import Response
    from src.tasks.blind_spectrum_monitoring.task import BlindSpectrumMonitoringTask

    out = args.output_dir / mode
    out.mkdir()
    task = (task_factory or BlindSpectrumMonitoringTask)(
        dataset_path=str(args.data_path), num_instances=args.num_scans,
        seed=args.seed, repeat_instructions=True)
    conversation = Conversation(mode)
    records = []
    input_tokens = output_tokens = 0
    journal_count = 0
    start = time.monotonic()
    query = task.reset()
    while query is not None and len(records) < args.num_scans:
        scan = len(records) + 1
        messages = conversation.begin(query)
        # Append each message exactly once; offsets reconstruct the complete input
        # without saving N duplicate copies of the expanding history.
        history_start = 0 if mode == "full_history" else journal_count
        append(out / "messages.jsonl", {"index": journal_count, **messages[-1]})
        journal_count += 1
        seed = query_seed(args.seed, query.instance_id or query.prompt)
        try:
            completion = model.generate(messages, seed)
        except Exception as exc:
            (out / "failure.json").write_text(json.dumps({
                "scan": scan, "completed": len(records), "error_type": type(exc).__name__,
                "error": str(exc), "history_truncated": False,
                "message_start": history_start, "message_end": journal_count}, indent=2))
            raise
        try:
            report = parse_report(completion["raw_response"], query.response_schema)
            if any(not math.isfinite(v) for row in report.transmitters
                   for v in (row.center_freq, row.bandwidth, row.estimated_power)):
                raise ValueError("Non-finite report values")
            error = None
        except ValueError as exc:
            report, error = None, str(exc)
        response = (Response(action=report) if report is not None else Response(
            action=query.response_schema(transmitters=[]), metadata={"latency_timeout": True}))
        step = task.step(response)
        feedback = step.observation.content
        record = {"scan": scan, "instance_id": query.instance_id, **completion,
                  "generation_seed": seed, "reward": float(step.instance_outcome.reward),
                  "report": report.model_dump() if report is not None else None,
                  "parse_error": error, "public_feedback": feedback,
                  "message_start": history_start, "message_end": journal_count,
                  "input_message_count": len(messages), "history_truncated": False}
        append(out / "responses.jsonl", record)
        records.append(record)
        before = len(conversation.messages)
        conversation.finish(completion["raw_response"], feedback)
        for message in conversation.messages[before:]:
            append(out / "messages.jsonl", {"index": journal_count, **message})
            journal_count += 1
        input_tokens += completion["input_tokens"]
        output_tokens += completion["output_tokens"]
        metrics = {"mode": mode, "completed": scan, "requested_scans": args.num_scans,
                   "mean_score": sum(r["reward"] for r in records) / scan,
                   "invalid_reports": sum(r["parse_error"] is not None for r in records),
                   "model_calls": scan, "reward_calls": scan, "parameter_updates": 0,
                   "total_input_tokens": input_tokens, "total_output_tokens": output_tokens,
                   "max_input_tokens": max(r["input_tokens"] for r in records),
                   "context_limit": completion["context_limit"], "history_truncated": False,
                   "scalar_reward_in_context": False, "elapsed_seconds": time.monotonic() - start}
        (out / "progress.json").write_text(json.dumps(metrics, indent=2))
        print(f"{mode} scan={scan}/{args.num_scans} reward={record['reward']:.4f} "
              f"mean={metrics['mean_score']:.4f} context_tokens={completion['input_tokens']}", flush=True)
        if step.done:
            break
        query = step.next_query
    metrics["score"] = task.evaluate().score
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (out / "history.json").write_text(json.dumps(conversation.messages, ensure_ascii=False, indent=2))
    return metrics


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=ROOT / "current_work/delta-Mem/model/Qwen3-4B-Instruct-2507")
    parser.add_argument("--data-path", type=Path, default=BENCH / "data/blind_spectrum_monitoring/mixed_grid_lifecycle.jsonl")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=["full_history", "independent", "both"], default="both")
    parser.add_argument("--num-scans", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--context-limit", type=int, help="Total context cap including output reservation; never truncate")
    parser.add_argument("--max-new-tokens", type=int, default=1536)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=0)
    args = parser.parse_args(argv)
    if min(args.num_scans, args.max_new_tokens) < 1 or (args.context_limit is not None and args.context_limit < 1):
        parser.error("counts and context limit must be positive")
    if (not math.isfinite(args.temperature) or args.temperature <= 0
            or not math.isfinite(args.top_p) or not 0 < args.top_p <= 1 or args.top_k < 0):
        parser.error("invalid sampling configuration")
    return args


def main():
    args = parse_args()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "config.json").write_text(json.dumps(vars(args), default=str, indent=2))
    model = LocalQwen(args)
    modes = ["independent", "full_history"] if args.mode == "both" else [args.mode]
    results = {mode: run_mode(args, mode, model) for mode in modes}
    if len(results) == 2:
        results["full_history_minus_independent"] = results["full_history"]["mean_score"] - results["independent"]["mean_score"]
    (args.output_dir / "comparison.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
