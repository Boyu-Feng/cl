"""Direct benchmark evaluation for a trained Delta-Mem adapter."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


TTCL_ROOT = Path(__file__).resolve().parents[1]
BENCH_ROOT = TTCL_ROOT.parent / "current_work" / "continual-learning-bench"
DELTA_ROOT = BENCH_ROOT.parent / "delta-Mem"
for import_root in (TTCL_ROOT.parent, BENCH_ROOT, DELTA_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a Delta-Mem adapter directly on benchmark scan data."
    )
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--delta-adapter", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=8192,
        help="Safety ceiling; generation still stops normally at EOS.",
    )
    parser.add_argument("--num-trajectories", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--mode",
        choices=["stateful", "stateless"],
        default="stateful",
        help="stateful keeps Delta-Mem/session history across scans; stateless resets it before every scan.",
    )
    return parser.parse_args()


def _load_dependencies():
    from deltamem.runtime.session import DeltaMemChatSession, load_delta_mem_chat_model
    from src.interface import Response
    from src.tasks.blind_spectrum_monitoring.task import (
        BlindSpectrumMonitoringTask,
        ScanReport,
    )

    return DeltaMemChatSession, load_delta_mem_chat_model, Response, BlindSpectrumMonitoringTask, ScanReport


def _parse_report(raw: str, report_type: type[Any]) -> Any:
    candidates = [raw.strip()]
    start = raw.find("{")
    if start >= 0:
        depth = 0
        in_string = False
        escaped = False
        for index in range(start, len(raw)):
            char = raw[index]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    candidates.insert(0, raw[start : index + 1])
                    break
    for candidate in candidates:
        try:
            return report_type.model_validate_json(candidate)
        except (ValueError, json.JSONDecodeError):
            continue
    raise ValueError(f"Delta-Mem output is not valid {report_type.__name__}: {raw[:400]!r}")


def run(args: argparse.Namespace) -> dict[str, Any]:
    DeltaMemChatSession, load_delta_mem_chat_model, Response, Task, ScanReport = _load_dependencies()
    data_path = Path(args.data_path).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not data_path.is_file():
        raise FileNotFoundError(f"Benchmark data not found: {data_path}")

    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("Require num_shards >= 1 and 0 <= shard_index < num_shards")

    source_lines = [line for line in data_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.num_shards > 1:
        source_lines = [
            line for index, line in enumerate(source_lines) if index % args.num_shards == args.shard_index
        ]
        data_path = output_dir / f"input_shard_{args.shard_index:02d}.jsonl"
        data_path.write_text("\n".join(source_lines) + "\n", encoding="utf-8")

    with data_path.open(encoding="utf-8") as handle:
        data_count = sum(1 for line in handle if line.strip())
    task = Task(
        dataset_path=str(data_path),
        num_instances=args.num_trajectories or data_count,
        seed=args.seed,
    )
    model, tokenizer = load_delta_mem_chat_model(
        model_path=args.base_model,
        device=args.device,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
        adapter_dir=args.delta_adapter,
    )
    session = DeltaMemChatSession(
        model=model,
        tokenizer=tokenizer,
        device=args.device,
    )

    query = task.reset()
    progress_path = output_dir / "progress.json"
    snapshot_dir = output_dir / "session_snapshot"
    records: list[dict[str, Any]] = []
    if args.resume and progress_path.is_file():
        saved = json.loads(progress_path.read_text(encoding="utf-8"))
        records = list(saved.get("trajectory_records", []))
        for record in records:
            report = ScanReport.model_validate(record["report"])
            step_result = task.step(Response(action=report))
            if step_result.done or step_result.next_query is None:
                query = None
                break
            query = step_result.next_query
        if snapshot_dir.is_dir() and records:
            session.load_snapshot_dir(snapshot_dir)
        print(f"resumed={len(records)} shard={args.shard_index}/{args.num_shards}", flush=True)

    if query is None:
        step_result = None
    while True:
        if query is None:
            break
        if args.mode == "stateless":
            # Independent form: no conversation, KV cache, or online Delta-Mem
            # state is carried from the previous scan.
            session.reset()
        result = session.generate_reply(
            query.prompt,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            write_enabled=True,
        )
        report = _parse_report(str(result["assistant_display"]), ScanReport)
        step_result = task.step(Response(action=report))
        reward = step_result.instance_outcome.reward if step_result.instance_outcome else 0.0
        records.append(
            {
                "trajectory": len(records) + 1,
                "reward": reward,
                "report": report.model_dump(mode="json"),
            }
        )
        scores = [float(item["reward"]) for item in records]
        running_mean = sum(scores) / len(scores)
        progress_path.write_text(
            json.dumps(
                {
                    "mode": args.mode,
                    "completed": len(records),
                    "total": task.num_instances,
                    "last_reward": reward,
                    "running_mean_score": round(running_mean, 4),
                    "trajectory_records": records,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        session.save_snapshot(snapshot_dir)
        state_stats = session.state_stats()
        print(
            f"mode={args.mode} trajectory={len(records)}/{task.num_instances} "
            f"reward={reward:.4f} running_mean={running_mean:.4f} "
            f"delta_state_norm={state_stats.get('mean_state_norm', 0.0):.4f}",
            flush=True,
        )
        if step_result.done or step_result.next_query is None:
            break
        query = step_result.next_query

    metrics = task.evaluate()
    # Preserve the benchmark's standardized metrics as well as task-specific
    # metrics.  The latter contains mean IoU, learning delta, score curve,
    # interaction counts, and per-scan results for Blind Spectrum Monitoring.
    eval_metrics = metrics.eval_metrics.model_dump(mode="json")
    payload = {
        "base_model": args.base_model,
        "delta_adapter": args.delta_adapter,
        "data_path": str(data_path),
        "num_trajectories": len(records),
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "trajectory_records": records,
        "benchmark_metrics": metrics.metrics,
        "eval_metrics": eval_metrics,
        "summary": metrics.summary,
        "score": metrics.score,
        "mode": args.mode,
    }
    (output_dir / "metrics.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    return payload


def main() -> None:
    print(json.dumps(run(parse_args()), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
