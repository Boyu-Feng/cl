"""Direct TTCL runner for the continual-learning-bench spectrum task.

This bypasses the clbench CLI/runner while reusing its task data and scoring.
Each scan is one trajectory. LoRA is updated after every ``update_every`` scans.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.tasks.blind_spectrum_monitoring.task import ScanReport


REPO_ROOT = Path(__file__).resolve().parents[2]
BENCH_ROOT = REPO_ROOT / "current_work" / "continual-learning-bench"
for import_root in (REPO_ROOT, BENCH_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run blind-spectrum benchmark directly with online LoRA updates."
    )
    parser.add_argument("--base-model", required=True, help="Local Hugging Face base model path.")
    parser.add_argument(
        "--experience-adapter",
        default=None,
        help="Optional existing PEFT LoRA adapter to load as the experience module.",
    )
    parser.add_argument("--data-path", required=True, help="Benchmark frozen-corpus JSONL path.")
    parser.add_argument("--output-dir", required=True, help="Directory for LoRA checkpoints and metrics.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--update-every", type=int, default=8)
    parser.add_argument("--replay-trajectories", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--train-epochs", type=int, default=1)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--target-modules", default="q_proj,k_proj,v_proj,o_proj")
    parser.add_argument("--num-trajectories", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _schema_prompt(prompt: str) -> str:
    from src.tasks.blind_spectrum_monitoring.task import ScanReport

    schema = json.dumps(ScanReport.model_json_schema(), ensure_ascii=False)
    return (
        f"{prompt}\n\n"
        "Return ONLY one valid JSON object matching this schema. "
        "Do not return YAML, Markdown, or explanations.\n"
        f"JSON schema: {schema}"
    )


def _parse_report(raw: str) -> ScanReport:
    from src.tasks.blind_spectrum_monitoring.task import ScanReport

    candidates = [raw.strip()]
    start, end = raw.find("{"), raw.rfind("}")
    if start >= 0 and end > start:
        candidates.append(raw[start : end + 1])
    for candidate in candidates:
        try:
            return ScanReport.model_validate_json(candidate)
        except (ValueError, json.JSONDecodeError):
            continue
    raise ValueError(f"Model did not return valid ScanReport JSON: {raw[:400]!r}")


def run(args: argparse.Namespace) -> dict[str, Any]:
    from src.interface import Response
    from src.tasks.blind_spectrum_monitoring.task import BlindSpectrumMonitoringTask
    from ttcl.online_lora.online_lora import OnlineLoRAMemory

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    data_path = Path(args.data_path).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not data_path.is_file():
        raise FileNotFoundError(f"Benchmark data not found: {data_path}")

    with data_path.open(encoding="utf-8") as data_file:
        data_count = sum(1 for line in data_file if line.strip())
    task = BlindSpectrumMonitoringTask(
        dataset_path=str(data_path),
        num_instances=args.num_trajectories or data_count,
        seed=args.seed,
    )
    memory = OnlineLoRAMemory(
        model_path=args.base_model,
        adapter_path=args.experience_adapter,
        output_dir=str(output_dir / "adapters"),
        device=args.device,
        dtype=args.dtype,
        update_every=args.update_every,
        replay_trajectories=args.replay_trajectories,
        learning_rate=args.learning_rate,
        train_epochs=args.train_epochs,
        max_seq_length=args.max_seq_length,
        max_new_tokens=args.max_new_tokens,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=tuple(item.strip() for item in args.target_modules.split(",") if item.strip()),
    )

    query = task.reset()
    records: list[dict[str, Any]] = []
    while True:
        raw_response = memory.respond(_schema_prompt(query.prompt))
        report = _parse_report(raw_response)
        response_payload = report.model_dump(mode="json")
        step_result = task.step(Response(action=report))
        reward = step_result.instance_outcome.reward if step_result.instance_outcome else 0.0
        trajectory = [
            {
                "prompt": query.prompt,
                "response": json.dumps(response_payload, ensure_ascii=False),
                "result": step_result.observation.content,
                "reward": reward,
            }
        ]
        memory.record_step(query.prompt, json.dumps(response_payload, ensure_ascii=False), step_result.observation.content)
        updated = memory.observe_trajectory(trajectory)
        records.append(
            {
                "trajectory": len(records) + 1,
                "reward": reward,
                "updated": updated,
                "update_count": memory.update_count,
            }
        )
        print(
            f"trajectory={len(records)} reward={reward:.4f} "
            f"updates={memory.update_count}",
            flush=True,
        )
        if step_result.done or step_result.next_query is None:
            break
        query = step_result.next_query

    metrics = task.evaluate()
    result = {
        "base_model": args.base_model,
        "experience_adapter": args.experience_adapter,
        "data_path": str(data_path),
        "update_every": args.update_every,
        "replay_trajectories": args.replay_trajectories,
        "num_trajectories": len(records),
        "num_updates": memory.update_count,
        "trajectory_records": records,
        "benchmark_metrics": metrics.metrics,
        "summary": metrics.summary,
        "score": metrics.score,
    }
    (output_dir / "metrics.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
    memory.save(output_dir / "latest")
    return result


def main() -> None:
    args = parse_args()
    result = run(args)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
