"""CLBench BSM adapter for task-independent LLM experience memory."""
from __future__ import annotations

import argparse
from functools import partial
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from ttcl.common.bsm import BENCH, parse_report  # noqa: E402
from ttcl.common.local_qwen import LocalQwen  # noqa: E402
from ttcl.llm_memory.memory import Episode, ExperienceMemory, answer_messages  # noqa: E402


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str, allow_nan=False), encoding="utf-8")


def append(path, value):
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")


def seed_for(base, identity, purpose):
    # Answer seeds exactly match the independent/full-history ICL experiment.
    payload = json.dumps([base, identity, 0 if purpose == "answer" else "memory"], ensure_ascii=False)
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], "big") % (2**63)


class RecordedCalls:
    def __init__(self, model, out):
        self.model, self.out = model, out
        self.completed = []
        self.attempted = 0

    def call(self, purpose, scan, messages, seed, **kwargs):
        self.attempted += 1
        meta = {"call_id": self.attempted, "purpose": purpose, "scan": scan, "seed": seed}
        append(self.out / "requests.jsonl", {**meta, "messages": messages, "options": kwargs})
        result = self.model.generate(messages, seed, **kwargs)
        item = {**meta, **result}
        append(self.out / "generations.jsonl", item)
        self.completed.append(item)
        return result


def run_mode(args, mode, model, task_factory=None):
    from src.interface import Response
    from src.tasks.blind_spectrum_monitoring.task import BlindSpectrumMonitoringTask

    out = args.output_dir / mode
    out.mkdir()
    calls = RecordedCalls(model, out)
    memory = (None if mode == "independent" else ExperienceMemory(
        include_reward=mode == "summary_reward", max_new_tokens=args.memory_max_new_tokens))
    task = (task_factory or BlindSpectrumMonitoringTask)(
        dataset_path=str(args.data_path), num_instances=args.num_scans,
        seed=args.seed, repeat_instructions=True)
    records = []
    start = time.monotonic()
    scan, phase = 0, "reset"
    try:
        query = task.reset()
        while query is not None and len(records) < args.num_scans:
            scan = len(records) + 1
            context = memory.text if memory is not None else ""
            identity = query.instance_id or query.prompt
            messages = answer_messages(query.prompt, query.response_schema.model_json_schema(), context)
            phase = "answer"
            completion = calls.call("answer", scan, messages, seed_for(args.seed, identity, "answer"))
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
            phase = "evaluation"
            step = task.step(response)
            reward = float(step.instance_outcome.reward)
            record = {"scan": scan, "instance_id": query.instance_id, **completion,
                      "reward": reward, "report": report.model_dump() if report else None,
                      "parse_error": error, "summary_context": context,
                      "memory_through_episode": memory.through_episode if memory else 0,
                      "public_feedback": step.observation.content}
            append(out / "responses.jsonl", record)
            records.append(record)
            if memory is not None:
                phase = "memory_update"
                # No task/query/observation metadata, hidden labels, or future
                # query are passed across the boundary into the generic core.
                episode = Episode(scan, query.instance_id, query.prompt,
                                  completion["raw_response"], step.observation.content,
                                  reward if memory.include_reward else None)
                append(out / "episodes.jsonl", episode.payload(memory.include_reward))
                update = memory.update(episode, partial(calls.call, "memory", scan),
                                       seed_for(args.seed, identity, "memory"))
                append(out / "memory_updates.jsonl", update)
                dump(out / "memory.json", memory.state_dict())
            metrics = {
                "mode": mode, "completed": len(records), "requested_scans": args.num_scans,
                "mean_score": sum(r["reward"] for r in records) / len(records),
                "invalid_reports": sum(r["parse_error"] is not None for r in records),
                "answer_calls": sum(c["purpose"] == "answer" for c in calls.completed),
                "memory_calls": sum(c["purpose"] == "memory" for c in calls.completed),
                "model_calls": len(calls.completed), "reward_calls": len(records), "parameter_updates": 0,
                "scalar_reward_supplied": mode == "summary_reward",
                "rejected_memory_updates": memory.rejected_updates if memory else 0,
                "memory_through_episode": memory.through_episode if memory else 0,
                "pending_memory_episodes": len(memory.pending) if memory else 0,
                "total_input_tokens": sum(c["input_tokens"] for c in calls.completed),
                "total_output_tokens": sum(c["output_tokens"] for c in calls.completed),
                "max_answer_input_tokens": max(r["input_tokens"] for r in records),
                "elapsed_seconds": time.monotonic() - start,
            }
            dump(out / "progress.json", metrics)
            print(f"{mode} scan={scan}/{args.num_scans} reward={reward:.4f} "
                  f"mean={metrics['mean_score']:.4f} memory_through={metrics['memory_through_episode']}", flush=True)
            if step.done:
                break
            query = step.next_query
        if not records:
            raise ValueError("Task returned no queries")
        metrics["score"] = task.evaluate().score
        dump(out / "metrics.json", metrics)
        return metrics
    except Exception as exc:
        dump(out / "failure.json", {"scan": scan, "phase": phase, "completed_answers": len(records),
                                    "attempted_model_calls": calls.attempted,
                                    "error_type": type(exc).__name__, "error": str(exc)})
        raise


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=ROOT / "current_work/delta-Mem/model/Qwen3-4B-Instruct-2507")
    parser.add_argument("--data-path", type=Path, default=BENCH / "data/blind_spectrum_monitoring/mixed_grid_lifecycle.jsonl")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=["independent", "summary", "summary_reward", "all"], default="all")
    parser.add_argument("--num-scans", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--context-limit", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=1536)
    parser.add_argument("--memory-max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=0)
    args = parser.parse_args(argv)
    if min(args.num_scans, args.max_new_tokens, args.memory_max_new_tokens) < 1:
        parser.error("counts must be positive")
    if args.context_limit is not None and args.context_limit < 1:
        parser.error("context limit must be positive")
    if (not math.isfinite(args.temperature) or args.temperature <= 0
            or not math.isfinite(args.top_p) or not 0 < args.top_p <= 1 or args.top_k < 0):
        parser.error("invalid sampling configuration")
    return args


def main():
    args = parse_args()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    dump(args.output_dir / "config.json", vars(args))
    source_paths = [Path(__file__), ROOT / "ttcl/llm_memory/memory.py", ROOT / "ttcl/common/local_qwen.py",
                    ROOT / "ttcl/common/bsm.py", ROOT / "ttcl/llm_memory/run.sh"]
    snapshot = args.output_dir / "code_snapshot"
    snapshot.mkdir()
    for src in source_paths:
        target = snapshot / src.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, target)
    dump(args.output_dir / "provenance.json", {
        str(src): hashlib.sha256(src.read_bytes()).hexdigest()
        for src in [*source_paths, args.data_path, args.model / "config.json"]})
    model = LocalQwen(args)
    modes = ["independent", "summary", "summary_reward"] if args.mode == "all" else [args.mode]
    results = {mode: run_mode(args, mode, model) for mode in modes}
    dump(args.output_dir / "comparison.json", results)
    lines = ["# LLM 生成的通用经验记忆", "", "| 方法 | 完成 | 平均分 | 答题调用 | 总结调用 | 评分次数 |",
             "|---|---:|---:|---:|---:|---:|"]
    for mode, m in results.items():
        lines.append(f"| {mode} | {m['completed']} | {m['mean_score']:.6f} | {m['answer_calls']} | "
                     f"{m['memory_calls']} | {m['reward_calls']} |")
    lines.extend(["", "模型参数冻结；每题只回答和评分一次。总结由同一个本地 Qwen3 贪心生成，没有额外评分。",
                  "summary 只接收公开反馈；summary_reward 额外接收已经获得的整份回答标量 reward。",
                  "每次回答仅能读取此前完成的记忆。最后一题后也生成记忆并计入调用量，便于审计和后续使用。",
                  "通用记忆核心无频率、带宽、候选聚类规则；当前实测适配器只覆盖 BSM，尚未验证跨任务效果。",
                  "记忆是模型生成的有损摘要，可能遗漏或误归因。若总结被截断，拒绝替换记忆并保留待总结交互。"])
    if "independent" in results:
        for mode in modes:
            if mode != "independent":
                delta = results[mode]["mean_score"] - results["independent"]["mean_score"]
                lines.append(f"{mode} 相对 independent 的平均分变化：{delta:+.6f}。")
    (args.output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
