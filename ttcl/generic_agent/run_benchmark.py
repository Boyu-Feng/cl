"""GenericAgent + local Qwen3, using original agent/history/memory and CLBench scoring."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import selectors
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from ttcl.common.bsm import BENCH, parse_report  # noqa: E402


def append(path, value):
    with path.open("a") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")


def seed_for(seed, identity):
    return int.from_bytes(hashlib.sha256(json.dumps([seed, identity, 0]).encode()).digest()[:8], "big") % (2**63)


def prepare_runtime(args, directory):
    shutil.copytree(args.ga_dir, directory, ignore=shutil.ignore_patterns(
        ".git", "__pycache__", "*.pyc", "temp", "node_modules", ".venv"))
    config = {"backend": "local_qwen", "model": "/model", "device": "cuda:0",
              "name": "local-Qwen3", "dtype": "bfloat16", "temperature": 0.7,
              "do_sample": True, "top_p": 0.9, "top_k": 0, "seed": args.seed,
              "max_tokens": args.max_new_tokens, "max_input_tokens": 16384,
              "context_win": 8192}
    (directory / "mykey.py").write_text("local_qwen_config = " + repr(config) + "\n")
    shutil.copy2(Path(__file__).with_name("worker.py"), directory / "benchmark_worker.py")


def sandbox_command(args, runtime):
    # No host workspace, benchmark corpus, old results or network is mounted.
    command = ["bwrap", "--die-with-parent", "--unshare-pid", "--unshare-net"]
    for name in ("/usr", "/bin", "/sbin", "/lib", "/lib64", "/etc", str(Path(sys.executable).resolve().parents[2])):
        if Path(name).exists():
            command += ["--ro-bind", name, name]
    command += ["--proc", "/proc", "--dev-bind", "/dev", "/dev", "--tmpfs", "/tmp",
                "--bind", str(runtime), "/agent", "--ro-bind", str(args.model), "/model",
                "--chdir", "/agent",
                "--setenv", "GA_LANG", "en", "--setenv", "TOKENIZERS_PARALLELISM", "false",
                "--setenv", "GA_INLINE_PROMPT", "1",
                sys.executable, "/agent/benchmark_worker.py"]
    return command


class Worker:
    def __init__(self, args, runtime):
        prepare_runtime(args, runtime)
        self.log = (runtime.parent / (runtime.name + ".log")).open("w")
        self.proc = subprocess.Popen(sandbox_command(args, runtime), stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, stderr=self.log, text=True,
                                     env={**os.environ, "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"})

    def ask(self, prompt, seed, timeout):
        self.proc.stdin.write(json.dumps({"prompt": prompt, "seed": seed, "timeout": timeout}) + "\n")
        self.proc.stdin.flush()
        with selectors.DefaultSelector() as selector:
            selector.register(self.proc.stdout, selectors.EVENT_READ)
            if not selector.select(timeout):
                raise TimeoutError("GenericAgent timed out; inspect its runtime log")
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError("GenericAgent worker exited; inspect its runtime log")
        return json.loads(line)

    def close(self):
        if self.proc.poll() is None:
            try:
                self.proc.stdin.write('{"stop":true}\n')
                self.proc.stdin.flush()
                self.proc.wait(timeout=10)
            except (BrokenPipeError, subprocess.TimeoutExpired):
                self.proc.terminate()
                try:
                    self.proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.proc.kill()
                    self.proc.wait()
        self.proc.stdin.close()
        self.proc.stdout.close()
        self.log.close()


def run_mode(args, mode, worker_factory=Worker):
    from src.interface import Response
    from src.tasks.blind_spectrum_monitoring.task import BlindSpectrumMonitoringTask

    out = args.output_dir / mode
    out.mkdir()
    task = BlindSpectrumMonitoringTask(dataset_path=str(args.data_path), num_instances=args.num_scans,
                                       seed=args.seed, repeat_instructions=True)
    query = task.reset()
    records, calls, tokens, tools, memory_calls = [], 0, 0, 0, 0
    worker = None
    start = time.monotonic()
    try:
        while query is not None and len(records) < args.num_scans:
            scan = len(records) + 1
            if worker is None:
                runtime = out / ("runtime" if mode == "stateful" else f"runtime_scan_{scan:03d}")
                worker = worker_factory(args, runtime)
            prompt = query.prompt + "\n\nReturn the final answer as one ScanReport JSON object. Schema:\n" + json.dumps(query.response_schema.model_json_schema())
            answer = worker.ask(prompt, seed_for(args.seed, query.instance_id), args.timeout)
            try:
                report = parse_report(answer["response"], query.response_schema)
                if any(not math.isfinite(v) for row in report.transmitters
                       for v in (row.center_freq, row.bandwidth, row.estimated_power)):
                    raise ValueError("Non-finite report")
                error = None
            except ValueError as exc:
                report, error = None, str(exc)
            response = (Response(action=report) if report is not None else
                        Response(action=query.response_schema(transmitters=[]), metadata={"latency_timeout": True}))
            step = task.step(response)
            # Score stays in the evaluator; no custom reward or reflection prompt.
            record = {"scan": scan, "instance_id": query.instance_id,
                      "reward": float(step.instance_outcome.reward), "parse_error": error,
                      "report": report.model_dump() if report is not None else None, **answer}
            append(out / "responses.jsonl", record)
            records.append(record)
            calls += len(answer["usage"])
            tokens += sum(x["output_tokens"] for x in answer["usage"])
            tools += sum(x["tool"] not in ("no_tool", "bad_json") for x in answer["tools"])
            memory_calls += sum(x["tool"] in ("start_long_term_update", "update_working_checkpoint")
                                for x in answer["tools"])
            metrics = {"mode": mode, "completed": len(records), "num_scans": args.num_scans,
                       "mean_score": sum(r["reward"] for r in records) / len(records),
                       "invalid_reports": sum(r["parse_error"] is not None for r in records),
                       "reward_calls": len(records), "model_calls": calls, "generation_tokens": tokens,
                       "tool_calls": tools, "memory_tool_calls": memory_calls,
                       "reward_supplied_to_agent": False, "elapsed_seconds": time.monotonic() - start}
            metrics["input_transport"] = "inline_official_prompt"
            (out / "progress.json").write_text(json.dumps(metrics, indent=2))
            print(f"{mode} scan={scan}/{args.num_scans} score={record['reward']:.4f} "
                  f"mean={metrics['mean_score']:.4f} calls={calls} tools={tools}", flush=True)
            if mode == "stateless":
                worker.close()
                worker = None
            if step.done:
                break
            query = step.next_query
    finally:
        if worker is not None:
            worker.close()
    metrics["score"] = task.evaluate().score
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=ROOT / "current_work/delta-Mem/model/Qwen3-4B-Instruct-2507")
    parser.add_argument("--ga-dir", type=Path, default=ROOT / "current_work/GenericAgent")
    parser.add_argument("--data-path", type=Path, default=BENCH / "data/blind_spectrum_monitoring/mixed_grid_lifecycle.jsonl")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=["stateful", "stateless", "both"], default="both")
    parser.add_argument("--num-scans", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=1536)
    parser.add_argument("--timeout", type=int, default=1800)
    args = parser.parse_args()
    if min(args.num_scans, args.max_new_tokens, args.timeout) < 1:
        parser.error("scan count, token limit and timeout must be positive")
    args.model, args.ga_dir = args.model.resolve(), args.ga_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "config.json").write_text(json.dumps(vars(args), default=str, indent=2))
    result = {}
    for mode in (["stateless", "stateful"] if args.mode == "both" else [args.mode]):
        result[mode] = run_mode(args, mode)
    if len(result) == 2:
        result["stateful_minus_stateless"] = result["stateful"]["mean_score"] - result["stateless"]["mean_score"]
    (args.output_dir / "comparison.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
