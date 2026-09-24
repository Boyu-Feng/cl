"""Run matched RAMP experiments on local GPUs, preserving every run and budget."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
VARIANTS = {
    "sample_frozen": ("frozen", "none", 1, "historical", "report"),
    "sample_ramp": ("ramp", "none", 1, "historical", "report"),
    "group_frozen": ("frozen", "none", 3, "group", "report"),
    "group_ramp": ("ramp", "none", 3, "group", "report"),
    "summary_frozen": ("frozen", "summary", 1, "historical", "report"),
    "summary_ramp": ("ramp", "summary", 1, "historical", "report"),
    "summary_group_frozen": ("frozen", "summary", 3, "group", "report"),
    "summary_group_ramp": ("ramp", "summary", 3, "group", "report"),
    "selection_frozen": ("frozen", "summary", 1, "historical", "selection"),
    "selection_ramp": ("ramp", "summary", 1, "historical", "selection"),
    "selection_group_frozen": ("frozen", "summary", 3, "group", "selection"),
    "selection_group_ramp": ("ramp", "summary", 3, "group", "selection"),
    "mutation_frozen": ("frozen", "summary", 3, "group", "selection"),
    "mutation_ramp": ("ramp", "summary", 3, "group", "selection"),
    "feedback_frozen": ("frozen", "summary", 1, "historical", "report"),
    "feedback_ramp": ("ramp", "summary", 1, "historical", "report"),
    "feedback_mutation_frozen": ("frozen", "summary", 3, "group", "selection"),
    "feedback_mutation_ramp": ("ramp", "summary", 3, "group", "selection"),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--num-scans", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-backtracks", type=int, default=0)
    parser.add_argument("--variants", default=",".join(name for name in VARIANTS if not name.startswith(("selection", "mutation", "feedback"))))
    parser.add_argument("--phase", default="pilot")
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--model", default=str(ROOT / "current_work/delta-Mem/model/Qwen3-4B-Instruct-2507"))
    args = parser.parse_args()
    names = args.variants.split(",")
    gpus = args.gpus.split(",")
    if len(set(names)) != len(names) or any(n not in VARIANTS for n in names):
        parser.error("Specify distinct known variants: " + ",".join(VARIANTS))
    if args.num_scans < 1 or len(set(gpus)) != len(gpus) or any(not gpu.isdigit() for gpu in gpus):
        parser.error("num-scans must be positive and GPUs must be distinct numeric indices")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out = args.run_root or ROOT / f"ttcl/results/ramp_experiments_{args.phase}_{timestamp}"
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    snapshots = out / "source_snapshot"
    snapshots.mkdir()
    hashes = {}
    for filename in ("run_reward_benchmark.py", "reward_memory.py", "spectrum_memory.py", "spectrum_actions.py", "feedback_memory.py", "run_ramp_experiments.py", "analyze_ramp_experiments.py", "../common/bsm.py"):
        path = Path(__file__).resolve().parent / filename
        if path.exists():
            shutil.copy2(path, snapshots / Path(filename).name)
            hashes[filename] = hashlib.sha256(path.read_bytes()).hexdigest()
    plan = {"phase": args.phase, "num_scans": args.num_scans, "seed": args.seed,
            "variants": names, "gpus": gpus, "runs": {}, "created_utc": timestamp,
            "source_sha256": hashes,
            "interpretation": "Candidate 0 is the official answer; extra candidates are diagnostic feedback. Summary variants use external public-observation memory."}
    pending = list(names)
    active = {}
    leases = {}
    failures = []
    last_progress = 0.0
    print(f"Experiment directory: {out}", flush=True)
    def handle_termination(signum, frame):
        raise SystemExit(f"Interrupted by signal {signum}")
    signal.signal(signal.SIGTERM, handle_termination)
    try:
        while pending or active:
            for gpu in gpus:
                if gpu in active or not pending:
                    continue
                lease = open(f"/tmp/ramp_experiment_gpu_{os.getuid()}_{gpu}.lock", "a+")
                try:
                    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    lease.close()
                    continue
                leases[gpu] = lease
                lease.seek(0)
                lease.truncate()
                lease.write(json.dumps({"supervisor_pid": os.getpid(), "run_root": str(out)}))
                lease.flush()
                name = pending.pop(0)
                for filename in ("run_reward_benchmark.py", "reward_memory.py", "spectrum_memory.py", "spectrum_actions.py", "feedback_memory.py", "../common/bsm.py"):
                    if hashlib.sha256((Path(__file__).resolve().parent / filename).read_bytes()).hexdigest() != hashes[filename]:
                        raise RuntimeError(f"Experiment code changed during suite: {filename}; use a new suite directory")
                method, memory, candidates, advantage, action = VARIANTS[name]
                command = [sys.executable, str(ROOT / "ttcl/ramp/run_reward_benchmark.py"),
                           "--model", args.model, "--output-dir", str(out / name),
                           "--method", method, "--memory-mode", memory,
                           "--action-mode", action,
                           "--candidates-per-scan", str(candidates), "--advantage-mode", advantage,
                           "--do-sample", "--temperature", "0.7", "--top-p", "0.9", "--top-k", "0",
                           "--num-scans", str(args.num_scans), "--seed", str(args.seed),
                           "--generation-seed", str(args.seed), "--max-new-tokens", "1536",
                           "--max-input-tokens", "8192", "--max-seq-length", "12288",
                           "--update-every", "4", "--epochs", "1", "--learning-rate", "2e-5",
                           "--min-pair-gap", "0.01",
                           "--max-backtracks", str(args.max_backtracks),
                           "--experiment-phase", args.phase]
                if "mutation" in name:
                    command += ["--proposal-mode", "mutate"]
                if name.startswith("feedback"):
                    command += ["--feedback-memory", "summary"]
                env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu, "TOKENIZERS_PARALLELISM": "false",
                       "PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"}
                log = (out / f"{name}.log").open("w")
                proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                active[gpu] = (name, proc, log)
                plan["runs"][name] = {"gpu": gpu, "command": command, "status": "running", "pid": proc.pid}
                print(f"Started {name} on GPU {gpu}", flush=True)
            for gpu, (name, proc, log) in list(active.items()):
                code = proc.poll()
                if code is None:
                    continue
                log.close()
                plan["runs"][name].update(status="complete" if code == 0 else "failed", exit_code=code)
                if code:
                    failures.append(name)
                print(f"Finished {name}: exit={code}", flush=True)
                del active[gpu]
                leases.pop(gpu).close()
            (out / "experiment_plan.json").write_text(json.dumps(plan, indent=2))
            if time.monotonic() - last_progress > 45:
                progress = {}
                for name, _, _ in active.values():
                    p = out / name / "progress.json"
                    if p.exists():
                        try:
                            d = json.loads(p.read_text())
                            progress[name] = {k: d.get(k) for k in ("completed", "total", "mean_score", "num_updates")}
                        except json.JSONDecodeError:
                            pass
                if progress:
                    print(json.dumps(progress), flush=True)
                last_progress = time.monotonic()
            if pending or active:
                time.sleep(2)
    finally:
        for name, proc, log in active.values():
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            log.close()
            plan["runs"][name].update(status="interrupted", exit_code=proc.returncode)
        for lease in leases.values():
            lease.close()
        (out / "experiment_plan.json").write_text(json.dumps(plan, indent=2))
    analyzer = ROOT / "ttcl/ramp/analyze_ramp_experiments.py"
    if analyzer.exists():
        subprocess.run([sys.executable, str(analyzer), str(out)], check=True, cwd=ROOT)
    if failures:
        raise SystemExit("Failed variants: " + ", ".join(failures))


if __name__ == "__main__":
    main()
