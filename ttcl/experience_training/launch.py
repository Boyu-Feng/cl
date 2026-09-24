"""Snapshot and supervise the complete utility-filtered writer training pilot."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
PYTHON = "/home/fengboyu/miniconda3/envs/seal_env/bin/python"


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    temp.replace(path)


def prepare(root, gpus):
    if len(gpus) != 2 or len(set(gpus)) != 2:
        raise ValueError("Specify two distinct available GPUs")
    source = ROOT / "ttcl/results/structured_memory/llm_online_bank_20260920_v2"
    prior = json.loads((source / "plan.json").read_text())
    if json.loads((source / "status.json").read_text())["status"] != "finished":
        raise ValueError("Source experiment must be finished")
    root.mkdir(parents=True, exist_ok=False)
    plan = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "gpus": gpus,
        "tasks": ["database_exploration", "cohort_studies"],
        "environment_seed": 42,
        "source_indices": list(range(11)),
        "test_indices": list(range(12, 20)),
        "candidate_temperatures": [0.0, 0.9],
        "collection_repeats": [101, 202],
        "evaluation_repeats": [303, 404],
        "training_seed": 812,
        "minimum_labels": 4,
        "minimum_positive_labels": 2,
        "writer_output_tokens": 4096,
        "bank": prior["bank"],
        "model": prior["model"],
        "training": {
            "rank": 8,
            "epochs": 2,
            "accumulation": 2,
            "learning_rate": 2e-5,
            "max_length": 32768,
        },
        "source_experiment": str(source),
    }
    package = root / "source/ttcl"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    for name in [
        "common",
        "llm_memory",
        "structured_memory",
        "memory_writer",
        "experience_training",
    ]:
        shutil.copytree(
            ROOT / "ttcl" / name,
            package / name,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
    for task in plan["tasks"]:
        for index in plan["source_indices"]:
            origin = source / task / "online_bank" / f"episode_{index + 1:03d}"
            dest = root / "data" / task / origin.name
            dest.mkdir(parents=True)
            shutil.copy2(origin / "trajectory.json", dest / "trajectory.json")
            update = json.loads((origin / "bank_update.json").read_text())
            write(dest / "bank_before.json", update["bank_before"])
    shutil.copy2(package / "experience_training/PROTOCOL.md", root / "PROTOCOL.md")
    shutil.copy2(
        package / "llm_memory/trajectory_extraction_prompt.md",
        root / "EXTRACTION_PROMPT.md",
    )
    write(root / "plan.json", plan)
    bench = ROOT / "current_work/continual-learning-bench"
    paths = [
        p
        for subtree in [root / "source", root / "data"]
        for p in subtree.rglob("*")
        if p.is_file() and p.suffix in {".py", ".md", ".json"}
    ]
    paths += list((bench / "src").rglob("*.py")) + [
        root / "plan.json",
        root / "EXTRACTION_PROMPT.md",
    ]
    write(
        root / "hashes.json",
        {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
    )
    environment = {
        "TTCL_BENCH": str(bench),
        "PYTHONUNBUFFERED": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "PYTHONPATH": os.pathsep.join(
            map(
                str,
                [ROOT / "ttcl/.runtime/structured_memory_deps", root / "source", bench],
            )
        ),
    }
    write(
        root / "manifest.json",
        {"environment": environment, "python": PYTHON, "workspace": str(ROOT)},
    )
    write(root / "status.json", {"phase": "prepared", "jobs": []})
    return environment


def supervise(root):
    from ttcl.experience_training.run import ARMS, assemble, report

    plan = json.loads((root / "plan.json").read_text())
    jobs, active = [], {}
    stopping = False

    def stop(signum, frame):
        nonlocal stopping
        stopping = True
        for process in active.values():
            process.terminate()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    def status(phase):
        write(
            root / "status.json",
            {
                "phase": phase,
                "jobs": jobs,
                "supervisor_pid": os.getpid(),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        report(root)

    def execute(phase, specifications):
        pending = []
        for name, arguments in specifications:
            job = {
                "name": name,
                "status": "queued",
                "command": [
                    PYTHON,
                    "-m",
                    "ttcl.experience_training.run",
                    phase,
                    "--root",
                    str(root),
                    *arguments,
                ],
            }
            jobs.append(job)
            pending.append(job)
        while pending or active:
            if stopping:
                raise InterruptedError("Stopped by signal")
            occupied = {j["gpu"] for j in jobs if j["status"] == "running"}
            for gpu in plan["gpus"]:
                if gpu in occupied or not pending:
                    continue
                job = pending.pop(0)
                log = root / "logs" / (job["name"] + ".log")
                log.parent.mkdir(exist_ok=True)
                with log.open("w") as stream:
                    process = subprocess.Popen(
                        job["command"],
                        stdout=stream,
                        stderr=subprocess.STDOUT,
                        cwd=root / "source",
                        env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)},
                    )
                job.update(status="running", gpu=gpu, pid=process.pid)
                active[job["name"]] = process
            for job in jobs:
                if job["status"] != "running":
                    continue
                process = active[job["name"]]
                code = process.poll()
                if code is not None:
                    job.update(
                        status="complete" if code == 0 else "failed", exit_code=code
                    )
                    del active[job["name"]]
            status(phase)
            if pending or active:
                time.sleep(10)
        if any(j["status"] == "failed" for j in jobs):
            raise RuntimeError("A stage failed; dependent stages were not started")

    try:
        execute("collect", [(f"collect_{t}", ["--task", t]) for t in plan["tasks"]])
        status("assemble")
        if not assemble(root):
            status("insufficient_training_signal")
            return
        execute(
            "train",
            [(f"train_{arm}", ["--arm", arm]) for arm in ARMS if arm.endswith("_sft")],
        )
        execute(
            "evaluate",
            [
                (
                    f"eval_{task}_{arm}_{repeat}",
                    ["--task", task, "--arm", arm, "--repeat", str(repeat)],
                )
                for task in plan["tasks"]
                for repeat in plan["evaluation_repeats"]
                for arm in ARMS
            ],
        )
        errors = []
        hashes = json.loads((root / "hashes.json").read_text())
        for filename, expected in hashes.items():
            if hashlib.sha256(Path(filename).read_bytes()).hexdigest() != expected:
                errors.append(filename)
        # At each task/repeat the first query must be identical with empty banks,
        # including the actor receiving no trained adapter in any arm.
        for task in plan["tasks"]:
            for repeat in plan["evaluation_repeats"]:
                rows = [
                    json.loads(
                        (
                            root
                            / "evaluation"
                            / task
                            / arm
                            / str(repeat)
                            / f"episode_{plan['test_indices'][0] + 1:03d}"
                            / "row.json"
                        ).read_text()
                    )
                    for arm in ARMS
                ]
                if len({r["first_prompt_sha256"] for r in rows}) != 1:
                    errors.append(f"empty-bank prompt mismatch {task} {repeat}")
                if len({r["reward"] for r in rows}) != 1:
                    errors.append(f"empty-bank reward mismatch {task} {repeat}")
        write(
            root / "audit.json",
            {"errors": errors, "checked_at": datetime.now(timezone.utc).isoformat()},
        )
        status("complete" if not errors else "audit_failed")
    except Exception:
        write(root / "supervisor_failure.json", {"traceback": traceback.format_exc()})
        status("stopped" if stopping else "failed")
        raise
    finally:
        for process in active.values():
            process.terminate()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--supervise", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    if args.supervise:
        supervise(root)
        return
    env = prepare(root, args.gpus.split(","))
    command = [
        PYTHON,
        str(root / "source/ttcl/experience_training/launch.py"),
        "--root",
        str(root),
        "--supervise",
    ]
    with (root / "supervisor.log").open("w") as stream:
        process = subprocess.Popen(
            command,
            cwd=root / "source",
            env={**os.environ, **env},
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    (root / "supervisor.pid").write_text(str(process.pid) + "\n")
    print(json.dumps({"root": str(root), "pid": process.pid, "command": command}))


if __name__ == "__main__":
    main()
