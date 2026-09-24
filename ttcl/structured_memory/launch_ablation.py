"""Snapshot and supervise the fixed experience-content pilot on local GPUs."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
PYTHON = "/home/fengboyu/miniconda3/envs/seal_env/bin/python"
TASKS = ["database_exploration", "cohort_studies", "exploitable_poker"]


def write(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temporary.replace(path)


def read(path):
    return json.loads(path.read_text()) if path.exists() else {}


def report(root, state):
    rows = [
        "# 经验内容开发消融",
        "",
        f"更新：{datetime.now(timezone.utc).isoformat()}",
        "",
        "| 任务 | 组别 | 状态 | 完成 | 平均 reward | 相对无经验 |",
        "|---|---|---|---:|---:|---:|",
    ]
    for task, job in state["jobs"].items():
        results = read(root / task / "results.json")
        status = read(root / task / "status.json")
        for variant in job["variants"]:
            mode = "independent" if variant == "independent" else "structured"
            row = results.get(variant) or read(
                root / task / variant / mode / "progress.json"
            )
            current = row.get(
                "status",
                "running"
                if status.get("variant") == variant and job["status"] == "running"
                else "pending",
            )
            score = "—" if row.get("mean_score") is None else f"{row['mean_score']:.6f}"
            delta = row.get("vs_independent", {}).get("structured_minus_independent")
            difference = "—" if delta is None else f"{delta:+.6f}"
            rows.append(
                f"| {task} | {variant} | {current} | {row.get('completed_instances', 0)} | {score} | {difference} |"
            )
    rows += [
        "",
        "仅完整配对组比较；进行中的均分不代表最终成绩。每任务前12个 canonical 实例，seed=42；已查看过的开发前缀，非独立最终测试。",
        "主线：independent / legacy / focused。reward 是附加消融，仅数据库和扑克运行 focused_reward。",
        "focused 与 focused_reward 使用相同结构、检索和历史行动案例；前者不显示官方标量，后者只增加已完成实例标量。公开反馈中已有的正确性或盈亏仍可能提供奖励信息，因此这是显式标量附加效果。",
        "所有组：冻结 Qwen3-4B，temperature=0.7，top_p=0.9，同实例/轮次种子；完整题内历史，65536 tokens 上限。保留官方工具预算和评分；格式包装修复和至多两次格式重试同样适用于所有组，成本全部记录。",
        "focused：数据库紧凑 schema 与问题关联查询；队列研究保留字段定义/入组范围的分析步骤；扑克按我方加注条件统计公开响应，不把输赢归因于最后动作。",
        "代码与销售尚未运行：Docker socket 的 Unix 权限不允许当前账户使用。未改动系统权限。",
        "命令、源码快照和 SHA256 见 manifest.json、source/、source_hashes.json；逐步模型输入/输出、公开观测、记忆和案例在各任务/组别目录。",
    ]
    temp = root / "RESULT.tmp"
    temp.write_text("\n".join(rows) + "\n")
    temp.replace(root / "RESULT.md")


def worker(root):
    manifest, state = read(root / "manifest.json"), read(root / "status.json")
    pending = list(state["jobs"])
    available, active = list(manifest["gpus"]), {}
    while pending or active:
        while pending and available:
            task, gpu = pending.pop(0), available.pop(0)
            env = os.environ.copy()
            env.update(manifest["environment"], CUDA_VISIBLE_DEVICES=gpu)
            with (root / f"{task}.log").open("a") as log:
                process = subprocess.Popen(
                    manifest["commands"][task],
                    cwd=manifest["environment"]["TTCL_BENCH"],
                    env=env,
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            active[task] = process, gpu
            state["jobs"][task].update(status="running", pid=process.pid, gpu=gpu)
        for task, (process, gpu) in list(active.items()):
            if process.poll() is not None:
                state["jobs"][task].update(
                    status="finished" if process.returncode == 0 else "failed",
                    exit_code=process.returncode,
                )
                available.append(gpu)
                del active[task]
        state["status"] = "running" if pending or active else "finished"
        write(root / "status.json", state)
        report(root, state)
        if pending or active:
            time.sleep(10)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--gpus", default="0,1,3")
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=TASKS)
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=[
            "independent",
            "legacy",
            "focused",
            "focused_reward",
            "focused_procedure",
        ],
    )
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--finalize-followup", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.finalize_followup:
        followup = args.finalize_followup.resolve()
        if (
            not (root / "manifest.json").exists()
            or not (followup / "manifest.json").exists()
        ):
            parser.error("Both existing experiment roots must contain manifest.json")
        with (root / "finalize.log").open("a") as log:
            process = subprocess.Popen(
                [
                    PYTHON,
                    "-m",
                    "ttcl.structured_memory.finalize_ablation",
                    str(root),
                    str(followup),
                    "--watch",
                ],
                cwd=ROOT,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        (root / "finalize.pid").write_text(str(process.pid) + "\n")
        print(json.dumps({"report_finalizer_pid": process.pid}))
        return
    if args.worker:
        worker(root)
        return
    if (
        args.variants
        and "focused_procedure" in args.variants
        and args.tasks != ["database_exploration"]
    ):
        parser.error("focused_procedure requires --tasks database_exploration")
    root.mkdir(parents=True, exist_ok=False)
    source = root / "source"
    package = source / "ttcl"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    for name in ["common", "structured_memory"]:
        shutil.copytree(
            ROOT / "ttcl" / name,
            package / name,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
    bench = ROOT / "current_work/continual-learning-bench"
    environment = {
        "TTCL_BENCH": str(bench),
        "PYTHONPATH": os.pathsep.join(
            map(str, [ROOT / "ttcl/.runtime/structured_memory_deps", source, bench])
        ),
        "PYTHONUNBUFFERED": "1",
        "TOKENIZERS_PARALLELISM": "false",
    }
    commands, jobs = {}, {}
    for task in args.tasks:
        variants = args.variants or ["independent", "legacy", "focused"] + (
            ["focused_reward"] if task != "cohort_studies" else []
        )
        commands[task] = [
            PYTHON,
            "-m",
            "ttcl.structured_memory.ablation",
            "--task",
            task,
            "--output-dir",
            str(root / task),
            "--model",
            str(ROOT / "current_work/delta-Mem/model/Qwen3-4B-Instruct-2507"),
            "--variants",
            *variants,
        ]
        jobs[task] = {"status": "pending", "variants": variants}
    manifest = {
        "commands": commands,
        "environment": environment,
        "gpus": args.gpus.split(","),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    write(root / "manifest.json", manifest)
    hashes = {
        str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in source.rglob("*.py")
    }
    hashes.update(
        {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (bench / "src").rglob("*.py")
        }
    )
    write(root / "source_hashes.json", hashes)
    state = {"status": "queued", "jobs": jobs}
    write(root / "status.json", state)
    report(root, state)
    with (root / "queue.log").open("a") as log:
        process = subprocess.Popen(
            [
                PYTHON,
                str(package / "structured_memory/launch_ablation.py"),
                "--worker",
                "--root",
                str(root),
            ],
            cwd=ROOT,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    (root / "queue.pid").write_text(str(process.pid) + "\n")
    print(json.dumps({"root": str(root), "pid": process.pid}))


if __name__ == "__main__":
    main()
