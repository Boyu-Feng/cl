"""Detached, snapshotted queue for structured-memory comparisons."""

from __future__ import annotations

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
READY = ["database_exploration", "cohort_studies", "exploitable_poker"]
PENDING = ["codebase_adaptation", "sales_prediction"]


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temporary.replace(path)


def read_json(path):
    return json.loads(path.read_text()) if path.exists() else None


def report(root, state):
    lines = [
        "# 多场景 Python 结构化经验实验",
        "",
        f"更新时间：{datetime.now(timezone.utc).isoformat()}",
        "",
        "| 任务 | 状态 | 独立组完成 | 经验组完成 | 独立 reward | 经验 reward | 配对差值 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for task, job in state["jobs"].items():
        values = []
        for mode in ("independent", "structured"):
            directory = root / task / mode
            values.append(
                read_json(directory / "metrics.json")
                or read_json(directory / "progress.json")
                or {}
            )
        baseline, memory = values
        final = read_json(root / task / "results.json") or {}
        delta = final.get("comparison", {}).get("structured_minus_independent")

        def fmt(value):
            return "—" if value is None else f"{value:.6f}"

        lines.append(
            f"| {task} | {job['status']} | {baseline.get('completed_instances', 0)} | "
            f"{memory.get('completed_instances', 0)} | {fmt(baseline.get('mean_score'))} | "
            f"{fmt(memory.get('mean_score'))} | {fmt(delta)} |"
        )
    lines += [
        "",
        "进行中的均分只是已完成部分，不能据此比较两组。仅当两组相同样本全部完成后报告配对差值。",
        "",
        "## 经验内容",
        "",
        "- 数据库：公开表结构、字段/值样例、已执行 SQL、错误与公开纠错。",
        "- 队列研究：各研究的量表与字段编码、样本分布、分组生存曲线、队列组成和不可观测项；保留研究范围。",
        "- 扑克：同名对手的公开动作计数、下注、回报与公开摊牌；不读取隐藏策略或手牌。",
        "- 代码：按仓库保存公开路径、命令和测试诊断。销售：公开表关联、历史销量、已到期预测误差。这两项待 Docker 权限与依赖就绪。",
        "",
        "## 对照协议",
        "",
        f"各任务 canonical 前 {state['num_instances']} 个样本，seed={state['seed']}；冻结本地 Qwen3-4B。",
        "independent 每样本新建官方任务并重置记忆；structured 连续任务间保留 Python 经验。两组每样本重建对话，保留完整题内工具交互。",
        "采样 temperature=0.7、top_p=0.9、top_k=0；按相同实例 ID 和动作轮数设相同随机种子。",
        "保留官方响应 schema、动作预算与评分器；不另加候选评分，不修改模型参数。按各任务自己的 reward 单位比较，不跨任务平均。",
        "公开题号/总数统一为 canonical 位置，任务简介两组一致。隐藏 metadata 与评分器内部状态不进入模型或经验。",
        "经验上下文上限 16000 字符，按记录裁剪并保存标记；题内历史不静默截断。总上下文上限 32768 tokens，超限标记失败。",
        "本地生成时间记录但不按墙钟超时判负；数据库 SQL 执行超时仍生效。实际调用数和 token 成本另列各任务 RESULT.md。",
        "",
        "## 解释范围",
        "",
        "这是规则提取经验的对照，不是模型自动总结或参数训练实验。单序列、单种子 pilot 不能证明跨场景泛化或统计显著提升。",
        "数据库使用 multi_group；队列研究前12覆盖 HERALD、MERIDIAN、MOSAIC；扑克前12均为同一公开对手，尚未测试切换对手。",
        "",
        "命令、源码快照/哈希、逐步输出和经验上下文分别保存在 manifest.json、source/、source_hashes.json 和各任务子目录。",
        "代码与销售按用户选择暂缓：当前账户无 Docker socket 权限，sudo 需要密码；这不是自动审批拒绝。",
    ]
    temporary = root / "RESULT.tmp"
    temporary.write_text("\n".join(lines) + "\n")
    temporary.replace(root / "RESULT.md")


def worker(root):
    manifest = read_json(root / "manifest.json")
    state = read_json(root / "status.json")
    state["status"] = "running"
    pending = list(manifest["commands"])
    active = {}
    available = list(manifest["gpus"])
    while pending or active:
        while available and pending:
            gpu, task = available.pop(0), pending.pop(0)
            command = manifest["commands"][task]
            environment = os.environ.copy()
            environment.update(manifest["environment"])
            environment["CUDA_VISIBLE_DEVICES"] = gpu
            log = (root / f"{task}.log").open("a")
            process = subprocess.Popen(
                command,
                cwd=manifest["environment"]["TTCL_BENCH"],
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            log.close()
            active[task] = (process, gpu)
            state["jobs"][task].update(
                status="running",
                pid=process.pid,
                gpu=gpu,
                started_at=datetime.now(timezone.utc).isoformat(),
            )
        for task, (process, gpu) in list(active.items()):
            code = process.poll()
            if code is None:
                continue
            state["jobs"][task].update(
                status="complete" if code == 0 else "failed",
                exit_code=code,
                finished_at=datetime.now(timezone.utc).isoformat(),
            )
            available.append(gpu)
            del active[task]
        state["updated_at"] = datetime.now(timezone.utc).isoformat()
        write_json(root / "status.json", state)
        report(root, state)
        if pending or active:
            time.sleep(10)
    state["status"] = "runnable_jobs_finished"
    write_json(root / "status.json", state)
    report(root, state)


def launch(args):
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    source = root / "source"
    package = source / "ttcl"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    for name in ("common", "structured_memory"):
        shutil.copytree(
            ROOT / "ttcl" / name,
            package / name,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
    bench = ROOT / "current_work/continual-learning-bench"
    dependencies = ROOT / "ttcl/.runtime/structured_memory_deps"
    environment = {
        "PYTHONPATH": os.pathsep.join(map(str, [dependencies, source, bench])),
        "TTCL_BENCH": str(bench),
        "PYTHONUNBUFFERED": "1",
        "TOKENIZERS_PARALLELISM": "false",
    }
    commands = {
        task: [
            args.python,
            str(package / "structured_memory/run_benchmark.py"),
            "--task",
            task,
            "--output-dir",
            str(root / task),
            "--num-instances",
            str(args.num_instances),
            "--seed",
            str(args.seed),
            "--model",
            str(ROOT / "current_work/delta-Mem/model/Qwen3-4B-Instruct-2507"),
        ]
        for task in args.tasks
    }
    hashes = {
        str(path.relative_to(source)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in source.rglob("*.py")
    }
    hashes.update(
        {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (bench / "src").rglob("*.py")
        }
    )
    write_json(root / "source_hashes.json", hashes)
    manifest = {
        "workspace": str(ROOT),
        "gpus": args.gpus.split(","),
        "commands": commands,
        "environment": environment,
        "python": args.python,
    }
    write_json(root / "manifest.json", manifest)
    state = {
        "status": "queued",
        "num_instances": args.num_instances,
        "seed": args.seed,
        "jobs": {task: {"status": "queued"} for task in args.tasks},
    }
    for task in PENDING:
        if task not in args.tasks:
            state["jobs"][task] = {"status": "pending_docker_permission"}
    write_json(root / "status.json", state)
    report(root, state)
    with (root / "queue.log").open("a") as log:
        process = subprocess.Popen(
            [
                args.python,
                str(package / "structured_memory/launch.py"),
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
    print(
        json.dumps({"root": str(root), "queue_pid": process.pid, "tasks": args.tasks})
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--tasks", nargs="+", choices=READY + PENDING, default=READY)
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--num-instances", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--python", default=PYTHON)
    args = parser.parse_args()
    if args.worker:
        worker(args.root)
    else:
        launch(args)


if __name__ == "__main__":
    main()
