"""Freeze and supervise the small model-authored trajectory-bank pilot."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "ttcl/.runtime/structured_memory_deps"))
from ttcl.structured_memory.online_bank import report, read  # noqa: E402
from ttcl.structured_memory.run_benchmark import write_json  # noqa: E402

PYTHON = "/home/fengboyu/miniconda3/envs/seal_env/bin/python"


def audit_sources(root):
    errors = []
    for filename, expected in read(root / "source_hashes.json").items():
        if hashlib.sha256(Path(filename).read_bytes()).hexdigest() != expected:
            errors.append(filename)
    write_json(
        root / "source_audit.json",
        {"errors": errors, "checked_at": datetime.now(timezone.utc).isoformat()},
    )
    return errors


def prepare(root, gpus, num_instances):
    if len(gpus) != 2 or len(set(gpus)) != 2:
        raise ValueError("Choose two distinct free GPUs")
    if not 1 <= num_instances <= 12:
        raise ValueError("This development pilot is limited to the first 12 instances")
    root.mkdir(parents=True, exist_ok=False)
    plan = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "tasks": ["database_exploration", "cohort_studies"],
        "num_instances": num_instances,
        "seed": 42,
        "arms": ["independent", "online_bank"],
        "bank": {"max_entries": 8, "max_chars": 9000, "max_tokens": 2048},
        "writer_output_tokens": 4096,
        "writer_temperature": 0.0,
        "model": {
            "model": str(ROOT / "current_work/delta-Mem/model/Qwen3-4B-Instruct-2507"),
            "device": "cuda:0",
            "dtype": "bfloat16",
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 0,
            "max_new_tokens": 4096,
            "context_limit": 65536,
            "memory_chars": 16000,
            "max_turns_per_instance": 64,
            "action_retries": 2,
            "normalize_action": True,
        },
        "protocol": {
            "experience_source": "The online arm's own chronological public queries, executed actions, tool feedback, public response schemas and terminal official scalar only.",
            "update_timing": "After every attempted sample, before the next sample. No next-query access. Incomplete attempts carry reward=null and an explicit local error.",
            "writer": "Same frozen model; KEEP or atomic ADD/REVISE/REMOVE. No domain-specific prefilled facts or automatic semantic selection.",
            "reader": "Entire compact bank at the next sample's first turn; within-sample public history retained; no online bank changes within a sample.",
            "control": "Fresh no-memory run on each same canonical instance, matching actor seed/settings/budgets. Writer calls are extra and separately counted.",
            "primary_metric": "Complete paired mean official reward difference; also report wins/ties/losses and paired difference excluding first empty-bank sample.",
            "validation": "JSON/entry/citation/token checks only. No semantic validator. Outer ADD/REVISE/REMOVE is normalized to UPDATE only if every supplied operation has that same type; all content is preserved and the repair logged. One writer format retry; reject atomically on continued error.",
            "failure": "Retain failed responses and missing scores, never replace missing reward with zero or claim a full comparison from survivors.",
            "feedback": "User-authorized augmented feedback: terminal official scalar supplied to writer, including for cohort where public observations omit it; no scoring metadata or hidden labels.",
            "limits": "Single-seed, previously inspected development prefix. Tests combined autonomous summarization plus reward; does not isolate reward's marginal effect or establish stable generalization.",
        },
    }
    write_json(root / "plan.json", plan)
    source = root / "source"
    package = source / "ttcl"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    for name in ("common", "structured_memory", "llm_memory"):
        shutil.copytree(
            ROOT / "ttcl" / name,
            package / name,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
    shutil.copy2(
        package / "llm_memory/trajectory_extraction_prompt.md",
        root / "EXTRACTION_PROMPT.md",
    )
    bench = ROOT / "current_work/continual-learning-bench"
    env = {
        "TTCL_BENCH": str(bench),
        "PYTHONPATH": os.pathsep.join(
            map(str, [ROOT / "ttcl/.runtime/structured_memory_deps", source, bench])
        ),
        "PYTHONUNBUFFERED": "1",
        "TOKENIZERS_PARALLELISM": "false",
    }
    commands = {
        task: [
            PYTHON,
            "-m",
            "ttcl.structured_memory.online_bank",
            "--root",
            str(root),
            "--task",
            task,
        ]
        for task in plan["tasks"]
    }
    write_json(
        root / "manifest.json",
        {"commands": commands, "gpus": gpus, "environment": env, "source": str(source)},
    )
    files = [
        *source.rglob("*.py"),
        *source.rglob("*.md"),
        *(bench / "src").rglob("*.py"),
        root / "plan.json",
        root / "EXTRACTION_PROMPT.md",
        root / "manifest.json",
    ]
    write_json(
        root / "source_hashes.json",
        {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
    )
    write_json(
        root / "status.json",
        {
            "status": "prepared",
            "jobs": {task: {"status": "pending"} for task in plan["tasks"]},
        },
    )
    report(root, plan)
    (root / "PROTOCOL.md").write_text("""# 轨迹压缩成经验：自主更新小测试

每个任务从空经验库开始，按原始顺序处理前12个实例（seed=42）。两组分别是无经验和模型经验库；两组每题重置环境与对话，后者只继承自己的经验库。全部使用冻结Qwen3-4B-Instruct-2507。

每题完整流程：读取当前银行→完成题内交互→获得最终官方reward→把所有实际操作、公开反馈、schema和reward交给同一模型→模型总结轨迹并自行决定KEEP/ADD/REVISE/REMOVE→将更新后的银行用于下一题。

程序不提供预先编写的领域经验、不读取未来任务、不读取评分器内部标签或metadata。工具已经公开的终局反馈保留。整题reward只在题目结束后提供给writer；不会当作每个动作的标签。生成思考保留在action中，prompt明确它不是已验证事实。

经验库最多8条、2048 tokens。每条含适用范围、内容、使用方式、局限与轨迹步骤引用。程序只校验格式、预算和引用是否存在，不决定内容是否正确。失败更新不修改旧库，最多一次格式重试；输入过长不会静默删掉轨迹。

每题都保存轨迹摘要和reward解释，即使KEEP。它们仅供审计，下一题只读银行。最后一题后也总结，计入成本。执行失败没有官方分数时记reward=null，可总结明确标注的不完整轨迹；不把缺失分数填0。

对照使用相同答题采样、工具预算和格式重试。经验组增加总结调用，单独计入总tokens和时间。因此这是额外总结计算带来的整体效果，不是等总计算成本比较，也未单独隔离reward效果。

主指标：完整同实例配对reward差值及胜/平/负。辅助报告去掉首次空库题后的配对差值、银行更新次数、拒绝次数和成本。只有完整组才给最终比较，负结果照常保留。单种子开发pilot不证明稳定泛化。

完整实际prompt见EXTRACTION_PROMPT.md；当前模型经验见BANKS.md；汇总见REPORT.md。逐题writer输入、原始输出、银行前后版本和决策均记录。
""")


def supervise(root):
    manifest, plan, state = (
        read(root / name) for name in ("manifest.json", "plan.json", "status.json")
    )
    if audit_sources(root):
        raise ValueError("Frozen source/plan changed")
    active = {}
    for (task, command), gpu in zip(
        manifest["commands"].items(), manifest["gpus"], strict=True
    ):
        env = os.environ.copy()
        env.update(manifest["environment"], CUDA_VISIBLE_DEVICES=gpu)
        with (root / f"{task}.log").open("a") as log:
            process = subprocess.Popen(
                command,
                cwd=manifest["source"],
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        active[task] = process
        state["jobs"][task] = {"status": "running", "pid": process.pid, "gpu": gpu}
    while active:
        for task, process in list(active.items()):
            if process.poll() is not None:
                state["jobs"][task].update(
                    status="finished" if process.returncode == 0 else "failed",
                    exit_code=process.returncode,
                )
                del active[task]
        state["status"] = (
            "running"
            if active
            else (
                "finished"
                if all(j["status"] == "finished" for j in state["jobs"].values())
                else "failed"
            )
        )
        state["updated_at"] = datetime.now(timezone.utc).isoformat()
        write_json(root / "status.json", state)
        report(root, plan)
        if active:
            time.sleep(10)
    audit_sources(root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--num-instances", type=int, default=12)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--prepare", action="store_true")
    modes.add_argument("--start", action="store_true")
    modes.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    if args.worker:
        supervise(root)
        return
    if not args.start:
        prepare(root, args.gpus.split(","), args.num_instances)
    if audit_sources(root):
        raise ValueError("Frozen source/plan changed")
    if args.prepare:
        print(json.dumps({"root": str(root), "status": "prepared"}))
        return
    if read(root / "status.json")["status"] != "prepared":
        raise ValueError("Refusing duplicate launch")
    manifest = read(root / "manifest.json")
    query = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    usage = {
        line.split(",")[0].strip(): int(line.split(",")[1])
        for line in query.splitlines()
    }
    if any(usage[gpu] > 1000 for gpu in manifest["gpus"]):
        raise RuntimeError("A selected GPU is occupied")
    env = os.environ.copy()
    env.update(manifest["environment"])
    with (root / "queue.log").open("a") as log:
        process = subprocess.Popen(
            [
                PYTHON,
                str(
                    Path(manifest["source"])
                    / "ttcl/structured_memory/launch_online_bank.py"
                ),
                "--root",
                str(root),
                "--worker",
            ],
            cwd=manifest["source"],
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    (root / "queue.pid").write_text(str(process.pid) + "\n")
    print(json.dumps({"root": str(root), "queue_pid": process.pid}))


if __name__ == "__main__":
    main()
