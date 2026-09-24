"""Freeze evidence, preregister the protocol, and supervise local experience tests."""

from __future__ import annotations

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
from ttcl.structured_memory.verified_experience import (  # noqa: E402
    ARMS,
    build_bank,
    core_context,
    digest,
    matched_context,
    read,
    render_report,
    write,
)

PYTHON = "/home/fengboyu/miniconda3/envs/seal_env/bin/python"


def prepare(root, source_root, gpus):
    from transformers import AutoTokenizer

    if len(gpus) != 2 or len(set(gpus)) != 2:
        raise ValueError("Provide two distinct GPUs, one per task")
    root.mkdir(parents=True, exist_ok=False)
    bank = build_bank(source_root)
    write(root / "bank.json", bank)
    plan = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_root": str(source_root),
        "source_indices": list(range(6)),
        "screen_seeds": [42, 43],
        "confirm_seeds": [101, 202, 303],
        "environment_seed": 42,
        "reference_context_tokens": 768,
        "tasks": {
            "database_exploration": {
                "canonical_total": 30,
                "screen_indices": list(range(6, 12)),
                "confirm_indices": list(range(12, 24)),
                "untouched_reserve": list(range(24, 30)),
            },
            "cohort_studies": {
                "canonical_total": 20,
                "screen_indices": list(range(6, 12)),
                "confirm_indices": list(range(12, 20)),
                "untouched_reserve": [],
            },
        },
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
        "screen_rule": "Against none, generic and irrelevant: positive mean in each of two decoding seeds, >=2 wins and wins>=losses on seed-averaged instances, positive mean after deleting any single instance. Select greatest mean gain vs none among eligible A/B/combined, using fixed arm order for ties. Missing pairs disqualify.",
        "confirm_rule": "Against each control, seed-averaged instance bootstrap 95% lower bound >0, each decoding seed mean >0, leave-one-instance-out mean >0; additionally both held-out cohort-study means >0. No rewriting or reselection after confirmation.",
        "causal_scope": "Effect of presenting this curated experience text to this frozen reader, with matched-length optional context; not proof that individual historical actions caused their episode reward.",
        "reward_role": "Offline screening and confirmation only. No scalar, evaluator metadata or future feedback enters memory. Compared with all prespecified unselected candidates on development data; no claim of a separate causal reward-learning effect.",
        "limitations": [
            "Development instances have been inspected previously.",
            "Manually curated factual evidence does not prove procedural recommendations improve reward.",
            "Decoding seeds are replicates on fixed data, not independent environments.",
            "One database and two held-out cohort study families limit generalization.",
            "Neutral shared padding controls approximate token length, not all semantic differences.",
            "No online memory growth in this phase.",
        ],
        "maximum_trials": 144 + 144 + 96,
    }
    for spec in plan["tasks"].values():
        assert not set(plan["source_indices"]) & set(spec["screen_indices"])
        assert not (set(plan["source_indices"]) | set(spec["screen_indices"])) & set(
            spec["confirm_indices"]
        )
        assert max(spec["confirm_indices"]) < spec["canonical_total"]
    tokenizer = AutoTokenizer.from_pretrained(
        plan["model"]["model"], local_files_only=True
    )
    contexts = {}
    for task in plan["tasks"]:
        contexts[task] = {}
        for arm in ARMS:
            core = core_context(bank, task, arm)
            text = matched_context(core, tokenizer, plan["reference_context_tokens"])
            contexts[task][arm] = {
                "text": text,
                "sha256": digest(text),
                "core_tokens": len(tokenizer.encode(core, add_special_tokens=False)),
                "total_tokens": len(tokenizer.encode(text, add_special_tokens=False)),
            }
    write(root / "contexts.json", contexts)
    write(root / "plan.json", plan)
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
    hashes = {
        str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source.rglob("*.py")
    }
    hashes.update(
        {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (bench / "src").rglob("*.py")
        }
    )
    hashes.update(
        {
            str(root / name): hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in ("bank.json", "contexts.json", "plan.json")
        }
    )
    write(root / "source_hashes.json", hashes)
    environment = {
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
            "ttcl.structured_memory.verified_experience",
            "--root",
            str(root),
            "--task",
            task,
        ]
        for task in plan["tasks"]
    }
    write(
        root / "manifest.json",
        {
            "commands": commands,
            "gpus": gpus,
            "environment": environment,
            "source": str(source),
        },
    )
    write(
        root / "status.json",
        {
            "status": "prepared",
            "jobs": {task: {"status": "pending"} for task in commands},
        },
    )
    protocol = """# 核验经验实验协议（运行前冻结）

经验定义：从过去公开交互中得到、有证据来源和适用边界、可指导未来判断或行动的知识单元。观察是否真实与经验是否提高 reward 分别检验。

本次是人工核验候选的使用实验。bank.json 保存精确公开工具证据、原始行号及 SHA256；不使用生成思考、终局答案或隐藏标签作为事实。建议步骤是从事实提出的待验证假设。

- 数据来源：两个任务 canonical 0–5（第1–6个）的已有公开轨迹。
- 开发筛选：canonical 6–11，生成种子42/43；每个任务6组×6实例×2种子=72次。
- 六组：无经验、通用提示、无关内容、经验A、经验B、A+B。
- 所有非空上下文约768 tokens，用相同无关背景补齐；不截断经验。无经验组不补齐，以保留真实无经验基线。
- 根据开发 reward 差值按 plan.json 的固定门槛选择一个候选。没有通过者就停止该任务，不以负结果继续调参后复用确认集。
- 确认：数据库 canonical 12–23（第13–24题）；cohort canonical 12–19（FORGE/CADENCE）。三个新生成种子101/202/303。选中候选与三个对照一起运行；同一任务种子、实例和官方工具预算。
- 固定数据库/患者数据：多个生成种子仅测模型采样波动，不代表新数据库或患者样本。
- 一题一份新任务状态，一题内完整历史。冻结经验每题第一步可见；各题之间不追加信息，确认中不修改经验。
- 模型固定Qwen3-4B-Instruct-2507，无参数更新。格式包装修复和最多两次格式重试对所有组相同；所有调用、tokens和失败保留。
- 主指标为官方reward；次指标为成功率（若任务提供）、工具调用、错误标识符、重复分组调用和tokens。次指标不用于事后更换胜负标准。
- 置信区间按实例重采样，先合并同一实例的生成种子。cohort额外分别报告两个留出研究；两个研究不足以宣称跨研究总体稳定。
- 后台自动执行筛选、必要的确认和报告。已经运行的旧实验不受影响。

reward 的作用是筛选值得复用的经验，不把整题reward当作最后动作的价值。当前实验不单独证明自动提取或在线reward学习能力。

参考：CBR https://www.iiia.csic.es/~enric/papers/Aamodt_1994_Case.pdf；ExpeL https://arxiv.org/abs/2308.10144；ReMe https://arxiv.org/abs/2512.10696。
"""
    (root / "PROTOCOL.md").write_text(protocol)
    lines = ["# 冻结候选经验", "", bank["authorship"], ""]
    for task in plan["tasks"]:
        for key in ("candidate_a", "candidate_b"):
            candidate = bank[task][key]
            lines += [
                f"## {task} / {key}: {candidate['name']}",
                "",
                candidate["scope"],
                "",
                candidate["content"],
                "",
                "失效条件：" + candidate["falsifier"],
                "",
                "预期机制（未验证）：" + candidate["mechanism"],
                "",
            ]
            for evidence in candidate["evidence"]:
                lines.append(
                    f"- 证据：{evidence['instance_id']} turn {evidence['turn']}；{evidence['observation_file']}:{evidence['observation_line']}"
                )
            lines.append("")
    (root / "CANDIDATES.md").write_text("\n".join(lines))
    render_report(root, plan)


def audit(root):
    errors = []
    for path, expected in read(root / "source_hashes.json").items():
        if (
            not Path(path).exists()
            or hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected
        ):
            errors.append(f"Frozen source/artifact changed: {path}")
    bank = read(root / "bank.json")
    for task in read(root / "plan.json")["tasks"]:
        for key in ("candidate_a", "candidate_b"):
            for evidence in bank[task][key]["evidence"]:
                for kind in ("observation", "response"):
                    line = (
                        Path(evidence[f"{kind}_file"])
                        .read_text()
                        .splitlines()[evidence[f"{kind}_line"] - 1]
                    )
                    if digest(line) != evidence[f"{kind}_sha256"]:
                        errors.append(f"Evidence changed: {task}/{key}/{kind}")
    result = {
        "errors": errors,
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "scope": "Frozen source, bank, contexts, protocol and source evidence; per-trial token/context checks are performed by workers.",
    }
    write(root / "audit.json", result)
    return errors


def supervise(root):
    manifest, plan = read(root / "manifest.json"), read(root / "plan.json")
    state = read(root / "status.json")
    if audit(root):
        raise ValueError("Preflight artifact audit failed")
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
        write(root / "status.json", state)
        render_report(root, plan)
        if active:
            time.sleep(10)
    audit(root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=ROOT / "ttcl/results/structured_memory/experience_ablation_20260920",
    )
    parser.add_argument("--gpus", default="0,3")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--start", action="store_true")
    mode.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    if args.worker:
        supervise(root)
        return
    if not args.start:
        prepare(root, args.source_root.resolve(), args.gpus.split(","))
    if args.prepare:
        print(
            json.dumps(
                {"root": str(root), "status": "prepared", "audit_errors": audit(root)}
            )
        )
        return
    if read(root / "status.json")["status"] != "prepared":
        raise ValueError("Experiment already started; refusing duplicate workers")
    if audit(root):
        raise ValueError("Preflight artifact audit failed")
    manifest = read(root / "manifest.json")
    # GPU access requires execution outside the filesystem/process sandbox.
    query = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    memory = {}
    for line in query.splitlines():
        index, used = line.split(",")
        memory[index.strip()] = int(used.strip())
    if any(memory[gpu] > 1000 for gpu in manifest["gpus"]):
        raise RuntimeError(
            "A selected GPU is occupied; choose free GPUs before preparing"
        )
    env = os.environ.copy()
    env.update(manifest["environment"])
    with (root / "queue.log").open("a") as log:
        process = subprocess.Popen(
            [
                PYTHON,
                str(
                    Path(manifest["source"])
                    / "ttcl/structured_memory/launch_verified.py"
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
