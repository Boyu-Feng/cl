"""Interactive hidden API-key entry; snapshot and launch a CPU-only experiment."""

import argparse
from datetime import datetime, timezone
import getpass
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import warnings

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from ttcl.common.openrouter import catalog, exact_model, write  # noqa: E402


def read_key():
    value = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if value:
        return value
    if not sys.stdin.isatty():
        raise ValueError(
            "未配置 OPENROUTER_API_KEY。请在本地终端运行入口，按隐藏提示输入；不要把 key 放在命令参数或聊天里。"
        )
    with warnings.catch_warnings():
        warnings.simplefilter("error", getpass.GetPassWarning)
        try:
            value = getpass.getpass(
                "OpenRouter API key（隐藏输入，仅本次进程使用）: "
            ).strip()
        except getpass.GetPassWarning:
            raise ValueError("当前终端无法隐藏输入；拒绝回显 API key。") from None
    if not value:
        raise ValueError("API key 为空；未启动实验。")
    return value


def prepare(args, models):
    root = args.root.resolve()
    writer = args.writer_model or args.model
    selected = list(
        dict.fromkeys(
            [args.model, writer]
            + ([args.comparison_writer_model] if args.comparison_writer_model else [])
        )
    )
    metadata = {identity: exact_model(models, identity) for identity in selected}
    if not 1 <= args.num_instances <= 12:
        raise ValueError("This matching pilot supports 1..12 instances per task")
    if (
        not math.isfinite(args.max_cost_usd)
        or args.max_cost_usd <= 0
        or args.max_requests <= 0
        or args.max_output_tokens <= 0
    ):
        raise ValueError("Budgets must be positive")
    if len(set(args.tasks)) != len(args.tasks):
        raise ValueError("Tasks must not be repeated")
    root.mkdir(parents=True, exist_ok=False)
    package = root / "source/ttcl"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    for name in ["common", "llm_memory", "structured_memory", "openrouter_memory"]:
        shutil.copytree(
            ROOT / "ttcl" / name,
            package / name,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
    arms = ["independent", "online_bank"]
    writers = {"online_bank": writer}
    if args.comparison_writer_model:
        arms.append("alternate_writer")
        writers["alternate_writer"] = args.comparison_writer_model
    plan = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "backend": "openrouter_http_cpu_only",
        "actor_model": args.model,
        "writer_models": writers,
        "models": metadata,
        "arms": arms,
        "tasks": args.tasks,
        "num_instances": args.num_instances,
        "seed": 42,
        "bank": {"max_entries": 8, "max_chars": 9000, "max_tokens": 2048},
        "bank_tokenizer": str(
            ROOT / "current_work/delta-Mem/model/Qwen3-4B-Instruct-2507"
        ),
        "api": {
            "output_tokens": args.max_output_tokens,
            "reasoning": args.reasoning,
            "provider": args.provider,
            "timeout": args.timeout,
            "retries": 2,
        },
        "writer_output_tokens": args.max_output_tokens,
        "budget": {
            "max_cost_usd": args.max_cost_usd,
            "max_requests": args.max_requests,
        },
        "actor": {
            "memory_chars": 16000,
            "max_turns_per_instance": 64,
            "action_retries": 2,
            "normalize_action": True,
            "allow_initial_experience": True,
        },
        "interpretation": "Primary test: bank vs no bank within the SAME larger actor. Comparison to old 4B is cross-model and does not isolate writer capacity. Optional alternate_writer keeps actor fixed.",
        "sampling": "Only supported parameters are sent. GPT-6 catalog does not advertise temperature/top_p; they are omitted. Seeds are best effort, not guaranteed deterministic.",
        "gpu": False,
    }
    write(root / "plan.json", plan)
    bench = ROOT / "current_work/continual-learning-bench"
    env = {
        "CUDA_VISIBLE_DEVICES": "",
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
        {"environment": env, "python": sys.executable, "cwd": str(root / "source")},
    )
    shutil.copy2(package / "openrouter_memory/README.md", root / "PROTOCOL.md")
    shutil.copy2(
        package / "llm_memory/trajectory_extraction_prompt.md",
        root / "EXTRACTION_PROMPT.md",
    )
    files = [
        *package.rglob("*.py"),
        *package.rglob("*.md"),
        *(bench / "src").rglob("*.py"),
        root / "plan.json",
        root / "manifest.json",
        root / "EXTRACTION_PROMPT.md",
    ]
    write(
        root / "hashes.json",
        {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
    )
    write(root / "status.json", {"status": "waiting_for_api_key", "gpu": False})
    (root / "REPORT.md").write_text(
        "# OpenRouter 自主经验库实验\n\n已准备，等待本地输入 API key。尚无付费模型调用或实验成绩。\n"
    )
    return root


def launch(root, key, foreground=False):
    state = json.loads((root / "status.json").read_text())
    if state["status"] != "waiting_for_api_key":
        raise ValueError("该目录已经启动过。请用新目录运行，避免覆盖结果或重复付费。")
    manifest = json.loads((root / "manifest.json").read_text())
    command = [
        manifest["python"],
        "-m",
        "ttcl.openrouter_memory.run",
        "--root",
        str(root),
    ]
    env = {**os.environ, **manifest["environment"], "OPENROUTER_API_KEY": key}
    write(root / "status.json", {"status": "starting", "gpu": False})
    if foreground:
        result = subprocess.run(command, cwd=manifest["cwd"], env=env)
        if result.returncode:
            raise RuntimeError("Experiment stopped; see status.json and REPORT.md")
    else:
        with (root / "run.log").open("w") as log:
            process = subprocess.Popen(
                command,
                cwd=manifest["cwd"],
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        (root / "run.pid").write_text(str(process.pid) + "\n")
        print(f"已启动后台 API 实验，PID={process.pid}；不使用 GPU。")
    print(f"结果：{root / 'REPORT.md'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="openai/gpt-6-astra")
    parser.add_argument("--writer-model", help="默认与答题模型相同")
    parser.add_argument(
        "--comparison-writer-model", help="可选第三组：固定答题模型，换另一写入器"
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=["database_exploration", "cohort_studies"],
        default=["database_exploration", "cohort_studies"],
    )
    parser.add_argument("--num-instances", type=int, default=12)
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT
        / "ttcl/results/openrouter_memory"
        / datetime.now(timezone.utc).strftime("gpt6_bank_%Y%m%d_%H%M%S"),
    )
    parser.add_argument(
        "--reasoning",
        choices=["default", "low", "medium", "high", "xhigh"],
        default="medium",
    )
    parser.add_argument(
        "--provider", help="可选 OpenRouter provider slug，用 only 固定路由"
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=8192,
        help="API completion总预算，包含可能的 reasoning tokens",
    )
    parser.add_argument("--max-cost-usd", type=float, default=50.0)
    parser.add_argument("--max-requests", type=int, default=1500)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--list-models", nargs="?", const="gpt-6", metavar="FILTER")
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="只生成冻结实验配置，不读取 key、不调用付费 API",
    )
    parser.add_argument(
        "--catalog-file",
        type=Path,
        help="仅 prepare-only/list-models 可用，读取已下载目录进行离线检查",
    )
    parser.add_argument(
        "--start", type=Path, help="输入 key 并启动之前 prepare-only 的目录"
    )
    parser.add_argument("--foreground", action="store_true")
    args = parser.parse_args()
    if args.start:
        if any(
            a.startswith("--") and a.split("=")[0] not in {"--start", "--foreground"}
            for a in sys.argv[1:]
        ):
            raise ValueError(
                "--start 使用已冻结配置；修改模型或预算请创建新的实验目录。"
            )
        prior = json.loads((args.start / "plan.json").read_text())
        print(
            f"启动已准备实验：{prior['actor_model']}，每任务 {prior['num_instances']} 题，费用上限 ${prior['budget']['max_cost_usd']:g}。"
        )
        launch(args.start.resolve(), read_key(), args.foreground)
        return
    if args.catalog_file and not (args.prepare_only or args.list_models is not None):
        raise ValueError(
            "--catalog-file is only available for non-billable preparation"
        )
    models = (
        json.loads(args.catalog_file.read_text())["data"]
        if args.catalog_file
        else catalog()
    )
    if args.list_models is not None:
        for model in models:
            if args.list_models.lower() in model["id"].lower():
                print(
                    f"{model['id']}  context={model.get('context_length')}  pricing={json.dumps(model.get('pricing', {}))}"
                )
        return
    root = prepare(args, models)
    print(f"答题模型：{args.model}；总结模型：{args.writer_model or args.model}")
    print(
        f"任务：{', '.join(args.tasks)}；每任务 {args.num_instances} 题；费用上限 ${args.max_cost_usd:g}（保守预留，可能提前停止）。"
    )
    if args.prepare_only:
        print(
            f"已准备，尚未调用付费 API。启动：bash ttcl/openrouter_memory/run.sh --start {root}"
        )
    else:
        launch(root, read_key(), args.foreground)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, RuntimeError, OSError) as exc:
        # Never dump process environment, Authorization or a key-bearing traceback.
        print(f"启动失败：{exc}", file=sys.stderr)
        sys.exit(1)
