"""Run the original trajectory-bank experiment entirely through OpenRouter."""

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import statistics
import time
from types import SimpleNamespace

from ttcl.common.openrouter import (
    APIStop,
    BankTokenizer,
    Ledger,
    OpenRouter,
    catalog,
    exact_model,
    write,
)
from ttcl.llm_memory.trajectory_bank import TrajectoryBank
from ttcl.structured_memory import run_benchmark as base
from ttcl.structured_memory.online_bank import audit_task, read, run_episode


def compare(rows, arms, count):
    groups = {
        arm: {
            r["canonical_index"]: r
            for r in rows
            if r["arm"] == arm and r["status"] == "complete"
        }
        for arm in arms
    }
    keys = set.intersection(*(set(g) for g in groups.values()))
    pairs = []
    for index in sorted(keys):
        if len({groups[a][index]["instance_id"] for a in arms}) != 1:
            raise ValueError("Instance mismatch")
        pairs.append(
            {"index": index, "rewards": {a: groups[a][index]["reward"] for a in arms}}
        )
    result = {
        "complete": len(pairs) == count,
        "paired_count": len(pairs),
        "pairs": pairs,
        "arms": {},
    }
    for arm in arms:
        ds = [p["rewards"][arm] - p["rewards"]["independent"] for p in pairs]
        later = [
            p["rewards"][arm] - p["rewards"]["independent"]
            for p in pairs
            if p["index"] > 0
        ]
        result["arms"][arm] = {
            "mean_reward": statistics.mean(p["rewards"][arm] for p in pairs)
            if pairs
            else None,
            "delta": statistics.mean(ds) if ds else None,
            "after_first_delta": statistics.mean(later) if later else None,
            "wins": sum(x > 1e-12 for x in ds),
            "ties": sum(abs(x) <= 1e-12 for x in ds),
            "losses": sum(x < -1e-12 for x in ds),
        }
    return result


def report(root):
    plan = read(root / "plan.json")
    state = read(root / "status.json", {})
    lines = [
        "# OpenRouter 自主经验库实验",
        "",
        f"状态：{state.get('status')}；更新时间：{state.get('updated_at', '—')}",
        "",
        f"答题模型：`{plan['actor_model']}`；写入器：`{json.dumps(plan['writer_models'])}`。CPU 本地环境，全部模型调用通过 API。",
        "",
        "| 任务 | 组别 | 完成 | 共同配对 | 配对均分 | 相对无经验 | 胜/平/负 |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for task in plan["tasks"]:
        rows = read(root / task / "results.json", [])
        paired = compare(rows, plan["arms"], plan["num_instances"])
        write(root / task / "comparison.json", paired)
        for arm, values in paired["arms"].items():
            completed = sum(r["status"] == "complete" and r["arm"] == arm for r in rows)
            lines.append(
                f"| {task} | {arm} | {completed}/{plan['num_instances']} | {paired['paired_count']}/{plan['num_instances']} | {values['mean_reward'] if values['mean_reward'] is not None else '—'} | {values['delta'] if values['delta'] is not None else '—'} | {values['wins']}/{values['ties']}/{values['losses']} |"
            )
        for arm in plan["arms"][1:]:
            decisions = [r.get("bank_decision") for r in rows if r["arm"] == arm]
            lines.append(
                f"\n{task}/{arm}：UPDATE={decisions.count('UPDATE')}，KEEP={decisions.count('KEEP')}，REJECTED={decisions.count('REJECTED')}；排除首次空库 Δ={paired['arms'][arm]['after_first_delta']}。\n"
            )
    usage = read(root / "api_usage.json", {})
    lines.extend(
        [
            "",
            "API 用量：`" + json.dumps(usage, ensure_ascii=False) + "`",
            "",
            "仅比较同一 API 答题模型的经验组与无经验组；旧4B结果不能直接作为GPT-6的无经验基线。",
            "默认同时增强答题与总结能力，不能单独证明旧结果是总结模型太小所致；alternate_writer 可固定 actor 比较不同 writer。",
            "seed 为尽力匹配，不保证远程服务完全确定；首题空库也可能有随机差异。",
            "仅发送目录支持的采样参数，具体请求、provider、实际model、token和reasoning用量见逐调用日志。",
            "API completion预算包括隐藏reasoning，因此与原Qwen输出预算不同；经验库仍使用原Qwen CPU tokenizer统一限制2048 tokens。",
            "未完成/网络故障不补零；只有完整配对才是完整实验结果。费用未知时预留保守上界。",
            "完整轨迹与公开反馈在每题结束后供writer使用，官方scalar只提供给该题结束后的writer；程序不核验经验语义。",
        ]
    )
    if state.get("error"):
        lines.extend(["", "停止原因：" + state["error"]])
    temp = root / "REPORT.tmp"
    temp.write_text("\n".join(lines) + "\n")
    temp.replace(root / "REPORT.md")
    notes = ["# 当前模型经验库", ""]
    for task in plan["tasks"]:
        for arm in plan["arms"][1:]:
            bank = read(root / task / arm / "bank.json")
            if bank:
                notes.extend(
                    [
                        f"## {task}/{arm}",
                        "",
                        "```json",
                        json.dumps(bank, ensure_ascii=False, indent=2),
                        "```",
                        "",
                    ]
                )
    (root / "BANKS.md").write_text("\n".join(notes))


def hash_audit(root):
    expected = read(root / "hashes.json", {})
    errors = [
        filename
        for filename, digest in expected.items()
        if hashlib.sha256(Path(filename).read_bytes()).hexdigest() != digest
    ]
    write(
        root / "source_audit.json",
        {"errors": errors, "checked_at": datetime.now(timezone.utc).isoformat()},
    )
    if errors:
        raise ValueError("Frozen source/plan changed; see source_audit.json")


def audit_all_arms(root, task, plan):
    result = audit_task(root, task, plan["num_instances"])
    rows = read(root / task / "results.json")
    first = [row for row in rows if row["episode"] == 1]
    if len({row["first_prompt_sha256"] for row in first}) > 1:
        result["errors"].append("First empty-bank actor prompts differ")
    for arm in plan["arms"][1:]:
        previous = []
        for index in range(plan["num_instances"]):
            directory = root / task / arm / f"episode_{index + 1:03d}"
            update = read(directory / "bank_update.json")
            episode = read(directory / "trajectory.json")
            if (
                update["bank_before"]["entries"] != previous
                or update["bank_before"]["last_observed"] != index
            ):
                result["errors"].append(f"{arm}/{index}: broken bank chronology")
            for attempt in update["attempts"]:
                supplied = json.loads(
                    attempt["messages"][1]["content"].split(
                        "\nYour previous response was rejected:"
                    )[0]
                )
                if (
                    supplied["trajectory"] != episode
                    or supplied["existing_bank"] != previous
                ):
                    result["errors"].append(f"{arm}/{index}: writer input mismatch")
            if not update["accepted"] and update["bank_after"]["entries"] != previous:
                result["errors"].append(
                    f"{arm}/{index}: rejected update changed memory"
                )
            previous = update["bank_after"]["entries"]
    write(root / task / "audit.json", result)
    return result


def execute(root, key, *, models=None, tokenizer=None, backend_class=OpenRouter):
    plan = read(root / "plan.json")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError(
            "This API-only entry requires CUDA_VISIBLE_DEVICES='' before execution"
        )
    hash_audit(root)
    current_models = catalog() if models is None else models
    metadata = {
        identity: exact_model(current_models, identity) for identity in plan["models"]
    }
    write(root / "runtime_models.json", metadata)
    tokenizer = tokenizer or BankTokenizer(plan["bank_tokenizer"])
    ledger = Ledger(root, **plan["budget"])
    backends = {
        identity: backend_class(key, info, tokenizer, ledger, **plan["api"])
        for identity, info in metadata.items()
    }
    actor = backends[plan["actor_model"]]

    def status(**kwargs):
        write(
            root / "status.json",
            {
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "gpu": False,
                **kwargs,
            },
        )
        report(root)

    try:
        os.chdir(base.BENCH)
        for task in plan["tasks"]:
            args = SimpleNamespace(
                **plan["actor"],
                task=task,
                seed=plan["seed"],
                num_instances=plan["num_instances"],
            )
            banks = {arm: TrajectoryBank(**plan["bank"]) for arm in plan["arms"][1:]}
            rows = []
            for index in range(plan["num_instances"]):
                for arm in plan["arms"]:
                    status(
                        status="running",
                        task=task,
                        arm=arm,
                        episode=index + 1,
                        phase="answer",
                    )
                    directory = root / task / arm / f"episode_{index + 1:03d}"
                    context = "" if arm == "independent" else banks[arm].context()
                    actor.purpose = {
                        "task": task,
                        "arm": arm,
                        "episode": index + 1,
                        "purpose": "actor",
                    }
                    row, episode = run_episode(args, actor, index, directory, context)
                    row["arm"] = arm
                    if ledger.stop_reason:
                        rows.append(row)
                        write(root / task / "results.json", rows)
                        raise APIStop(ledger.stop_reason)
                    if arm != "independent":
                        status(
                            status="running",
                            task=task,
                            arm=arm,
                            episode=index + 1,
                            phase="summarize",
                        )
                        writer = backends[plan["writer_models"][arm]]
                        writer.purpose = {
                            "task": task,
                            "arm": arm,
                            "episode": index + 1,
                            "purpose": "writer",
                        }

                        def generate(messages, seed, **kwargs):
                            base.append(
                                directory / "writer_requests.jsonl",
                                {
                                    "messages": copy.deepcopy(messages),
                                    "seed": seed,
                                    "options": kwargs,
                                },
                            )
                            start = time.monotonic()
                            result = writer.generate(messages, seed, **kwargs)
                            base.append(
                                directory / "writer_generations.jsonl",
                                {**result, "seconds": time.monotonic() - start},
                            )
                            return result

                        update = banks[arm].update(
                            episode,
                            generate,
                            base.generation_seed(
                                plan["seed"], f"episode_{index + 1}", "bank_writer"
                            ),
                            lambda text: len(
                                tokenizer.encode(text, add_special_tokens=False)
                            ),
                            output_tokens=plan["writer_output_tokens"],
                        )
                        write(directory / "bank_update.json", update)
                        write(root / task / arm / "bank.json", banks[arm].state_dict())
                        row.update(
                            bank_decision=update["decision"],
                            bank_entries=len(banks[arm].entries),
                            bank_version=banks[arm].version,
                            writer_error=update["error"],
                            writer_calls=len(update["attempts"]),
                            writer_input_tokens=sum(
                                a.get("completion", {}).get("input_tokens", 0)
                                for a in update["attempts"]
                            ),
                            writer_output_tokens=sum(
                                a.get("completion", {}).get("output_tokens", 0)
                                for a in update["attempts"]
                            ),
                        )
                    rows.append(row)
                    write(root / task / "results.json", rows)
                    status(
                        status="running",
                        task=task,
                        arm=arm,
                        episode=index + 1,
                        phase="episode_complete",
                    )
                    print(
                        f"{task} {arm} episode={index + 1} reward={row['reward']} decision={row.get('bank_decision')}",
                        flush=True,
                    )
                    if ledger.stop_reason:
                        raise APIStop(ledger.stop_reason)
            audit = audit_all_arms(root, task, plan)
            if audit["errors"]:
                raise ValueError(f"Trajectory/bank audit failed: {task}")
            write(
                root / task / "status.json", {"status": "finished", "audit_errors": []}
            )
        hash_audit(root)
        ledger.persist()
        status(status="finished")
    except APIStop as exc:
        status(status="stopped_api_or_budget", error=str(exc))
        raise
    except Exception as exc:
        # Provider exceptions never contain a key; still redact unexpected ones.
        status(
            status="failed",
            error=(type(exc).__name__ + ": " + str(exc)).replace(key, "[REDACTED]"),
        )
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    key = os.environ.get("OPENROUTER_API_KEY", "")
    try:
        execute(root, key)
    except Exception as exc:
        safe = (
            (type(exc).__name__ + ": " + str(exc)).replace(key, "[REDACTED]")
            if key
            else type(exc).__name__
        )
        if read(root / "status.json", {}).get("status") not in {
            "failed",
            "stopped_api_or_budget",
        }:
            write(
                root / "status.json", {"status": "failed", "error": safe, "gpu": False}
            )
            report(root)
        print(safe, flush=True)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
