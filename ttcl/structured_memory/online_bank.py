"""Pilot: full public trajectories + terminal reward -> model-maintained bank."""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import statistics
import time
import traceback
from types import SimpleNamespace

from ttcl.llm_memory.trajectory_bank import TrajectoryBank
from ttcl.structured_memory import run_benchmark as base

ARMS = ("independent", "online_bank")


def read(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


class BankSystem(base.StructuredSystem):
    def __init__(self, args, model, output, brief, context):
        self.fixed_bank_context = context
        super().__init__(args, "independent", model, output, brief)

    def reset(self):
        super().reset()
        self.public_steps = []
        self.public_schemas = []
        self.active_schema = None

    def experience_context(self, prompt):
        return self.fixed_bank_context

    def respond(self, query):
        schema = query.response_schema.model_json_schema()
        if schema not in self.public_schemas:
            self.public_schemas.append(schema)
        self.active_schema = self.public_schemas.index(schema) + 1
        return super().respond(query)

    def observe(self, observation, next_query=None):
        if self.last is not None:
            prompt, action, _ = self.last
            self.public_steps.append(
                {
                    "step": len(self.public_steps) + 1,
                    "query": prompt,
                    "response_schema_id": self.active_schema,
                    "action": copy.deepcopy(action),
                    "public_feedback": observation.content,
                    "instance_complete": base.observation_marks_instance_complete(
                        observation
                    ),
                }
            )
        # This system uses independent mode: only within-episode history changes.
        super().observe(observation, next_query)

    def get_run_artifacts(self):
        return {
            "actor_bank_context_sha256": sha(self.fixed_bank_context),
            "public_steps": self.public_steps,
        }


def run_episode(args, model, index, output, context):
    output.mkdir(parents=True, exist_ok=False)
    task = base.make_task(args.task, args.seed, independent=True)
    query = task.reset_baseline_instance(index)
    brief = task.get_agent_brief()
    brief_text = base.format_task_agent_brief(brief) if brief else ""
    system = BankSystem(args, model, output, brief_text, context)
    system.requested_canonical_index = index
    recorder = base.Recorder(output, system, 1)
    record = {
        "canonical_index": index,
        "episode": index + 1,
        "instance_id": query.instance_id,
        "reward": None,
        "status": "failed",
    }
    try:
        result = base.run_task(
            task,
            system,
            trace_recorder=recorder,
            show_progress=False,
            reset_system=False,
            initial_query=query,
        )
        if len(result.instance_outcomes) != 1:
            raise ValueError("Expected exactly one official outcome")
        outcome = result.instance_outcomes[0]
        record.update(
            status="complete", reward=float(outcome.reward), success=outcome.success
        )
    except Exception as exc:
        record.update(error=repr(exc), traceback=traceback.format_exc())
    finally:
        connection = getattr(task, "_conn", None)
        if connection is not None:
            connection.close()
    events = (
        [
            json.loads(line)
            for line in (output / "responses.jsonl").read_text().splitlines()
        ]
        if (output / "responses.jsonl").exists()
        else []
    )
    episode = {
        "episode": index + 1,
        "public_task_brief": brief_text,
        "initial_public_query": query.prompt,
        "response_schemas": system.public_schemas,
        "steps": system.public_steps,
        "completed": record["status"] == "complete",
        "reward": record["reward"],
        "reward_scope": "Official scalar for this whole episode only; higher is better. No per-step scalar was supplied.",
        "local_execution_error": record.get("error"),
        "format_failures": [
            {
                "turn": e["turn"],
                "attempt": e["format_retry"],
                "unexecuted_output": e["raw_response"],
                "error": e["parse_error"],
            }
            for e in events
            if e.get("parse_error")
        ],
    }
    record.update(
        actor_calls=system.calls,
        actor_input_tokens=system.input_tokens,
        actor_output_tokens=system.output_tokens,
        actor_seconds=recorder.metrics()["elapsed_seconds"],
        bank_context_sha256=sha(context),
        bank_context_chars=len(context),
        first_prompt_sha256=events[0].get("rendered_prompt_sha256") if events else None,
        writer_calls=0,
        writer_input_tokens=0,
        writer_output_tokens=0,
    )
    if (
        sum(e["input_tokens"] for e in events) != system.input_tokens
        or sum(e["output_tokens"] for e in events) != system.output_tokens
    ):
        raise ValueError("Actor token accounting mismatch")
    contexts = read_lines(output / "memory_contexts.jsonl")
    if contexts and contexts[0]["memory_context"] != context:
        raise ValueError("Actor did not receive the intended bank")
    base.write_json(output / "trajectory.json", episode)
    base.write_json(output / "metrics.json", {**recorder.metrics(), **record})
    return record, episode


def read_lines(path):
    return (
        [json.loads(line) for line in path.read_text().splitlines()]
        if path.exists()
        else []
    )


def paired(rows, count):
    groups = {
        arm: {
            r["canonical_index"]: r
            for r in rows
            if r["arm"] == arm and r["status"] == "complete"
        }
        for arm in ARMS
    }
    keys = sorted(set(groups[ARMS[0]]) & set(groups[ARMS[1]]))
    pairs = []
    for index in keys:
        left, right = (groups[arm][index] for arm in ARMS)
        if left["instance_id"] != right["instance_id"]:
            raise ValueError("Pair identity mismatch")
        pairs.append(
            {
                "canonical_index": index,
                "independent": left["reward"],
                "online_bank": right["reward"],
                "delta": right["reward"] - left["reward"],
            }
        )
    later = [p["delta"] for p in pairs if p["canonical_index"] > 0]
    return {
        "complete_comparison": len(pairs) == count,
        "paired_count": len(pairs),
        "mean_delta": statistics.mean(p["delta"] for p in pairs) if pairs else None,
        "after_first_mean_delta": statistics.mean(later) if later else None,
        "wins": sum(p["delta"] > 1e-12 for p in pairs),
        "ties": sum(abs(p["delta"]) <= 1e-12 for p in pairs),
        "losses": sum(p["delta"] < -1e-12 for p in pairs),
        "pairs": pairs,
    }


def audit_task(root, task, count):
    directory = root / task
    rows = read(directory / "results.json", [])
    errors = []
    previous_entries = []
    for index in range(count):
        output = directory / "online_bank" / f"episode_{index + 1:03d}"
        update = read(output / "bank_update.json")
        if update is None:
            continue
        if update["bank_before"]["entries"] != previous_entries:
            errors.append(f"Bank chain broken at {index + 1}")
        if update["bank_before"]["last_observed"] != index:
            errors.append(f"Wrong bank age at {index + 1}")
        from ttcl.llm_memory.trajectory_bank import render

        matching = [
            r for r in rows if r["arm"] == "online_bank" and r["episode"] == index + 1
        ]
        if matching and matching[0]["bank_context_sha256"] != sha(
            render(previous_entries)
        ):
            errors.append(f"Actor bank mismatch at {index + 1}")
        episode = read(output / "trajectory.json")
        for attempt in update["attempts"]:
            supplied = json.loads(
                attempt["messages"][1]["content"].split(
                    "\nYour previous response was rejected:"
                )[0]
            )
            if (
                supplied["trajectory"] != episode
                or supplied["existing_bank"] != previous_entries
            ):
                errors.append(
                    f"Writer payload differs from public trajectory/bank at {index + 1}"
                )
        if matching and episode["reward"] != matching[0]["reward"]:
            errors.append(f"Wrong scalar at {index + 1}")
        previous_entries = update["bank_after"]["entries"]
        if (
            not update["accepted"]
            and previous_entries != update["bank_before"]["entries"]
        ):
            errors.append(f"Rejected update mutated bank at {index + 1}")
        for entry in previous_entries:
            if any(ref["episode"] > index + 1 for ref in entry["evidence"]):
                errors.append(f"Future evidence at {index + 1}")
    first = [r for r in rows if r["episode"] == 1]
    if (
        len(first) == 2
        and first[0]["first_prompt_sha256"] != first[1]["first_prompt_sha256"]
    ):
        errors.append("Empty-bank first prompts differ")
    result = {
        "errors": errors,
        "completed_trials": len(rows),
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "checks": "Chronological bank chain, actor bank hash, full public writer payload, exact scalar, no future citations, rejected-update atomicity, first-prompt equality. Does not certify semantic truth.",
    }
    base.write_json(directory / "audit.json", result)
    return result


def report(root, plan):
    lines = [
        "# 模型自主维护经验库：在线小测试",
        "",
        "状态与部分均分只用于监控；完整同实例配对才用于最终比较。",
        "",
        "| 任务 | 组别 | 完成/计划 | 失败 | 平均reward | 答题调用 | 总结调用 | 总输入/输出tokens |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    notes, bank_notes = [], ["# 模型经验库更新", ""]
    for task in plan["tasks"]:
        rows = read(root / task / "results.json", [])
        for arm in ARMS:
            all_rows = [r for r in rows if r["arm"] == arm]
            complete = [r for r in all_rows if r["status"] == "complete"]
            mean = (
                f"{statistics.mean(r['reward'] for r in complete):.6f}"
                if complete
                else "—"
            )
            lines.append(
                f"| {task} | {arm} | {len(complete)}/{plan['num_instances']} | {len(all_rows) - len(complete)} | {mean} | "
                f"{sum(r['actor_calls'] for r in all_rows)} | {sum(r['writer_calls'] for r in all_rows)} | "
                f"{sum(r['actor_input_tokens'] + r['writer_input_tokens'] for r in all_rows)}/{sum(r['actor_output_tokens'] + r['writer_output_tokens'] for r in all_rows)} |"
            )
        comparison = paired(rows, plan["num_instances"])
        if comparison["paired_count"]:
            label = "完整" if comparison["complete_comparison"] else "进行中，非最终"
            notes.append(
                f"{task}：{label}配对 {comparison['paired_count']}/{plan['num_instances']}，Δ={comparison['mean_delta']:+.6f}；胜/平/负={comparison['wins']}/{comparison['ties']}/{comparison['losses']}。"
            )
            if comparison["after_first_mean_delta"] is not None:
                notes.append(
                    f"排除首次空库样本后的配对 Δ={comparison['after_first_mean_delta']:+.6f}。"
                )
        updates = [r for r in rows if r["arm"] == "online_bank"]
        notes.append(
            f"{task} 总结决策：UPDATE={sum(r.get('bank_decision') == 'UPDATE' for r in updates)}，KEEP={sum(r.get('bank_decision') == 'KEEP' for r in updates)}，格式/长度等拒绝={sum(r.get('bank_decision') == 'REJECTED' for r in updates)}。"
        )
        state = read(root / task / "bank.json", {})
        bank_notes += [
            f"## {task}",
            "",
            f"已处理轨迹：{state.get('last_observed', 0)}；版本：{state.get('version', 0)}",
            "",
        ]
        for entry in state.get("entries", []):
            bank_notes += [
                f"### {entry['id']} — {entry['title']} ({entry['type']})",
                "",
                entry["lesson"],
                "",
                "适用范围：" + entry["scope"],
                "",
                "使用方式：" + entry["application"],
                "",
                "局限：" + entry["limitations"],
                "",
                "证据：" + json.dumps(entry["evidence"]),
                "",
            ]
    lines += [
        "",
        *notes,
        "",
        "每个样本结束后同一冻结Qwen3读取完整公开轨迹和最终官方scalar，自行选择新增/修改/删除/不更新。只向下一样本提供银行内容，不追加原轨迹或总结报告。",
        "两组答题采样、工具预算和格式重试一致；经验组额外使用总结调用，成本已计入。没有参数更新或额外候选评分。",
        "最多8条经验、2048 tokens；无人工领域事实预置。程序只校验格式、引用范围和预算，不核验经验语义或奖励归因。",
        "前12个实例、单种子开发pilot，不能认定稳定泛化。官方反馈中已公开的信息照常保留；额外标量仅提供给当前样本结束后的总结模型。",
        "没有官方结果的执行失败标记为缺失reward，不补零；其可见部分轨迹可总结，明确为不完整。",
        "完整提取prompt见 EXTRACTION_PROMPT.md；模型银行见 BANKS.md；逐样本更新和原始writer输出见各任务 online_bank/episode_*/bank_update.json。",
    ]
    (root / "REPORT.tmp").write_text("\n".join(lines) + "\n")
    (root / "REPORT.tmp").replace(root / "REPORT.md")
    (root / "BANKS.tmp").write_text("\n".join(bank_notes) + "\n")
    (root / "BANKS.tmp").replace(root / "BANKS.md")


def worker(root, task):
    plan = read(root / "plan.json")
    args = SimpleNamespace(
        **plan["model"],
        task=task,
        seed=plan["seed"],
        num_instances=plan["num_instances"],
        allow_initial_experience=True,
    )
    directory = root / task
    directory.mkdir(exist_ok=False)
    os.chdir(base.BENCH)
    model = base.LocalQwen(args)
    bank = TrajectoryBank(**plan["bank"])
    base.write_json(directory / "bank.json", bank.state_dict())
    rows = []

    def count_tokens(text):
        return len(model.tokenizer.encode(text, add_special_tokens=False))

    for index in range(args.num_instances):
        for arm in ARMS:
            output = directory / arm / f"episode_{index + 1:03d}"
            base.write_json(
                directory / "status.json",
                {
                    "status": "running",
                    "phase": "answer",
                    "episode": index + 1,
                    "arm": arm,
                },
            )
            context = bank.context() if arm == "online_bank" else ""
            row, episode = run_episode(args, model, index, output, context)
            if arm == "online_bank":
                base.write_json(
                    directory / "status.json",
                    {
                        "status": "running",
                        "phase": "summarize",
                        "episode": index + 1,
                        "arm": arm,
                    },
                )

                def generate(messages, seed, **kwargs):
                    base.append(
                        output / "writer_requests.jsonl",
                        {
                            "messages": copy.deepcopy(messages),
                            "seed": seed,
                            "options": kwargs,
                        },
                    )
                    start = time.monotonic()
                    result = model.generate(messages, seed, **kwargs)
                    base.append(
                        output / "writer_generations.jsonl",
                        {**result, "seconds": time.monotonic() - start},
                    )
                    return result

                update = bank.update(
                    episode,
                    generate,
                    base.generation_seed(
                        args.seed, f"episode_{index + 1}", "bank_writer"
                    ),
                    count_tokens,
                    output_tokens=plan["writer_output_tokens"],
                )
                base.write_json(output / "bank_update.json", update)
                base.write_json(directory / "bank.json", bank.state_dict())
                generations = read_lines(output / "writer_generations.jsonl")
                row.update(
                    writer_calls=len(read_lines(output / "writer_requests.jsonl")),
                    writer_input_tokens=sum(x["input_tokens"] for x in generations),
                    writer_output_tokens=sum(x["output_tokens"] for x in generations),
                    writer_seconds=sum(x["seconds"] for x in generations),
                    bank_decision=update["decision"],
                    bank_entries=len(bank.entries),
                    bank_version=bank.version,
                    writer_error=update["error"],
                )
            rows.append({"arm": arm, **row})
            base.write_json(directory / "results.json", rows)
            base.write_json(
                directory / "comparison.json", paired(rows, args.num_instances)
            )
            print(
                f"{task} {arm} episode={index + 1} reward={row['reward']} update={row.get('bank_decision')} entries={row.get('bank_entries')}",
                flush=True,
            )
    audit = audit_task(root, task, args.num_instances)
    base.write_json(
        directory / "status.json",
        {"status": "finished", "trials": len(rows), "audit_errors": audit["errors"]},
    )
    if audit["errors"]:
        raise RuntimeError("Bank chronology/feedback audit failed; see audit.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--task", choices=["database_exploration", "cohort_studies"], required=True
    )
    args = parser.parse_args()
    worker(args.root.resolve(), args.task)


if __name__ == "__main__":
    main()
