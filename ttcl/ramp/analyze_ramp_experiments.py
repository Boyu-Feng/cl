"""Summarize recorded experiment outcomes without selecting best-of-K as scores."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics

CONTROLS = {
    "sample_ramp": "sample_frozen",
    "group_ramp": "group_frozen",
    "summary_ramp": "summary_frozen",
    "summary_group_ramp": "summary_group_frozen",
    "selection_ramp": "selection_frozen",
    "selection_group_ramp": "selection_group_frozen",
    "mutation_ramp": "mutation_frozen",
    "feedback_ramp": "feedback_frozen",
    "feedback_mutation_ramp": "feedback_mutation_frozen",
}

FEEDBACK_CONTROLS = {
    "feedback_frozen": "summary_frozen",
    "feedback_mutation_frozen": "mutation_frozen",
}


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def summarize(root):
    records = {}
    responses = {}
    configs = {}
    for path in sorted(root.glob("*/metrics.json")):
        name = path.parent.name
        metrics = json.loads(path.read_text())
        config = json.loads((path.parent / "config.json").read_text())
        configs[name] = config
        rows = read_jsonl(path.parent / "responses.jsonl")
        updates = read_jsonl(path.parent / "updates.jsonl")
        candidates = read_jsonl(path.parent / "candidates.jsonl")
        responses[name] = rows
        quarter = max(1, len(rows) // 4)
        records[name] = {
            "variant": name,
            "seed": config["seed"],
            "completed": len(rows),
            "method": config["method"],
            "memory_mode": config.get("memory_mode", "none"),
            "feedback_memory": config.get("feedback_memory", "none"),
            "action_mode": config.get("action_mode", "report"),
            "proposal_mode": config.get("proposal_mode", "sampled"),
            "candidates_per_scan": config.get("candidates_per_scan", 1),
            "mean_score": statistics.mean(r["reward"] for r in rows),
            "first_quarter_score": statistics.mean(r["reward"] for r in rows[:quarter]),
            "last_quarter_score": statistics.mean(r["reward"] for r in rows[-quarter:]),
            "accepted_updates": sum(bool(u.get("accepted")) for u in updates),
            "rejected_updates": sum(not u.get("accepted") and u.get("reason") not in ("no_reward_signal", "no_new_data") for u in updates),
            "skipped_updates": sum(u.get("reason") in ("no_reward_signal", "no_new_data") for u in updates),
            "optimizer_steps": sum(u.get("optimizer_steps", len(u.get("losses", []))) for u in updates),
            "retained_optimizer_steps": sum(u.get("retained_optimizer_steps", len(u.get("losses", [])) if u.get("accepted") else 0) for u in updates),
            "total_pair_count": sum(u.get("pair_count", 0) for u in updates),
            "invalid_reports": sum(r["parse_error"] is not None for r in rows),
            "reward_calls": metrics.get("reward_calls", len(candidates) or len(rows)),
            "learner_feedback_count": metrics.get("learner_feedback_count"),
            "feedback_memory_reward_count": metrics.get("feedback_memory_reward_count", 0),
            "generation_tokens": metrics.get("generation_tokens"),
            "elapsed_seconds": metrics.get("elapsed_seconds"),
            "current_peak_copy_count": metrics.get("current_peak_copy_count"),
            "oracle_mean_score": metrics.get("oracle_mean_score"),
            "mean_distinct_candidate_geometries": metrics.get("mean_distinct_candidate_geometries"),
            "mean_candidate_reward_spread": metrics.get("mean_candidate_reward_spread"),
            "history_supported_extra_transmitters": metrics.get("history_supported_extra_transmitters"),
            "control": None, "delta_vs_control": None,
        }
    comparisons = {}
    feedback_comparisons = {}
    for name, control in {**CONTROLS, **FEEDBACK_CONTROLS}.items():
        if name not in records or control not in records:
            continue
        a, b = responses[name], responses[control]
        matching_keys = ("model", "adapter_path", "data_path", "num_scans", "seed", "generation_seed",
                         "memory_mode", "action_mode", "proposal_mode", "candidates_per_scan", "do_sample", "temperature",
                         "top_p", "top_k", "max_new_tokens", "max_input_tokens", "dtype")
        if name in CONTROLS:
            matching_keys += ("feedback_memory", "feedback_window", "feedback_min_gap")
        elif (configs[name].get("feedback_memory") != "summary"
              or configs[control].get("feedback_memory", "none") != "none"
              or configs[name]["method"] != "frozen" or configs[control]["method"] != "frozen"):
            raise ValueError("Feedback ablation requires frozen models with/without feedback memory")
        mismatched = [key for key in matching_keys if configs[name].get(key) != configs[control].get(key)]
        if mismatched:
            raise ValueError(f"Mismatched configurations {name}, {control}: {mismatched}")
        if len(a) != len(b) or any(x["instance_id"] != y["instance_id"] for x, y in zip(a, b)):
            raise ValueError(f"Mismatched scan order: {name}, {control}")
        changes = [x["reward"] - y["reward"] for x, y in zip(a, b)]
        delta = statistics.mean(changes)
        destination = comparisons if name in CONTROLS else feedback_comparisons
        if name in CONTROLS:
            records[name].update(control=control, delta_vs_control=delta)
        destination[name] = {"control": control, "mean_delta": delta,
                             "wins": sum(x > 0 for x in changes),
                             "ties": sum(x == 0 for x in changes),
                             "losses": sum(x < 0 for x in changes)}
    plan_path = root / "experiment_plan.json"
    plan = json.loads(plan_path.read_text()) if plan_path.exists() else {}
    unfinished = {name: entry.get("status", "unknown") for name, entry in plan.get("runs", {}).items() if name not in records}
    for name in plan.get("variants", []):
        if name not in records and name not in unfinished:
            unfinished[name] = "not_started"
    result = {"runs": records, "comparisons": comparisons,
              "feedback_memory_comparisons": feedback_comparisons, "unfinished_runs": unfinished,
              "notes": ["Only candidate 0 contributes to official mean_score.",
                        "Oracle best-of-K scores use extra feedback and are diagnostics, not deployable performance.",
                        "Summary variants retain external public-observation memory; compare with summary frozen controls.",
                        "A single ordered corpus is an exploratory result, not independent-task generalization."]}
    (root / "comparison.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    if records:
        with (root / "comparison.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(next(iter(records.values()))))
            writer.writeheader()
            writer.writerows(records.values())
    lines = ["# RAMP 对照实验", "", "正式得分只使用评分前已确定的第一个回答。多候选最优分仅作诊断，不替换正式回答。", "",
             "| 实验 | 扫描数 | 候选/题 | 评分次数 | 平均 IoU | 对匹配 frozen 的变化（百分点） | 接受/回滚/跳过 | 偏好对总数 | 无效报告 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, r in records.items():
        delta = "—" if r["delta_vs_control"] is None else f'{r["delta_vs_control"] * 100:+.3f}'
        lines.append(f'| {name} | {r["completed"]} | {r["candidates_per_scan"]} | {r["reward_calls"]} | {r["mean_score"]:.3%} | {delta} | {r["accepted_updates"]}/{r["rejected_updates"]}/{r["skipped_updates"]} | {r["total_pair_count"]} | {r["invalid_reports"]} |')
    lines += ["", "## 探索诊断", "", "| 实验 | 完全照抄当前峰的扫描数 | 平均不同候选几何数 | 平均同题分差 | best-of-K IoU（仅诊断） |",
              "|---|---:|---:|---:|---:|"]
    for name, r in records.items():
        distinct = r["mean_distinct_candidate_geometries"]
        spread = r["mean_candidate_reward_spread"]
        oracle = r["oracle_mean_score"]
        lines.append(f'| {name} | {r["current_peak_copy_count"]} | {distinct if distinct is not None else "—"} | {spread if spread is not None else "—"} | {format(oracle, ".3%") if oracle is not None else "—"} |')
    lines += ["", "## 解释边界", "", "- `summary` 组使用外部历史观测摘要，收益不能全部归因于参数学习。",
              "- `feedback` 组把已支付的历史标量反馈写入外部经验记忆；frozen 只表示参数冻结。",
              "- `group` 组每题使用三次评分，不能与单次反馈组当作相同反馈预算比较。",
              "- 相同输入条件下的 frozen 对照用于分离 LoRA 更新带来的变化。",
              "- 当前只有一个按顺序运行的数据集；不同采样种子也不等于独立任务数据。", ""]
    if feedback_comparisons:
        lines += ["## 反馈经验记忆消融（参数均冻结、相同评分预算）", "",
                  "| 反馈记忆组 | 无反馈记忆对照 | 变化（百分点） | 胜/平/负 |",
                  "|---|---|---:|---:|"]
        for name, row in feedback_comparisons.items():
            lines.append(f"| {name} | {row['control']} | {row['mean_delta'] * 100:+.3f} | "
                         f"{row['wins']}/{row['ties']}/{row['losses']} |")
        lines.append("")
    if unfinished:
        lines += ["## 未完成的实验", ""] + [f"- {name}: {status}" for name, status in unfinished.items()] + [""]
    (root / "report.md").write_text("\n".join(lines))
    print("\n".join(lines[:5 + len(records) + 2]))
    print(f"Saved: {root / 'report.md'}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    summarize(parser.parse_args().run_root)
