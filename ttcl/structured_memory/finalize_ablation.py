"""Keep a consolidated report current, and audit traces after jobs finish."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

from ttcl.structured_memory.audit_ablation import audit


def read(path):
    return json.loads(path.read_text()) if path.exists() else {}


def update(root, followup):
    lines = [
        "# 经验内容实验汇总",
        "",
        f"更新：{datetime.now(timezone.utc).isoformat()}",
        "",
        "主目标是验证哪些经验有用；reward 只作为附加可见性消融。所有组均为冻结 Qwen3-4B、前12个实例、seed=42。",
        "",
        "| 任务 | 经验组 | 完成状态 | 无经验 reward | 本组 reward | 配对差值 | 胜/平/负 |",
        "|---|---|---|---:|---:|---:|---|",
    ]
    for task in ["database_exploration", "cohort_studies", "exploitable_poker"]:
        results = read(root / task / "results.json")
        baseline = results.get("independent", {})
        if task == "database_exploration":
            results.update(read(followup / task / "results.json"))
        variants = ["legacy", "focused"] + (
            ["focused_reward"] if task != "cohort_studies" else []
        )
        if task == "database_exploration":
            variants += ["focused_procedure"]
        for variant in variants:
            r = results.get(variant, {})

            def score(x):
                return "—" if x.get("mean_score") is None else f"{x['mean_score']:.6f}"

            difference, counts = "—", "—"
            if r.get("status") == baseline.get("status") == "complete":
                a = {o["instance_id"]: o["reward"] for o in baseline["outcomes"]}
                b = {o["instance_id"]: o["reward"] for o in r["outcomes"]}
                if a.keys() == b.keys():
                    deltas = [b[k] - a[k] for k in a]
                    difference = f"{sum(deltas) / len(deltas):+.6f}"
                    counts = f"{sum(d > 1e-12 for d in deltas)}/{sum(abs(d) <= 1e-12 for d in deltas)}/{sum(d < -1e-12 for d in deltas)}"
            state = r.get("status", "运行中/待运行")
            lines.append(
                f"| {task} | {variant} | {state} | {score(baseline)} | {score(r)} | {difference} | {counts} |"
            )
    lines += [
        "",
        "legacy：原版结构化经验。focused：任务针对性事实/统计与历史行动案例。focused_reward：同结构加已完成实例标量。",
        "focused_procedure：看到数据库主对照失败后追加的开发组；只用公开已见 schema 构造类别、日期单位等候选排查 SQL，不预置查询结果。该组去掉旧行动案例，由模型在官方预算内决定是否执行建议查询；不是单一字段消融，也不是独立确认实验。无经验控制复用本轮完整同种子结果，未重新计费调用。",
        "当前已查实的失败模式：数据库会复用错误类别过滤、未验证的日期解释和连接字段；扑克模型既可能误读经验，也存在基础牌力判断错误。改进提取成功不等于下游收益。",
        "所有负结果保留。仅完整同实例组计算差值；无跨任务平均。这是单序列开发实验，后续若有提升仍需留出序列/更多种子验证。",
        "队列任务未增加隐藏 reward；数据库/扑克显式标量是增强反馈协议。无模型参数更新、无额外候选评分、无隐藏标签或策略输入。",
        "代码适应和销售预测尚未运行：Docker Unix socket 权限仍不允许当前账户使用。",
        "明细：各任务 RESULT.md、results.json、responses.jsonl、memory_contexts.jsonl；后续组位于 ../database_procedure_20260920。所有任务完成后自动生成/刷新 audit.json。",
    ]
    temp = root / "FINDINGS.tmp"
    temp.write_text("\n".join(lines) + "\n")
    temp.replace(root / "FINDINGS.md")
    return all(
        read(p / "status.json").get("status") == "finished" for p in [root, followup]
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("followup", type=Path)
    parser.add_argument("--watch", action="store_true")
    args = parser.parse_args()
    while True:
        finished = update(args.root, args.followup)
        if finished or not args.watch:
            for root in [args.root, args.followup]:
                (root / "audit.json").write_text(
                    json.dumps(audit(root), ensure_ascii=False, indent=2) + "\n"
                )
            break
        time.sleep(20)
