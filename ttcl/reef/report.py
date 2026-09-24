"""Summarize completed, paired Reef/CLBench runs without filling missing scores."""

import argparse
import json
from pathlib import Path
import statistics


def report(root):
    rows, all_pairs = [], []
    complete = True
    for task in ["database_exploration", "cohort_studies"]:
        output = root / task
        status_path = output / "status.json"
        status = json.loads(status_path.read_text()) if status_path.exists() else {}
        pairs_path = output / "paired_results.json"
        pairs = json.loads(pairs_path.read_text()) if pairs_path.exists() else []
        if status.get("status") != "complete":
            complete = False
            rows.append(
                f"| {task} | {status.get('status', 'running')} | {len(pairs)} | — | — | — | — |"
            )
            continue
        summary = json.loads((output / "summary.json").read_text())
        if len(pairs) != summary["n"]:
            raise ValueError(f"Pair count differs from summary for {task}")
        all_pairs.extend(pairs)
        wtl = "/".join(str(n) for n in summary["wins_ties_losses"])
        rows.append(
            f"| {task} | complete | {len(pairs)} | {summary['none_mean']:.6f} | {summary['reef_gepa_mean']:.6f} | {summary['mean_delta']:+.6f} | {wtl} |"
        )
    if complete:
        wtl = [
            sum(p["delta"] > 1e-9 for p in all_pairs),
            sum(abs(p["delta"]) <= 1e-9 for p in all_pairs),
            sum(p["delta"] < -1e-9 for p in all_pairs),
        ]
        rows.append(
            f"| 合计 | complete | {len(all_pairs)} | {statistics.mean(p['none'] for p in all_pairs):.6f} | {statistics.mean(p['reef_gepa'] for p in all_pairs):.6f} | {statistics.mean(p['delta'] for p in all_pairs):+.6f} | {'/'.join(map(str, wtl))} |"
        )
    text = "# REEF-GEPA / CLBench\n\n"
    text += "| 任务 | 状态 | 完整配对数 | 无经验 reward | REEF-GEPA reward | 差值 | 胜/平/负 |\n|---|---|---:|---:|---:|---:|---|\n"
    text += "\n".join(rows)
    text += "\n\n采用 REEF 原生 GEPAProposer、Archive 和 GEPASelectorMixin；适配层使用官方 CLBench 任务执行器和评分器。没有运行 REEF HTTP 服务、部署或权重训练。\n\n"
    text += "模型：本地 Qwen3-4B-Instruct-2507，冻结权重，temperature=0.7，环境 seed=42。每任务 canonical indices 0–2 为候选生成数据，3–5 为验证集；两轮 GEPA 后冻结选出的提示词，在 indices 12–19 上与无经验组逐题配对。两组 actor 采样设置、实例、工具预算一致，GEPA 的额外开发和反思开销单独保留。\n\n"
    text += "这是单种子、小规模的提示词优化试验，不能等同于全部 REEF 方法、在线训练或完整 CLBench 成绩，也不能与其他训练协议的旧表直接混为同一个实验。未完成配对不计作零分。\n\n"
    text += "逐题记录见各任务的 paired_results.json；提示词候选、选择过程见 archive.json、decisions.jsonl、selected_rules.md；全部模型输入输出与官方轨迹见 episodes/。源码快照及哈希位于 source/ 和 source_hashes.json。\n"
    (root / "REPORT.md").write_text(text)
    print(text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    report(parser.parse_args().root)
