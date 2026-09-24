"""Compare independently launched GenericAgent runs with matched evaluation inputs."""
import argparse
import json
from pathlib import Path


def compare(stateful_root, stateless_root, output):
    roots = {"stateful": stateful_root, "stateless": stateless_root}
    configs = {mode: json.loads((root / "config.json").read_text()) for mode, root in roots.items()}
    keys = ("model", "ga_dir", "data_path", "seed", "num_scans", "max_new_tokens", "timeout")
    if any(configs["stateful"][key] != configs["stateless"][key] for key in keys):
        raise ValueError("Mismatched GenericAgent configurations")
    metrics = {mode: json.loads((root / mode / "metrics.json").read_text()) for mode, root in roots.items()}
    rows = {mode: [json.loads(line) for line in (root / mode / "responses.jsonl").read_text().splitlines()]
            for mode, root in roots.items()}
    for mode, records in rows.items():
        metrics[mode]["memory_file_writes"] = sum(
            event["tool"] in ("file_write", "file_patch")
            and "memory" in Path(event.get("args", {}).get("path", "")).parts
            for record in records for event in record["tools"])
    if metrics["stateful"].get("input_transport") != metrics["stateless"].get("input_transport"):
        raise ValueError("Mismatched input transport")
    if [r["instance_id"] for r in rows["stateful"]] != [r["instance_id"] for r in rows["stateless"]]:
        raise ValueError("Mismatched scan order or incomplete run")
    delta = [a["reward"] - b["reward"] for a, b in zip(rows["stateful"], rows["stateless"])]
    result = {"runs": metrics, "mean_delta": sum(delta) / len(delta),
              "wins": sum(d > 0 for d in delta), "ties": sum(d == 0 for d in delta),
              "losses": sum(d < 0 for d in delta)}
    output.mkdir(parents=True, exist_ok=True)
    (output / "comparison.json").write_text(json.dumps(result, indent=2))
    lines = ["# GenericAgent + 本地 Qwen3", "",
             "使用原生 agent loop、工具、会话与记忆机制。没有加入人工历史摘要、reward 记忆或额外反思提示。", "",
             "| 方式 | 扫描数 | 平均 IoU | 无效报告 | 评分次数 | 模型调用 | 工具调用 | 检查点/长期记忆工具调用 | 记忆文件写入请求 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for mode in ("stateless", "stateful"):
        r = metrics[mode]
        lines.append(f"| {mode} | {r['completed']} | {r['mean_score']:.4%} | {r['invalid_reports']} | "
                     f"{r['reward_calls']} | {r['model_calls']} | {r['tool_calls']} | {r['memory_tool_calls']} | {r['memory_file_writes']} |")
    lines += ["", f"连续运行减去逐样本独立运行：**{result['mean_delta'] * 100:+.4f} 个百分点**；"
              f"逐题胜/平/负：{result['wins']}/{result['ties']}/{result['losses']}。", "",
              "- 评分只在外部评估器计算，未额外发送给 Agent。每题只评分一次，内部工具和模型调用预算单列。",
              "- stateful 保留框架自身会话、工作记忆及文件；stateless 每题新进程和工作目录。",
              "- 除专门的记忆工具外，模型也可能通过原生文件工具写 memory 文件；写入请求是否成功需查工具结果。",
              "- 单序列、单采样种子结果不能推出跨任务结论；短测不代表完整 90 条表现。", ""]
    (output / "report.md").write_text("\n".join(lines))
    print("\n".join(lines))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stateful", type=Path, required=True)
    parser.add_argument("--stateless", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    compare(args.stateful, args.stateless, args.output)
