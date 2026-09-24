"""Run fixed public-experience ablations without updating model parameters."""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import traceback

from ttcl.structured_memory import run_benchmark as base
from ttcl.structured_memory.focused_memory import (
    FOCUSED,
    ProceduralDatabaseMemory,
    bounded,
    encode,
    words,
)
from ttcl.structured_memory.poker_memory import _line


def without_reasoning(action):
    if isinstance(action, dict):
        return {
            k: without_reasoning(v)
            for k, v in action.items()
            if k not in {"thinking", "thought", "reasoning"}
        }
    if isinstance(action, list):
        return [without_reasoning(v) for v in action]
    return action


class AblationSystem(base.StructuredSystem):
    def reset(self):
        super().reset()
        if self.args.variant.startswith("focused"):
            memory_class = (
                ProceduralDatabaseMemory
                if self.args.variant == "focused_procedure"
                else FOCUSED[self.args.task]
            )
            self.memory = memory_class(max_chars=self.args.memory_chars - 4000)
        self.public_steps = {}
        self.finished_cases = {}
        self.aliases = {}

    def respond(self, query):
        response = super().respond(query)
        self.aliases[query.instance_id] = self.public_episode
        return response

    def observe(self, observation, next_query=None):
        if self.last and self.args.variant.startswith("focused"):
            prompt, action, identity = self.last
            # Never store the generated reasoning as factual experience.
            episode = self.public_steps.setdefault(
                identity,
                {
                    "public_episode": self.public_episode,
                    "scope": _line(prompt, "Opponent")
                    if self.args.task == "exploitable_poker"
                    else str(getattr(self.memory, "epoch", 0)),
                    "question": prompt,
                    "steps": [],
                    "complete": False,
                },
            )
            episode["steps"].append(
                {
                    "query": prompt,
                    "action": without_reasoning(action),
                    "public_observation": observation.content,
                }
            )
            episode["complete"] = base.observation_marks_instance_complete(observation)
        super().observe(observation, next_query)

    def receive_scalar(self, identity, reward):
        """Explicit research protocol: only a completed episode's scalar passes."""
        episode = self.public_steps.get(identity)
        if not episode or not episode["complete"] or identity in self.finished_cases:
            return
        case = copy.deepcopy(episode)
        if self.args.variant == "focused_reward":
            case["reward"] = float(reward)
        self.finished_cases[identity] = case
        base.append(self.output / "experience_cases.jsonl", case)

    def experience_context(self, prompt):
        factual = super().experience_context(prompt)
        if self.args.variant == "focused_procedure":
            return factual
        if not self.args.variant.startswith("focused") or not self.finished_cases:
            return factual
        scope = (
            _line(prompt, "Opponent")
            if self.args.task == "exploitable_poker"
            else str(getattr(self.memory, "epoch", 0))
        )
        candidates = [c for c in self.finished_cases.values() if c["scope"] == scope]
        if self.args.task == "database_exploration" and self.memory._drift_notice(
            prompt
        ):
            candidates = []
        target = words(prompt)
        candidates = sorted(
            enumerate(candidates),
            key=lambda c: (len(target & words(c[1]["question"])), c[0]),
            reverse=True,
        )
        records = []
        for _, case in candidates[:3]:
            steps = case["steps"]
            # Both focused variants expose exactly this same record structure;
            # only the reward field is revealed in the reward variant.
            record = {
                "source_episode": case["public_episode"],
                "scope": case["scope"],
                "question_excerpt": case["question"][:650],
                "actions": [s["action"] for s in steps],
                "reward": case.get("reward", "withheld_in_this_ablation"),
                "interpretation": "Outcome of the whole episode, not a label for its last action. Different episodes have different difficulty and random outcomes.",
            }
            if len(encode(record)) > 2800:
                record["actions"] = [s["action"] for s in steps[:2] + steps[-2:]]
                record["omitted_middle_actions"] = max(0, len(steps) - 4)
            records.append(record)
        if not records:
            return factual
        cases = bounded(
            {
                "type": "past_episode_cases",
                "reward_protocol": "Completed official scalar, higher is better; no hidden labels or metadata"
                if self.args.variant == "focused_reward"
                else "Scalar withheld",
                "use": "Use cases as scoped hypotheses to re-check, not verified action-value rules. Never compare rewards of different questions as if they were a controlled action comparison.",
            },
            records,
            4000,
        )
        return factual + "\n" + cases

    def get_run_artifacts(self):
        return {
            "memory": self.memory.state_dict(),
            "cases": list(self.finished_cases.values()),
        }


class AblationRecorder(base.Recorder):
    def sync_instance_outcomes(self, outcomes):
        # No outcome.success, raw metrics, canonical identity text, or metadata
        # is passed into the prompt/memory. identity is an internal join key.
        for outcome in outcomes:
            if self.system.args.variant.startswith("focused"):
                self.system.receive_scalar(outcome.instance_id, outcome.reward)
        super().sync_instance_outcomes(outcomes)


def report(root, task, results):
    lines = [
        f"# 经验内容消融：{task}",
        "",
        "| 组别 | 状态 | 完成 | 平均 reward | 相对无经验 | 胜/平/负 | 调用 / 输入 tokens |",
        "|---|---|---:|---:|---:|---|---|",
    ]
    baseline = results.get("independent", {})
    for variant, row in results.items():
        delta, counts = "—", "—"
        if (
            row.get("status") == baseline.get("status") == "complete"
            and variant != "independent"
        ):
            paired = base.comparison({"independent": baseline, "structured": row})
            row["vs_independent"] = paired
            delta = f"{paired['structured_minus_independent']:+.6f}"
            counts = f"{paired['improved']}/{paired['tied']}/{paired['worse']}"
        score = "—" if row.get("mean_score") is None else f"{row['mean_score']:.6f}"
        lines.append(
            f"| {variant} | {row.get('status')} | {row.get('completed_instances', 0)} | {score} | {delta} | {counts} | {row.get('model_calls', 0)} / {row.get('input_tokens', 0)} |"
        )
    if (
        results.get("focused", {}).get("status")
        == results.get("focused_reward", {}).get("status")
        == "complete"
    ):
        paired = base.comparison(
            {"independent": results["focused"], "structured": results["focused_reward"]}
        )
        results["focused_reward"]["vs_focused"] = paired
        lines += [
            "",
            f"reward 附加组相对 focused：{paired['structured_minus_independent']:+.6f}。两组使用相同经验提取/检索/案例结构；仅 reward 可见性不同。后续行动和收集到的证据会随之分化。",
        ]
    lines += [
        "",
        "independent：每题重置；legacy：旧结构化记忆；focused：任务针对性经验与历史案例，标量隐藏；focused_reward：同结构额外读取已完成实例的官方标量。",
        "reward 组属于显式增强反馈协议，尤其 cohort 的官方公开反馈本不包含该分数。未读取隐藏标签、策略、未来数据或评分器 metadata。",
        "所有组参数冻结、每步一次实际行动、相同采样与实例种子；格式修复仅调整包装，最多两次格式重试，全部调用计入成本。无额外候选评分。",
        "上下文上限 65536 tokens，题内历史不截断；经验上限 16000 字符。未完成的组不能拿部分均分与完整组比较。",
        "这是已查看过的前12个实例上的开发消融，单种子；不能据此声称泛化或统计显著。未选择最佳 checkpoint 或删除失败样本。",
    ]
    (root / "RESULT.md").write_text("\n".join(lines) + "\n")
    base.write_json(root / "results.json", results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=FOCUSED, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--num-instances", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["independent", "legacy", "focused", "focused_reward"],
        choices=[
            "independent",
            "legacy",
            "focused",
            "focused_reward",
            "focused_procedure",
        ],
    )
    args = parser.parse_args()
    if "focused_procedure" in args.variants and args.task != "database_exploration":
        parser.error("focused_procedure is a database-only development ablation")
    args.device, args.dtype = "cuda:0", "bfloat16"
    args.temperature, args.top_p, args.top_k = 0.7, 0.9, 0
    args.max_new_tokens, args.context_limit = 4096, 65536
    args.memory_chars, args.max_turns_per_instance = 16000, 64
    args.action_retries, args.normalize_action = 2, True
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=False)
    base.write_json(root / "config.json", {**vars(args), "output_dir": str(root)})
    os.chdir(base.BENCH)
    base.StructuredSystem, base.Recorder = AblationSystem, AblationRecorder
    model = base.LocalQwen(args)
    results = {}
    for variant in args.variants:
        args.variant = variant
        args.output_dir = root / variant
        args.output_dir.mkdir()
        base.write_json(
            root / "status.json",
            {
                "status": "running",
                "variant": variant,
                "completed_variants": list(results),
            },
        )
        mode = "independent" if variant == "independent" else "structured"
        try:
            results[variant] = base.run_mode(args, mode, model)
        except Exception as exc:
            failure = args.output_dir / mode / "failure.json"
            results[variant] = (
                json.loads(failure.read_text())
                if failure.exists()
                else {
                    "status": "failed",
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                }
            )
            print(f"{variant} FAILED: {exc}", flush=True)
        report(root, args.task, results)
    base.write_json(
        root / "status.json",
        {"status": "finished", "completed_variants": list(results)},
    )


if __name__ == "__main__":
    main()
