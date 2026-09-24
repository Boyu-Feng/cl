"""Official CLBench loops with public structured memory and isolated controls."""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
BENCH = Path(
    os.environ.get("TTCL_BENCH", ROOT / "current_work/continual-learning-bench")
)
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(BENCH))

from src.interface import (  # noqa: E402
    ContinualLearningSystem,
    Response,
    format_task_agent_brief,
    observation_marks_instance_complete,
    run_task,
)
from ttcl.common.local_qwen import LocalQwen  # noqa: E402
from ttcl.structured_memory.cohort_memory import CohortMemory  # noqa: E402
from ttcl.structured_memory.codebase_memory import CodebaseMemory  # noqa: E402
from ttcl.structured_memory.database_memory import DatabaseMemory  # noqa: E402
from ttcl.structured_memory.poker_memory import PokerMemory  # noqa: E402
from ttcl.structured_memory.sales_memory import SalesMemory  # noqa: E402

MEMORIES = {
    "database_exploration": DatabaseMemory,
    "cohort_studies": CohortMemory,
    "exploitable_poker": PokerMemory,
    "sales_prediction": SalesMemory,
    "codebase_adaptation": CodebaseMemory,
}


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
    )
    temporary.replace(path)


def append(path, value):
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")


def generation_seed(seed, identity, turn):
    payload = json.dumps([seed, identity, turn], ensure_ascii=False)
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], "big") % (
        2**63
    )


def parse_action(raw, schema):
    """Accept one schema-valid JSON object, optionally fenced or prefaced."""
    decoder = json.JSONDecoder()
    for index, character in enumerate(raw):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(raw[index:])
            return schema.model_validate(value)
        except ValueError:
            continue
    raise ValueError("Model output contains no schema-valid JSON action")


def normalize_action(raw, schema):
    """Opt-in packaging repair only; never invent SQL, values, or decisions."""
    try:
        return parse_action(raw, schema), None
    except ValueError:
        pass
    decoder = json.JSONDecoder()
    for index, character in enumerate(raw):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(raw[index:])
        except ValueError:
            continue
        if not isinstance(value, dict):
            continue
        repaired = copy.deepcopy(value)
        reason = None
        call = repaired.get("tool_call")
        if isinstance(call, dict) and isinstance(call.get("tool_call_params"), dict):
            params = call["tool_call_params"]
            if not (set(params) & (set(call) - {"tool_call_params"})):
                call.pop("tool_call_params")
                call.update(params)
                reason = "flatten_tool_call_params"
        if set(repaired) == {"action"} and isinstance(repaired["action"], str):
            sql = repaired["action"]
            if re.match(
                r"^\s*(?:SELECT\b|WITH\b|PRAGMA\b|\.tables\b|\.schema\b)", sql, re.I
            ):
                repaired = {"action": "QUERY", "content": sql}
                reason = "move_sql_to_content"
        if reason:
            try:
                return schema.model_validate(repaired), reason
            except ValueError:
                pass
    raise ValueError(
        "Model output contains no schema-valid JSON action after packaging repair"
    )


def make_task(name, seed, independent=False):
    if name == "database_exploration":
        from src.tasks.database_exploration.task import DatabaseExploration

        # Local generation latency is logged, not used as a task failure.
        # SQL execution timeouts and the official 15-query budget remain active.
        return DatabaseExploration(
            variant="multi_group", seed=seed, response_timeout_seconds=0
        )
    if name == "cohort_studies":
        from src.tasks.cohort_studies.task import CohortStudiesTask

        return CohortStudiesTask(seed=seed, repeat_instructions=True)
    if name == "exploitable_poker":
        from src.tasks.exploitable_poker.task import Poker

        return Poker(seed=seed, schedule="default")
    if name == "sales_prediction":
        from src.tasks.sales_prediction.task import SalesPredictionTask

        return SalesPredictionTask(
            seed=seed, clean_workspace_between_instances=independent
        )
    if name == "codebase_adaptation":
        from src.tasks.codebase_adaptation.task import CodebaseAdaptationTask

        return CodebaseAdaptationTask(seed=seed, schedule="default")
    raise ValueError(name)


class StructuredSystem(ContinualLearningSystem):
    """Only public text enters model context and the Python memory modules."""

    def __init__(self, args, mode, model, output, brief):
        self.args = args
        self.mode = mode
        self.model = model
        self.output = output
        self.brief = brief
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.max_input_tokens = 0
        self.requested_canonical_index = None
        self.reset()

    @property
    def name(self):
        return "frozen_qwen_python_memory_" + self.mode

    def reset(self):
        self.memory = MEMORIES[self.args.task](max_chars=self.args.memory_chars)
        self.messages = []
        self.identity = None
        self.public_episode = None
        self.episodes_seen = 0
        self.turn = 0
        self.last = None

    def experience_context(self, prompt):
        return self.memory.context(prompt) if self.mode == "structured" else ""

    def respond(self, query):
        # Normalize only display ordinals/briefs: isolated tasks otherwise
        # display 1/1 or Hand #1, confounding a comparison with memory alone.
        canonical_index = self.requested_canonical_index
        if canonical_index is None:
            canonical_index = query.instance_index
        prompt = query.prompt
        if self.brief:
            prompt = prompt.replace(self.brief + "\n\n", "")
        if canonical_index is not None:
            ordinal = canonical_index + 1
            prompt = re.sub(
                r"\bQuestion \d+/\d+",
                f"Question {ordinal}/{self.args.num_instances}",
                prompt,
            )
            prompt = re.sub(
                r"## Study\s+\d+/\d+",
                f"## Study {ordinal}/{self.args.num_instances}",
                prompt,
            )
            prompt = re.sub(r"\bHand #\d+", f"Hand #{ordinal}", prompt)
        if query.instance_id != self.identity:
            self.identity = query.instance_id
            self.episodes_seen += 1
            # Canonical IDs sometimes encode a hidden opponent policy. They
            # stay in logs/seeding; memory provenance receives an opaque alias.
            self.public_episode = f"experience_{self.episodes_seen}"
            self.turn = 0
            context = self.experience_context(prompt)
            instruction = (
                "Solve the current task using its permitted tools. Return exactly one JSON "
                "action matching the schema. Tool results are evidence; your earlier guesses "
                "are not verified facts. Keep each action within the stated budget."
            )
            if self.brief:
                instruction += "\n\n" + self.brief
            self.messages = [{"role": "system", "content": instruction}]
            expose_context = self.episodes_seen > 1 or getattr(
                self.args, "allow_initial_experience", False
            )
            if context and expose_context:
                self.messages.append(
                    {
                        "role": "user",
                        "content": "Structured experience from earlier public interactions:\n"
                        + context,
                    }
                )
            append(
                self.output / "memory_contexts.jsonl",
                {
                    "instance_id": query.instance_id,
                    "memory_context": context if expose_context else "",
                    "prior_episodes": self.episodes_seen - 1,
                    "memory_state": self.memory.state_dict()
                    if self.mode == "structured"
                    else None,
                },
            )
        self.turn += 1
        if self.turn > self.args.max_turns_per_instance:
            raise RuntimeError(
                "Safety cap exceeded; mark run incomplete, do not invent a score"
            )
        self.messages.append(
            {
                "role": "user",
                "content": prompt
                + "\n\nReturn only JSON. Action schema:\n"
                + json.dumps(
                    query.response_schema.model_json_schema(),
                    ensure_ascii=False,
                    sort_keys=True,
                ),
            }
        )
        attempts = 1 + getattr(self.args, "action_retries", 0)
        for attempt in range(attempts):
            turn_seed = (
                self.turn if attempt == 0 else f"{self.turn}:format_retry:{attempt}"
            )
            seed = generation_seed(self.args.seed, query.instance_id, turn_seed)
            start = time.monotonic()
            completion = self.model.generate(copy.deepcopy(self.messages), seed)
            self.calls += 1
            self.input_tokens += completion["input_tokens"]
            self.output_tokens += completion["output_tokens"]
            self.max_input_tokens = max(
                self.max_input_tokens, completion["input_tokens"]
            )
            event = {
                "call": self.calls,
                "instance_id": query.instance_id,
                "turn": self.turn,
                "format_retry": attempt,
                "generation_seed": seed,
                "query": query.prompt,
                "messages": copy.deepcopy(self.messages),
                **completion,
                "latency_seconds": time.monotonic() - start,
            }
            try:
                if getattr(self.args, "normalize_action", False):
                    action, repair = normalize_action(
                        completion["raw_response"], query.response_schema
                    )
                else:
                    action, repair = (
                        parse_action(completion["raw_response"], query.response_schema),
                        None,
                    )
            except ValueError as exc:
                event["parse_error"] = str(exc)
                append(self.output / "responses.jsonl", event)
                if attempt + 1 == attempts:
                    raise
                self.messages.extend(
                    [
                        {"role": "assistant", "content": completion["raw_response"]},
                        {
                            "role": "user",
                            "content": (
                                "Your reply does not match the action schema above. Return one complete JSON "
                                "object with all required fields and the exact nesting shown. Put tool arguments "
                                "alongside the tool field, not inside tool_call_params. Do not change the task "
                                "or invent tool results. This is a formatting retry; no action was executed."
                            ),
                        },
                    ]
                )
                continue
            event["action"] = action.model_dump()
            event["parse_error"] = None
            event["packaging_repair"] = repair
            append(self.output / "responses.jsonl", event)
            break
        self.last = (prompt, action.model_dump(), query.instance_id)
        self.messages.append(
            {
                "role": "assistant",
                "content": json.dumps(action.model_dump())
                if repair
                else completion["raw_response"],
            }
        )
        print(
            f"{self.mode} call={self.calls} id={query.instance_id} turn={self.turn} "
            f"tokens={completion['input_tokens']}+{completion['output_tokens']}",
            flush=True,
        )
        return Response(action=action)

    def observe(self, observation, next_query=None):
        complete = observation_marks_instance_complete(observation)
        if self.last is None:
            return
        query, action, identity = self.last
        if self.mode == "structured":
            self.memory.observe(
                query,
                action,
                observation.content,
                instance_id=self.public_episode,
                instance_complete=complete,
            )
        append(
            self.output / "public_observations.jsonl",
            {
                "instance_id": identity,
                "turn": self.turn,
                "content": observation.content,
                "instance_complete": complete,
            },
        )
        if not complete and observation.content:
            self.messages.append(
                {"role": "user", "content": "Tool feedback:\n" + observation.content}
            )

    def get_run_artifacts(self):
        return {"memory": self.memory.state_dict()}


class Recorder:
    """Log official outcomes separately; never pass them to the system."""

    def __init__(self, output, system, requested):
        self.output = output
        self.system = system
        self.requested = requested
        self.outcomes = {}
        self.start = time.monotonic()

    def record_interaction(self, *args, **kwargs):
        pass  # Public input/action/observation logs are written by the system.

    def record_system_artifacts(self, artifacts):
        write_json(self.output / "memory_final.json", artifacts)

    def sync_instance_outcomes(self, outcomes):
        for item in outcomes:
            self.outcomes[item.instance_id] = asdict(item)
        write_json(self.output / "progress.json", self.metrics())

    def metrics(self):
        values = list(self.outcomes.values())
        return {
            "mode": self.system.mode,
            "requested_instances": self.requested,
            "completed_instances": len(values),
            "mean_score": sum(x["reward"] for x in values) / len(values)
            if values
            else None,
            "model_calls": self.system.calls,
            "parameter_updates": 0,
            "input_tokens": self.system.input_tokens,
            "output_tokens": self.system.output_tokens,
            "max_input_tokens": self.system.max_input_tokens,
            "elapsed_seconds": time.monotonic() - self.start,
            "outcomes": values,
            "evaluator_metadata_in_context": False,
            "within_instance_history_truncated": False,
            "memory_context_pruning": "bounded whole records; extractor flags saved in context",
        }


def run_mode(args, mode, model):
    output = args.output_dir / mode
    output.mkdir()
    task = make_task(args.task, args.seed, independent=mode == "independent")
    brief = task.get_agent_brief()
    system = StructuredSystem(
        args, mode, model, output, format_task_agent_brief(brief) if brief else ""
    )
    recorder = Recorder(output, system, args.num_instances)
    try:
        if mode == "structured":
            task.build_canonical_run_state()
            task.select_run_instances(list(range(args.num_instances)))
            query = task.build_current_query()
            run_task(
                task,
                system,
                trace_recorder=recorder,
                show_progress=False,
                reset_system=False,
                initial_query=query,
            )
        else:
            # Selection can mutate stored deal schedules. Each independent
            # item gets a fresh task object, as in CLBench's baseline worker.
            for index in range(args.num_instances):
                task = make_task(args.task, args.seed, independent=True)
                query = task.reset_baseline_instance(index)
                system.reset()
                system.requested_canonical_index = index
                result = run_task(
                    task,
                    system,
                    trace_recorder=recorder,
                    show_progress=False,
                    reset_system=False,
                    initial_query=query,
                )
                for outcome in result.instance_outcomes:
                    if len(result.instance_outcomes) != 1:
                        raise RuntimeError(
                            "Independent reset produced unexpected outcome count"
                        )
                    recorder.outcomes[outcome.instance_id][
                        "requested_canonical_index"
                    ] = index
        metrics = recorder.metrics()
        if metrics["completed_instances"] != args.num_instances:
            raise RuntimeError(
                "Official outcome count does not match requested instance count"
            )
        metrics["status"] = "complete"
        write_json(output / "metrics.json", metrics)
        return metrics
    except Exception as exc:
        metrics = recorder.metrics()
        metrics.update(
            status="failed", error=repr(exc), traceback=traceback.format_exc()
        )
        write_json(output / "failure.json", metrics)
        raise


def comparison(results):
    left = {x["instance_id"]: x for x in results["independent"]["outcomes"]}
    right = {x["instance_id"]: x for x in results["structured"]["outcomes"]}
    if left.keys() != right.keys():
        raise ValueError("Cannot compare different canonical instance sets")
    pairs = [
        {
            "instance_id": key,
            "independent": left[key]["reward"],
            "structured": right[key]["reward"],
            "difference": right[key]["reward"] - left[key]["reward"],
        }
        for key in left
    ]
    return {
        "paired_instances": len(pairs),
        "structured_minus_independent": sum(x["difference"] for x in pairs)
        / len(pairs),
        "improved": sum(x["difference"] > 1e-12 for x in pairs),
        "worse": sum(x["difference"] < -1e-12 for x in pairs),
        "tied": sum(abs(x["difference"]) <= 1e-12 for x in pairs),
        "pairs": pairs,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=MEMORIES, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--model",
        default=str(ROOT / "current_work/delta-Mem/model/Qwen3-4B-Instruct-2507"),
    )
    parser.add_argument(
        "--mode", choices=["both", "independent", "structured"], default="both"
    )
    parser.add_argument("--num-instances", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--context-limit", type=int, default=32768)
    parser.add_argument("--memory-chars", type=int, default=16000)
    parser.add_argument("--max-turns-per-instance", type=int, default=64)
    args = parser.parse_args(argv)
    if (
        min(
            args.num_instances,
            args.max_new_tokens,
            args.memory_chars,
            args.max_turns_per_instance,
        )
        < 1
    ):
        parser.error("counts and limits must be positive")
    if not math.isfinite(args.temperature) or args.temperature < 0:
        parser.error("temperature must be finite and nonnegative")
    return args


def main():
    args = parse_args()
    args.output_dir = args.output_dir.resolve()
    # Named official variants can specify paths relative to the benchmark.
    # Resolve CLI outputs first, then run the task from its expected directory.
    os.chdir(BENCH)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_json(
        args.output_dir / "config.json",
        vars(args)
        | {
            "output_dir": str(args.output_dir),
            "protocol": "official_task_loop_public_memory",
            "response_wall_timeout_enforced": False,
            "note": "Frozen model, official action budgets/scorers; timing logged separately.",
        },
    )
    model = LocalQwen(args)
    results = {}
    modes = ["independent", "structured"] if args.mode == "both" else [args.mode]
    for mode in modes:
        results[mode] = run_mode(args, mode, model)
    if len(results) == 2:
        results["comparison"] = comparison(results)
    write_json(args.output_dir / "results.json", results)
    rows = [
        "# Python 结构化经验：" + args.task,
        "",
        "| 模式 | 完成样本 | 平均官方 reward | 模型调用 | 输入 tokens |",
        "|---|---:|---:|---:|---:|",
    ]
    for mode in modes:
        row = results[mode]
        rows.append(
            f"| {mode} | {row['completed_instances']} | {row['mean_score']:.6f} | {row['model_calls']} | {row['input_tokens']} |"
        )
    if "comparison" in results:
        delta = results["comparison"]["structured_minus_independent"]
        rows += [
            "",
            f"配对平均差值（structured − independent）：**{delta:+.6f}**。",
            "",
            "这是固定序列前缀、单随机种子的 pilot；差值不能直接推广到全部场景或视为统计显著。",
        ]
    rows += [
        "",
        "只将已发生的公开输入、动作与工具反馈整理成经验；模型参数冻结。",
        "每题均重新开始对话，structured 仅额外读取之前的结构化经验。",
        "工具与提交预算相同，实际调用数由模型决定；完整上下文不静默截断。",
        "本地生成耗时单独记录，未启用按生成延迟判负；SQL 工具超时仍使用官方规则。",
    ]
    (args.output_dir / "RESULT.md").write_text("\n".join(rows) + "\n")
    print(json.dumps(results.get("comparison", {})), flush=True)


if __name__ == "__main__":
    main()
