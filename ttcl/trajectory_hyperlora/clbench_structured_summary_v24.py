"""Full CLBench comparison: shared LoRA versus structured summary evidence.

Each domain has an independent, chronological source stream written from the
LoRA control's completed public trajectories. Both arms read exactly that
stream before a task; a frozen writer first turns completed public trajectories
into fixed-field, source-bound typed records. Only the hybrid actor receives
retrieved records. No evidence-use gate is trained or invoked.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import statistics

from jsonschema import Draft202012Validator

from ttcl.trajectory_hyperlora.alfworld_online_from_empty import (
    digest, file_hash,
)
from ttcl.trajectory_hyperlora.clbench_online_parameter_memory_v2 import (
    Actor, BENCH, DOMAINS, ROOT, benchmark, run_episode,
)
from ttcl.trajectory_hyperlora.clbench_online_typed_gate_v21 import (
    TrainedActor as ExistingActor, reward_gate,
)
from ttcl.trajectory_hyperlora.structured_summary_evidence_v24 import (
    build_summary, digest as evidence_digest, render, select, writer_input,
    writer_messages,
)
from ttcl.trajectory_hyperlora.online_parameter_memory_v2 import ParameterMemory


class ComposedActor(Actor):
    encode_episode = ExistingActor.encode_episode

    def __init__(self, args):
        super().__init__(args)
        self.compose_enabled = False
        self.summaries = []
        self.response_cache = {}

    def generate(self, messages, random_seed):
        marker = "Return only JSON. Action schema:\n"
        text = next((item["content"] for item in reversed(messages)
            if item["role"] == "user" and marker in item["content"]), "")
        query = text.split(marker, 1)[0]
        chosen = (select(self.summaries, target_index=self.current_index,
                         target_query=query) if self.compose_enabled else [])
        evidence_text = render(chosen) if chosen else ""
        injected = deepcopy(messages)
        if evidence_text:
            injected[0]["content"] += "\n\n" + evidence_text
        cache_key = digest([injected, random_seed, self.active_adapter])
        cached = self.response_cache.get(cache_key) if self.compose_enabled else None
        response = deepcopy(cached) if cached is not None else \
            Actor.generate(self, injected, random_seed)
        if cached is not None:
            response["cached_from_lora"] = True
        try:
            schema_text = text.split(marker, 1)[1]
            schema = json.loads(schema_text)
            current = json.loads(response["raw_response"])
            valid = isinstance(current, dict) and \
                Draft202012Validator(schema).is_valid(current)
        except (StopIteration, ValueError, TypeError):
            valid = False
        if not valid and self.active_adapter:
            failed = response["raw_response"]
            self.agent.set_source(None)
            self.active_adapter = False
            response = Actor.generate(self, injected, random_seed)
            response["lora_invalid_action_sha256"] = digest(failed)
            response["safety_fallback"] = "frozen_base_for_remaining_task"
            try:
                current = json.loads(response["raw_response"])
                valid = isinstance(current, dict) and \
                    Draft202012Validator(schema).is_valid(current)
            except (ValueError, TypeError, UnboundLocalError):
                valid = False
        if not self.compose_enabled and "safety_fallback" not in response:
            self.response_cache[cache_key] = deepcopy(response)
        if evidence_text:
            response["typed_evidence_prompt"] = {
                "source_indices": [item["source_index"] for item in chosen],
                "evidence_sha256": evidence_digest(evidence_text),
                "rendered_chars": len(evidence_text)}
        return response


def settings(args):
    return {"method": "same_source_lora_plus_structured_summary_v24",
        "runner_sha256": file_hash(Path(__file__)),
        "evidence_module_sha256": file_hash(Path(__file__).with_name(
            "structured_summary_evidence_v24.py")),
        "model_config_sha256": file_hash(args.model / "config.json"),
        "checkpoint_sha256": file_hash(args.checkpoint),
        "domain": args.domain, "seed": args.seed,
        "action_tokens": args.action_tokens,
        "context_limit": args.context_limit,
        "context_tokens": args.context_tokens,
        "history_turns": args.history_turns,
        "temperature": args.temperature, "top_p": args.top_p,
        "capacity": args.capacity,
        "read_temperature": args.read_temperature,
        "attention_mix": args.attention_mix,
        "writer_temperature": 0.0, "writer_max_tokens": 512,
        "writer_attempts": 2,
        "max_turns_per_instance": 64, "action_retries": 2}


def targets(args):
    os.chdir(BENCH)
    base = benchmark()
    probe = base.make_task(args.domain, args.seed, independent=True)
    count = probe.num_instances
    conn = getattr(probe, "_conn", None)
    if conn is not None:
        conn.close()
    output = []
    for index in range(count):
        task = base.make_task(args.domain, args.seed, independent=True)
        try:
            query = task.reset_baseline_instance(index)
            output.append({"index": index, "instance_id": query.instance_id,
                           "initial_query_sha256": digest(query.prompt)})
        finally:
            conn = getattr(task, "_conn", None)
            if conn is not None:
                conn.close()
    return output


def prepare(args):
    if args.review.exists():
        raise FileExistsError(args.review)
    target_rows = targets(args)
    binding = settings(args)
    value = {"protocol": "Complete current local official domain sequence, "
        "index 0 through last; paired same-source LoRA and model-distilled "
        "fixed-field typed evidence; "
        "prior published evidence gate training exposure remains separate in analysis",
        "binding": binding, "targets": target_rows,
        "input_content_sha256": digest({"binding": binding,
                                        "targets": target_rows})}
    args.review.parent.mkdir(parents=True, exist_ok=True)
    args.review.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"domain": args.domain, "targets": len(target_rows)}), flush=True)


def checked(args):
    review = json.loads(args.review.read_text())
    if (review["binding"] != settings(args) or
            review["input_content_sha256"] != digest({"binding": review["binding"],
                                                         "targets": review["targets"]}) or
            review["targets"] != targets(args)):
        raise ValueError("Changed reviewed domain targets or method")
    return review


def summarize_source(actor, records: list[dict], *, index: int,
                     reward: float, initial_query: str):
    """Call the frozen actor as a bounded JSON writer, retaining failures."""
    value = writer_input(records, source_index=index, reward=reward,
                         initial_query=initial_query)
    messages = writer_messages(value)
    attempts = []
    actor.agent.set_source(None)
    actor.active_adapter = False
    original_tokens, original_temperature = (actor.args.action_tokens,
                                             actor.args.temperature)
    try:
        actor.args.action_tokens = 512
        actor.args.temperature = 0.0
        for attempt in range(2):
            seed = actor.args.seed + 900000 + index * 2 + attempt
            try:
                response = Actor.generate(actor, messages, seed)
            except Exception as error:
                attempts.append({"generation_seed": seed,
                    "status": "generation_error", "error": repr(error)})
                return None, {"status": "generation_error",
                    "writer_input": value, "attempts": attempts}
            raw = response["raw_response"]
            item = {"generation_seed": seed,
                    "writer_response_sha256": evidence_digest(raw),
                    "raw_response": raw,
                    "rendered_prompt_sha256": response["rendered_prompt_sha256"]}
            try:
                summary = build_summary(raw, value,
                    writer_prompt_sha256=response["rendered_prompt_sha256"],
                    writer_response_sha256=evidence_digest(raw))
                item["status"] = "valid"
                attempts.append(item)
                return summary, {"status": "valid",
                    "writer_input": value, "attempts": attempts}
            except (ValueError, TypeError) as error:
                item.update(status="invalid", error=str(error))
                attempts.append(item)
                messages = [*writer_messages(value), {"role": "user",
                    "content": "Previous JSON failed validation: " + str(error) +
                               ". Return corrected JSON only."}]
        return None, {"status": "invalid", "writer_input": value,
                      "attempts": attempts}
    finally:
        actor.args.action_tokens = original_tokens
        actor.args.temperature = original_temperature


def evaluate(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    review = checked(args)
    args.episodes = len(review["targets"])
    actor = ComposedActor(args)
    actor.memory = ParameterMemory(capacity=args.capacity,
        temperature=args.read_temperature, attention_mix=args.attention_mix)
    actor.summaries = []
    rewards = []
    report = {"protocol": review["protocol"],
        "review_sha256": file_hash(args.review),
        "binding": review["binding"], "rows": [], "failures": [],
        "summary_failures": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    for target in review["targets"]:
        index = target["index"]
        actor.response_cache = {}
        actor.current_index = index
        before = actor.memory.digest()
        source_count = len(actor.summaries)
        pair = []
        source_episode = None
        for arm, composed in (("lora", False), ("lora_evidence", True)):
            actor.arm = "online"
            actor.compose_enabled = composed
            directory = (args.output.parent /
                (args.output.stem + "_episodes") / arm /
                f"episode_{index+1:03}")
            row, episode = run_episode(args, actor, args.domain, index,
                                       arm, directory)
            if (row["instance_id"] != target["instance_id"] or
                    row["initial_query_sha256"] != target["initial_query_sha256"]):
                raise ValueError("Changed official target")
            pair.append(row)
            if arm == "lora":
                source_episode = episode
        write = {"written": False, "reason": "reward_gate_or_incomplete"}
        source_summary = None
        writer = {"status": "source_incomplete"}
        if pair[0]["status"] == "complete":
            reward = float(pair[0]["reward"])
            records = [{"observation": step.get("query", ""),
                        "action": step["action"],
                        "feedback": step.get("public_feedback", "")}
                       for step in source_episode["steps"]]
            if records:
                source_summary, writer = summarize_source(actor, records,
                    index=index, reward=reward,
                    initial_query=source_episode["initial_public_query"])
                if source_summary is not None:
                    actor.summaries.append(source_summary)
                else:
                    report["summary_failures"].append({"index": index,
                        "reason": writer["status"]})
            else:
                writer = {"status": "no_public_actions"}
            if (index + 1 < len(review["targets"]) and
                    reward_gate(reward, rewards)):
                write = actor.encode_episode(source_episode)
            rewards.append(reward)
        if any(row["status"] != "complete" for row in pair):
            report["failures"].append({"index": index, "rows": pair})
        report["rows"].append({"index": index, "target": target,
            "lora": pair[0], "lora_evidence": pair[1],
            "source_evidence_before": source_count,
            "source_evidence_after": len(actor.summaries),
            "source_summary": source_summary,
            "writer": writer,
            "memory_sha256_before": before,
            "memory_sha256_after": actor.memory.digest(), "write": write})
        report["summary"] = summary(report["rows"])
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps({"domain": args.domain, "index": index,
            "lora": pair[0]["reward"], "hybrid": pair[1]["reward"],
            "failures": len(report["failures"])}), flush=True)
    return report


def summary(rows):
    paired = [r for r in rows if
              r["lora"]["status"] == r["lora_evidence"]["status"] == "complete"]
    return {"completed_targets": len(rows), "paired": len(paired),
        "lora_mean": statistics.mean(r["lora"]["reward"] for r in paired)
                     if paired else None,
        "lora_evidence_mean": statistics.mean(r["lora_evidence"]["reward"]
                                         for r in paired) if paired else None}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "evaluate"))
    parser.add_argument("--domain", choices=DOMAINS, required=True)
    parser.add_argument("--model", type=Path, default=ROOT /
        "current_work/delta-Mem/model/Qwen3-4B-Instruct-2507")
    parser.add_argument("--checkpoint", type=Path, default=ROOT /
        "results/trajectory_hyperlora/clbench_hyperlora_v2_20261008.pt")
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpu-fraction", type=float, default=.85)
    parser.add_argument("--action-tokens", type=int, default=512)
    parser.add_argument("--context-limit", type=int, default=16384)
    parser.add_argument("--context-tokens", type=int, default=2048)
    parser.add_argument("--history-turns", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=.7)
    parser.add_argument("--top-p", type=float, default=.9)
    parser.add_argument("--capacity", type=int, default=64)
    parser.add_argument("--read-temperature", type=float, default=.05)
    parser.add_argument("--attention-mix", type=float, default=.75)
    args = parser.parse_args()
    args.event_only = False
    args.disable_events = False
    for name in ("model", "checkpoint", "review", "output"):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.resolve())
    if args.command == "evaluate" and args.output is None:
        parser.error("evaluate requires --output")
    if args.domain == "cohort_studies" and \
            (args.action_tokens < 8192 or args.context_limit < 32768):
        parser.error("Cohort requires its reviewed long-output budget")
    (prepare if args.command == "prepare" else evaluate)(args)


if __name__ == "__main__":
    main()
