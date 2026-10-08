"""Positive-evidence CLBench parameter memory with action-schema fallback.

Only completed trajectories with positive official reward write a parameter
memory. Invalid LoRA actions fall back to the frozen base for the rest of the
official task. This isolates the effect of rejecting harmful cold-start writes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics

from jsonschema import Draft202012Validator
import torch

from ttcl.trajectory_hyperlora.alfworld_online_from_empty import digest, file_hash
from ttcl.trajectory_hyperlora.clbench_online_parameter_memory_v2 import (
    Actor, BENCH, DOMAINS, ROOT, benchmark, bounded_text, run_episode, target_text,
)
from ttcl.trajectory_hyperlora.contextual_alf_source import (
    contextual_text_fields, source_text, task_context_text,
)
from ttcl.trajectory_hyperlora.online_parameter_memory_v2 import ParameterMemory
from ttcl.trajectory_hyperlora.train_clbench_hyperlora_v1 import clean_records
from ttcl.trajectory_hyperlora.train_clbench_cumulative_hyperlora_v4 import merge_object_arrays


def reward_gate(reward: float, earlier: list[float]) -> bool:
    """Outcome-only admission with a no-write choice and no domain thresholds."""
    return reward > 0 and (not earlier or reward >= statistics.median(earlier))


class TrainedActor(Actor):
    def __init__(self, args):
        super().__init__(args)
        self.records = []
        self.historical_actions = []

    def start_episode(self, context):
        if self.args.event_only and self.arm == "online":
            self.agent.set_source(None)
            self.active_adapter = False
            self.current_read = {"entries": len(self.memory.entries),
                                 "weights": [], "cosine": []}
            return
        super().start_episode(context)

    def generate(self, messages, random_seed):
        previous = self.historical_actions
        if self.args.disable_events:
            self.historical_actions = []
        try:
            response = super().generate(messages, random_seed)
            if self.active_adapter:
                try:
                    action = json.loads(response["raw_response"])
                    valid = isinstance(action, dict)
                    schema_marker = "Return only JSON. Action schema:\n"
                    schema_text = next((item["content"].split(schema_marker, 1)[1]
                        for item in reversed(messages)
                        if item["role"] == "user" and
                        schema_marker in item["content"]), None)
                    if valid and schema_text is not None:
                        valid = Draft202012Validator(
                            json.loads(schema_text)).is_valid(action)
                except (ValueError, TypeError):
                    valid = False
                if not valid:
                    failed = response["raw_response"]
                    self.agent.set_source(None)
                    self.active_adapter = False
                    response = super().generate(messages, random_seed)
                    response["lora_invalid_action_sha256"] = digest(failed)
                    response["lora_invalid_raw_response"] = failed
                    response["safety_fallback"] = "frozen_base_for_remaining_task"
        finally:
            self.historical_actions = previous
        if self.args.disable_events:
            return response
        if self.arm != "online" or not self.historical_actions:
            return response
        try:
            current = json.loads(response["raw_response"])
            if not isinstance(current, dict):
                return response
            merged = merge_object_arrays(current, self.historical_actions)
        except (ValueError, TypeError):
            return response
        if merged != current:
            response["generated_response_sha256"] = digest(response["raw_response"])
            response["generated_raw_response"] = response["raw_response"]
            response["raw_response"] = json.dumps(merged, ensure_ascii=False,
                                                   sort_keys=True)
            response["public_array_items_added"] = sum(
                len(value) - len(current.get(key, []))
                for key, value in merged.items()
                if isinstance(value, list) and isinstance(current.get(key), list))
        return response

    def encode_episode(self, episode):
        records = clean_records(episode)
        if not records:
            return {"written": False, "reason": "no_public_actions"}
        self.records.extend(records)
        source, original, retained = bounded_text(self.tokenizer,
            source_text(self.records), self.args.context_tokens)
        fields = contextual_text_fields(self.agent, self.tokenizer, source,
            self.args.device, self.args.context_tokens, pooling="both")
        context = target_text(episode)
        compact, _, _ = bounded_text(self.tokenizer,
            task_context_text(context, context), self.args.context_tokens)
        target = contextual_text_fields(self.agent, self.tokenizer, compact,
            self.args.device, self.args.context_tokens, pooling="both")
        with torch.no_grad():
            self.agent.set_source(fields, target_fields=target)
            factors = [layer.b.detach().cpu().float().clone()
                       for layer in self.agent.adapters]
            self.agent.set_source(None)
        self.memory.entries.clear()
        slot = self.memory.write(target["pair_contextual"], factors)
        return {"written": True, "slot": slot,
                "source_records_sha256": digest(self.records),
                "source_episodes": len(self.records),
                "source_tokens_original": original,
                "source_tokens_retained": retained,
                "factor_norms": [float(x.norm()) for x in factors]}


def prepare(args):
    if args.review.exists():
        raise FileExistsError(args.review)
    import os
    os.chdir(BENCH)
    base = benchmark()
    targets = []
    for domain in args.domains:
        for index in range(args.start, args.stop):
            task = base.make_task(domain, args.seed, independent=True)
            query = task.reset_baseline_instance(index)
            targets.append({"domain": domain, "index": index,
                "instance_id": query.instance_id,
                "initial_query_sha256": digest(query.prompt)})
            connection = getattr(task, "_conn", None)
            if connection is not None:
                connection.close()
    binding = {"checkpoint_sha256": file_hash(args.checkpoint),
        "model_config_sha256": file_hash(args.model / "config.json"),
        "actor_script_sha256": file_hash(Path(__file__)),
        "domains": args.domains, "seed": args.seed,
        "start": args.start, "stop": args.stop,
        "context_limit": args.context_limit, "context_tokens": args.context_tokens,
        "action_tokens": args.action_tokens, "history_turns": args.history_turns,
        "temperature": args.temperature, "top_p": args.top_p,
        "capacity": args.capacity, "read_temperature": args.read_temperature,
        "attention_mix": args.attention_mix,
        "event_only": args.event_only,
        "disable_events": args.disable_events,
        "max_turns_per_instance": 64, "action_retries": 2}
    if args.max_writes is not None:
        binding["max_writes"] = args.max_writes
    value = {"protocol": "Held-out CLBench positive-evidence parameter memory with frozen-base schema fallback",
             "binding": binding, "targets": targets,
             "input_content_sha256": digest({"binding": binding, "targets": targets})}
    args.review.parent.mkdir(parents=True, exist_ok=True)
    args.review.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"targets": len(targets)}), flush=True)


def evaluate(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    review = json.loads(args.review.read_text())
    expected_binding = {"checkpoint_sha256": file_hash(args.checkpoint),
        "model_config_sha256": file_hash(args.model / "config.json"),
        "actor_script_sha256": file_hash(Path(__file__)),
        "domains": args.domains, "seed": args.seed,
        "start": args.start, "stop": args.stop,
        "context_limit": args.context_limit, "context_tokens": args.context_tokens,
        "action_tokens": args.action_tokens, "history_turns": args.history_turns,
        "temperature": args.temperature, "top_p": args.top_p,
        "capacity": args.capacity, "read_temperature": args.read_temperature,
        "attention_mix": args.attention_mix,
        "event_only": args.event_only,
        "disable_events": args.disable_events,
        "max_turns_per_instance": 64, "action_retries": 2}
    if args.max_writes is not None:
        expected_binding["max_writes"] = args.max_writes
    if review["binding"] != expected_binding or \
       review["input_content_sha256"] != digest({"binding": review["binding"],
                                                   "targets": review["targets"]}):
        raise ValueError("Held-out target binding changed")
    import os
    os.chdir(BENCH)
    actor = TrainedActor(args)
    report = {"protocol": review["protocol"],
              "review_sha256": file_hash(args.review),
              "checkpoint_sha256": file_hash(args.checkpoint),
              "rows": [], "failures": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for domain in args.domains:
        actor.memory = ParameterMemory(capacity=args.capacity,
            temperature=args.read_temperature, attention_mix=args.attention_mix)
        actor.records = []
        actor.historical_actions = []
        rewards = []
        writes_so_far = 0
        for index in range(args.start, args.stop):
            target = next(x for x in review["targets"] if
                          x["domain"] == domain and x["index"] == index)
            before = actor.memory.digest()
            pair = []
            episode = None
            for arm in ("base", "online"):
                actor.arm = arm
                directory = args.output.parent / (args.output.stem + "_episodes") / domain / arm / f"episode_{index+1:03}"
                row, current = run_episode(args, actor, domain, index, arm, directory)
                if row["instance_id"] != target["instance_id"] or \
                   row["initial_query_sha256"] != target["initial_query_sha256"]:
                    raise ValueError("Held-out CLBench target changed")
                pair.append(row)
                if arm == "online":
                    episode = current
            write = {"written": False, "reason": "reward_gate_or_incomplete"}
            if pair[1]["status"] == "complete":
                reward = float(pair[1]["reward"])
                actor.historical_actions.extend(
                    step["action"] for step in episode["steps"]
                    if isinstance(step.get("action"), dict))
                if (not args.event_only and index + 1 < args.stop and
                        reward > 0 and
                        (args.max_writes is None or writes_so_far < args.max_writes)):
                    write = actor.encode_episode(episode)
                    writes_so_far += int(write["written"])
                rewards.append(reward)
            if any(x["status"] != "complete" for x in pair):
                report["failures"].append({"domain": domain, "index": index,
                                           "rows": pair})
            if index == args.start and pair[0]["first_prompt_sha256"] != pair[1]["first_prompt_sha256"]:
                raise ValueError("First held-out prompts differ before a write")
            report["rows"].append({"domain": domain, "index": index,
                "target": target, "base": pair[0], "online": pair[1],
                "write": write, "memory_sha256_before": before,
                "memory_sha256_after": actor.memory.digest()})
            args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
            print(json.dumps({"domain": domain, "index": index,
                "base": pair[0]["reward"], "online": pair[1]["reward"],
                "write": write["written"]}), flush=True)
    report["summary"] = {}
    for domain in args.domains:
        rows = [x for x in report["rows"] if x["domain"] == domain and
                x["base"]["status"] == x["online"]["status"] == "complete"]
        report["summary"][domain] = {"paired": len(rows),
            "base_mean": statistics.mean(x["base"]["reward"] for x in rows) if rows else None,
            "online_mean": statistics.mean(x["online"]["reward"] for x in rows) if rows else None,
            "writes": sum(x["write"]["written"] for x in rows)}
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report["summary"]), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "evaluate"))
    parser.add_argument("--domains", nargs="+", choices=DOMAINS, default=["cohort_studies"])
    parser.add_argument("--start", type=int, default=8)
    parser.add_argument("--stop", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", type=Path, default=ROOT / "current_work/delta-Mem/model/Qwen3-4B-Instruct-2507")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "results/trajectory_hyperlora/clbench_hyperlora_v2_20261008.pt")
    parser.add_argument("--review", type=Path, default=ROOT / "data/annotations/clbench_hyperlora_v1_test_20261008.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results/trajectory_hyperlora/clbench_hyperlora_v1_test_20261008.json")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpu-fraction", type=float, default=.85)
    parser.add_argument("--action-tokens", type=int, default=8192)
    parser.add_argument("--context-limit", type=int, default=32768)
    parser.add_argument("--context-tokens", type=int, default=2048)
    parser.add_argument("--history-turns", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=.7)
    parser.add_argument("--top-p", type=float, default=.9)
    parser.add_argument("--capacity", type=int, default=1)
    parser.add_argument("--read-temperature", type=float, default=.05)
    parser.add_argument("--attention-mix", type=float, default=.75)
    parser.add_argument("--max-writes", type=int)
    parser.add_argument("--event-only", action="store_true")
    parser.add_argument("--disable-events", action="store_true")
    args = parser.parse_args()
    args.episodes = args.stop
    if (args.start < 8 or args.stop <= args.start or args.history_turns < 1 or
            args.max_writes is not None and args.max_writes < 1):
        parser.error("Held-out index range must start after training indices 0-7")
    (prepare if args.command == "prepare" else evaluate)(args)


if __name__ == "__main__":
    main()
