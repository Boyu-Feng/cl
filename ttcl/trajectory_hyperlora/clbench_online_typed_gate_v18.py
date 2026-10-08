"""Mixed CLBench memory with typed provenance and operator-specific gate.

Retains v10's exact-evidence precedence while guaranteeing that a source
trajectory decoded and retokenized for the hypernetwork fits its budget.
Every generated LoRA action is checked against the current public action
schema; invalid actions fall back to the frozen actor for that task.
The evidence selector was trained only on previously exposed task chains;
test task rewards never train or tune it during evaluation.
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
from ttcl.trajectory_hyperlora.structured_evidence_gate_v16 import (
    candidate, features, predict,
)
from ttcl.trajectory_hyperlora.typed_evidence_v17 import (
    extract, proposals, route,
)
from ttcl.trajectory_hyperlora.clbench_online_parameter_memory_v3 import (
    TrainedActor as SingleSourceActor, reward_gate,
)


def reward_gate(reward: float, earlier: list[float]) -> bool:
    """Outcome-only admission with a no-write choice and no domain thresholds."""
    return reward > 0 and (not earlier or reward >= statistics.median(earlier))


class TrainedActor(Actor):
    def __init__(self, args):
        super().__init__(args)
        self.records = []
        self.historical_actions = []
        self.evidence = []
        self.current_index = None
        self.gate_model = json.loads(args.gate_model.read_text())

    def start_episode(self, context):
        if self.args.event_only and self.arm == "online":
            self.agent.set_source(None)
            self.active_adapter = False
            self.current_read = {"entries": len(self.memory.entries),
                                 "weights": [], "cosine": []}
            return
        super().start_episode(context)

    def generate(self, messages, random_seed):
        if (self.arm == "online" and self.active_adapter and
                not self.args.disable_events and
                self._has_exact_array_evidence(messages)):
            self.agent.set_source(None)
            self.active_adapter = False
        previous = self.historical_actions
        if self.args.disable_events:
            self.historical_actions = []
        try:
            response = super().generate(messages, random_seed)
            if self.active_adapter:
                try:
                    action = json.loads(response["raw_response"])
                    valid = isinstance(action, dict)
                    marker = "Return only JSON. Action schema:\n"
                    schema_text = next((item["content"].split(marker, 1)[1]
                        for item in reversed(messages)
                        if item["role"] == "user" and marker in item["content"]), None)
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
            marker = "Return only JSON. Action schema:\n"
            text = next(item["content"] for item in reversed(messages)
                if item["role"] == "user" and marker in item["content"])
            query, schema_text = text.split(marker, 1)
            schema = json.loads(schema_text)
            properties = schema.get("properties", {})
        except (ValueError, TypeError, StopIteration):
            return response
        scored = []
        for item in self.evidence:
            key = item.schema_path[1:].replace("~1", "/").replace("~0", "~")
            if key not in properties or properties[key].get("type") != "array":
                continue
            vector = features(current, item, target_index=self.current_index,
                              target_query=query)
            if vector is not None:
                scored.append((predict(self.gate_model, vector), item))
        if self.args.evidence_policy == "learned":
            chosen = sorted((x for x in scored if x[0] > 0),
                            key=lambda x: x[0], reverse=True)[:1]
        elif self.args.evidence_policy == "highest_reward":
            chosen = sorted(scored, key=lambda x: x[1].source_reward,
                            reverse=True)[:1]
        else:
            chosen = scored
        merged = current
        for _, item in chosen:
            merged = candidate(merged, item)
        if merged != current and not Draft202012Validator(schema).is_valid(merged):
            response["evidence_rejected"] = "current_action_schema"
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
            response["selected_evidence"] = [{
                "source_index": item.source_index,
                "source_trajectory_sha256": item.source_trajectory_sha256,
                "source_action_sha256": item.source_action_sha256,
                "schema_path": item.schema_path,
                "predicted_utility": value}
                for value, item in chosen]
        return response

    def _has_exact_array_evidence(self, messages):
        marker = "Return only JSON. Action schema:\n"
        try:
            schema = json.loads(next(item["content"].split(marker, 1)[1]
                for item in reversed(messages)
                if item["role"] == "user" and marker in item["content"]))
        except (StopIteration, ValueError, TypeError):
            return False
        properties = schema.get("properties", {})
        return any(
            key in properties and properties[key].get("type") == "array" and
            isinstance(value, list) and value and
            all(isinstance(item, dict) for item in value)
            for prior in self.historical_actions
            for key, value in prior.items())

    def encode_episode(self, episode):
        records = clean_records(episode)
        if not records:
            return {"written": False, "reason": "no_public_actions"}
        source, original, retained = safe_bounded_text(self.tokenizer,
            source_text(records), self.args.context_tokens)
        source_fields = contextual_text_fields(self.agent, self.tokenizer,
            source, self.args.device, self.args.context_tokens, pooling="both")
        context = target_text(episode)
        compact, _, _ = safe_bounded_text(self.tokenizer,
            task_context_text(context, context), self.args.context_tokens)
        target_fields = contextual_text_fields(self.agent, self.tokenizer,
            compact, self.args.device, self.args.context_tokens, pooling="both")
        with torch.no_grad():
            self.agent.set_source(source_fields, target_fields=target_fields)
            factors = [layer.b.detach().cpu().float().clone()
                       for layer in self.agent.adapters]
            self.agent.set_source(None)
        slot = self.memory.write(target_fields["pair_contextual"], factors)
        return {"written": True, "slot": slot,
            "source_tokens_original": original,
            "source_tokens_retained": retained,
            "source_records_sha256": digest(records),
            "factor_norms": [float(value.norm()) for value in factors]}


def safe_bounded_text(tokenizer, value: str, limit: int):
    """Retry with a smaller window if decode/re-encode crosses the limit."""
    budget = limit
    while budget >= 32:
        try:
            text, original, retained = bounded_text(tokenizer, value, budget)
            if retained > limit:
                raise ValueError("Retained source exceeds actor budget")
            return text, original, retained
        except ValueError as error:
            if "Decoded CLBench source exceeds" not in str(error):
                raise
            budget = max(0, int(budget * .9))
    raise ValueError("Unable to bound CLBench source even after shrinking")


class TypedActor(TrainedActor):
    """Use the learned typed gate only for operators with reviewed labels."""

    def generate(self, messages, random_seed):
        if self.arm != "online":
            return Actor.generate(self, messages, random_seed)
        fitted = self.gate_model.get("operators", {})
        marker = "Return only JSON. Action schema:\n"
        try:
            text = next(item["content"] for item in reversed(messages)
                if item["role"] == "user" and marker in item["content"])
            query, schema_text = text.split(marker, 1)
            schema = json.loads(schema_text)
        except (StopIteration, ValueError, TypeError):
            return Actor.generate(self, messages, random_seed)

        # Start with the existing parameter policy. An invalid LoRA action
        # falls back to the frozen actor and remains visible in the trace.
        response = Actor.generate(self, messages, random_seed)
        if self.active_adapter:
            try:
                valid = Draft202012Validator(schema).is_valid(
                    json.loads(response["raw_response"]))
            except (ValueError, TypeError):
                valid = False
            if not valid:
                failed = response["raw_response"]
                self.agent.set_source(None)
                self.active_adapter = False
                response = Actor.generate(self, messages, random_seed)
                response["lora_invalid_action_sha256"] = digest(failed)
                response["lora_invalid_raw_response"] = failed
                response["safety_fallback"] = "frozen_base_for_remaining_task"
        if (self.args.disable_events or not self.evidence or not fitted):
            return response

        # Generate a frozen-policy candidate on the same action prompt. This
        # permits a real gate choice between parameter and evidence actions.
        factors = ([layer.b.detach().clone() for layer in self.agent.adapters]
                   if self.active_adapter else None)
        if factors is not None:
            self.agent.set_source(None)
        base_response = Actor.generate(self, messages, random_seed)
        try:
            try:
                current = json.loads(base_response["raw_response"])
                if not isinstance(current, dict) or not \
                        Draft202012Validator(schema).is_valid(current):
                    raise ValueError("Invalid frozen evidence candidate")
            except (ValueError, TypeError):
                response["typed_gate"] = {"decision": "lora",
                    "reason": "invalid_frozen_candidate"}
                return response
            choices = []
            for evidence in self.evidence:
                if evidence.source_index >= self.current_index:
                    continue
                choices.extend(item for item in proposals(current, evidence, schema)
                    if item.operator in fitted)
            operator, selected, lower = route(choices, current, self.gate_model,
                target_index=self.current_index, target_query=query)
            if selected is None:
                response["typed_gate"] = {"decision": "lora",
                    "considered": len(choices)}
                return response
            if selected.action is None:
                # A text hint needs a task-specific prompt adapter; CLBench
                # JSON actions must not silently reinterpret it as an action.
                response["typed_gate"] = {"decision": "lora",
                    "reason": "text_hint_adapter_unavailable"}
                return response
            base_response["generated_raw_response"] = base_response["raw_response"]
            base_response["generated_response_sha256"] = digest(
                base_response["raw_response"])
            base_response["raw_response"] = json.dumps(selected.action,
                ensure_ascii=False, sort_keys=True)
            base_response["typed_gate"] = {"decision": operator,
                "lower_utility": lower, "considered": len(choices)}
            base_response["selected_evidence"] = [{
                "source_index": selected.evidence.source_index,
                "source_trajectory_sha256": selected.evidence.trajectory_sha256,
                "source_action_sha256": selected.evidence.action_sha256,
                "schema_path": selected.evidence.path,
                "operator": operator}]
            self.active_adapter = False
            return base_response
        finally:
            if self.active_adapter and factors is not None:
                for layer, value in zip(self.agent.adapters, factors, strict=True):
                    layer.b = value


TrainedActor = TypedActor


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
        "gate_model_sha256": file_hash(args.gate_model),
        "evidence_policy": args.evidence_policy,
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
    value = {"protocol": "Held-out CLBench provenance evidence with learned or controlled read policy; LoRA retained",
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
        "gate_model_sha256": file_hash(args.gate_model),
        "evidence_policy": args.evidence_policy,
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
        actor.evidence = []
        rewards = []
        writes_so_far = 0
        for index in range(args.start, args.stop):
            target = next(x for x in review["targets"] if
                          x["domain"] == domain and x["index"] == index)
            actor.current_index = index
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
                online_path = (args.output.parent /
                    (args.output.stem + "_episodes") / domain / "online" /
                    f"episode_{index+1:03}" / "trajectory.json")
                trajectory_hash = file_hash(online_path)
                for step in episode["steps"]:
                    actor.evidence.extend(extract(step["action"], source_index=index,
                        trajectory_sha256=trajectory_hash,
                        feedback=step.get("public_feedback"), reward=reward,
                        query=step.get("query", "")))
                if (not args.event_only and index + 1 < args.stop and
                        reward_gate(reward, rewards) and
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
    parser.add_argument("--domains", nargs="+", choices=DOMAINS, default=list(DOMAINS[:3]))
    parser.add_argument("--start", type=int, default=8)
    parser.add_argument("--stop", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", type=Path, default=ROOT / "current_work/delta-Mem/model/Qwen3-4B-Instruct-2507")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "results/trajectory_hyperlora/clbench_hyperlora_v2_20261008.pt")
    parser.add_argument("--gate-model", type=Path,
        default=ROOT / "results/trajectory_hyperlora/clbench_typed_gate_v17_20261008.json")
    parser.add_argument("--evidence-policy", choices=("learned", "highest_reward", "all"),
                        default="learned")
    parser.add_argument("--review", type=Path, default=ROOT / "data/annotations/clbench_hyperlora_v1_test_20261008.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results/trajectory_hyperlora/clbench_hyperlora_v1_test_20261008.json")
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
