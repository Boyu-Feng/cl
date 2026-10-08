"""Exploratory CLBench transfer of the frozen ALFWorld parameter-memory v2.

Uses official CLBench task resets, JSON action schemas and run_task scoring.
Each domain starts empty. A completed own episode writes LoRA factors only when
the official outcome.success is true; no task-specific scoring rule is added.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import traceback
from types import SimpleNamespace

import torch

from ttcl.trajectory_hyperlora.alfworld_online_from_empty import digest, file_hash
from ttcl.trajectory_hyperlora.alfworld_zero_shot_probe import load_agent
from ttcl.trajectory_hyperlora.contextual_alf_source import (
    contextual_text_fields, source_text, task_context_text,
)
from ttcl.trajectory_hyperlora.online_parameter_memory_v2 import ParameterMemory

ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / "current_work/continual-learning-bench"
sys.path.insert(0, str(BENCH))

DOMAINS = ("blind_spectrum_monitoring", "exploitable_poker",
           "database_exploration", "cohort_studies")


def benchmark():
    from ttcl.structured_memory import run_benchmark as base
    if "blind_spectrum_monitoring" not in base.MEMORIES:
        original = base.make_task

        def make_task(name, seed, independent=False):
            if name == "blind_spectrum_monitoring":
                from src.tasks.blind_spectrum_monitoring.task import BlindSpectrumMonitoringTask
                return BlindSpectrumMonitoringTask(seed=seed, schedule="default",
                    response_timeout_seconds=0)
            return original(name, seed, independent)

        base.make_task = make_task
        base.MEMORIES["blind_spectrum_monitoring"] = base.DatabaseMemory
    return base


def bounded_text(tokenizer, value: str, limit: int):
    ids = tokenizer(value, add_special_tokens=False).input_ids
    original = len(ids)
    if original > limit:
        head = limit // 2
        ids = ids[:head] + ids[-(limit - head):]
    kept = tokenizer.decode(ids, skip_special_tokens=False)
    retained = len(tokenizer(kept, add_special_tokens=False).input_ids)
    if retained > limit:
        raise ValueError("Decoded CLBench source exceeds declared budget")
    return kept, original, retained


def public_records(episode):
    if not episode["steps"]:
        return []
    return [{"observation": step["query"],
             "action": json.dumps(step["action"], ensure_ascii=False, sort_keys=True),
             "feedback": step.get("public_feedback", "")}
            for step in episode["steps"]]


def target_text(episode):
    return episode["public_task_brief"] + "\n\n" + episode["initial_public_query"]


def compact_actor_messages(messages, history_turns):
    """Preserve the initial task and latest turns under a fixed actor budget."""
    if (len(messages) < 2 or messages[0]["role"] != "system" or
            messages[1]["role"] != "user" or messages[-1]["role"] != "user" or
            history_turns < 1):
        raise ValueError("Invalid CLBench actor history window")
    if len(messages) == 2:
        return messages
    return [messages[0], messages[1],
            *messages[max(2, len(messages) - (2 * history_turns + 1)):]]


class Actor:
    def __init__(self, args):
        self.args = args
        self.agent, self.tokenizer = load_agent(args.model, args.checkpoint,
            args.device, args.gpu_fraction)
        if (self.agent.encoder_kind != "contextual" or
                not self.agent.task_conditioned or
                self.agent.task_pair_pooling != "mean"):
            raise ValueError("Expected contextual task-conditioned ALFWorld checkpoint")
        self.memory = None
        self.arm = "base"
        self.current_read = {}
        self.active_adapter = False
        self.context_limit = min(args.context_limit,
            self.agent.model.config.max_position_embeddings)

    def _fields(self, text):
        bounded, original, retained = bounded_text(
            self.tokenizer, text, self.args.context_tokens)
        fields = contextual_text_fields(self.agent, self.tokenizer, bounded,
            self.args.device, self.args.context_tokens, pooling="both")
        return fields, original, retained

    def start_episode(self, context):
        self.agent.set_source(None)
        self.active_adapter = False
        self.current_read = {"entries": 0, "weights": [], "cosine": []}
        if self.arm != "online" or self.memory is None or not self.memory.entries:
            return
        fields, _, _ = self._fields(task_context_text(context, context))
        factors, self.current_read = self.memory.read(fields["pair_contextual"])
        if factors is not None:
            for layer, value in zip(self.agent.adapters, factors, strict=True):
                if tuple(value.shape) != (1, layer.base.out_features, layer.rank):
                    raise ValueError("CLBench LoRA factor shape changed")
                layer.b = value.to(self.args.device)
            self.active_adapter = True

    def encode_success(self, episode):
        records = public_records(episode)
        if not records:
            return {"written": False, "reason": "no_public_actions"}
        source, original, retained = bounded_text(self.tokenizer,
            source_text(records), self.args.context_tokens)
        source_fields = contextual_text_fields(self.agent, self.tokenizer,
            source, self.args.device, self.args.context_tokens, pooling="both")
        context = target_text(episode)
        target_fields, _, _ = self._fields(task_context_text(context, context))
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

    def generate(self, messages, random_seed):
        from ttcl.common.local_qwen import check_context
        tokenizer = self.tokenizer
        if self.args.history_turns is not None:
            messages = compact_actor_messages(messages, self.args.history_turns)
        rendered = tokenizer.apply_chat_template(messages, tokenize=False,
                                                  add_generation_prompt=True)
        batch = tokenizer(rendered, add_special_tokens=False,
                          return_tensors="pt", truncation=False)
        count = batch.input_ids.shape[1]
        check_context(count, self.args.action_tokens, self.context_limit)
        devices = [torch.device(self.args.device).index or 0]
        with torch.random.fork_rng(devices=devices), torch.inference_mode():
            torch.manual_seed(random_seed)
            torch.cuda.manual_seed(random_seed)
            output = self.agent.model.generate(**batch.to(self.args.device),
                do_sample=self.args.temperature > 0,
                temperature=self.args.temperature,
                top_p=self.args.top_p,
                top_k=50,
                max_new_tokens=self.args.action_tokens,
                pad_token_id=tokenizer.eos_token_id, use_cache=True)
        tokens = output[0, count:].tolist()
        eos = self.agent.model.generation_config.eos_token_id
        eos = eos if isinstance(eos, list) else [eos]
        return {"raw_response": tokenizer.decode(tokens, skip_special_tokens=True).strip(),
            "input_tokens": count, "output_tokens": len(tokens),
            "context_limit": self.context_limit,
            "finish_reason": "stop" if tokens and tokens[-1] in eos else "length",
            "rendered_prompt_sha256": hashlib.sha256(rendered.encode()).hexdigest(),
            "actual_generation_seed": random_seed,
            "adapter_enabled": self.active_adapter}


def run_episode(args, actor, domain, index, arm, output):
    base = benchmark()
    from ttcl.structured_memory.online_bank import BankSystem
    output.mkdir(parents=True, exist_ok=False)
    task = base.make_task(domain, args.seed, independent=True)
    query = task.reset_baseline_instance(index)
    brief = task.get_agent_brief()
    brief_text = base.format_task_agent_brief(brief) if brief else ""

    class ParameterSystem(BankSystem):
        def respond(self, current):
            if not self.public_schemas:
                actor.start_episode(brief_text + "\n\n" + current.prompt)
            return super().respond(current)

    settings = SimpleNamespace(**vars(args), task=domain,
        num_instances=args.episodes, memory_chars=20000,
        max_turns_per_instance=64, action_retries=2,
        normalize_action=True, allow_initial_experience=True)
    system = ParameterSystem(settings, actor, output, brief_text, "")
    system.requested_canonical_index = index
    recorder = base.Recorder(output, system, 1)
    row = {"domain": domain, "index": index, "arm": arm,
           "instance_id": query.instance_id,
           "initial_query_sha256": digest(query.prompt),
           "status": "failed", "reward": None, "success": None}
    try:
        result = base.run_task(task, system, trace_recorder=recorder,
                               show_progress=False, reset_system=False,
                               initial_query=query)
        if len(result.instance_outcomes) != 1:
            raise ValueError("Expected one official CLBench outcome")
        outcome = result.instance_outcomes[0]
        row.update(status="complete", reward=float(outcome.reward),
                   success=bool(outcome.success))
    except Exception as error:
        row.update(error=repr(error), traceback=traceback.format_exc())
    finally:
        connection = getattr(task, "_conn", None)
        if connection is not None:
            connection.close()
        actor.agent.set_source(None)
    episode = {"public_task_brief": brief_text,
        "initial_public_query": query.prompt,
        "response_schemas": system.public_schemas,
        "steps": system.public_steps,
        "reward": row["reward"], "success": row["success"],
        "completed": row["status"] == "complete"}
    row.update(actor_calls=system.calls, actor_input_tokens=system.input_tokens,
        actor_output_tokens=system.output_tokens,
        read=copy.deepcopy(actor.current_read),
        adapter_enabled=actor.active_adapter,
        first_prompt_sha256=(json.loads((output / "responses.jsonl").read_text().splitlines()[0])
            ["rendered_prompt_sha256"] if (output / "responses.jsonl").exists() else None))
    base.write_json(output / "trajectory.json", episode)
    base.write_json(output / "row.json", row)
    return row, episode


def prepare(args):
    if args.review.exists():
        raise FileExistsError(args.review)
    base = benchmark()
    os.chdir(BENCH)
    targets = []
    for domain in args.domains:
        for index in range(args.episodes):
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
        "seed": args.seed, "domains": args.domains,
        "episodes": args.episodes, "action_tokens": args.action_tokens,
        "max_turns_per_instance": 64, "action_retries": 2,
        "context_limit": args.context_limit,
        "context_tokens": args.context_tokens,
        "temperature": args.temperature, "top_p": args.top_p,
        "capacity": args.capacity, "read_temperature": args.read_temperature,
        "attention_mix": args.attention_mix}
    if args.history_turns is not None:
        binding["history_turns"] = args.history_turns
    review = {"protocol": "Frozen official CLBench target/action interface for zero-training ALFWorld-hypernetwork transfer",
              "binding": binding, "targets": targets,
              "input_content_sha256": digest({"binding": binding, "targets": targets})}
    args.review.parent.mkdir(parents=True, exist_ok=True)
    args.review.write_text(json.dumps(review, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"targets": len(targets)}), flush=True)


def checked(args):
    value = json.loads(args.review.read_text())
    binding = value["binding"]
    expected = {"checkpoint_sha256": file_hash(args.checkpoint),
        "model_config_sha256": file_hash(args.model / "config.json"),
        "seed": args.seed, "domains": args.domains,
        "episodes": args.episodes, "action_tokens": args.action_tokens,
        "max_turns_per_instance": 64, "action_retries": 2,
        "context_limit": args.context_limit,
        "context_tokens": args.context_tokens,
        "temperature": args.temperature, "top_p": args.top_p,
        "capacity": args.capacity, "read_temperature": args.read_temperature,
        "attention_mix": args.attention_mix}
    if args.history_turns is not None:
        expected["history_turns"] = args.history_turns
    if (binding != expected or
            value["input_content_sha256"] != digest({
                "binding": binding, "targets": value["targets"]})):
        raise ValueError("Changed CLBench v2 input binding")
    return value


def evaluate(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    review = checked(args)
    os.chdir(BENCH)
    actor = Actor(args)
    report = {"protocol": "Official CLBench task/reset/scoring, paired base vs frozen ALFWorld-hypernetwork online parameter memory; each domain starts empty; only own official success writes; no CLBench training",
        "review_sha256": file_hash(args.review),
        "checkpoint_sha256": file_hash(args.checkpoint),
        "domains": args.domains, "episodes_per_domain": args.episodes,
        "history_turns": args.history_turns,
        "rows": [], "failures": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for domain in args.domains:
        actor.memory = ParameterMemory(capacity=args.capacity,
            temperature=args.read_temperature, attention_mix=args.attention_mix)
        for index in range(args.episodes):
            target = next(row for row in review["targets"] if
                          row["domain"] == domain and row["index"] == index)
            before = actor.memory.digest()
            pair = []
            own_episode = None
            for arm in ("base", "online"):
                actor.arm = arm
                directory = args.output.parent / (args.output.stem + "_episodes") / domain / arm / f"episode_{index+1:03}"
                row, episode = run_episode(args, actor, domain, index, arm, directory)
                if (row["instance_id"] != target["instance_id"] or
                        row["initial_query_sha256"] != target["initial_query_sha256"]):
                    raise ValueError("Official CLBench target changed")
                pair.append(row)
                if arm == "online":
                    own_episode = episode
            if (pair[0]["status"] != "complete" or
                    pair[1]["status"] != "complete"):
                report["failures"].append({"domain": domain, "index": index,
                                           "rows": pair})
            if index == 0 and pair[0]["first_prompt_sha256"] != pair[1]["first_prompt_sha256"]:
                raise ValueError("First task prompts differ before any memory write")
            write = {"written": False, "reason": "not_successful_or_incomplete"}
            if pair[1]["status"] == "complete" and pair[1]["success"] and index+1 < args.episodes:
                write = actor.encode_success(own_episode)
            entry = {"domain": domain, "index": index,
                "target": target, "memory_sha256_before": before,
                "base": pair[0], "online": pair[1], "write": write,
                "memory_sha256_after": actor.memory.digest()}
            report["rows"].append(entry)
            args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
            print(json.dumps({"domain": domain, "n": index+1,
                "base": pair[0]["reward"], "online": pair[1]["reward"],
                "write": write["written"]}), flush=True)
    summary = {}
    for domain in args.domains:
        rows = [row for row in report["rows"] if row["domain"] == domain and
                row["base"]["status"] == row["online"]["status"] == "complete"]
        summary[domain] = {"paired": len(rows),
            "base_mean": statistics.mean(row["base"]["reward"] for row in rows) if rows else None,
            "online_mean": statistics.mean(row["online"]["reward"] for row in rows) if rows else None,
            "base_successes": sum(row["base"]["success"] for row in rows),
            "online_successes": sum(row["online"]["success"] for row in rows),
            "writes": sum(row["write"]["written"] for row in rows)}
    report["summary"] = summary
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "evaluate"))
    parser.add_argument("--model", type=Path, default=ROOT /
        "current_work/delta-Mem/model/Qwen3-4B-Instruct-2507")
    parser.add_argument("--checkpoint", type=Path, default=ROOT /
        "results/trajectory_hyperlora/alf_sibling_simple40_onpolicy600_20261006.pt")
    parser.add_argument("--review", type=Path, default=ROOT /
        "data/annotations/clbench_online_parameter_memory_v2_reviewed_20261008.json")
    parser.add_argument("--output", type=Path, default=ROOT /
        "results/trajectory_hyperlora/clbench_online_parameter_memory_v2_20261008.json")
    parser.add_argument("--domains", nargs="+", choices=DOMAINS, default=list(DOMAINS))
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpu-fraction", type=float, default=.85)
    parser.add_argument("--action-tokens", type=int, default=512)
    parser.add_argument("--context-limit", type=int, default=8192)
    parser.add_argument("--context-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=.7)
    parser.add_argument("--top-p", type=float, default=.9)
    parser.add_argument("--capacity", type=int, default=64)
    parser.add_argument("--read-temperature", type=float, default=.05)
    parser.add_argument("--attention-mix", type=float, default=.75)
    parser.add_argument("--history-turns", type=int)
    args = parser.parse_args()
    if (args.episodes < 2 or args.action_tokens < 1 or args.context_tokens < 2 or
            args.history_turns is not None and args.history_turns < 1):
        parser.error("Invalid CLBench evaluation budget")
    (prepare if args.command == "prepare" else evaluate)(args)


if __name__ == "__main__":
    main()
