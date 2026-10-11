"""CLBench paired continual LoRA state plus typed SQL feedback evidence.

Each domain has an independent, chronological source stream written from the
LoRA control's completed public trajectories. Both arms read exactly that
stream before a task; only the hybrid arm composes typed evidence onto the
LoRA-generated action. This is a same-source read comparison, not two separate
on-policy memory processes.
"""

from __future__ import annotations

import argparse
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
from ttcl.trajectory_hyperlora.general_evidence_v22 import (
    compose_structured, digest as evidence_digest, extract_records, verify,
)
from ttcl.trajectory_hyperlora.continual_lora_state_v26 import ContinualLoRAState
from ttcl.trajectory_hyperlora.sql_feedback_evidence_v25 import (
    apply as apply_sql_feedback, missing_identifiers,
)


class ComposedActor(ExistingActor):
    def __init__(self, args):
        super().__init__(args)
        self.compose_enabled = False
        self.sql_gate_model = json.loads(args.sql_gate_model.read_text())
        self.sql_applied_this_episode = False

    def start_episode(self, context):
        super().start_episode(context)
        self.sql_applied_this_episode = False

    def generate(self, messages, random_seed):
        response = Actor.generate(self, messages, random_seed)
        marker = "Return only JSON. Action schema:\n"
        try:
            text = next(item["content"] for item in reversed(messages)
                if item["role"] == "user" and marker in item["content"])
            query, schema_text = text.split(marker, 1)
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
            response = Actor.generate(self, messages, random_seed)
            response["lora_invalid_action_sha256"] = digest(failed)
            response["safety_fallback"] = "frozen_base_for_remaining_task"
            try:
                current = json.loads(response["raw_response"])
                valid = isinstance(current, dict) and \
                    Draft202012Validator(schema).is_valid(current)
            except (ValueError, TypeError, UnboundLocalError):
                valid = False
        if not self.compose_enabled or not valid or not self.evidence:
            return response
        composed, decision = compose_structured(current, self.evidence,
            schema=schema, gate_model=self.gate_model,
            target_index=self.current_index, target_query=query)
        response["composed_evidence"] = decision
        if composed != current:
            response["generated_response_sha256"] = digest(response["raw_response"])
            response["generated_raw_response"] = response["raw_response"]
            response["raw_response"] = json.dumps(composed, ensure_ascii=False,
                                                    sort_keys=True)
        known_bad = set()
        for item in self.evidence:
            known_bad.update(missing_identifiers(item.feedback))
        if self.sql_applied_this_episode:
            sql_action, sql_decision = composed, {
                "decision": "unchanged", "reason": "one_sql_evidence_use_per_episode"}
        else:
            sql_action, sql_decision = apply_sql_feedback(composed, self.evidence,
                self.current_index, known_bad, self.sql_gate_model)
            if sql_action != composed and any(
                    sql_action["content"] in str(message.get("content", ""))
                    for message in messages[2:] if message.get("role") == "assistant"):
                sql_action, sql_decision = composed, {
                    "decision": "unchanged", "reason": "query_already_used_this_episode"}
        if sql_action != composed:
            if not Draft202012Validator(schema).is_valid(sql_action):
                raise ValueError("SQL evidence action violates current schema")
            response["sql_feedback_evidence"] = sql_decision
            response["pre_sql_response_sha256"] = digest(composed)
            response["pre_sql_raw_response"] = json.dumps(composed,
                ensure_ascii=False, sort_keys=True)
            response["raw_response"] = json.dumps(sql_action, ensure_ascii=False,
                                                    sort_keys=True)
            self.sql_applied_this_episode = True
        else:
            response["sql_feedback_evidence"] = sql_decision
        return response


def settings(args):
    return {"method": "same_source_continual_lora_plus_sql_feedback_v26",
        "runner_sha256": file_hash(Path(__file__)),
        "evidence_module_sha256": file_hash(Path(__file__).with_name(
            "general_evidence_v22.py")),
        "model_config_sha256": file_hash(args.model / "config.json"),
        "checkpoint_sha256": file_hash(args.checkpoint),
        "gate_model_sha256": file_hash(args.gate_model),
        "sql_gate_model_sha256": file_hash(args.sql_gate_model),
        "sql_operator_sha256": file_hash(Path(__file__).with_name(
            "sql_feedback_evidence_v25.py")),
        "continual_state_sha256": file_hash(Path(__file__).with_name(
            "continual_lora_state_v26.py")),
        "domain": args.domain, "seed": args.seed,
        "max_targets": args.max_targets,
        "action_tokens": args.action_tokens,
        "context_limit": args.context_limit,
        "context_tokens": args.context_tokens,
        "history_turns": args.history_turns,
        "temperature": args.temperature, "top_p": args.top_p,
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
    if args.max_targets is not None:
        if args.max_targets < 1 or args.max_targets > count:
            raise ValueError("Requested target count outside official domain")
        count = args.max_targets
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
    value = {"protocol": "Reviewed chronological official domain prefix, "
        "index 0 through reviewed last; paired same-source persistent LoRA and evidence; "
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


def evaluate(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    review = checked(args)
    args.episodes = len(review["targets"])
    actor = ComposedActor(args)
    actor.memory = ContinualLoRAState()
    actor.evidence = []
    rewards = []
    report = {"protocol": review["protocol"],
        "review_sha256": file_hash(args.review),
        "binding": review["binding"], "rows": [], "failures": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    for target in review["targets"]:
        index = target["index"]
        actor.current_index = index
        before = actor.memory.digest()
        source_count = len(actor.evidence)
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
        if pair[0]["status"] == "complete":
            reward = float(pair[0]["reward"])
            records = [{"observation": step.get("query", ""),
                        "action": step["action"],
                        "feedback": step.get("public_feedback", "")}
                       for step in source_episode["steps"]]
            if records:
                source_sha = evidence_digest(records)
                added = extract_records(records, source_index=index,
                    source_sha256=source_sha, reward=reward)
                if not all(verify(item, records, source_sha) for item in added):
                    raise ValueError("Changed public evidence provenance")
                actor.evidence.extend(added)
            if (index + 1 < len(review["targets"]) and
                    reward_gate(reward, rewards)):
                write = actor.encode_episode(source_episode)
            rewards.append(reward)
        if any(row["status"] != "complete" for row in pair):
            report["failures"].append({"index": index, "rows": pair})
        report["rows"].append({"index": index, "target": target,
            "lora": pair[0], "lora_evidence": pair[1],
            "source_evidence_before": source_count,
            "source_evidence_after": len(actor.evidence),
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
    parser.add_argument("--gate-model", type=Path, default=ROOT /
        "results/trajectory_hyperlora/clbench_typed_gate_v19_20261008.json")
    parser.add_argument("--sql-gate-model", type=Path, default=ROOT /
        "results/trajectory_hyperlora/clbench_sql_feedback_gate_v25_20261009.json")
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-targets", type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpu-fraction", type=float, default=.85)
    parser.add_argument("--action-tokens", type=int, default=512)
    parser.add_argument("--context-limit", type=int, default=16384)
    parser.add_argument("--context-tokens", type=int, default=2048)
    parser.add_argument("--history-turns", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=.7)
    parser.add_argument("--top-p", type=float, default=.9)
    args = parser.parse_args()
    args.event_only = False
    args.disable_events = False
    for name in ("model", "checkpoint", "gate_model", "sql_gate_model",
                 "review", "output"):
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
