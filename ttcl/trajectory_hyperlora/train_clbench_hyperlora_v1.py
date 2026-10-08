"""CLBench reward-filtered next-episode distillation for a trajectory hyper-LoRA.

Train and development labels are new content-bound records from official
completed episodes. The official scalar rewards select positive next-episode
actions; cross-entropy is a proxy objective, not policy-gradient RL.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

import torch

from ttcl.trajectory_hyperlora.alfworld_online_from_empty import digest, file_hash
from ttcl.trajectory_hyperlora.clbench_online_parameter_memory_v2 import (
    ROOT, bounded_text, compact_actor_messages, target_text,
)
from ttcl.trajectory_hyperlora.clbench_online_parameter_memory_v2 import public_records
from ttcl.trajectory_hyperlora.contextual_alf_source import (
    contextual_text_fields, source_text, task_context_text,
)
from ttcl.trajectory_hyperlora.alfworld_zero_shot_probe import load_agent
from ttcl.trajectory_hyperlora.train_alf_next_task import target_loss

DOMAINS = ("blind_spectrum_monitoring", "exploitable_poker", "database_exploration")
OLD = ROOT / "results/trajectory_hyperlora/clbench_online_parameter_memory_v2_r2_20261008_episodes"
DB = ROOT / "results/trajectory_hyperlora/clbench_online_parameter_memory_v2_db_cohort16k_20261008_episodes"
NEW = ROOT / "results/trajectory_hyperlora/clbench_hyperlora_train_collect_r2_20261008_episodes"


def episode_dir(domain: str, index: int) -> Path:
    if index < 4:
        root = DB if domain == "database_exploration" else OLD
        return root / domain / "base" / f"episode_{index+1:03}"
    return NEW / domain / f"episode_{index+1:03}"


def clean_records(episode: dict) -> list[dict[str, str]]:
    result = []
    for record in public_records(episode):
        action = json.loads(record["action"])
        if isinstance(action, dict):
            action.pop("thinking", None)
        result.append({"observation": record["observation"][-1800:],
                       "action": json.dumps(action, ensure_ascii=False, sort_keys=True),
                       "feedback": record["feedback"][-800:]})
    return result


def load_episode(domain: str, index: int):
    path = episode_dir(domain, index)
    row = json.loads((path / "row.json").read_text())
    episode = json.loads((path / "trajectory.json").read_text())
    events = [json.loads(line) for line in (path / "responses.jsonl").read_text().splitlines()]
    if (row["status"] != "complete" or not episode["completed"] or
            float(row["reward"]) != float(episode["reward"])):
        raise ValueError(f"Incomplete/mismatched official episode: {path}")
    return row, episode, events, path


def build_candidates(*, include_thinking: bool = False):
    labels = []
    provenance = {}
    for domain in DOMAINS:
        for index in range(8):
            row, episode, events, path = load_episode(domain, index)
            provenance[f"{domain}:{index}"] = {
                "row_sha256": file_hash(path / "row.json"),
                "trajectory_sha256": file_hash(path / "trajectory.json"),
                "responses_sha256": file_hash(path / "responses.jsonl"),
                "reward": row["reward"], "success": row["success"]}
            if index == 0:
                continue
            if float(row["reward"]) <= 0:
                continue
            source_row, source, _, source_path = load_episode(domain, index - 1)
            if source_row["status"] != "complete" or not source["steps"]:
                continue
            approved_events = [event for event in events if
                               event.get("action") is not None and
                               event.get("parse_error") is None]
            for event in approved_events[:2]:
                action = event["action"]
                if not isinstance(action, dict):
                    continue
                if not include_thinking:
                    action.pop("thinking", None)
                messages = compact_actor_messages(event["messages"], 2)
                if messages[-1]["role"] != "user":
                    raise ValueError("Invalid reviewed CLBench target prompt")
                content = {"domain": domain, "source_index": index - 1,
                    "target_index": index,
                    "source_trajectory_sha256": file_hash(source_path / "trajectory.json"),
                    "target_trajectory_sha256": file_hash(path / "trajectory.json"),
                    "target_responses_sha256": file_hash(path / "responses.jsonl"),
                    "source_records": clean_records(source),
                    "source_context": target_text(source),
                    "target_context": target_text(episode),
                    "target_messages": messages,
                    "target_action": json.dumps(action, ensure_ascii=False, sort_keys=True),
                    "target_reward": float(row["reward"]),
                    "split": "train" if index <= 5 else "dev"}
                labels.append({**content, "input_content_sha256": digest(content)})
    return {"protocol": "Official CLBench completed own-trajectory source to positive-reward future action; ordered train targets 1-5, dev 6-7, test reserved 8-11",
            "provenance": provenance, "labels": labels,
            "input_content_sha256": digest({"provenance": provenance, "labels": labels})}


def prepare(args, *, include_thinking: bool = False):
    if args.candidates.exists() or args.review.exists():
        raise FileExistsError("Fresh candidate and review paths required")
    result = build_candidates(include_thinking=include_thinking)
    args.candidates.parent.mkdir(parents=True, exist_ok=True)
    args.candidates.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    review = {"candidates_sha256": file_hash(args.candidates),
              "annotations": [{"input_content_sha256": label["input_content_sha256"],
                               "approved": False, "review_basis": "pending"}
                              for label in result["labels"]]}
    args.review.parent.mkdir(parents=True, exist_ok=True)
    args.review.write_text(json.dumps(review, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"labels": len(result["labels"]),
                      "train": sum(x["split"] == "train" for x in result["labels"]),
                      "dev": sum(x["split"] == "dev" for x in result["labels"])}), flush=True)


def checked(args, *, include_thinking: bool = False):
    data = json.loads(args.candidates.read_text())
    if data != build_candidates(include_thinking=include_thinking):
        raise ValueError("Training candidate content or official lineage changed")
    review = json.loads(args.review.read_text())
    if review["candidates_sha256"] != file_hash(args.candidates):
        raise ValueError("Review candidate hash changed")
    annotations = review["annotations"]
    if (len(annotations) != len(data["labels"]) or
            [x["input_content_sha256"] for x in annotations] !=
            [x["input_content_sha256"] for x in data["labels"]] or
            not all(x["approved"] and x["review_basis"] != "pending" for x in annotations)):
        raise ValueError("Training targets require new reviewed content bindings")
    return data


def train(args, *, include_thinking: bool = False, reviewed_data=None):
    if args.checkpoint_out.exists() or args.output.exists():
        raise FileExistsError("Fresh checkpoint and report paths required")
    data = (reviewed_data if reviewed_data is not None else
            checked(args, include_thinking=include_thinking))
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    agent, tokenizer = load_agent(args.model, args.checkpoint, args.device, args.gpu_fraction)
    agent.model.config.use_cache = False
    if agent.encoder_kind != "contextual" or not agent.task_conditioned:
        raise ValueError("Expected contextual task-conditioned warm start")
    for name, parameter in agent.named_parameters():
        parameter.requires_grad_(name.startswith(("contextual_latent.", "task_pair_latent.", "b_heads.")))
    parameters = [p for p in agent.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=0.0)
    prepared = []
    for row in data["labels"]:
        source, _, _ = bounded_text(tokenizer, source_text(row["source_records"]), args.source_tokens)
        source_fields = contextual_text_fields(agent, tokenizer, source, args.device,
                                                 args.source_tokens, pooling="both")
        context, _, _ = bounded_text(tokenizer,
            task_context_text(row["source_context"], row["source_context"]), args.source_tokens)
        target_fields = contextual_text_fields(agent, tokenizer, context, args.device,
                                                 args.source_tokens, pooling="both")
        prompt = tokenizer.apply_chat_template(row["target_messages"], tokenize=False,
                                                add_generation_prompt=True)
        prefix = tokenizer(prompt, add_special_tokens=False).input_ids
        answer = tokenizer(row["target_action"] + tokenizer.eos_token,
                           add_special_tokens=False).input_ids
        if len(prefix) + len(answer) > args.context_limit:
            raise ValueError("Reviewed target exceeds training context budget")
        prepared.append((row, source_fields, target_fields, prefix))
    train_rows = [x for x in prepared if x[0]["split"] == "train"]
    dev_rows = [x for x in prepared if x[0]["split"] == "dev"]
    if not train_rows:
        raise ValueError("Train action labels missing")
    delta_scales = {}
    for domain in {x[0]["domain"] for x in train_rows}:
        values = [abs(float(x[0]["reward_delta"])) for x in train_rows
                  if x[0]["domain"] == domain and "reward_delta" in x[0]]
        if values:
            delta_scales[domain] = max(1e-6, sum(values) / len(values))
    def score(rows):
        agent.eval()
        values = []
        for row, source, target, prefix in rows:
            with torch.no_grad():
                agent.set_source(None)
                base = float(target_loss(agent, tokenizer, row, prefix, args.device))
                agent.set_source(source, target_fields=target)
                memory = float(target_loss(agent, tokenizer, row, prefix, args.device))
                agent.set_source(None)
            values.append({"domain": row["domain"], "hash": row["input_content_sha256"],
                           "base_ce": base, "lora_ce": memory})
        return values
    before = score(dev_rows)
    losses = []
    for step in range(args.steps):
        row, source, target, prefix = rng.choice(train_rows)
        agent.train()
        optimizer.zero_grad(set_to_none=True)
        agent.set_source(source, target_fields=target)
        ce = target_loss(agent, tokenizer, row, prefix, args.device)
        weight = (min(2.0, max(.5, abs(float(row["reward_delta"])) /
                    delta_scales[row["domain"]]))
                  if "reward_delta" in row else 1.0)
        (weight * ce).backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        agent.set_source(None)
        losses.append(float(ce.detach()))
        if (step + 1) % args.log_every == 0:
            print(json.dumps({"step": step + 1,
                              "mean_ce": sum(losses[-args.log_every:]) / args.log_every}), flush=True)
    after = score(dev_rows)
    original = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    state = {name: value.detach().cpu() for name, value in agent.named_parameters()
             if name in original["trainable_state"]}
    original["trainable_state"] = state
    original.update(training_domain=("clbench_counterfactual_first_divergence"
                    if "reward_delta" in train_rows[0][0] else
                    "clbench_reward_filtered_future_action"),
                    source_checkpoint_sha256=file_hash(args.checkpoint),
                    train_candidates_sha256=file_hash(args.candidates),
                    reviewed_annotations_sha256=file_hash(args.review),
                    training_steps=args.steps,
                    schema_conforming_targets=include_thinking)
    args.checkpoint_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(original, args.checkpoint_out)
    report = {"protocol": "CLBench positive-official-reward next-episode action distillation; no policy-gradient RL",
              "train_labels": len(train_rows), "dev_labels": len(dev_rows),
              "checkpoint_sha256": file_hash(args.checkpoint_out),
              "candidates_sha256": file_hash(args.candidates),
              "review_sha256": file_hash(args.review),
              "steps": args.steps, "lr": args.lr,
              "dev_before": before, "dev_after": after,
              "last_losses": losses[-10:]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"dev_before_mean_ce": sum(x["lora_ce"] for x in before)/len(before) if before else None,
                      "dev_after_mean_ce": sum(x["lora_ce"] for x in after)/len(after) if after else None}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "train"))
    parser.add_argument("--candidates", type=Path, default=ROOT / "data/annotations/clbench_hyperlora_v1_candidates_20261008.json")
    parser.add_argument("--review", type=Path, default=ROOT / "data/annotations/clbench_hyperlora_v1_reviewed_20261008.json")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "results/trajectory_hyperlora/alf_sibling_simple40_onpolicy600_20261006.pt")
    parser.add_argument("--checkpoint-out", type=Path, default=ROOT / "results/trajectory_hyperlora/clbench_hyperlora_v1_20261008.pt")
    parser.add_argument("--output", type=Path, default=ROOT / "results/trajectory_hyperlora/clbench_hyperlora_v1_train_20261008.json")
    parser.add_argument("--model", type=Path, default=ROOT / "current_work/delta-Mem/model/Qwen3-4B-Instruct-2507")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpu-fraction", type=float, default=.85)
    parser.add_argument("--source-tokens", type=int, default=2048)
    parser.add_argument("--context-limit", type=int, default=16384)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=10)
    args = parser.parse_args()
    (prepare if args.command == "prepare" else train)(args)


if __name__ == "__main__":
    main()
