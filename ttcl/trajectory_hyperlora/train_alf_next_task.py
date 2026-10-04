"""Train raw-trajectory hyper-LoRA on reviewed ALFWorld train-game pairs.

This is an offline next-task action warm start. Exact action imitation is not
the official ALFWorld reward; a later paired environment rollout is required.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ttcl.experience_evolution.environment import clean_command
from ttcl.trajectory_hyperlora.direct_composition_pilot import DirectRelationHyperLoRA
from ttcl.trajectory_hyperlora.prepare_alf_next_task import validate_review
from ttcl.trajectory_hyperlora.relational_router_pilot import tokenize_records


def compact_messages(messages: list[dict], history_turns: int) -> list[dict]:
    if (len(messages) < 2 or messages[0]["role"] != "system" or
            messages[-1]["role"] != "user"):
        raise ValueError("Invalid reviewed actor prompt")
    tail = messages[max(2, len(messages) - (2 * history_turns + 1)):]
    result = [messages[0], messages[1]]
    if messages[-1] is not messages[1]:
        result.extend(tail)
    return result


def query_ids(tokenizer, row: dict, *, history_turns: int,
              max_prompt_tokens: int) -> list[int]:
    messages = compact_messages(row["target_messages"], history_turns)
    prompt = tokenizer.apply_chat_template(messages, tokenize=False,
                                           add_generation_prompt=True)
    ids = tokenizer(prompt, add_special_tokens=False).input_ids
    if len(ids) > max_prompt_tokens:
        raise ValueError(f"Reviewed prompt exceeds declared token budget: {len(ids)}")
    return ids


def target_loss(agent, tokenizer, row: dict, prefix: list[int],
                device: str) -> torch.Tensor:
    answer = tokenizer(row["target_action"] + tokenizer.eos_token,
                       add_special_tokens=False).input_ids
    ids = torch.tensor([prefix + answer], device=device)
    labels = torch.tensor([[-100] * len(prefix) + answer], device=device)
    return agent.model(input_ids=ids, labels=labels, use_cache=False).loss


def generate(agent, tokenizer, prefix: list[int], device: str,
             max_new_tokens: int) -> str:
    ids = torch.tensor([prefix], device=device)
    with torch.no_grad():
        output = agent.model.generate(input_ids=ids,
                                      attention_mask=torch.ones_like(ids),
                                      do_sample=False,
                                      max_new_tokens=max_new_tokens,
                                      pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(output[0, len(prefix):],
                            skip_special_tokens=True).strip()


def evaluate(agent, tokenizer, rows: list[dict], prefixes: dict[str, list[int]],
             device: str, max_new_tokens: int) -> dict:
    agent.eval()
    output = []
    for row in rows:
        key = row["input_content_sha256"]
        fields = tokenize_records(tokenizer, row["source_records"], device)
        available = row["target_messages"][-1]["content"].split(
            "\nAvailable commands:\n", 1)[-1].splitlines()
        result = {"input_content_sha256": key, "split": row["split"],
                  "family": row["family"], "source_game": row["source_game"],
                  "target_game": row["target_game"],
                  "target_action": row["target_action"]}
        for arm, enabled in (("base", False), ("generated", True)):
            with torch.no_grad():
                agent.set_source(fields if enabled else None)
            answer = generate(agent, tokenizer, prefixes[key], device,
                              max_new_tokens)
            command = clean_command(answer, available)
            result[arm] = {"answer": answer, "command": command,
                           "valid": command in available,
                           "exact_target": command == row["target_action"]}
            agent.set_source(None)
        output.append(result)
    return {"n": len(output),
            "summary": {arm: {
                "exact_target": sum(row[arm]["exact_target"] for row in output),
                "valid": sum(row[arm]["valid"] for row in output)}
                for arm in ("base", "generated")},
            "rows": output}


def run(args: argparse.Namespace) -> dict:
    if args.steps < 1 or args.history_turns < 0 or args.max_prompt_tokens < 1:
        raise ValueError("Invalid training or context budget")
    manifest = json.loads(args.candidates.read_text())
    review = json.loads(args.reviewed_annotations.read_text())
    approved = validate_review(manifest, review)
    train = [row for row in approved if row["split"] == "train"]
    dev = [row for row in approved if row["split"] == "dev"]
    if not train or not dev:
        raise ValueError("Need reviewed train and development targets")
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    if args.device.startswith("cuda"):
        torch.cuda.set_per_process_memory_fraction(args.gpu_fraction,
                                                  device=args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    prefixes = {row["input_content_sha256"]: query_ids(
        tokenizer, row, history_turns=args.history_turns,
        max_prompt_tokens=args.max_prompt_tokens) for row in approved}
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, local_files_only=True,
        attn_implementation="sdpa").to(args.device)
    model.config.use_cache = False
    model.generation_config.temperature = 1.0
    model.generation_config.top_p = 1.0
    model.generation_config.top_k = 50
    agent = DirectRelationHyperLoRA(model, rank=args.rank,
                                    layers=args.layers).to(args.device)
    optimizer = torch.optim.AdamW((p for p in agent.parameters() if p.requires_grad),
                                  lr=args.lr)
    losses = []
    order = list(range(len(train)))
    for step in range(args.steps):
        if step % len(order) == 0:
            rng.shuffle(order)
        row = train[order[step % len(order)]]
        key = row["input_content_sha256"]
        fields = tokenize_records(tokenizer, row["source_records"], args.device)
        agent.set_source(fields)
        loss = target_loss(agent, tokenizer, row, prefixes[key], args.device)
        loss.backward()
        torch.nn.utils.clip_grad_norm_((p for p in agent.parameters()
                                        if p.requires_grad), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        agent.set_source(None)
        losses.append(float(loss.detach()))
        if (step + 1) % 20 == 0:
            print(json.dumps({"step": step + 1,
                              "mean_loss": sum(losses[-20:]) / 20}), flush=True)
    evaluation = evaluate(agent, tokenizer, dev, prefixes, args.device,
                          args.max_new_tokens)
    result = {"protocol": "Exploratory ALFWorld train-game only, reviewed source-to-other-game final-action warm start; exact imitation is not environment reward; no official test games",
              "seed": args.seed, "steps": args.steps,
              "train_targets": len(train), "dev_targets": len(dev),
              "history_turns": args.history_turns,
              "max_prompt_tokens": args.max_prompt_tokens,
              "last_20_loss": sum(losses[-20:]) / min(20, len(losses)),
              "candidate_manifest_sha256": hashlib.sha256(args.candidates.read_bytes()).hexdigest(),
              "reviewed_annotations_sha256": hashlib.sha256(
                  args.reviewed_annotations.read_bytes()).hexdigest(),
              "evaluation": evaluation}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    if args.save_checkpoint:
        args.save_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"trainable_state": {name: param.detach().cpu()
                                       for name, param in agent.named_parameters()
                                       if param.requires_grad},
                    "rank": args.rank, "layers": args.layers,
                    "encoder_kind": "covariance",
                    "relation_bottleneck": "none", "source_encoder": "raw",
                    "training_domain": "alfworld"}, args.save_checkpoint)
    print(json.dumps({"evaluation": evaluation["summary"],
                      "output": str(args.output)}), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="current_work/delta-Mem/model/Qwen3-4B-Instruct-2507")
    parser.add_argument("--candidates", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_next_task_candidates_20261005.json"))
    parser.add_argument("--reviewed-annotations", type=Path, default=Path(
        "data/annotations/trajectory_hyperlora_alf_20261005_reviewed.json"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpu-fraction", type=float, default=.40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=124)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=.0003)
    parser.add_argument("--history-turns", type=int, default=2)
    parser.add_argument("--max-prompt-tokens", type=int, default=1800)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--output", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_next_task_warmstart_20261005/seed42.json"))
    parser.add_argument("--save-checkpoint", type=Path)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
