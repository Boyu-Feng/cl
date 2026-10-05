"""Read-only failure probe for rejected actions followed by corrections."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ttcl.trajectory_hyperlora.general_relation_hyperlora import first_logits
from ttcl.trajectory_hyperlora.pretrained_relation_hyperlora import PretrainedRelationHyperLoRA
from ttcl.trajectory_hyperlora.slot_lora_oracle_pilot import (
    CUES, EVAL_NUMBERS, EVAL_QUERY, TEST_RULES, action, source_records,
)


def corrected_records(rule: int, seed: int,
                      feedback_variant: str = "standard") -> list[dict[str, str]]:
    if feedback_variant not in ("standard", "novel"):
        raise ValueError(feedback_variant)
    clean = source_records(rule, random.Random(seed), test=True)
    out = []
    for row in clean:
        rejected_action = row["action"].replace("LEFT", "__ACTION__").replace(
            "RIGHT", "LEFT").replace("__ACTION__", "RIGHT")
        if rejected_action == row["action"]:
            raise ValueError("Could not reverse source action")
        out.append({"observation": row["observation"],
                    "action": rejected_action,
                    "feedback": ("blocked operation" if feedback_variant == "novel"
                                 else "invalid attempt")})
        out.append({**row, "feedback": ("approved operation"
                                        if feedback_variant == "novel"
                                        else row["feedback"])})
    return out


def run(args: argparse.Namespace) -> dict:
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.device.startswith("cuda"):
        torch.cuda.set_per_process_memory_fraction(args.gpu_fraction,
                                                   device=args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    base = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, local_files_only=True,
        attn_implementation="sdpa").to(args.device)
    relation_checkpoint = torch.load(args.relation_checkpoint,
                                     map_location="cpu", weights_only=True)
    head_checkpoint = torch.load(args.head_checkpoint,
                                 map_location="cpu", weights_only=True)
    if head_checkpoint["relation_checkpoint_sha256"] != hashlib.sha256(
            args.relation_checkpoint.read_bytes()).hexdigest():
        raise ValueError("Relation/head checkpoint lineage mismatch")
    agent = PretrainedRelationHyperLoRA(
        base, relation_checkpoint, head_checkpoint["rank"],
        head_checkpoint["layers"], head_checkpoint["latent_width"]).to(args.device)
    parameters = dict(agent.named_parameters())
    with torch.no_grad():
        for name, value in head_checkpoint["trainable_state"].items():
            if name not in parameters or parameters[name].shape != value.shape:
                raise ValueError(f"Checkpoint mismatch: {name}")
            parameters[name].copy_(value.to(args.device))
    agent.eval()
    ids = {label: tokenizer(label, add_special_tokens=False).input_ids[0]
           for label in ("LEFT", "RIGHT")}
    rows = []
    with torch.no_grad():
        for rule in TEST_RULES:
            for source_seed in (3001, 3002):
                seed = source_seed + 31 * rule + 10000
                source = corrected_records(rule, seed, args.feedback_variant)
                wrong = corrected_records(rule ^ 15, seed, args.feedback_variant)
                if any(left["observation"] != right["observation"] or
                       left["feedback"] != right["feedback"]
                       for left, right in zip(source, wrong, strict=True)):
                    raise ValueError("Corrected source swap changed observation/feedback")
                questions = [EVAL_QUERY["test"].format(number=number, cue=cue)
                             for cue in CUES for number in EVAL_NUMBERS["test"]]
                relation_logits, _ = agent.relation(
                    tokenizer, source, questions, args.device)
                wrong_relation_logits, _ = agent.relation(
                    tokenizer, wrong, questions, args.device)
                factors = agent.source_factors((tokenizer, source, args.device))
                wrong_factors = agent.source_factors((tokenizer, wrong, args.device))
                source_hash = hashlib.sha256(json.dumps(
                    source, sort_keys=True).encode()).hexdigest()
                for cue_index, cue in enumerate(CUES):
                    for index, number in enumerate(EVAL_NUMBERS["test"]):
                        position = cue_index * len(EVAL_NUMBERS["test"]) + index
                        question = questions[position]
                        expected = action(rule, cue_index)
                        row = {"rule": rule, "source_seed": source_seed,
                               "cue": cue, "number": number, "expected": expected,
                               "source_sha256": source_hash,
                               "query_sha256": hashlib.sha256(
                                   question.encode()).hexdigest(),
                               "relation_correct": int(relation_logits[position].argmax()) ==
                                                   int(expected == "RIGHT"),
                               "relation_wrong": int(wrong_relation_logits[position].argmax()) ==
                                                 int(expected == "RIGHT")}
                        for arm, b in (("lora_correct", factors),
                                       ("lora_wrong", wrong_factors)):
                            agent.mount(b)
                            logits = first_logits(agent, tokenizer, question, args.device)
                            choice = int(logits.argmax(-1)[0])
                            row[arm] = {"first_token": tokenizer.decode(choice).strip(),
                                        "correct": choice == ids[expected]}
                        agent.mount(None)
                        rows.append(row)
    result = {"protocol": "Read-only correction probe: each successful step preceded by the opposite rejected action at identical observation; test-only, no retraining",
              "feedback_variant": args.feedback_variant,
              "relation_checkpoint_sha256": hashlib.sha256(
                  args.relation_checkpoint.read_bytes()).hexdigest(),
              "head_checkpoint_sha256": hashlib.sha256(
                  args.head_checkpoint.read_bytes()).hexdigest(),
              "n": len(rows),
              "relation_correct": sum(row["relation_correct"] for row in rows),
              "relation_wrong": sum(row["relation_wrong"] for row in rows),
              "lora_correct": sum(row["lora_correct"]["correct"] for row in rows),
              "lora_wrong": sum(row["lora_wrong"]["correct"] for row in rows),
              "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in (
        "n", "relation_correct", "relation_wrong", "lora_correct", "lora_wrong")}),
          flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path(
        "current_work/delta-Mem/model/Qwen3-4B-Instruct-2507"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpu-fraction", type=float, default=.35)
    parser.add_argument("--relation-checkpoint", type=Path, required=True)
    parser.add_argument("--head-checkpoint", type=Path, required=True)
    parser.add_argument("--feedback-variant", choices=("standard", "novel"),
                        default="standard")
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
