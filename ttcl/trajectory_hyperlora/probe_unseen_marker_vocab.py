"""Read-only probe of a slot-free LoRA on unseen cue words."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ttcl.trajectory_hyperlora.feedback_relation_pretrain import add_corrections
from ttcl.trajectory_hyperlora.general_relation_hyperlora import first_logits
from ttcl.trajectory_hyperlora.pretrained_relation_hyperlora import PretrainedRelationHyperLoRA
from ttcl.trajectory_hyperlora.slot_lora_oracle_pilot import (
    CUES, EVAL_NUMBERS, EVAL_QUERY, TEST_RULES, action, source_records,
)


NEW_CUES = ("OMEGA", "SIGMA", "THETA", "KAPPA")


def replace_cues(value: str) -> str:
    for old, new in zip(CUES, NEW_CUES, strict=True):
        value = value.replace(old, new)
    return value


def changed_source(rule: int, seed: int, corrected: bool) -> list[dict]:
    original = source_records(rule, random.Random(seed), test=True)
    source = [{**row, "observation": replace_cues(row["observation"])}
              for row in original]
    return add_corrections(source, test=True) if corrected else source


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
        raise ValueError("Relation/head lineage mismatch")
    agent = PretrainedRelationHyperLoRA(
        base, relation_checkpoint, head_checkpoint["rank"],
        head_checkpoint["layers"], head_checkpoint["latent_width"]).to(args.device)
    params = dict(agent.named_parameters())
    with torch.no_grad():
        for name, value in head_checkpoint["trainable_state"].items():
            if name not in params or params[name].shape != value.shape:
                raise ValueError(f"Head checkpoint mismatch: {name}")
            params[name].copy_(value.to(args.device))
    agent.eval()
    ids = {label: tokenizer(label, add_special_tokens=False).input_ids[0]
           for label in ("LEFT", "RIGHT")}
    rows = []
    with torch.no_grad():
        for rule in TEST_RULES:
            for source_seed in (3001, 3002):
                seed = source_seed + 31 * rule + 10000
                source = changed_source(rule, seed, args.corrected)
                wrong = changed_source(rule ^ 15, seed, args.corrected)
                if any(left["observation"] != right["observation"] or
                       left["feedback"] != right["feedback"]
                       for left, right in zip(source, wrong, strict=True)):
                    raise ValueError("Source swap changed observation or feedback")
                correct_factors = agent.source_factors((tokenizer, source,
                                                        args.device))
                wrong_factors = agent.source_factors((tokenizer, wrong,
                                                      args.device))
                source_hash = hashlib.sha256(json.dumps(
                    source, sort_keys=True).encode()).hexdigest()
                for cue_index, cue in enumerate(NEW_CUES):
                    for number in EVAL_NUMBERS["test"]:
                        question = EVAL_QUERY["test"].format(number=number, cue=cue)
                        expected = action(rule, cue_index)
                        row = {"rule": rule, "source_seed": source_seed,
                               "cue": cue, "number": number, "expected": expected,
                               "source_sha256": source_hash,
                               "query_sha256": hashlib.sha256(
                                   question.encode()).hexdigest()}
                        for arm, factors in (("base", None),
                                             ("correct", correct_factors),
                                             ("wrong", wrong_factors)):
                            agent.mount(factors)
                            logits = first_logits(agent, tokenizer, question,
                                                  args.device)
                            choice = int(logits.argmax(-1)[0])
                            row[arm] = {"first_token": tokenizer.decode(choice).strip(),
                                        "correct": choice == ids[expected]}
                        agent.mount(None)
                        rows.append(row)
    result = {"protocol": "Read-only OOD cue-vocabulary probe; source and target cue words both replaced with unseen words; same frozen actor, relation encoder and generated LoRA head; no retraining",
              "new_cues": NEW_CUES, "corrected": args.corrected,
              "relation_checkpoint_sha256": hashlib.sha256(
                  args.relation_checkpoint.read_bytes()).hexdigest(),
              "head_checkpoint_sha256": hashlib.sha256(
                  args.head_checkpoint.read_bytes()).hexdigest(),
              "n": len(rows),
              "success": {arm: sum(row[arm]["correct"] for row in rows)
                          for arm in ("base", "correct", "wrong")},
              "source_swap_action_changes": sum(
                  row["correct"]["first_token"] != row["wrong"]["first_token"]
                  for row in rows),
              "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"success": result["success"],
                      "source_swap_action_changes": result[
                          "source_swap_action_changes"]}), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path(
        "current_work/delta-Mem/model/Qwen3-4B-Instruct-2507"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpu-fraction", type=float, default=.35)
    parser.add_argument("--relation-checkpoint", type=Path, required=True)
    parser.add_argument("--head-checkpoint", type=Path, required=True)
    parser.add_argument("--corrected", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
