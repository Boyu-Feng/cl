"""Frozen slot-free LoRA probe with unseen numbers of cue/action relations.

This changes the number of independent relations in each trajectory, rather
than merely renaming the four cues used during training. It remains a synthetic
binary-action task and is an exploratory read-only probe.
"""

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
from ttcl.trajectory_hyperlora.slot_lora_oracle_pilot import EVAL_QUERY


CUES = ("OMEGA", "SIGMA", "THETA", "KAPPA", "LAMBDA")
RULES = {3: (1, 2, 5, 6), 5: (5, 10, 22, 25)}
EVAL_NUMBERS = (151, 164)


def records(rule: int, cue_count: int, seed: int, corrected: bool) -> list[dict]:
    if cue_count not in RULES or rule >= 1 << cue_count:
        raise ValueError("Invalid cue count or rule")
    rng = random.Random(seed)
    numbers = rng.sample(range(1, 90), 3 * cue_count)
    steps = []
    for index, cue in enumerate(CUES[:cue_count]):
        label = "RIGHT" if (rule >> index) & 1 else "LEFT"
        for number in numbers[3 * index:3 * (index + 1)]:
            steps.append({
                "observation": rng.choice((
                    "Archived panel state: {number} ({cue}).",
                    "At measurement {number}, the display carried {cue}."
                )).format(number=number, cue=cue),
                "action": rng.choice((
                    "Selected lever {label}.", "The agent chose {label}."
                )).format(label=label),
                "feedback": "confirmed correct",
            })
    rng.shuffle(steps)
    return add_corrections(steps, test=True) if corrected else steps


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
    relation_bytes = args.relation_checkpoint.read_bytes()
    head_bytes = args.head_checkpoint.read_bytes()
    relation = torch.load(args.relation_checkpoint, map_location="cpu",
                          weights_only=True)
    head = torch.load(args.head_checkpoint, map_location="cpu",
                      weights_only=True)
    relation_sha = hashlib.sha256(relation_bytes).hexdigest()
    if head["relation_checkpoint_sha256"] != relation_sha:
        raise ValueError("Relation/head lineage mismatch")
    agent = PretrainedRelationHyperLoRA(
        base, relation, head["rank"], head["layers"],
        head["latent_width"]).to(args.device)
    params = dict(agent.named_parameters())
    with torch.no_grad():
        for name, value in head["trainable_state"].items():
            if name not in params or params[name].shape != value.shape:
                raise ValueError(f"Head checkpoint mismatch: {name}")
            params[name].copy_(value.to(args.device))
    agent.eval()
    ids = {label: tokenizer(label, add_special_tokens=False).input_ids[0]
           for label in ("LEFT", "RIGHT")}
    rows = []
    with torch.no_grad():
        for cue_count, rules in RULES.items():
            for rule in rules:
                for source_seed in (3001, 3002):
                    seed = 10000 + 31 * rule + source_seed
                    source = records(rule, cue_count, seed, args.corrected)
                    wrong = records(rule ^ ((1 << cue_count) - 1),
                                    cue_count, seed, args.corrected)
                    if any(left["observation"] != right["observation"] or
                           left["feedback"] != right["feedback"]
                           for left, right in zip(source, wrong, strict=True)):
                        raise ValueError("Source swap changed observations or feedback")
                    correct_factors = agent.source_factors((tokenizer, source,
                                                            args.device))
                    wrong_factors = agent.source_factors((tokenizer, wrong,
                                                          args.device))
                    source_sha = hashlib.sha256(json.dumps(
                        source, sort_keys=True).encode()).hexdigest()
                    for index, cue in enumerate(CUES[:cue_count]):
                        expected = "RIGHT" if (rule >> index) & 1 else "LEFT"
                        for number in EVAL_NUMBERS:
                            question = EVAL_QUERY["test"].format(
                                number=number, cue=cue)
                            row = {"cue_count": cue_count, "rule": rule,
                                   "source_seed": source_seed, "cue": cue,
                                   "number": number, "expected": expected,
                                   "source_sha256": source_sha,
                                   "query_sha256": hashlib.sha256(
                                       question.encode()).hexdigest()}
                            for arm, factors in (("base", None),
                                                 ("correct", correct_factors),
                                                 ("wrong", wrong_factors)):
                                agent.mount(factors)
                                logits = first_logits(agent, tokenizer,
                                                      question, args.device)
                                choice = int(logits.argmax(-1)[0])
                                row[arm] = {
                                    "first_token": tokenizer.decode(choice).strip(),
                                    "correct": choice == ids[expected],
                                }
                            agent.mount(None)
                            rows.append(row)
    groups = {}
    for count in RULES:
        items = [row for row in rows if row["cue_count"] == count]
        groups[str(count)] = {
            "n": len(items),
            "success": {arm: sum(row[arm]["correct"] for row in items)
                        for arm in ("base", "correct", "wrong")},
            "source_swap_action_changes": sum(
                row["correct"]["first_token"] != row["wrong"]["first_token"]
                for row in items),
        }
    result = {
        "protocol": "Frozen exploratory cardinality shift from training four cues to three or five; same binary actions and question family; first-token exact scoring",
        "cues": CUES, "rules": RULES, "corrected": args.corrected,
        "relation_checkpoint_sha256": relation_sha,
        "head_checkpoint_sha256": hashlib.sha256(head_bytes).hexdigest(),
        "groups": groups, "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(groups), flush=True)
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
