"""Learn feedback-weighted relation memory from future-task actions only.

Training mixes clean successes with rejected-opposite-action then correction
histories. The feedback gate is never given step validity labels; it is trained
through next-query action loss. There are no named cue parameter slots.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random

import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from ttcl.trajectory_hyperlora.generic_relation_pretrain import RelationMemory
from ttcl.trajectory_hyperlora.slot_lora_oracle_pilot import (
    CUES, DEV_RULES, EVAL_NUMBERS, EVAL_QUERY, TEST_RULES,
    TRAIN_QUERY, TRAIN_RULES, action, source_records,
)


def add_corrections(clean: list[dict], *, test: bool) -> list[dict]:
    output = []
    for row in clean:
        opposite = row["action"].replace("LEFT", "__SWAP__").replace(
            "RIGHT", "LEFT").replace("__SWAP__", "RIGHT")
        if opposite == row["action"]:
            raise ValueError("Source action lacks a reversible action word")
        output.append({"observation": row["observation"],
                       "action": opposite,
                       "feedback": "invalid attempt" if test else "rejected"})
        output.append(row)
    return output


class FeedbackRelationMemory(RelationMemory):
    def __init__(self, embedding: nn.Embedding, width: int = 32) -> None:
        super().__init__(embedding, width)
        size = embedding.embedding_dim
        self.feedback = nn.Sequential(nn.LayerNorm(size),
                                      nn.Linear(size, width), nn.Tanh(),
                                      nn.Linear(width, 1))

    def relation(self, tokenizer, source: list[dict], device: str) -> torch.Tensor:
        observation = self.observation(self.field(
            tokenizer, [row["observation"] for row in source], device))
        action_vector = self.action(self.field(
            tokenizer, [row["action"] for row in source], device))
        feedback = self.feedback(self.field(
            tokenizer, [row["feedback"] for row in source], device)).sigmoid()
        weighted = torch.einsum("s,si,sj->ij", feedback.squeeze(-1),
                                observation, action_vector)
        return weighted / feedback.sum().clamp_min(1e-6)


def evaluate(memory: FeedbackRelationMemory, tokenizer, device: str,
             split: str, corrected: bool,
             feedback_variant: str = "standard") -> dict:
    rows = []
    memory.eval()
    with torch.no_grad():
        for rule in DEV_RULES if split == "dev" else TEST_RULES:
            for source_seed in (3001, 3002):
                seed = source_seed + 31 * rule + (0 if split == "dev" else 10000)
                source = source_records(rule, random.Random(seed), test=True)
                wrong = source_records(rule ^ 15, random.Random(seed), test=True)
                if corrected:
                    source, wrong = (add_corrections(source, test=True),
                                     add_corrections(wrong, test=True))
                    if feedback_variant == "novel":
                        for episode in (source, wrong):
                            for row in episode:
                                row["feedback"] = (
                                    "approved operation" if row["feedback"] ==
                                    "confirmed correct" else "blocked operation")
                questions = [EVAL_QUERY[split].format(number=number, cue=cue)
                             for cue in CUES for number in EVAL_NUMBERS[split]]
                logits, _ = memory(tokenizer, source, questions, device)
                wrong_logits, _ = memory(tokenizer, wrong, questions, device)
                source_hash = hashlib.sha256(json.dumps(
                    source, sort_keys=True).encode()).hexdigest()
                for cue_index, cue in enumerate(CUES):
                    for number_index, number in enumerate(EVAL_NUMBERS[split]):
                        index = cue_index * len(EVAL_NUMBERS[split]) + number_index
                        expected = int(action(rule, cue_index) == "RIGHT")
                        rows.append({"rule": rule, "source_seed": source_seed,
                                     "cue": cue, "number": number,
                                     "source_sha256": source_hash,
                                     "query_sha256": hashlib.sha256(
                                         questions[index].encode()).hexdigest(),
                                     "expected": expected,
                                     "correct": int(logits[index].argmax()) == expected,
                                     "wrong": int(wrong_logits[index].argmax()) == expected,
                                     "changed": int(logits[index].argmax()) !=
                                                int(wrong_logits[index].argmax())})
    return {"n": len(rows), "correct_source_success": sum(r["correct"] for r in rows),
            "wrong_source_success": sum(r["wrong"] for r in rows),
            "source_swap_action_changes": sum(r["changed"] for r in rows),
            "rows": rows}


def run(args: argparse.Namespace) -> dict:
    if args.output.exists() or args.checkpoint.exists() or args.steps < 1:
        raise ValueError("Fresh outputs and positive steps required")
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    if args.device.startswith("cuda"):
        torch.cuda.set_per_process_memory_fraction(args.gpu_fraction,
                                                   device=args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    actor = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, local_files_only=True,
        attn_implementation="sdpa").to(args.device)
    for parameter in actor.parameters():
        parameter.requires_grad_(False)
    memory = FeedbackRelationMemory(actor.get_input_embeddings(), args.width).to(args.device)
    if args.init_clean_checkpoint:
        clean = torch.load(args.init_clean_checkpoint, map_location="cpu",
                           weights_only=True)
        if clean["width"] != args.width:
            raise ValueError("Clean relation width mismatch")
        parameters = dict(memory.named_parameters())
        with torch.no_grad():
            for name, value in clean["trainable_state"].items():
                if name not in parameters or parameters[name].shape != value.shape:
                    raise ValueError(f"Clean relation checkpoint mismatch: {name}")
                parameters[name].copy_(value.to(args.device))
        if args.freeze_clean_core:
            for name, parameter in memory.named_parameters():
                if not name.startswith("feedback."):
                    parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        [p for p in memory.parameters() if p.requires_grad], lr=args.lr,
        weight_decay=0)
    losses = []
    for step in range(args.steps):
        rule = TRAIN_RULES[step % len(TRAIN_RULES)]
        source = source_records(rule, rng, test=False)
        if rng.random() < args.correction_prob:
            source = add_corrections(source, test=False)
        if args.diverse_feedback:
            for row in source:
                row["feedback"] = rng.choice(
                    ("success", "confirmed correct", "accepted")
                    if row["feedback"] == "success" else
                    ("rejected", "invalid attempt", "incorrect"))
        questions = [rng.choice(TRAIN_QUERY).format(
            number=rng.randrange(90, 120), cue=cue) for cue in CUES]
        targets = torch.tensor([int(action(rule, index) == "RIGHT")
                                for index in range(len(CUES))], device=args.device)
        logits, _ = memory(tokenizer, source, questions, args.device)
        loss = F.cross_entropy(logits, targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in memory.parameters() if p.requires_grad], 1.0)
        optimizer.step()
        losses.append(float(loss.detach()))
        if (step + 1) % 100 == 0:
            print(json.dumps({"step": step + 1,
                              "mean_loss": sum(losses[-100:]) / 100}), flush=True)
    dev = {name: evaluate(memory, tokenizer, args.device, "dev", corrected)
           for name, corrected in (("clean", False), ("corrected", True))}
    test = {name: evaluate(memory, tokenizer, args.device, "test", corrected)
            for name, corrected in (("clean", False), ("corrected", True))}
    if args.diverse_feedback:
        test["novel_feedback"] = evaluate(memory, tokenizer, args.device,
                                           "test", True, "novel")
    result = {"protocol": "No named cue slots; feedback gate and relation memory trained on mixed clean/correction train histories only via future-query action loss; rejected-action validity never supervised directly; odd-parity whole-rule holdout",
              "seed": args.seed, "steps": args.steps, "lr": args.lr,
              "width": args.width, "correction_prob": args.correction_prob,
              "diverse_feedback": args.diverse_feedback,
              "init_clean_checkpoint_sha256": (hashlib.sha256(
                  args.init_clean_checkpoint.read_bytes()).hexdigest()
                  if args.init_clean_checkpoint else None),
              "freeze_clean_core": args.freeze_clean_core,
              "last_100_loss": sum(losses[-100:]) / 100,
              "dev": dev, "test": test}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    torch.save({"trainable_state": {name: p.detach().cpu()
                                   for name, p in memory.named_parameters()
                                   if name != "embedding.weight"},
                "kind": "feedback_relation", "seed": args.seed,
                "width": args.width, "steps": args.steps,
                "init_clean_checkpoint_sha256": result["init_clean_checkpoint_sha256"]},
               args.checkpoint)
    print(json.dumps({split: {name: {key: value for key, value in item.items()
                                       if key != "rows"}
                              for name, item in groups.items()}
                      for split, groups in (("dev", dev), ("test", test))}), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="current_work/delta-Mem/model/Qwen3-4B-Instruct-2507")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpu-fraction", type=float, default=.35)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--lr", type=float, default=.001)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--correction-prob", type=float, default=.5)
    parser.add_argument("--diverse-feedback", action="store_true")
    parser.add_argument("--init-clean-checkpoint", type=Path)
    parser.add_argument("--freeze-clean-core", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    if args.freeze_clean_core and not args.init_clean_checkpoint:
        parser.error("Freezing the clean core requires its checkpoint")
    run(args)


if __name__ == "__main__":
    main()
