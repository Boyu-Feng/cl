"""Pairwise source-swap training of a direct trajectory-to-LoRA generator.

This is a synthetic mechanism experiment. The same query is evaluated under
opposite completed source trajectories; only train-policy combinations update
the LoRA factor heads. Held-out combinations and fresh query wording are used
only after training. The actor and trajectory encoder remain frozen.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random

import torch
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from ttcl.trajectory_hyperlora.direct_composition_pilot import (
    CUES, HELDOUT_POLICIES, TRAIN_POLICIES, TRAIN_QUERY_TEMPLATES,
    DirectRelationHyperLoRA,
    action, records,
)
from ttcl.trajectory_hyperlora.pilot import prompt
from ttcl.trajectory_hyperlora.relational_router_pilot import tokenize_records


FRESH_NUMBERS = (151, 152, 163, 164)
FRESH_QUERY = (
    "The new panel reports {number} beside symbol {cue}. From its earlier "
    "behavior, decide which lever to press. Reply LEFT or RIGHT only."
)


def first_logits(agent, tokenizer, question: str, device: str) -> torch.Tensor:
    ids = tokenizer(prompt(tokenizer, question), add_special_tokens=False).input_ids
    return agent.model(input_ids=torch.tensor([ids], device=device),
                       use_cache=False).logits[:, -1, :].float()


def paired_objective(source_logits: torch.Tensor, opposite_logits: torch.Tensor,
                     correct_id: int, opposite_id: int, margin: float,
                     rank_weight: float) -> torch.Tensor:
    """Both directions must work; ranking alone admits constant-answer collapse."""
    correct = torch.tensor([correct_id], device=source_logits.device)
    opposite = torch.tensor([opposite_id], device=source_logits.device)
    source_ce = F.cross_entropy(source_logits, correct)
    opposite_ce = F.cross_entropy(opposite_logits, opposite)
    source_logp = source_logits.log_softmax(-1)[0, correct_id]
    opposite_logp = opposite_logits.log_softmax(-1)[0, correct_id]
    rank = F.softplus(margin - (source_logp - opposite_logp))
    return (source_ce + opposite_ce) / 2 + rank_weight * rank


def mount(agent, b_factors: list[torch.Tensor] | None) -> None:
    for adapter, factor in zip(agent.adapters,
                               b_factors or [None] * len(agent.adapters), strict=True):
        adapter.b = factor


def generated_factors(agent, tokenizer, source: list[dict], device: str
                      ) -> list[torch.Tensor]:
    agent.set_source(tokenize_records(tokenizer, source, device))
    factors = [adapter.b.detach().clone() for adapter in agent.adapters]
    agent.set_source(None)
    return factors


def evaluate(agent, tokenizer, device: str) -> dict:
    agent.eval()
    ids = {label: tokenizer(label, add_special_tokens=False).input_ids[0]
           for label in ("LEFT", "RIGHT")}
    rows = []
    with torch.no_grad():
        public_factors = []
        for policy in TRAIN_POLICIES:
            source, _ = records(policy, random.Random(5000 + policy), test=False)
            public_factors.append(generated_factors(agent, tokenizer, source, device))
        public = [torch.stack([factors[layer] for factors in public_factors]).mean(0)
                  for layer in range(len(agent.adapters))]
        for split, policies in (("seen", (1, 6)), ("unseen", HELDOUT_POLICIES)):
            for policy in policies:
                for source_seed in (3001, 3002):
                    seed = source_seed + 37 * policy
                    source, used = records(policy, random.Random(seed), test=True)
                    wrong, _ = records(policy ^ 7, random.Random(seed), test=True)
                    assert all(n not in used for n in FRESH_NUMBERS)
                    source_factors = generated_factors(agent, tokenizer, source, device)
                    wrong_factors = generated_factors(agent, tokenizer, wrong, device)
                    source_hash = hashlib.sha256(json.dumps(
                        source, sort_keys=True).encode()).hexdigest()
                    wrong_hash = hashlib.sha256(json.dumps(
                        wrong, sort_keys=True).encode()).hexdigest()
                    for cue_index, cue in enumerate(CUES):
                        for number in FRESH_NUMBERS:
                            question = FRESH_QUERY.format(number=number, cue=cue)
                            expected = action(policy, cue_index)
                            row = {"split": split, "policy": policy,
                                   "source_seed": source_seed, "cue": cue,
                                   "number": number, "expected": expected,
                                   "source_sha256": source_hash,
                                   "wrong_source_sha256": wrong_hash,
                                   "query_sha256": hashlib.sha256(
                                       question.encode()).hexdigest()}
                            for arm, factors in (("base", None), ("public", public),
                                                 ("correct", source_factors),
                                                 ("wrong", wrong_factors)):
                                mount(agent, factors)
                                logits = first_logits(agent, tokenizer, question, device)
                                decision = tokenizer.decode(int(logits.argmax(-1)[0])).strip()
                                row[arm] = {"first_token": decision,
                                            "correct": int(logits.argmax(-1)[0]) == ids[expected],
                                            "answer_logprob": float(
                                                logits.log_softmax(-1)[0, ids[expected]])}
                            mount(agent, None)
                            rows.append(row)
    summary = {split: {arm: sum(row[arm]["correct"] for row in rows
                                if row["split"] == split)
                       for arm in ("base", "public", "correct", "wrong")}
               for split in ("seen", "unseen")}
    return {"summary": summary, "rows": rows}


def run(args: argparse.Namespace) -> dict:
    if args.output.exists() or args.checkpoint_output.exists():
        raise FileExistsError("Use fresh output paths for this experiment")
    if args.steps < 0 or args.margin < 0 or args.rank_weight < 0:
        raise ValueError("Invalid training budget or loss coefficients")
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    if args.device.startswith("cuda"):
        torch.cuda.set_per_process_memory_fraction(args.gpu_fraction,
                                                   device=args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    action_ids = {label: tokenizer(label, add_special_tokens=False).input_ids
                  for label in ("LEFT", "RIGHT")}
    if any(len(ids) != 1 for ids in action_ids.values()):
        raise ValueError("Action words must each have one token")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, local_files_only=True,
        attn_implementation="sdpa").to(args.device)
    model.config.use_cache = False
    checkpoint = torch.load(args.initial_checkpoint, map_location="cpu",
                            weights_only=True)
    if checkpoint["source_encoder"] != "raw":
        raise ValueError("Expected learned trajectory encoder")
    agent = DirectRelationHyperLoRA(
        model, rank=checkpoint["rank"], layers=checkpoint["layers"],
        encoder_kind=checkpoint["encoder_kind"],
        relation_bottleneck=args.relation_bottleneck).to(args.device)
    saved = checkpoint["relation_state"] | checkpoint["trainable_state"]
    actual = dict(agent.named_parameters())
    with torch.no_grad():
        for name, tensor in saved.items():
            if name not in actual or actual[name].shape != tensor.shape:
                raise ValueError(f"Checkpoint tensor mismatch: {name}")
            actual[name].copy_(tensor.to(args.device))
    for name, parameter in agent.named_parameters():
        parameter.requires_grad_(name.startswith("b_heads.") or
                                 (args.relation_bottleneck != "none" and
                                  name.startswith("oracle_latent.")))
    optimizer = torch.optim.AdamW(
        [parameter for parameter in agent.parameters() if parameter.requires_grad],
        lr=args.lr, weight_decay=0)
    losses = []
    for step in range(args.steps):
        policy = TRAIN_POLICIES[step % len(TRAIN_POLICIES)]
        cue_index = (step // len(TRAIN_POLICIES)) % len(CUES)
        pair_seed = rng.randrange(2**32)
        source, used = records(policy, random.Random(pair_seed), test=False)
        wrong, wrong_used = records(policy ^ 7, random.Random(pair_seed), test=False)
        if used != wrong_used or any(left["observation"] != right["observation"]
                                     or left["feedback"] != right["feedback"]
                                     for left, right in zip(source, wrong, strict=True)):
            raise ValueError("Counterfactual source differs beyond actions")
        number = rng.choice([n for n in range(1, 90) if n not in used])
        question = rng.choice(TRAIN_QUERY_TEMPLATES).format(
            number=number, cue=CUES[cue_index])
        target = action(policy, cue_index)
        agent.set_source(tokenize_records(tokenizer, source, args.device))
        source_logits = first_logits(agent, tokenizer, question, args.device)
        agent.set_source(tokenize_records(tokenizer, wrong, args.device))
        opposite_logits = first_logits(agent, tokenizer, question, args.device)
        agent.set_source(None)
        loss = paired_objective(
            source_logits, opposite_logits, action_ids[target][0],
            action_ids[action(policy ^ 7, cue_index)][0], args.margin,
            args.rank_weight)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in agent.parameters() if parameter.requires_grad], 1.0)
        optimizer.step()
        losses.append(float(loss.detach()))
        if (step + 1) % 50 == 0:
            print(json.dumps({"step": step + 1,
                              "mean_loss": sum(losses[-50:]) / 50}), flush=True)
    result = evaluate(agent, tokenizer, args.device)
    result.update({"protocol": "Synthetic same-query opposite-source paired loss; Qwen and trajectory encoder frozen; direct B-factor heads tuned on six train rules; two whole combinations held out; fresh source seeds, wording and numbers at evaluation",
                   "seed": args.seed, "steps": args.steps, "lr": args.lr,
                   "relation_bottleneck": args.relation_bottleneck,
                   "margin": args.margin, "rank_weight": args.rank_weight,
                   "initial_checkpoint_sha256": hashlib.sha256(
                       args.initial_checkpoint.read_bytes()).hexdigest(),
                   "last_50_loss": (sum(losses[-50:]) / min(50, len(losses))
                                    if losses else None)})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.checkpoint_output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    torch.save({"head_state": {name: tensor.detach().cpu()
                               for name, tensor in agent.named_parameters()
                               if tensor.requires_grad},
                "relation_bottleneck": args.relation_bottleneck,
                "initial_checkpoint_sha256": result["initial_checkpoint_sha256"]},
               args.checkpoint_output)
    print(json.dumps({"summary": result["summary"],
                      "output": str(args.output)}), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="current_work/delta-Mem/model/Qwen3-4B-Instruct-2507")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpu-fraction", type=float, default=.35)
    parser.add_argument("--initial-checkpoint", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--relation-bottleneck", choices=("none", "soft", "hard"),
                        default="none")
    parser.add_argument("--lr", type=float, default=.0003)
    parser.add_argument("--margin", type=float, default=2.0)
    parser.add_argument("--rank-weight", type=float, default=.3)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint-output", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
