"""Generate a new LoRA from raw steps for unseen combinations of subrules.

Unlike the two-expert router, there is no per-policy LoRA bank. A shared
hypernetwork predicts the B factor of a rank-r update from each completed
trajectory; the A factor is shared and trainable. Entire policy combinations
are held out during training. This is a synthetic mechanism test only.
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

from ttcl.trajectory_hyperlora.pilot import prompt
from ttcl.trajectory_hyperlora.relational_router_pilot import (
    RelationalEncoder, centered_relation, tokenize_records,
)


CUES = ("ALPHA", "BETA", "GAMMA")
TRAIN_POLICIES = (0, 1, 3, 4, 6, 7)
HELDOUT_POLICIES = (2, 5)
TEST_NUMBERS = (101, 102, 113, 114)
TRAIN_OBSERVATION = (
    "Panel reading {number}; marker {cue}.",
    "The sensor shows {number} with label {cue}.",
)
TEST_OBSERVATION = (
    "Archived panel state: {number} ({cue}).",
    "At measurement {number}, the display carried {cue}.",
)
TRAIN_ACTION = ("Agent pressed {action}.", "Command sent: {action}.")
TEST_ACTION = ("Selected lever {action}.", "The agent chose {action}.")
TRAIN_QUERY = (
    "Panel reading {number} has marker {cue}. Under this panel's convention, "
    "which lever should be pressed? Reply with exactly LEFT or RIGHT."
)
TEST_QUERY = (
    "New measurement {number}, labeled {cue}. Choose its correct lever. "
    "Reply with only LEFT or RIGHT."
)


def action(policy: int, cue_index: int) -> str:
    """Private synthetic environment function, never supplied to generator."""
    return "RIGHT" if (policy >> cue_index) & 1 else "LEFT"


def records(policy: int, rng: random.Random, *, test: bool,
            per_cue: int = 4) -> tuple[list[dict[str, str]], set[int]]:
    observations = TEST_OBSERVATION if test else TRAIN_OBSERVATION
    actions = TEST_ACTION if test else TRAIN_ACTION
    steps = []
    chosen_numbers = set()
    for cue_index, cue in enumerate(CUES):
        numbers = rng.sample([n for n in range(1, 90) if n not in chosen_numbers],
                             per_cue)
        chosen_numbers.update(numbers)
        for number in numbers:
            steps.append({
                "observation": rng.choice(observations).format(
                    number=number, cue=cue),
                "action": rng.choice(actions).format(
                    action=action(policy, cue_index)),
                "feedback": "confirmed correct" if test else "success",
            })
    rng.shuffle(steps)
    return steps, chosen_numbers


class GeneratedDownProjection(nn.Module):
    def __init__(self, base: nn.Linear, rank: int) -> None:
        super().__init__()
        self.base = base
        self.rank = rank
        self.a = nn.Parameter(torch.randn(rank, base.in_features) * 0.02)
        self.b: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.base(x)
        if self.b is None:
            return output
        low = F.linear(x.float(), self.a)
        update = torch.einsum("bsr,bor->bso", low, self.b)
        return output + (update / self.rank).to(output.dtype)


class DirectRelationHyperLoRA(nn.Module):
    def __init__(self, model: nn.Module, rank: int = 4,
                 layers: int = 2, width: int = 24) -> None:
        super().__init__()
        self.model = model
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        self.encoder = RelationalEncoder(model.get_input_embeddings(),
                                         width=width, experts=2)
        self.latent = nn.Sequential(nn.LayerNorm(width * width + width),
                                    nn.Linear(width * width + width, 128),
                                    nn.Tanh())
        self.adapters = nn.ModuleList()
        self.b_heads = nn.ModuleList()
        for block in model.model.layers[-layers:]:
            adapter = GeneratedDownProjection(block.mlp.down_proj, rank)
            block.mlp.down_proj = adapter
            self.adapters.append(adapter)
            head = nn.Linear(128, adapter.base.out_features * rank)
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
            self.b_heads.append(head)

    def encode(self, fields: dict) -> torch.Tensor:
        encoder = self.encoder
        observation = encoder.observation(encoder.field_mean(fields["observation"]))
        action_vector = encoder.action(encoder.field_mean(fields["action"]))
        feedback = encoder.feedback(encoder.field_mean(fields["feedback"]))
        covariance = centered_relation(observation, action_vector, feedback)
        action_mean = (action_vector * feedback).mean(0, keepdim=True)
        return self.latent(torch.cat((covariance, action_mean), dim=-1))

    def set_source(self, fields: dict | None) -> None:
        if fields is None:
            for adapter in self.adapters:
                adapter.b = None
            return
        latent = self.encode(fields)
        for adapter, head in zip(self.adapters, self.b_heads, strict=True):
            adapter.b = head(latent).reshape(1, adapter.base.out_features,
                                             adapter.rank)


def target_loss(agent: DirectRelationHyperLoRA, tokenizer: object,
                policy: int, cue_index: int, number: int, device: str,
                *, train: bool) -> torch.Tensor:
    template = TRAIN_QUERY if train else TEST_QUERY
    question = template.format(number=number, cue=CUES[cue_index])
    prefix = tokenizer(prompt(tokenizer, question), add_special_tokens=False).input_ids
    suffix = tokenizer(" " + action(policy, cue_index) + tokenizer.eos_token,
                       add_special_tokens=False).input_ids
    ids = torch.tensor([prefix + suffix], device=device)
    labels = torch.tensor([[-100] * len(prefix) + suffix], device=device)
    return agent.model(input_ids=ids, labels=labels, use_cache=False).loss


def generate(agent: DirectRelationHyperLoRA, tokenizer: object,
             question: str, device: str, history: str | None = None) -> str:
    inputs = tokenizer(prompt(tokenizer, question, history),
                       return_tensors="pt").to(device)
    with torch.no_grad():
        output = agent.model.generate(**inputs, do_sample=False, max_new_tokens=5,
                                      pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(output[0, inputs.input_ids.shape[1]:],
                            skip_special_tokens=True).strip()


def evaluate(agent: DirectRelationHyperLoRA, tokenizer: object,
             device: str) -> dict:
    agent.eval()
    rows = []
    for split, policies in (("seen", (1, 6)), ("unseen", HELDOUT_POLICIES)):
        for policy in policies:
            for source_seed in (2001, 2002):
                source, used_numbers = records(policy,
                                               random.Random(source_seed + 31 * policy),
                                               test=True)
                wrong, _ = records(policy ^ 7,
                                   random.Random(source_seed + 31 * policy),
                                   test=True)
                source_hash = hashlib.sha256(json.dumps(
                    source, sort_keys=True).encode()).hexdigest()
                history = "\n".join(
                    f"Observation: {step['observation']} Action: {step['action']} "
                    f"Feedback: {step['feedback']}" for step in source)
                for cue_index, cue in enumerate(CUES):
                    for number in TEST_NUMBERS:
                        assert number not in used_numbers
                        question = TEST_QUERY.format(number=number, cue=cue)
                        expected = action(policy, cue_index)
                        row = {"split": split, "policy": policy,
                               "source_seed": source_seed,
                               "cue": cue, "number": number,
                               "expected": expected,
                               "source_sha256": source_hash,
                               "query_sha256": hashlib.sha256(
                                   question.encode()).hexdigest()}
                        for arm in ("base", "text", "generated", "wrong_source"):
                            selected = source if arm == "generated" else (
                                wrong if arm == "wrong_source" else None)
                            agent.set_source(tokenize_records(tokenizer, selected, device)
                                             if selected is not None else None)
                            answer = generate(agent, tokenizer, question, device,
                                              history=history if arm == "text" else None)
                            first = answer.upper().split()[0].strip(".,:;!") if answer else ""
                            row[arm] = {"answer": answer, "correct": first == expected}
                        agent.set_source(None)
                        rows.append(row)
    arms = ("base", "text", "generated", "wrong_source")
    return {"n": len(rows),
            "accuracy": {split: {arm: sum(row[arm]["correct"] for row in rows
                                          if row["split"] == split) /
                                 sum(row["split"] == split for row in rows)
                                for arm in arms} for split in ("seen", "unseen")},
            "by_policy": {str(policy): {arm: sum(row[arm]["correct"] for row in rows
                                               if row["policy"] == policy) / 24
                                        for arm in arms}
                          for policy in (*((1, 6)), *HELDOUT_POLICIES)},
            "rows": rows}


def run(args: argparse.Namespace) -> dict:
    if args.steps < 1 or args.per_cue < 1:
        raise ValueError("Invalid training size")
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    if args.device.startswith("cuda"):
        torch.cuda.set_per_process_memory_fraction(args.gpu_fraction,
                                                  device=args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
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
    for step in range(args.steps):
        policy = TRAIN_POLICIES[step % len(TRAIN_POLICIES)]
        cue_index = (step // len(TRAIN_POLICIES)) % len(CUES)
        source, used_numbers = records(policy, rng, test=False,
                                       per_cue=args.per_cue)
        source_fields = tokenize_records(tokenizer, source, args.device)
        queried_cues = range(len(CUES)) if args.balanced_queries else (cue_index,)
        losses_this_step = []
        for index in queried_cues:
            agent.set_source(source_fields)
            number = rng.choice([n for n in range(1, 90) if n not in used_numbers])
            loss = target_loss(agent, tokenizer, policy, index, number,
                               args.device, train=True) / len(queried_cues)
            loss.backward()
            losses_this_step.append(float(loss.detach()))
            agent.set_source(None)
        torch.nn.utils.clip_grad_norm_((p for p in agent.parameters()
                                        if p.requires_grad), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        agent.set_source(None)
        losses.append(sum(losses_this_step))
        if (step + 1) % 100 == 0:
            print(json.dumps({"step": step + 1,
                              "mean_loss": sum(losses[-100:]) / 100}), flush=True)
    result = evaluate(agent, tokenizer, args.device)
    result.update({"seed": args.seed, "steps": args.steps,
                   "rank": args.rank, "layers": args.layers,
                   "per_cue": args.per_cue,
                   "balanced_queries": args.balanced_queries,
                   "train_policies": TRAIN_POLICIES,
                   "heldout_policies": HELDOUT_POLICIES,
                   "last_100_loss": sum(losses[-100:]) / min(100, len(losses)),
                   "protocol": "Synthetic three-condition policy; direct generated LoRA B factor; whole policies 2 and 5 held out; no per-policy expert bank"})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"accuracy": result["accuracy"],
                      "by_policy": result["by_policy"],
                      "output": str(args.output)}), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="current_work/delta-Mem/model/Qwen3-4B-Instruct-2507")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpu-fraction", type=float, default=0.40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--per-cue", type=int, default=4)
    parser.add_argument("--balanced-queries", action="store_true")
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--output", type=Path, default=Path(
        "results/trajectory_hyperlora/direct_composition_20261005/seed42.json"))
    run(parser.parse_args())


if __name__ == "__main__":
    main()
