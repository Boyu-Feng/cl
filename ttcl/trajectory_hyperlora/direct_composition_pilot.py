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


def oracle_relation_bits(steps: list[dict[str, str]], device: str) -> torch.Tensor:
    """Diagnostic parser of source records, never a deployable learned encoder."""
    values: dict[str, set[str]] = {cue: set() for cue in CUES}
    for step in steps:
        cues = [cue for cue in CUES if cue in step["observation"]]
        actions = [label for label in ("LEFT", "RIGHT") if label in step["action"]]
        if len(cues) != 1 or len(actions) != 1:
            raise ValueError("Ambiguous diagnostic source record")
        values[cues[0]].add(actions[0])
    if any(len(actions) != 1 for actions in values.values()):
        raise ValueError("Missing or inconsistent diagnostic source records")
    return torch.tensor([[1.0 if "RIGHT" in values[cue] else -1.0
                          for cue in CUES]], device=device)


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
        # A linear map enforces composition of the three observed relations.
        # This oracle branch isolates LoRA generation from trajectory parsing.
        self.oracle_latent = nn.Linear(len(CUES), 128, bias=False)
        self.relation_head = nn.Linear(128, len(CUES))
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

    def set_source(self, fields: dict | None,
                   oracle_bits: torch.Tensor | None = None) -> None:
        if fields is None and oracle_bits is None:
            for adapter in self.adapters:
                adapter.b = None
            return
        latent = self.oracle_latent(oracle_bits) if oracle_bits is not None else self.encode(fields)
        for adapter, head in zip(self.adapters, self.b_heads, strict=True):
            adapter.b = head(latent).reshape(1, adapter.base.out_features,
                                             adapter.rank)

    def relation_loss(self, fields: dict,
                      target_bits: torch.Tensor) -> torch.Tensor:
        logits = self.relation_head(self.encode(fields))
        return F.binary_cross_entropy_with_logits(logits,
                                                  (target_bits + 1.0) / 2.0)


def target_loss(agent: DirectRelationHyperLoRA, tokenizer: object,
                policy: int, cue_index: int, number: int, device: str,
                *, train: bool, action_contrastive: bool = False,
                action_token_ce: bool = False,
                action_sequence_ce: bool = False) -> torch.Tensor:
    template = TRAIN_QUERY if train else TEST_QUERY
    question = template.format(number=number, cue=CUES[cue_index])
    prefix = tokenizer(prompt(tokenizer, question), add_special_tokens=False).input_ids
    if action_contrastive or action_token_ce:
        action_ids = [tokenizer(label, add_special_tokens=False).input_ids
                      for label in ("LEFT", "RIGHT")]
        if any(len(ids) != 1 for ids in action_ids):
            raise ValueError("Contrastive action labels must be single tokens")
        logits = agent.model(
            input_ids=torch.tensor([prefix], device=device),
            use_cache=False).logits[:, -1, :].float()
        if action_contrastive:
            logits = logits[:, [ids[0] for ids in action_ids]]
            target_id = int(action(policy, cue_index) == "RIGHT")
        else:
            target_id = action_ids[int(action(policy, cue_index) == "RIGHT")][0]
        target = torch.tensor([target_id], device=device)
        return F.cross_entropy(logits, target)
    answer = action(policy, cue_index)
    suffix = tokenizer((answer if action_sequence_ce else " " + answer) +
                       tokenizer.eos_token,
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
             device: str, *, oracle_relations: bool = False) -> dict:
    agent.eval()
    rows = []
    relation_rows = []
    for split, policies in (("seen", (1, 6)), ("unseen", HELDOUT_POLICIES)):
        for policy in policies:
            for source_seed in (2001, 2002):
                source, used_numbers = records(policy,
                                               random.Random(source_seed + 31 * policy),
                                               test=True)
                wrong, _ = records(policy ^ 7,
                                   random.Random(source_seed + 31 * policy),
                                   test=True)
                if not oracle_relations:
                    with torch.no_grad():
                        fields = tokenize_records(tokenizer, source, device)
                        prediction = torch.sigmoid(
                            agent.relation_head(agent.encode(fields)))[0].tolist()
                    expected_bits = oracle_relation_bits(source, device)[0].tolist()
                    relation_rows.append({
                        "split": split, "policy": policy,
                        "source_seed": source_seed,
                        "probabilities": prediction,
                        "expected": [int(bit > 0) for bit in expected_bits],
                        "correct": [int(prob >= .5) == int(bit > 0)
                                    for prob, bit in zip(prediction, expected_bits, strict=True)],
                    })
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
                            if oracle_relations and selected is not None:
                                agent.set_source(None, oracle_relation_bits(selected, device))
                            else:
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
            "relation_rows": relation_rows,
            "relation_bit_accuracy": (sum(sum(r["correct"]) for r in relation_rows) /
                                      (len(relation_rows) * len(CUES))
                                      if relation_rows else None),
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
    if args.oracle_relations and args.relation_aux_weight:
        raise ValueError("Relation auxiliary loss requires raw source encoding")
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
        source_fields = (None if args.oracle_relations else
                         tokenize_records(tokenizer, source, args.device))
        source_bits = (oracle_relation_bits(source, args.device)
                       if args.oracle_relations else None)
        queried_cues = range(len(CUES)) if args.balanced_queries else (cue_index,)
        losses_this_step = []
        for index in queried_cues:
            agent.set_source(source_fields, source_bits)
            number = rng.choice([n for n in range(1, 90) if n not in used_numbers])
            loss = target_loss(agent, tokenizer, policy, index, number,
                               args.device, train=True,
                               action_contrastive=args.action_contrastive,
                               action_token_ce=args.action_token_ce,
                               action_sequence_ce=args.action_sequence_ce) / len(queried_cues)
            loss.backward()
            losses_this_step.append(float(loss.detach()))
            agent.set_source(None)
        if args.relation_aux_weight:
            aux = args.relation_aux_weight * agent.relation_loss(
                source_fields, oracle_relation_bits(source, args.device))
            aux.backward()
            losses_this_step.append(float(aux.detach()))
        torch.nn.utils.clip_grad_norm_((p for p in agent.parameters()
                                        if p.requires_grad), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        agent.set_source(None)
        losses.append(sum(losses_this_step))
        if (step + 1) % 100 == 0:
            print(json.dumps({"step": step + 1,
                              "mean_loss": sum(losses[-100:]) / 100}), flush=True)
    result = evaluate(agent, tokenizer, args.device,
                      oracle_relations=args.oracle_relations)
    result.update({"seed": args.seed, "steps": args.steps,
                   "rank": args.rank, "layers": args.layers,
                   "per_cue": args.per_cue,
                   "balanced_queries": args.balanced_queries,
                   "action_contrastive": args.action_contrastive,
                   "action_token_ce": args.action_token_ce,
                   "action_sequence_ce": args.action_sequence_ce,
                   "oracle_relations": args.oracle_relations,
                   "relation_aux_weight": args.relation_aux_weight,
                   "train_policies": TRAIN_POLICIES,
                   "heldout_policies": HELDOUT_POLICIES,
                   "last_100_loss": sum(losses[-100:]) / min(100, len(losses)),
                   "protocol": "Synthetic three-condition policy; direct generated LoRA B factor; whole policies 2 and 5 held out; no per-policy expert bank"})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    if args.save_checkpoint is not None:
        args.save_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"trainable_state": {name: param.detach().cpu()
                                       for name, param in agent.named_parameters()
                                       if param.requires_grad},
                    "rank": args.rank, "layers": args.layers,
                    "source_encoder": "oracle" if args.oracle_relations else "raw"},
                   args.save_checkpoint)
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
    parser.add_argument("--action-contrastive", action="store_true")
    parser.add_argument("--action-token-ce", action="store_true")
    parser.add_argument("--action-sequence-ce", action="store_true")
    parser.add_argument("--oracle-relations", action="store_true")
    parser.add_argument("--relation-aux-weight", type=float, default=0.0)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--output", type=Path, default=Path(
        "results/trajectory_hyperlora/direct_composition_20261005/seed42.json"))
    parser.add_argument("--save-checkpoint", type=Path)
    args = parser.parse_args()
    if sum((args.action_contrastive, args.action_token_ce,
            args.action_sequence_ce)) > 1:
        parser.error("Select at most one action loss")
    run(args)


if __name__ == "__main__":
    main()
