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
import math
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
TRAIN_QUERY_TEMPLATES = (
    TRAIN_QUERY,
    "A panel displays {number} and tag {cue}. Based on this panel's "
    "convention, press which lever? Answer LEFT or RIGHT only.",
    "For the {cue} marker at reading {number}, select the correct lever. "
    "Return just LEFT or RIGHT.",
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
        self.scale = 1.0
        self.a = nn.Parameter(torch.randn(rank, base.in_features) * 0.02)
        self.b: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.base(x)
        if self.b is None:
            return output
        low = F.linear(x.float(), self.a)
        update = torch.einsum("bsr,bor->bso", low, self.b)
        return output + (self.scale * update / self.rank).to(output.dtype)


class DirectRelationHyperLoRA(nn.Module):
    def __init__(self, model: nn.Module, rank: int = 4,
                 layers: int = 2, width: int = 24,
                 encoder_kind: str = "covariance",
                 relation_bottleneck: str = "none",
                 context_mode: str = "none",
                 context_strength: float | None = None) -> None:
        super().__init__()
        if encoder_kind not in ("covariance", "slots", "token_slots",
                                "factorized", "contextual"):
            raise ValueError(f"Unknown source encoder: {encoder_kind}")
        if relation_bottleneck not in ("none", "soft", "hard"):
            raise ValueError(f"Unknown relation bottleneck: {relation_bottleneck}")
        if context_mode not in ("none", "initial"):
            raise ValueError(f"Unknown initial-context mode: {context_mode}")
        if context_strength is not None and not 0 <= context_strength <= 1:
            raise ValueError("Initial-context strength must be between zero and one")
        self.encoder_kind = encoder_kind
        self.relation_bottleneck = relation_bottleneck
        self.context_mode = context_mode
        self.context_strength = (float(context_strength)
            if context_strength is not None else
            1.0 if context_mode == "initial" else 0.0)
        self.model = model
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        self.encoder = RelationalEncoder(model.get_input_embeddings(),
                                         width=width, experts=2)
        if encoder_kind in ("covariance", "factorized"):
            self.latent = nn.Sequential(nn.LayerNorm(width * width + width),
                                        nn.Linear(width * width + width, 128),
                                        nn.Tanh())
        elif encoder_kind == "contextual":
            self.contextual_latent = nn.Sequential(
                nn.LayerNorm(model.get_input_embeddings().embedding_dim),
                nn.Linear(model.get_input_embeddings().embedding_dim, 128))
        else:
            self.slot_queries = nn.Parameter(torch.randn(len(CUES), width))
            self.slot_latent = nn.Sequential(nn.LayerNorm(len(CUES) * width),
                                             nn.Linear(len(CUES) * width, 128),
                                             nn.Tanh())
        # A linear map enforces composition of the three observed relations.
        # This oracle branch isolates LoRA generation from trajectory parsing.
        self.oracle_latent = nn.Linear(len(CUES), 128, bias=False)
        self.relation_head = nn.Linear(128, len(CUES))
        if encoder_kind == "contextual":
            for module in (self.encoder, self.oracle_latent,
                           self.relation_head):
                for parameter in module.parameters():
                    parameter.requires_grad_(False)
        if encoder_kind == "factorized":
            self.cue_head = nn.Linear(width, len(CUES))
            self.action_head = nn.Linear(width, 1)
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
        if self.encoder_kind == "contextual":
            vector = fields["contextual"]
            if vector.ndim != 2 or vector.shape[0] != 1:
                raise ValueError("Contextual source must be one pooled episode vector")
            return self.contextual_latent(vector.float())
        encoder = self.encoder
        observation = encoder.observation(encoder.field_mean(fields["observation"]))
        action_vector = encoder.action(encoder.field_mean(fields["action"]))
        feedback = encoder.feedback(encoder.field_mean(fields["feedback"]))
        if self.encoder_kind in ("slots", "token_slots"):
            if self.encoder_kind == "token_slots":
                ids, mask = fields["observation"]
                with torch.no_grad():
                    embedded = encoder.embedding(ids).float()
                tokens = encoder.observation(embedded)
                token_scores = torch.einsum(
                    "qw,slw->qsl", self.slot_queries,
                    tokens) / math.sqrt(tokens.shape[-1])
                token_scores = token_scores.masked_fill(
                    ~mask.bool().unsqueeze(0), -1e4)
                scores = torch.logsumexp(token_scores, dim=-1) - \
                    mask.sum(-1).float().log().unsqueeze(0)
            else:
                scores = torch.einsum("qw,sw->qs", self.slot_queries,
                                      observation) / math.sqrt(observation.shape[-1])
            weights = torch.softmax(scores, dim=-1)
            slot_actions = weights @ (action_vector * feedback)
            return self.slot_latent(slot_actions.reshape(1, -1))
        covariance = centered_relation(observation, action_vector, feedback)
        action_mean = (action_vector * feedback).mean(0, keepdim=True)
        if self.context_mode == "initial":
            action_mean = action_mean + self.context_strength * observation[:1]
        return self.latent(torch.cat((covariance, action_mean), dim=-1))

    def set_source(self, fields: dict | None,
                   oracle_bits: torch.Tensor | None = None) -> None:
        if fields is None and oracle_bits is None:
            for adapter in self.adapters:
                adapter.b = None
            return
        if oracle_bits is not None:
            latent = self.oracle_latent(oracle_bits)
        elif self.relation_bottleneck == "none":
            latent = self.encode(fields)
        else:
            logits = self.predict_relation(fields)
            if self.relation_bottleneck == "hard":
                bits = torch.where(logits >= 0, 1.0, -1.0)
            else:
                bits = 2.0 * torch.sigmoid(logits) - 1.0
            latent = self.oracle_latent(bits)
        for adapter, head in zip(self.adapters, self.b_heads, strict=True):
            adapter.b = head(latent).reshape(1, adapter.base.out_features,
                                             adapter.rank)

    def predict_relation(self, fields: dict) -> torch.Tensor:
        if self.encoder_kind != "factorized":
            return self.relation_head(self.encode(fields))
        encoder = self.encoder
        observation = encoder.observation(
            encoder.field_mean(fields["observation"]))
        action_vector = encoder.action(encoder.field_mean(fields["action"]))
        cue_prob = torch.softmax(self.cue_head(observation), dim=-1)
        action_sign = torch.tanh(self.action_head(action_vector))
        relation = (cue_prob.T @ action_sign).squeeze(-1) / \
            cue_prob.sum(0).clamp_min(1e-6)
        return (5.0 * relation).unsqueeze(0)

    def relation_loss(self, fields: dict,
                      target_bits: torch.Tensor) -> torch.Tensor:
        logits = self.predict_relation(fields)
        return F.binary_cross_entropy_with_logits(logits,
                                                  (target_bits + 1.0) / 2.0)

    def set_adapter_scale(self, scale: float) -> None:
        if not math.isfinite(scale) or scale < 0:
            raise ValueError("Adapter scale must be finite and nonnegative")
        for adapter in self.adapters:
            adapter.scale = scale

    def factorized_step_loss(self, fields: dict,
                             source: list[dict[str, str]]) -> torch.Tensor:
        if self.encoder_kind != "factorized":
            raise ValueError("Step supervision requires factorized encoding")
        cue_targets = []
        action_targets = []
        for step in source:
            cues = [index for index, cue in enumerate(CUES)
                    if cue in step["observation"]]
            actions = [int(label == "RIGHT") for label in ("LEFT", "RIGHT")
                       if label in step["action"]]
            if len(cues) != 1 or len(actions) != 1:
                raise ValueError("Ambiguous factorized training record")
            cue_targets.append(cues[0])
            action_targets.append(actions[0])
        device = fields["observation"][0].device
        observation = self.encoder.observation(
            self.encoder.field_mean(fields["observation"]))
        action_vector = self.encoder.action(
            self.encoder.field_mean(fields["action"]))
        cue_loss = F.cross_entropy(self.cue_head(observation),
                                   torch.tensor(cue_targets, device=device))
        action_loss = F.binary_cross_entropy_with_logits(
            self.action_head(action_vector).squeeze(-1),
            torch.tensor(action_targets, device=device, dtype=torch.float32))
        return cue_loss + action_loss + self.relation_loss(
            fields, oracle_relation_bits(source, device))


def target_loss(agent: DirectRelationHyperLoRA, tokenizer: object,
                policy: int, cue_index: int, number: int, device: str,
                *, train: bool, action_contrastive: bool = False,
                action_token_ce: bool = False,
                action_sequence_ce: bool = False,
                query_template: str | None = None) -> torch.Tensor:
    template = query_template or (TRAIN_QUERY if train else TEST_QUERY)
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
                            agent.predict_relation(fields))[0].tolist()
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
    if args.relation_pretrain_steps < 0:
        raise ValueError("Relation pretraining steps cannot be negative")
    if any(value is not None and value <= 0 for value in
           (args.relation_pretrain_lr, args.lora_lr)):
        raise ValueError("Learning rates must be positive")
    if args.oracle_relations and args.relation_aux_weight:
        raise ValueError("Relation auxiliary loss requires raw source encoding")
    if args.freeze_relation_after_pretrain and args.relation_pretrain_steps < 1:
        raise ValueError("Freezing requires relation pretraining")
    if args.freeze_relation_after_pretrain and args.relation_aux_weight:
        raise ValueError("Frozen relation encoder cannot use auxiliary loss")
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
                                    layers=args.layers,
                                    encoder_kind=args.encoder_kind,
                                    relation_bottleneck=args.relation_bottleneck).to(args.device)
    relation_prefixes = ("encoder.", "latent.", "slot_", "relation_head.",
                         "cue_head.", "action_head.")
    pretrain_losses = []
    if args.relation_pretrain_steps:
        relation_optimizer = torch.optim.AdamW(
            [param for name, param in agent.named_parameters()
             if param.requires_grad and name.startswith(relation_prefixes)],
            lr=args.relation_pretrain_lr or args.lr)
        for step in range(args.relation_pretrain_steps):
            policy = TRAIN_POLICIES[step % len(TRAIN_POLICIES)]
            source, _ = records(policy, rng, test=False,
                                per_cue=args.per_cue)
            fields = tokenize_records(tokenizer, source, args.device)
            target = oracle_relation_bits(source, args.device)
            loss = (agent.factorized_step_loss(fields, source)
                    if args.encoder_kind == "factorized" else
                    agent.relation_loss(fields, target))
            loss.backward()
            relation_optimizer.step()
            relation_optimizer.zero_grad(set_to_none=True)
            pretrain_losses.append(float(loss.detach()))
        if args.freeze_relation_after_pretrain:
            for name, param in agent.named_parameters():
                if name.startswith(relation_prefixes):
                    param.requires_grad_(False)
    optimizer = torch.optim.AdamW((p for p in agent.parameters() if p.requires_grad),
                                  lr=args.lora_lr or args.lr)
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
                               action_sequence_ce=args.action_sequence_ce,
                               query_template=(rng.choice(TRAIN_QUERY_TEMPLATES)
                                               if args.query_augmentation else None)) / len(queried_cues)
            loss.backward()
            losses_this_step.append(float(loss.detach()))
            agent.set_source(None)
        if args.relation_aux_weight:
            aux_loss = (agent.factorized_step_loss(source_fields, source)
                        if args.encoder_kind == "factorized" else
                        agent.relation_loss(
                            source_fields, oracle_relation_bits(source, args.device)))
            aux = args.relation_aux_weight * aux_loss
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
                   "encoder_kind": args.encoder_kind,
                   "relation_bottleneck": args.relation_bottleneck,
                   "query_augmentation": args.query_augmentation,
                   "relation_pretrain_steps": args.relation_pretrain_steps,
                   "relation_pretrain_lr": args.relation_pretrain_lr or args.lr,
                   "lora_lr": args.lora_lr or args.lr,
                   "freeze_relation_after_pretrain": args.freeze_relation_after_pretrain,
                   "pretrain_last_100_loss": (sum(pretrain_losses[-100:]) /
                                              min(100, len(pretrain_losses))
                                              if pretrain_losses else None),
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
                    "relation_state": {name: param.detach().cpu()
                                       for name, param in agent.named_parameters()
                                       if name.startswith(relation_prefixes)},
                    "rank": args.rank, "layers": args.layers,
                    "encoder_kind": args.encoder_kind,
                    "relation_bottleneck": args.relation_bottleneck,
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
    parser.add_argument("--relation-pretrain-steps", type=int, default=0)
    parser.add_argument("--relation-pretrain-lr", type=float)
    parser.add_argument("--lora-lr", type=float)
    parser.add_argument("--freeze-relation-after-pretrain", action="store_true")
    parser.add_argument("--encoder-kind", choices=("covariance", "slots",
                                                    "token_slots", "factorized"),
                        default="covariance")
    parser.add_argument("--relation-bottleneck", choices=("none", "soft", "hard"),
                        default="none")
    parser.add_argument("--query-augmentation", action="store_true")
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
