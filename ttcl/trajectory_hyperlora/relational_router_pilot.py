"""Preserve observation/action pairing when compiling toy trajectories to LoRA.

Inputs are generic step fields (observation, action, feedback). No parity or
lever vocabulary is hard-coded into the encoder. This still mixes a frozen
two-adapter basis, so it tests amortized selection, not open-ended skill birth.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random

import torch
from torch import nn
from peft import LoraConfig, TaskType
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from ttcl.trajectory_hyperlora.experience_pilot import (
    TEST_NUMBERS, TRAIN_EVEN, TRAIN_ODD, action, generate, parity, query,
)
from ttcl.trajectory_hyperlora.pilot import TrajectoryHyperLoRA
from ttcl.trajectory_hyperlora.stepwise_router_pilot import load_expert_bank, mount
from ttcl.trajectory_hyperlora.two_stage_pilot import target_loss


TRAIN_OBSERVATION = (
    "Panel reading {number}; indicator {parity}.",
    "The sensor shows {number}, marked {parity}.",
)
TEST_OBSERVATION = (
    "At reading {number}, the panel displayed {parity}.",
    "Archived sensor state: {number} ({parity}).",
)
TRAIN_ACTION = ("Agent pressed {action}.", "Command sent: {action}.")
TEST_ACTION = ("Selected lever {action}.", "The agent chose {action}.")


def records(policy: int, rng: random.Random, count_per_parity: int,
            *, test: bool) -> tuple[list[dict[str, str]], tuple[int, ...]]:
    numbers = (rng.sample(TRAIN_ODD, count_per_parity)
               + rng.sample(TRAIN_EVEN, count_per_parity))
    rng.shuffle(numbers)
    observations = TEST_OBSERVATION if test else TRAIN_OBSERVATION
    actions = TEST_ACTION if test else TRAIN_ACTION
    out = [{"observation": rng.choice(observations).format(
                number=n, parity=parity(n)),
            "action": rng.choice(actions).format(action=action(policy, n)),
            "feedback": "correct" if test else "success"}
           for n in numbers]
    return out, tuple(numbers)


def add_rejected_attempts(steps: list[dict[str, str]],
                          numbers: tuple[int, ...], policy: int,
                          *, test: bool) -> list[dict[str, str]]:
    """Synthetic environment emits an unsuccessful attempt before correction."""
    output = []
    action_template = TEST_ACTION[0] if test else TRAIN_ACTION[0]
    for item, number in zip(steps, numbers, strict=True):
        output.append({"observation": item["observation"],
                       "action": action_template.format(
                           action=action(1 - policy, number)),
                       "feedback": "rejected"})
        output.append(item)
    return output


def tokenize_records(tokenizer: object, steps: list[dict[str, str]], device: str,
                     max_tokens: int = 40) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    if not steps or any(not all(step.get(key) for key in
                            ("observation", "action", "feedback")) for step in steps):
        raise ValueError("Completed steps need observation, action, and feedback")
    output = {}
    for key in ("observation", "action", "feedback"):
        encoded = tokenizer([step[key] for step in steps],
                            add_special_tokens=False, padding=True,
                            truncation=True, max_length=max_tokens,
                            return_tensors="pt")
        output[key] = (encoded.input_ids.to(device),
                       encoded.attention_mask.to(device))
    return output


class RelationalEncoder(nn.Module):
    def __init__(self, embedding: nn.Embedding, width: int = 24,
                 experts: int = 2) -> None:
        super().__init__()
        self.embedding = embedding
        size = embedding.embedding_dim
        self.observation = nn.Sequential(nn.LayerNorm(size), nn.Linear(size, width),
                                         nn.Tanh())
        self.action = nn.Sequential(nn.LayerNorm(size), nn.Linear(size, width),
                                    nn.Tanh())
        self.feedback = nn.Sequential(nn.LayerNorm(size), nn.Linear(size, width),
                                      nn.Sigmoid())
        self.head = nn.Sequential(nn.LayerNorm(width * width),
                                  nn.Linear(width * width, 64), nn.Tanh(),
                                  nn.Linear(64, experts))

    def field_mean(self, item: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        ids, mask = item
        if ids.ndim != 2 or mask.shape != ids.shape or torch.any(mask.sum(-1) == 0):
            raise ValueError("Invalid step-field token batch")
        with torch.no_grad():
            embedded = self.embedding(ids).float()
        weight = mask.unsqueeze(-1)
        return (embedded * weight).sum(1) / weight.sum(1)

    def forward(self, fields: dict[str, tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
        observation = self.observation(self.field_mean(fields["observation"]))
        action_vector = self.action(self.field_mean(fields["action"]))
        feedback = self.feedback(self.field_mean(fields["feedback"]))
        pooled = centered_relation(observation, action_vector, feedback)
        return torch.softmax(self.head(pooled), dim=-1)


def centered_relation(observation: torch.Tensor, action_vector: torch.Tensor,
                      feedback: torch.Tensor) -> torch.Tensor:
    if (observation.ndim != 2 or action_vector.ndim != 2 or
            feedback.shape != observation.shape or
            observation.shape != action_vector.shape or
            observation.shape[0] < 2):
        raise ValueError("Need aligned observation/action/feedback step matrices")
    # Remove episode-wide vocabulary and template bias before combining a
    # step's observation with its action. Balanced opposite rules have
    # nearly identical marginal token counts; their covariance differs.
    observation = observation - observation.mean(0, keepdim=True)
    action_vector = action_vector - action_vector.mean(0, keepdim=True)
    relation = torch.einsum("si,sj->sij", observation * feedback, action_vector)
    return relation.mean(0).reshape(1, -1)


def export_generated_adapter(agent: TrajectoryHyperLoRA,
                             coefficients: torch.Tensor, source_hash: str,
                             model_path: str, output_dir: Path) -> dict:
    """Materialize the generated two-basis mixture as a standard PEFT LoRA."""
    if output_dir.exists():
        raise FileExistsError(output_dir)
    coefficient = coefficients.detach().float().cpu().reshape(-1)
    experts = len(coefficient)
    rank = agent.adapters[0].a.shape[1]
    total_rank = rank * experts
    first_layer = len(agent.model.model.layers) - len(agent.adapters)
    targets = []
    tensors = {}
    for offset, adapter in enumerate(agent.adapters):
        name = f"model.layers.{first_layer + offset}.mlp.down_proj"
        targets.append(name)
        a = adapter.a.detach().float().cpu().reshape(total_rank, -1).contiguous()
        b = (adapter.b.detach().float().cpu()
             * coefficient[:, None, None] / rank)
        b = b.permute(1, 0, 2).reshape(adapter.b.shape[1], total_rank).contiguous()
        prefix = f"base_model.model.{name}"
        tensors[f"{prefix}.lora_A.weight"] = a
        tensors[f"{prefix}.lora_B.weight"] = b
    output_dir.mkdir(parents=True)
    config = LoraConfig(r=total_rank, lora_alpha=total_rank, lora_dropout=0.0,
                        target_modules=targets, bias="none",
                        task_type=TaskType.CAUSAL_LM, inference_mode=True,
                        base_model_name_or_path=str(Path(model_path).resolve()))
    config.save_pretrained(output_dir)
    save_file(tensors, output_dir / "adapter_model.safetensors")
    manifest = {"source_sha256": source_hash,
                "coefficients": coefficient.tolist(),
                "base_model": str(Path(model_path).resolve()),
                "rank": total_rank, "targets": targets}
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def evaluate(agent: TrajectoryHyperLoRA, encoder: RelationalEncoder,
             tokenizer: object, device: str) -> dict:
    agent.eval()
    encoder.eval()
    rows = []
    for policy in (0, 1):
        for source_seed in (2001, 2002):
            steps, support = records(policy, random.Random(source_seed + 17 * policy),
                                     8, test=True)
            wrong, _ = records(1 - policy,
                               random.Random(source_seed + 17 * policy),
                               8, test=True)
            raw = json.dumps(steps, ensure_ascii=False, sort_keys=True)
            with torch.no_grad():
                coefficients = encoder(tokenize_records(tokenizer, steps, device))
                wrong_coefficients = encoder(tokenize_records(tokenizer, wrong, device))
            history = "\n".join(
                f"Observation: {step['observation']} Action: {step['action']} "
                f"Feedback: {step['feedback']}" for step in steps)
            for number in TEST_NUMBERS:
                assert number not in support
                question = query(number, train=False)
                expected = action(policy, number)
                row = {"policy": policy, "source_seed": source_seed,
                       "number": number, "expected": expected,
                       "source_sha256": hashlib.sha256(raw.encode()).hexdigest(),
                       "query_sha256": hashlib.sha256(question.encode()).hexdigest(),
                       "coefficients": coefficients.detach().float().cpu().tolist()[0]}
                for arm in ("base", "text", "generated", "wrong_source", "expert"):
                    selected = (coefficients if arm == "generated" else
                                wrong_coefficients if arm == "wrong_source" else
                                torch.tensor([[float(i == policy) for i in (0, 1)]],
                                             device=device) if arm == "expert" else None)
                    mount(agent, selected)
                    answer = generate(agent, tokenizer, question, device,
                                      history=history if arm == "text" else None)
                    first = answer.upper().split()[0].strip(".,:;!") if answer else ""
                    row[arm] = {"answer": answer, "correct": first == expected}
                mount(agent, None)
                rows.append(row)
    arms = ("base", "text", "generated", "wrong_source", "expert")
    return {"n": len(rows),
            "accuracy": {arm: sum(row[arm]["correct"] for row in rows) / len(rows)
                         for arm in arms},
            "by_policy": {str(policy): {arm: sum(row[arm]["correct"] for row in rows
                                               if row["policy"] == policy) / 8
                                        for arm in arms} for policy in (0, 1)},
            "rows": rows}


def evaluate_corrections(agent: TrajectoryHyperLoRA, encoder: RelationalEncoder,
                         tokenizer: object, device: str) -> dict:
    """Test rejected action followed by accepted correction, never trained here."""
    agent.eval()
    encoder.eval()
    rows = []
    for policy in (0, 1):
        for source_seed in (2001, 2002):
            clean, numbers = records(policy,
                                     random.Random(source_seed + 17 * policy),
                                     8, test=True)
            wrong_clean, wrong_numbers = records(
                1 - policy, random.Random(source_seed + 17 * policy),
                8, test=True)

            source = add_rejected_attempts(clean, numbers, policy, test=True)
            wrong = add_rejected_attempts(wrong_clean, wrong_numbers,
                                          1 - policy, test=True)
            with torch.no_grad():
                coefficients = encoder(tokenize_records(tokenizer, source, device))
                wrong_coefficients = encoder(tokenize_records(tokenizer, wrong, device))
            raw = json.dumps(source, ensure_ascii=False, sort_keys=True)
            history = "\n".join(
                f"Observation: {step['observation']} Action: {step['action']} "
                f"Feedback: {step['feedback']}" for step in source)
            for number in TEST_NUMBERS:
                expected = action(policy, number)
                question = query(number, train=False)
                row = {"policy": policy, "source_seed": source_seed,
                       "number": number, "expected": expected,
                       "source_sha256": hashlib.sha256(raw.encode()).hexdigest(),
                       "query_sha256": hashlib.sha256(question.encode()).hexdigest(),
                       "coefficients": coefficients.detach().float().cpu().tolist()[0]}
                for arm in ("text", "generated", "wrong_source"):
                    mount(agent, coefficients if arm == "generated" else
                          wrong_coefficients if arm == "wrong_source" else None)
                    answer = generate(agent, tokenizer, question, device,
                                      history=history if arm == "text" else None)
                    first = answer.upper().split()[0].strip(".,:;!") if answer else ""
                    row[arm] = {"answer": answer, "correct": first == expected}
                mount(agent, None)
                rows.append(row)
    return {"n": len(rows),
            "accuracy": {arm: sum(row[arm]["correct"] for row in rows) / len(rows)
                         for arm in ("text", "generated", "wrong_source")},
            "rows": rows}


def run(args: argparse.Namespace) -> dict:
    if args.steps < 1 or not 1 <= args.support_per_parity < len(TRAIN_EVEN):
        raise ValueError("Invalid step or support count")
    if not 0.0 <= args.correction_prob <= 1.0:
        raise ValueError("Correction probability must be in [0, 1]")
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
    agent = TrajectoryHyperLoRA(model, experts=2, rank=4, layers=2).to(args.device)
    load_expert_bank(agent, args.bank)
    encoder = RelationalEncoder(agent.model.get_input_embeddings()).to(args.device)
    optimizer = torch.optim.AdamW((p for p in encoder.parameters() if p.requires_grad),
                                  lr=args.lr)
    losses = []
    for step in range(args.steps):
        policy = step % 2
        source, source_numbers = records(policy, rng, args.support_per_parity,
                                         test=False)
        if rng.random() < args.correction_prob:
            source = add_rejected_attempts(source, source_numbers, policy,
                                           test=False)
        coefficients = encoder(tokenize_records(tokenizer, source, args.device))
        mount(agent, coefficients)
        odd = rng.choice([n for n in TRAIN_ODD if n not in source_numbers])
        even = rng.choice([n for n in TRAIN_EVEN if n not in source_numbers])
        loss = (target_loss(agent, tokenizer, policy, odd, args.device)
                + target_loss(agent, tokenizer, policy, even, args.device)) / 2
        loss.backward()
        torch.nn.utils.clip_grad_norm_((p for p in encoder.parameters()
                                        if p.requires_grad), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        mount(agent, None)
        losses.append(float(loss.detach()))
        if (step + 1) % 50 == 0:
            print(json.dumps({"step": step + 1,
                              "mean_loss": sum(losses[-50:]) / 50}), flush=True)
    result = evaluate(agent, encoder, tokenizer, args.device)
    result["correction_probe"] = evaluate_corrections(
        agent, encoder, tokenizer, args.device)
    result.update({"seed": args.seed, "steps": args.steps,
                   "support_per_parity": args.support_per_parity,
                   "correction_prob": args.correction_prob,
                   "last_50_loss": sum(losses[-50:]) / min(50, len(losses)),
                   "protocol": "Synthetic structured raw-step trajectory to frozen LoRA-basis mixture; no handcrafted action semantics or RL"})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    weights_path = args.output.with_suffix(".pt")
    torch.save({name: parameter.detach().cpu()
                for name, parameter in encoder.named_parameters()
                if parameter.requires_grad}, weights_path)
    result["local_encoder_weights"] = str(weights_path)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"accuracy": result["accuracy"],
                      "correction_accuracy": result["correction_probe"]["accuracy"],
                      "by_policy": result["by_policy"],
                      "output": str(args.output)}), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="current_work/delta-Mem/model/Qwen3-4B-Instruct-2507")
    parser.add_argument("--bank", type=Path, default=Path(
        "results/trajectory_hyperlora/experience_pilot_20261004"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpu-fraction", type=float, default=0.40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--support-per-parity", type=int, default=8)
    parser.add_argument("--correction-prob", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--output", type=Path, default=Path(
        "results/trajectory_hyperlora/relational_router_20261005/seed42.json"))
    run(parser.parse_args())


if __name__ == "__main__":
    main()
