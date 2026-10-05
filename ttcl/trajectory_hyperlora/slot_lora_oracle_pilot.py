"""Four-relation compositional LoRA capacity test with diagnostic oracle readout.

The source rule is parsed from completed synthetic steps. This intentionally
isolates LoRA composition from trajectory understanding. Whole four-bit rules
with odd parity are never used for training; dev/test are fixed disjoint sets.
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

from ttcl.trajectory_hyperlora.direct_composition_pilot import GeneratedDownProjection
from ttcl.trajectory_hyperlora.pilot import prompt


CUES = ("ALPHA", "BETA", "GAMMA", "DELTA")
TRAIN_RULES = (0, 3, 5, 6, 9, 10, 12, 15)  # even parity
DEV_RULES = (1, 2, 4, 7)
TEST_RULES = (8, 11, 13, 14)
TRAIN_QUERY = (
    "Panel reading {number} has marker {cue}. Under this panel's convention, "
    "which lever should be pressed? Reply with exactly LEFT or RIGHT.",
    "At {number}, the device displays {cue}. Which lever is correct? "
    "Return LEFT or RIGHT only.",
)
EVAL_QUERY = {
    "dev": "New measurement {number} bears {cue}. Select LEFT or RIGHT.",
    "test": "A fresh device shows {number} with symbol {cue}. Based on its "
            "previous behavior, choose the lever. Answer LEFT or RIGHT only.",
}
EVAL_NUMBERS = {"dev": (121, 122, 133, 134),
                "test": (151, 152, 163, 164)}


def action(rule: int, cue_index: int) -> str:
    return "RIGHT" if (rule >> cue_index) & 1 else "LEFT"


def source_records(rule: int, rng: random.Random, *, test: bool) -> list[dict[str, str]]:
    observation_templates = (
        ("Panel reading {number}; marker {cue}.",
         "The sensor shows {number} with label {cue}.") if not test else
        ("Archived panel state: {number} ({cue}).",
         "At measurement {number}, the display carried {cue}."))
    action_templates = (("Agent pressed {action}.", "Command sent: {action}.")
                        if not test else
                        ("Selected lever {action}.", "The agent chose {action}."))
    numbers = rng.sample(range(1, 90), 3 * len(CUES))
    steps = []
    for cue_index, cue in enumerate(CUES):
        for number in numbers[3 * cue_index:3 * (cue_index + 1)]:
            steps.append({"observation": rng.choice(observation_templates).format(
                              number=number, cue=cue),
                          "action": rng.choice(action_templates).format(
                              action=action(rule, cue_index)),
                          "feedback": "confirmed correct" if test else "success"})
    rng.shuffle(steps)
    return steps


def oracle_bits(steps: list[dict[str, str]]) -> tuple[int, ...]:
    """Diagnostic source parser, never a learned trajectory encoder."""
    found: dict[str, set[int]] = {cue: set() for cue in CUES}
    for step in steps:
        cues = [cue for cue in CUES if cue in step["observation"]]
        labels = [int(label == "RIGHT") for label in ("LEFT", "RIGHT")
                  if label in step["action"]]
        if len(cues) != 1 or len(labels) != 1 or step["feedback"] not in (
                "success", "confirmed correct"):
            raise ValueError("Ambiguous or unsuccessful diagnostic source step")
        found[cues[0]].add(labels[0])
    if any(len(found[cue]) != 1 for cue in CUES):
        raise ValueError("Source rule missing or inconsistent")
    return tuple(next(iter(found[cue])) for cue in CUES)


class SlotLoRA(nn.Module):
    def __init__(self, model: nn.Module, rank_per_cue: int = 2,
                 layers: int = 2) -> None:
        super().__init__()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        self.model = model
        self.rank_per_cue = rank_per_cue
        rank = rank_per_cue * len(CUES)
        self.adapters = nn.ModuleList()
        self.b_experts = nn.ParameterList()
        for block in model.model.layers[-layers:]:
            adapter = GeneratedDownProjection(block.mlp.down_proj, rank)
            adapter.scale = len(CUES)
            block.mlp.down_proj = adapter
            self.adapters.append(adapter)
            self.b_experts.append(nn.Parameter(torch.zeros(
                len(CUES), 2, adapter.base.out_features, rank_per_cue)))

    def mount(self, bits: tuple[int, ...] | None,
              public: list[torch.Tensor] | None = None) -> None:
        if bits is not None and (len(bits) != len(CUES) or
                                 any(bit not in (0, 1) for bit in bits)):
            raise ValueError("Expected four binary source relations")
        for layer, (adapter, experts) in enumerate(zip(
                self.adapters, self.b_experts, strict=True)):
            if public is not None:
                adapter.b = public[layer]
            elif bits is None:
                adapter.b = None
            else:
                selected = torch.stack([experts[cue, bits[cue]]
                                        for cue in range(len(CUES))], dim=0)
                adapter.b = selected.permute(1, 0, 2).reshape(
                    1, adapter.base.out_features, -1)

    def public_factors(self) -> list[torch.Tensor]:
        factors = []
        for experts in self.b_experts:
            mean = experts.mean(1)
            factors.append(mean.permute(1, 0, 2).reshape(1, mean.shape[1], -1))
        return factors


def first_logits(agent: SlotLoRA, tokenizer, question: str,
                 device: str) -> torch.Tensor:
    ids = tokenizer(prompt(tokenizer, question), add_special_tokens=False).input_ids
    return agent.model(input_ids=torch.tensor([ids], device=device),
                       use_cache=False).logits[:, -1, :].float()


def evaluate(agent: SlotLoRA, tokenizer, device: str, split: str,
             relation_reader=None) -> dict:
    if split not in ("dev", "test"):
        raise ValueError(split)
    agent.eval()
    public = agent.public_factors()
    action_ids = {label: tokenizer(label, add_special_tokens=False).input_ids[0]
                  for label in ("LEFT", "RIGHT")}
    rows = []
    with torch.no_grad():
        for rule in DEV_RULES if split == "dev" else TEST_RULES:
            for source_seed in (3001, 3002):
                seed = source_seed + 31 * rule + (0 if split == "dev" else 10000)
                source = source_records(rule, random.Random(seed), test=True)
                wrong = source_records(rule ^ 15, random.Random(seed), test=True)
                if any(left["observation"] != right["observation"]
                       or left["feedback"] != right["feedback"]
                       for left, right in zip(source, wrong, strict=True)):
                    raise ValueError("Test source swap changes more than action")
                expected_bits, expected_wrong_bits = oracle_bits(source), oracle_bits(wrong)
                correct_bits = (relation_reader(source) if relation_reader else
                                expected_bits)
                wrong_bits = (relation_reader(wrong) if relation_reader else
                              expected_wrong_bits)
                source_hash = hashlib.sha256(json.dumps(
                    source, sort_keys=True).encode()).hexdigest()
                wrong_hash = hashlib.sha256(json.dumps(
                    wrong, sort_keys=True).encode()).hexdigest()
                for cue_index, cue in enumerate(CUES):
                    for number in EVAL_NUMBERS[split]:
                        question = EVAL_QUERY[split].format(number=number, cue=cue)
                        expected = action(rule, cue_index)
                        row = {"rule": rule, "source_seed": source_seed,
                               "cue": cue, "number": number, "expected": expected,
                               "readout_bits": correct_bits,
                               "wrong_readout_bits": wrong_bits,
                               "oracle_bits": expected_bits,
                               "wrong_oracle_bits": expected_wrong_bits,
                               "source_sha256": source_hash,
                               "wrong_source_sha256": wrong_hash,
                               "query_sha256": hashlib.sha256(
                                   question.encode()).hexdigest()}
                        for arm, bits, mean in (("base", None, None),
                                                ("public", None, public),
                                                ("correct", correct_bits, None),
                                                ("wrong", wrong_bits, None)):
                            agent.mount(bits, mean)
                            logits = first_logits(agent, tokenizer, question, device)
                            choice = int(logits.argmax(-1)[0])
                            row[arm] = {"first_token": tokenizer.decode(choice).strip(),
                                        "correct": choice == action_ids[expected]}
                        agent.mount(None)
                        rows.append(row)
    return {"n": len(rows), "success": {arm: sum(r[arm]["correct"] for r in rows)
                                         for arm in ("base", "public", "correct", "wrong")},
            "rows": rows}


def run(args: argparse.Namespace) -> dict:
    if args.output.exists() or args.checkpoint.exists() or args.steps < 1:
        raise ValueError("Fresh outputs and positive step budget required")
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    if args.device.startswith("cuda"):
        torch.cuda.set_per_process_memory_fraction(args.gpu_fraction,
                                                   device=args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    action_ids = {label: tokenizer(label, add_special_tokens=False).input_ids
                  for label in ("LEFT", "RIGHT")}
    if any(len(ids) != 1 for ids in action_ids.values()):
        raise ValueError("Action labels must be one token")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, local_files_only=True,
        attn_implementation="sdpa").to(args.device)
    model.config.use_cache = False
    agent = SlotLoRA(model, args.rank_per_cue, args.layers).to(args.device)
    optimizer = torch.optim.AdamW(
        [p for p in agent.parameters() if p.requires_grad], lr=args.lr,
        weight_decay=0)
    losses = []
    for step in range(args.steps):
        rule = TRAIN_RULES[step % len(TRAIN_RULES)]
        source = source_records(rule, rng, test=False)
        bits = oracle_bits(source)
        optimizer.zero_grad(set_to_none=True)
        for cue_index, cue in enumerate(CUES):
            number = rng.randrange(90, 120)
            question = rng.choice(TRAIN_QUERY).format(number=number, cue=cue)
            agent.mount(bits)
            logits = first_logits(agent, tokenizer, question, args.device)
            target = torch.tensor([action_ids[action(rule, cue_index)][0]],
                                  device=args.device)
            loss = F.cross_entropy(logits, target) / len(CUES)
            loss.backward()
            losses.append(float(loss.detach()) * len(CUES))
            agent.mount(None)
        torch.nn.utils.clip_grad_norm_(
            [p for p in agent.parameters() if p.requires_grad], 1.0)
        optimizer.step()
        if (step + 1) % 100 == 0:
            print(json.dumps({"step": step + 1,
                              "mean_loss": sum(losses[-400:]) / min(400, len(losses))}),
                  flush=True)
    agent.mount(None)
    dev = evaluate(agent, tokenizer, args.device, "dev")
    test = evaluate(agent, tokenizer, args.device, "test")
    result = {"protocol": "Four-condition diagnostic oracle-to-slot-LoRA capacity experiment; even-parity rules only train, odd-parity dev/test frozen and disjoint; Qwen base frozen; first-token exact scoring; no learned source encoder or RL",
              "seed": args.seed, "steps": args.steps, "lr": args.lr,
              "rank_per_cue": args.rank_per_cue, "layers": args.layers,
              "train_rules": TRAIN_RULES, "dev_rules": DEV_RULES,
              "test_rules": TEST_RULES, "last_100_step_loss":
              sum(losses[-400:]) / min(400, len(losses)),
              "dev": dev, "test": test}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    torch.save({"trainable_state": {name: p.detach().cpu()
                                   for name, p in agent.named_parameters()
                                   if p.requires_grad},
                "seed": args.seed, "rank_per_cue": args.rank_per_cue,
                "layers": args.layers, "steps": args.steps}, args.checkpoint)
    print(json.dumps({"dev": dev["success"], "test": test["success"]}), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="current_work/delta-Mem/model/Qwen3-4B-Instruct-2507")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpu-fraction", type=float, default=.35)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--lr", type=float, default=.001)
    parser.add_argument("--rank-per-cue", type=int, default=2)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
