"""Controlled trajectory-to-LoRA pilot on a frozen local causal LM.

This is a mechanics/transfer check, not an ALFWorld or CLBench score.  Every
support trajectory is completed before its independent query is presented.
The query never contains the support trajectory in the adapter arm.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


COLORS = ("amber", "cobalt", "jade", "violet")
TRAIN_TEMPLATES = (
    "Episode {episode}. Instruction: use the private color convention. Agent replied {color}. Feedback: accepted.",
    "In completed run {episode}, the agent was asked for its convention color and answered {color}; the environment accepted it.",
    "Past task {episode}: report the agreed color. Action: {color}. Observation: correct response.",
    "Trajectory {episode}: user requested the color for this session. Assistant: {color}. Tool result: success.",
)
TEST_TEMPLATES = (
    "Finished interaction {episode}: the assistant responded with {color} to the color request and received success feedback.",
    "Completed trial {episode}: the answer to the convention-color question was {color}; the judge accepted the answer.",
)
QUERY = "What is the color convention established in the previous completed episode? Reply with exactly one color word."


class ConditionalLoRALinear(nn.Module):
    """Frozen linear layer plus a trajectory-weighted bank of rank-r updates."""

    def __init__(self, base: nn.Linear, experts: int, rank: int) -> None:
        super().__init__()
        self.base = base
        self.a = nn.Parameter(torch.randn(experts, rank, base.in_features) * 0.02)
        self.b = nn.Parameter(torch.zeros(experts, base.out_features, rank))
        self.coefficients: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        if self.coefficients is None:
            return out
        low = torch.einsum("bsi,kri->bskr", x.float(), self.a)
        update = torch.einsum("bskr,kor->bsko", low, self.b)
        update = torch.einsum("bsko,bk->bso", update, self.coefficients)
        return out + (update / self.a.shape[1]).to(out.dtype)


class TrajectoryHyperLoRA(nn.Module):
    def __init__(self, model: nn.Module, experts: int = 4, rank: int = 4,
                 layers: int = 2, encoder_kind: str = "attention") -> None:
        super().__init__()
        self.model = model
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        self.encoder_kind = encoder_kind
        if encoder_kind == "gru":
            self.encoder = nn.GRU(model.config.hidden_size, 96, batch_first=True)
            self.projection = nn.Linear(96, experts)
        elif encoder_kind == "attention":
            self.encoder = nn.Sequential(
                nn.LayerNorm(model.config.hidden_size),
                nn.Linear(model.config.hidden_size, 128),
                nn.Tanh(),
            )
            self.token_attention = nn.Linear(128, 1)
            self.projection = nn.Linear(128, experts)
        else:
            raise ValueError(f"Unknown encoder: {encoder_kind}")
        self.adapters = nn.ModuleList()
        for block in model.model.layers[-layers:]:
            adapter = ConditionalLoRALinear(block.mlp.down_proj, experts, rank)
            block.mlp.down_proj = adapter
            self.adapters.append(adapter)

    def encode(self, source_ids: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            embeddings = self.model.get_input_embeddings()(source_ids).float()
        if self.encoder_kind == "gru":
            _, hidden = self.encoder(embeddings)
            pooled = hidden[-1]
        else:
            states = self.encoder(embeddings)
            weights = torch.softmax(self.token_attention(states), dim=1)
            pooled = (weights * states).sum(dim=1)
        return torch.softmax(self.projection(pooled), dim=-1)

    def set_source(self, source_ids: torch.Tensor | None) -> None:
        coefficients = self.encode(source_ids) if source_ids is not None else None
        for adapter in self.adapters:
            adapter.coefficients = coefficients

    def clear(self) -> None:
        for adapter in self.adapters:
            adapter.coefficients = None


def prompt(tokenizer: object, query: str, history: str | None = None) -> str:
    content = (f"Previous completed episode:\n{history}\n\n" if history else "") + query
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True
    )


def example(template: str, color: str, episode: int) -> str:
    return template.format(episode=episode, color=color)


def evaluate(agent: TrajectoryHyperLoRA, tokenizer: object, cases: list[dict], device: str) -> dict:
    agent.eval()
    rows = []
    for case in cases:
        row = {"label": case["color"], "source_hash": hashlib.sha256(case["source"].encode()).hexdigest()}
        for arm in ("base", "text", "hyper", "shuffled"):
            history = case["source"] if arm == "text" else None
            source = case["source"] if arm == "hyper" else case["shuffled"] if arm == "shuffled" else None
            agent.set_source(tokenizer(source, return_tensors="pt").input_ids.to(device) if source else None)
            inputs = tokenizer(prompt(tokenizer, QUERY, history), return_tensors="pt").to(device)
            with torch.no_grad():
                generated = agent.model.generate(**inputs, do_sample=False, max_new_tokens=5,
                                                 pad_token_id=tokenizer.eos_token_id)
            answer = tokenizer.decode(generated[0, inputs.input_ids.shape[1]:], skip_special_tokens=True).strip().lower()
            first = answer.split()[0].strip(".,:;!") if answer else ""
            row[arm] = {"answer": answer, "correct": first == case["color"]}
        agent.clear()
        rows.append(row)
    return {"n": len(rows), "accuracy": {arm: sum(row[arm]["correct"] for row in rows) / len(rows)
                                      for arm in ("base", "text", "hyper", "shuffled")}, "rows": rows}


def run(args: argparse.Namespace) -> dict:
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    if args.device.startswith("cuda"):
        torch.cuda.set_per_process_memory_fraction(args.gpu_fraction, device=args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, local_files_only=True,
        attn_implementation="sdpa"
    ).to(args.device)
    model.config.use_cache = False
    model.generation_config.temperature = 1.0
    model.generation_config.top_p = 1.0
    model.generation_config.top_k = 50
    agent = TrajectoryHyperLoRA(model, experts=args.experts, rank=args.rank,
                                layers=args.layers, encoder_kind=args.encoder).to(args.device)
    optimizer = torch.optim.AdamW((p for p in agent.parameters() if p.requires_grad), lr=args.lr)
    losses = []
    for step in range(args.steps):
        color = COLORS[step % len(COLORS)]
        source = example(rng.choice(TRAIN_TEMPLATES), color, step)
        source_ids = tokenizer(source, return_tensors="pt").input_ids.to(args.device)
        agent.set_source(source_ids)
        prefix = tokenizer(prompt(tokenizer, QUERY), add_special_tokens=False).input_ids
        suffix = tokenizer(" " + color + tokenizer.eos_token, add_special_tokens=False).input_ids
        ids = torch.tensor([prefix + suffix], device=args.device)
        labels = torch.tensor([[-100] * len(prefix) + suffix], device=args.device)
        agent.train()
        loss = agent.model(input_ids=ids, labels=labels, use_cache=False).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_((p for p in agent.parameters() if p.requires_grad), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        agent.clear()
        losses.append(float(loss.detach()))
        if (step + 1) % 10 == 0:
            print(json.dumps({"step": step + 1, "loss": sum(losses[-10:]) / 10}), flush=True)
    cases = []
    for color in COLORS:
        for index, template in enumerate(TEST_TEMPLATES):
            source = example(template, color, 1000 + index)
            other = COLORS[(COLORS.index(color) + 1) % len(COLORS)]
            cases.append({"color": color, "source": source,
                          "shuffled": example(template, other, 1000 + index)})
    metrics = evaluate(agent, tokenizer, cases, args.device)
    metrics.update({"model": args.model, "seed": args.seed, "steps": args.steps,
                    "encoder": args.encoder,
                    "rank": args.rank, "experts": args.experts, "layers": args.layers,
                    "loss_first_10": sum(losses[:10]) / min(10, len(losses)),
                    "loss_last_10": sum(losses[-10:]) / min(10, len(losses)),
                    "task": "controlled four-color trajectory-to-adapter transfer; no benchmark claim"})
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps({"accuracy": metrics["accuracy"], "output": str(output)}), flush=True)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="current_work/delta-Mem/model/Qwen3-4B-Instruct-2507")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpu-fraction", type=float, default=0.40)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--experts", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--encoder", choices=("gru", "attention"), default="attention")
    parser.add_argument("--output", default="results/trajectory_hyperlora/pilot_20261004/metrics.json")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
