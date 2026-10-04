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
from peft import LoraConfig, TaskType
from safetensors.torch import save_file
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
FRESH_TEMPLATES = (
    "Archived episode {episode} ended after the assistant gave {color} as the session color. The response passed validation.",
    "A previous run, numbered {episode}, received a request for the conventional color. Its final response was {color}, and the task succeeded.",
)
DISTRACTOR_TEMPLATES = (
    "Run {episode}: the agent tried {other} and got rejected; it then replied {color} and the environment accepted the response.",
    "History {episode}: first answer {other}, feedback incorrect. Revised answer {color}, feedback correct. The episode ended.",
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
                 layers: int = 2, encoder_kind: str = "attention",
                 pretrain_tokens: bool = False) -> None:
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
        self.reconstruction_head = (
            nn.Linear(96 if encoder_kind == "gru" else 128, model.config.hidden_size)
            if pretrain_tokens else None
        )
        self.source_features: torch.Tensor | None = None
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
        self.source_features = pooled
        return torch.softmax(self.projection(pooled), dim=-1)

    def reconstruction_loss(self, token_id: int) -> torch.Tensor:
        if self.reconstruction_head is None or self.source_features is None:
            raise RuntimeError("Source reconstruction was not enabled")
        vector = self.reconstruction_head(self.source_features)
        logits = self.model.lm_head(vector.to(self.model.lm_head.weight.dtype)).float()
        target = torch.tensor([token_id], device=logits.device)
        return F.cross_entropy(logits, target)

    def set_source(self, source_ids: torch.Tensor | None) -> None:
        coefficients = self.encode(source_ids) if source_ids is not None else None
        for adapter in self.adapters:
            adapter.coefficients = coefficients

    def clear(self) -> None:
        for adapter in self.adapters:
            adapter.coefficients = None
        self.source_features = None


def prompt(tokenizer: object, query: str, history: str | None = None) -> str:
    content = (f"Previous completed episode:\n{history}\n\n" if history else "") + query
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True
    )


def example(template: str, color: str, episode: int, other: str = "") -> str:
    return template.format(episode=episode, color=color, other=other)


def evaluate(agent: TrajectoryHyperLoRA, tokenizer: object, cases: list[dict], device: str) -> dict:
    agent.eval()
    rows = []
    for case in cases:
        row = {"label": case["color"], "group": case.get("group", "original"),
               "source_hash": hashlib.sha256(case["source"].encode()).hexdigest()}
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
    arms = ("base", "text", "hyper", "shuffled")
    groups = sorted({row["group"] for row in rows})
    return {"n": len(rows),
            "accuracy": {arm: sum(row[arm]["correct"] for row in rows) / len(rows) for arm in arms},
            "group_accuracy": {
                group: {arm: sum(row[arm]["correct"] for row in rows if row["group"] == group)
                        / sum(row["group"] == group for row in rows) for arm in arms}
                for group in groups
            }, "rows": rows}


def export_peft_lora(agent: TrajectoryHyperLoRA, tokenizer: object, source: str,
                     output_dir: Path, model_path: str, device: str) -> dict:
    """Materialize one generated update as a standard PEFT LoRA adapter."""
    agent.eval()
    source_ids = tokenizer(source, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        agent.set_source(source_ids)
        coefficients = agent.adapters[0].coefficients[0].detach().float().cpu()
    rank = agent.adapters[0].a.shape[1]
    experts = len(coefficients)
    total_rank = rank * experts
    last_layer = len(agent.model.model.layers) - len(agent.adapters)
    targets = []
    tensors = {}
    for offset, adapter in enumerate(agent.adapters):
        layer_index = last_layer + offset
        name = f"model.layers.{layer_index}.mlp.down_proj"
        targets.append(name)
        a = adapter.a.detach().float().cpu().reshape(total_rank, -1).contiguous()
        b = (adapter.b.detach().float().cpu()
             * coefficients[:, None, None] / rank)
        b = b.permute(1, 0, 2).reshape(adapter.b.shape[1], total_rank).contiguous()
        prefix = f"base_model.model.{name}"
        tensors[f"{prefix}.lora_A.weight"] = a
        tensors[f"{prefix}.lora_B.weight"] = b
    output_dir.mkdir(parents=True, exist_ok=True)
    config = LoraConfig(
        r=total_rank, lora_alpha=total_rank, lora_dropout=0.0,
        target_modules=targets, bias="none", task_type=TaskType.CAUSAL_LM,
        inference_mode=True, base_model_name_or_path=str(Path(model_path).resolve()),
    )
    config.save_pretrained(output_dir)
    save_file(tensors, output_dir / "adapter_model.safetensors")
    manifest = {"source_sha256": hashlib.sha256(source.encode()).hexdigest(),
                "base_model": str(Path(model_path).resolve()),
                "targets": targets, "rank": total_rank,
                "coefficient_values": coefficients.tolist()}
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    agent.clear()
    return manifest


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
                                layers=args.layers, encoder_kind=args.encoder,
                                pretrain_tokens=args.warmup_steps > 0).to(args.device)
    optimizer = torch.optim.AdamW((p for p in agent.parameters() if p.requires_grad), lr=args.lr)
    warmup_losses = []
    for step in range(args.warmup_steps):
        color = COLORS[step % len(COLORS)]
        source = example(rng.choice(TRAIN_TEMPLATES), color, step)
        source_ids = tokenizer(source, return_tensors="pt").input_ids.to(args.device)
        agent.set_source(source_ids)
        target_token = tokenizer(" " + color, add_special_tokens=False).input_ids[0]
        loss = agent.reconstruction_loss(target_token)
        loss.backward()
        torch.nn.utils.clip_grad_norm_((p for p in agent.parameters() if p.requires_grad), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        agent.clear()
        warmup_losses.append(float(loss.detach()))
        if (step + 1) % 50 == 0:
            print(json.dumps({"warmup_step": step + 1,
                              "loss": sum(warmup_losses[-50:]) / 50}), flush=True)
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
    template_groups = {
        "original": TEST_TEMPLATES,
        "fresh": FRESH_TEMPLATES,
        "distractor": DISTRACTOR_TEMPLATES,
    }
    selected_groups = tuple(template_groups) if args.test_set == "combined" else (args.test_set,)
    cases = []
    for group in selected_groups:
        for color in COLORS:
            for index, template in enumerate(template_groups[group]):
                other = COLORS[(COLORS.index(color) + 1) % len(COLORS)]
                source = example(template, color, 1000 + index, other=other)
                shuffled_other = COLORS[(COLORS.index(other) + 1) % len(COLORS)]
                cases.append({"color": color, "group": group, "source": source,
                              "shuffled": example(template, other, 1000 + index,
                                                  other=shuffled_other)})
    metrics = evaluate(agent, tokenizer, cases, args.device)
    metrics.update({"model": args.model, "seed": args.seed, "steps": args.steps,
                    "encoder": args.encoder,
                    "test_set": args.test_set,
                    "warmup_steps": args.warmup_steps,
                    "warmup_loss_last_10": sum(warmup_losses[-10:]) / min(10, len(warmup_losses))
                    if warmup_losses else None,
                    "rank": args.rank, "experts": args.experts, "layers": args.layers,
                    "loss_first_10": sum(losses[:10]) / min(10, len(losses)),
                    "loss_last_10": sum(losses[-10:]) / min(10, len(losses)),
                    "task": "controlled four-color trajectory-to-adapter transfer; no benchmark claim"})
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2) + "\n")
    if args.checkpoint_path:
        checkpoint = Path(args.checkpoint_path)
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"trainable_state": {name: parameter.detach().cpu()
                                          for name, parameter in agent.named_parameters()
                                          if parameter.requires_grad},
                    "architecture": {"encoder": args.encoder, "experts": args.experts,
                                     "rank": args.rank, "layers": args.layers,
                                     "pretrain_tokens": args.warmup_steps > 0}}, checkpoint)
    if args.export_case is not None:
        if not 0 <= args.export_case < len(cases):
            raise ValueError("export-case index is outside the test cases")
        adapter_dir = output.parent / f"generated_adapter_case_{args.export_case}"
        metrics["exported_adapter"] = str(adapter_dir)
        export_peft_lora(agent, tokenizer, cases[args.export_case]["source"],
                         adapter_dir, args.model, args.device)
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
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--test-set", choices=("original", "fresh", "distractor", "combined"),
                        default="original")
    parser.add_argument("--output", default="results/trajectory_hyperlora/pilot_20261004/metrics.json")
    parser.add_argument("--checkpoint-path", default=None)
    parser.add_argument("--export-case", type=int, default=None)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
