"""Train a raw-trajectory hypernetwork that mounts LoRA on frozen Qwen.

The target prompt contains public state/goal only. Source history affects
Qwen solely through generated LoRA factors on its final MLP down projections.
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
from ttcl.trajectory_hyperlora.train_xland_raw_hyperlora import (
    RawHyperLoRA, checked_query, make_split,
)


ACTIONS = (0, 3, 4)
ARMS = ("hold", "near", "tile_near")


def target_question(content: dict) -> str:
    state = content["target_initial_state"]
    rows = [" ".join(f"({tile[0]},{tile[1]})" for tile in row)
            for row in state["observation"]]
    question = (
        "In this grid environment, select the ONE action that produces the "
        "target tile after a single step. Actions: 0=move forward, 3=pick up, "
        "4=put down. Tiles are (type,color) pairs. Reply with only 0, 3, or 4.\n"
        f"Target tile: ({content['goal'][0]},{content['goal'][1]})\n"
        f"Held tile: ({state['pocket'][0]},{state['pocket'][1]})\n"
        "Current 5x5 observation (top to bottom):\n" + "\n".join(rows) +
        "\nAction:")
    return question


def target_prompt(tokenizer, content: dict) -> list[int]:
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": target_question(content)}], tokenize=False,
        add_generation_prompt=True)
    return tokenizer(text, add_special_tokens=False).input_ids


def queries(items: list[dict], tokenizer) -> list[list[int]]:
    output = []
    for item in items:
        first, _ = checked_query(item["arms"][ARMS[0]]["queries"][0])
        for arm in ARMS[1:]:
            other, _ = checked_query(item["arms"][arm]["queries"][0])
            if (other["goal"] != first["goal"] or
                    other["target_initial_state"] != first["target_initial_state"]):
                raise ValueError("Arms do not share a public target query")
        output.append(target_prompt(tokenizer, first))
    return output


def group(data: dict, index: int) -> dict:
    begin = index * 3
    return {key: value[begin:begin + 3] if isinstance(value, torch.Tensor)
            else value for key, value in data.items()}


class QwenRawHyperLoRA(nn.Module):
    def __init__(self, base: nn.Module, rank: int = 8, layers: int = 2,
                 width: int = 64, order_invariant_source: bool = False) -> None:
        super().__init__()
        for parameter in base.parameters():
            parameter.requires_grad_(False)
        self.base = base
        self.encoder = RawHyperLoRA(width=width, rank=rank,
            order_invariant_source=order_invariant_source)
        for module in (self.encoder.hyper, self.encoder.policy,
                       self.encoder.base_head, self.encoder.lora_a):
            for parameter in module.parameters():
                parameter.requires_grad_(False)
        self.adapters = nn.ModuleList()
        self.heads = nn.ModuleList()
        for block in base.model.layers[-layers:]:
            adapter = GeneratedDownProjection(block.mlp.down_proj, rank)
            adapter.scale = rank / 2
            block.mlp.down_proj = adapter
            self.adapters.append(adapter)
            head = nn.Linear(width, adapter.base.out_features * rank)
            nn.init.normal_(head.weight, std=.02)
            nn.init.zeros_(head.bias)
            self.heads.append(head)

    def compile_adapters(self, source: dict) -> list[torch.Tensor]:
        latent = self.encoder.source_latent(source)
        return [head(latent).reshape(len(latent), adapter.base.out_features,
                                     adapter.rank)
                for head, adapter in zip(self.heads, self.adapters, strict=True)]

    def mount(self, factors: list[torch.Tensor] | None) -> None:
        for index, adapter in enumerate(self.adapters):
            adapter.b = None if factors is None else factors[index]

    def choice_logits(self, ids: list[int], choice_ids: list[int],
                      batch: int, device: str) -> torch.Tensor:
        tokens = torch.tensor(ids, device=device).unsqueeze(0).expand(batch, -1)
        logits = self.base(input_ids=tokens, use_cache=False).logits[:, -1]
        return logits[:, choice_ids].float()


def evaluate(agent, data: dict, prompts: list[list[int]], choice_ids: list[int],
             device: str, limit: int | None = None) -> dict:
    agent.eval()
    counts = {kind: 0 for kind in ("correct", "wrong", "none")}
    changed = 0
    n_groups = min(len(prompts), limit or len(prompts))
    with torch.no_grad():
        for i in range(n_groups):
            source = group(data, i)
            factors = agent.compile_adapters(source)
            outputs = {}
            for kind in counts:
                mounted = ([factor[[1, 2, 0]] for factor in factors]
                           if kind == "wrong" else None if kind == "none"
                           else factors)
                agent.mount(mounted)
                outputs[kind] = agent.choice_logits(
                    prompts[i], choice_ids, 3, device).argmax(-1)
                counts[kind] += int((outputs[kind] == torch.tensor(
                    [ACTIONS.index(int(x)) for x in source["target"]],
                    device=device)).sum())
            changed += int((outputs["correct"] != outputs["wrong"]).sum())
    agent.mount(None)
    return {"n": 3 * n_groups, **counts, "swap_changes": changed}


def run(args: argparse.Namespace) -> dict:
    if args.output.exists() or args.checkpoint.exists():
        raise FileExistsError("Fresh output required")
    raw = args.annotations.read_bytes()
    annotations = json.loads(raw)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if args.device.startswith("cuda"):
        torch.cuda.set_per_process_memory_fraction(args.gpu_fraction,
                                                   device=args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    action_tokens = [tokenizer(str(action), add_special_tokens=False).input_ids
                     for action in ACTIONS]
    if any(len(tokens) != 1 for tokens in action_tokens):
        raise ValueError("Action choices must each be a single token")
    choice_ids = [tokens[0] for tokens in action_tokens]
    split = {name: make_split(annotations["split"][name], args.device)
             for name in ("train", "dev", "test")}
    prompts = {name: queries(annotations["split"][name], tokenizer)
               for name in split}
    base = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, local_files_only=True,
        attn_implementation="sdpa").to(args.device)
    base.config.use_cache = False
    agent = QwenRawHyperLoRA(base, args.rank, args.layers, args.width,
                             args.order_invariant_source).to(args.device)
    optimizer = torch.optim.AdamW((p for p in agent.parameters() if p.requires_grad),
                                  lr=args.lr, weight_decay=0)
    history, best_dev, best_step, best_state = [], -1, 0, None
    order = list(range(len(prompts["train"])))
    for step in range(1, args.steps + 1):
        agent.train()
        if (step - 1) % len(order) == 0:
            random.shuffle(order)
        index = order[(step - 1) % len(order)]
        data = group(split["train"], index)
        factors = agent.compile_adapters(data)
        agent.mount(factors)
        logits = agent.choice_logits(prompts["train"][index], choice_ids,
                                     3, args.device)
        targets = torch.tensor([ACTIONS.index(int(x)) for x in data["target"]],
                               device=args.device)
        competing = targets.roll(-1)
        advantage = logits.gather(1, targets[:, None]).squeeze(1) - \
            logits.gather(1, competing[:, None]).squeeze(1)
        loss = F.cross_entropy(logits, targets) + args.pair_weight * F.softplus(
            args.pair_margin - advantage).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in agent.parameters() if p.requires_grad], 1.0)
        optimizer.step()
        agent.mount(None)
        if step % args.eval_every == 0 or step == args.steps:
            dev = evaluate(agent, split["dev"], prompts["dev"], choice_ids,
                           args.device, args.dev_probe)
            history.append({"step": step, "loss": float(loss.detach()),
                            "dev_probe": dev})
            print(json.dumps(history[-1]), flush=True)
            if dev["correct"] > best_dev:
                best_dev, best_step = dev["correct"], step
                best_state = {name: p.detach().cpu().clone()
                              for name, p in agent.named_parameters()
                              if p.requires_grad}
    if best_state is None:
        raise RuntimeError("No checkpoint selected")
    params = dict(agent.named_parameters())
    with torch.no_grad():
        for name, value in best_state.items():
            params[name].copy_(value.to(params[name].device))
    final = {name: evaluate(agent, split[name], prompts[name], choice_ids,
                            args.device) for name in ("dev", "test")}
    result = {"protocol": "Raw XLand trajectories -> generated Qwen LoRA; frozen Qwen; train-only future action CE plus paired margin; dev-selected checkpoint; test once",
              "annotations_sha256": hashlib.sha256(raw).hexdigest(),
              "model_config_sha256": hashlib.sha256((args.model / "config.json").read_bytes()).hexdigest(),
              "seed": args.seed, "steps": args.steps, "best_step": best_step,
              "rank": args.rank, "layers": args.layers, "width": args.width,
              "order_invariant_source": args.order_invariant_source,
              "pair_weight": args.pair_weight, "pair_margin": args.pair_margin,
              "choice_token_ids": choice_ids, "history": history, **final}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    torch.save({"trainable_state": best_state,
                "annotations_sha256": result["annotations_sha256"],
                "order_invariant_source": args.order_invariant_source,
                "seed": args.seed}, args.checkpoint)
    print(json.dumps(final), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path(
        "current_work/delta-Mem/model/Qwen3-4B-Instruct-2507"))
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpu-fraction", type=float, default=.55)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--dev-probe", type=int, default=12)
    parser.add_argument("--lr", type=float, default=.0003)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--order-invariant-source", action="store_true")
    parser.add_argument("--pair-weight", type=float, default=1.)
    parser.add_argument("--pair-margin", type=float, default=1.)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    if (args.steps < 1 or args.eval_every < 1 or args.dev_probe < 1 or
            args.lr <= 0 or args.pair_weight < 0 or
            not 0 < args.gpu_fraction <= 1):
        parser.error("Invalid training settings")
    run(args)


if __name__ == "__main__":
    main()
