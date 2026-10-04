"""Train only a centered trajectory-to-LoRA residual on next-game actions.

The previously trained public LoRA component and frozen Qwen actor are kept
fixed. A source-dependent head is optimized to improve reviewed next-game
actions relative to a different-family source, with no development labels in
the optimizer. This is an offline meta-learning proxy, not environment RL.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random

import torch
from torch import nn
from torch.nn import functional as F

from ttcl.experience_evolution.environment import clean_command
from ttcl.trajectory_hyperlora.alfworld_frozen_eval_probe import mean_adapter
from ttcl.trajectory_hyperlora.alfworld_zero_shot_probe import load_agent
from ttcl.trajectory_hyperlora.prepare_alf_next_task import (
    digest_json, partition, validate_review,
)
from ttcl.trajectory_hyperlora.relational_router_pilot import tokenize_records
from ttcl.trajectory_hyperlora.train_alf_next_task import (
    generate, query_ids, target_loss,
)


def reviewed_sources(rows: list[dict]) -> dict[str, dict]:
    sources = {}
    for row in rows:
        if partition(row["source_game"]) != "train":
            raise ValueError("Source outside reviewed training partition")
        previous = sources.get(row["source_game"])
        if previous and (previous["source_episode_sha256"] != row["source_episode_sha256"]
                         or previous["source_records"] != row["source_records"]):
            raise ValueError("Conflicting reviewed source content")
        sources[row["source_game"]] = row
    return sources


class CenteredResidual(nn.Module):
    def __init__(self, agent, mean_factors: list[torch.Tensor],
                 mean_latent: torch.Tensor) -> None:
        super().__init__()
        self.agent = agent
        self.register_buffer("mean_latent", mean_latent.detach().float())
        self.heads = nn.ModuleList()
        for adapter in agent.adapters:
            head = nn.Linear(mean_latent.numel(),
                             adapter.base.out_features * adapter.rank,
                             bias=False)
            nn.init.zeros_(head.weight)
            self.heads.append(head)
        for index, factor in enumerate(mean_factors):
            self.register_buffer(f"mean_factor_{index}", factor.detach().float())
        self.mean_count = len(mean_factors)
        self.mean_square = sum(float(factor.float().square().sum())
                               for factor in mean_factors)

    def mean_factors(self) -> list[torch.Tensor]:
        return [getattr(self, f"mean_factor_{index}")
                for index in range(self.mean_count)]

    def mount(self, latent: torch.Tensor | None) -> torch.Tensor:
        residual_square = torch.zeros((), device=self.mean_latent.device)
        for adapter, head, mean in zip(self.agent.adapters, self.heads,
                                       self.mean_factors(), strict=True):
            if latent is None:
                residual = torch.zeros_like(mean)
            else:
                residual = head((latent - self.mean_latent).float()).reshape_as(mean)
            adapter.b = mean + residual
            residual_square = residual_square + residual.square().sum()
        return residual_square / max(self.mean_square, 1e-12)

    def clear(self) -> None:
        self.agent.set_source(None)


def evaluate(agent, residual: CenteredResidual, tokenizer,
             rows: list[dict], prefixes: dict[str, list[int]],
             latents: dict[str, torch.Tensor], wrong_source: dict[str, str],
             device: str, max_new_tokens: int) -> dict:
    output = []
    for row in rows:
        source = row["source_game"]
        wrong = wrong_source[row["input_content_sha256"]]
        key = row["input_content_sha256"]
        available = row["target_messages"][-1]["content"].split(
            "\nAvailable commands:\n", 1)[-1].splitlines()
        result = {"input_content_sha256": key, "family": row["family"],
                  "target_game": row["target_game"],
                  "source_game": source, "wrong_source_game": wrong,
                  "target_action": row["target_action"]}
        for arm in ("base", "mean", "original", "centered_correct",
                    "centered_wrong"):
            with torch.no_grad():
                if arm == "base":
                    residual.clear()
                elif arm == "mean":
                    residual.mount(None)
                elif arm == "original":
                    fields = tokenize_records(tokenizer, row["source_records"], device)
                    agent.set_source(fields)
                elif arm == "centered_correct":
                    residual.mount(latents[source])
                else:
                    residual.mount(latents[wrong])
                answer = generate(agent, tokenizer, prefixes[key], device,
                                  max_new_tokens)
                command = clean_command(answer, available)
            result[arm] = {"answer": answer, "command": command,
                           "valid": command in available,
                           "exact_target": command == row["target_action"]}
            residual.clear()
        output.append(result)
    arms = ("base", "mean", "original", "centered_correct", "centered_wrong")
    return {"n": len(output),
            "summary": {arm: {"exact_target": sum(row[arm]["exact_target"]
                                                 for row in output),
                              "valid": sum(row[arm]["valid"] for row in output)}
                        for arm in arms},
            "rows": output}


def run(args: argparse.Namespace) -> dict:
    if args.output.exists() or args.steps < 1 or args.dev_limit < 0 or args.margin < 0 or \
            args.rank_weight < 0 or args.residual_penalty < 0:
        raise ValueError("Need fresh output and valid training budgets")
    manifest = json.loads(args.candidates.read_text())
    review = json.loads(args.review.read_text())
    approved = validate_review(manifest, review)
    train = [row for row in approved if row["split"] == "train"]
    dev = [row for row in approved if row["split"] == "dev"]
    sources = reviewed_sources(approved)
    by_family = defaultdict(list)
    for game, row in sources.items():
        by_family[row["family"]].append(game)
    wrong_source = {}
    for row in approved:
        choices = [game for fam, games in by_family.items()
                   if fam != row["family"] for game in games]
        wrong_source[row["input_content_sha256"]] = min(
            choices, key=lambda game: digest_json(
                ["source_residual_wrong", row["input_content_sha256"], game]))
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    agent, tokenizer = load_agent(args.model, args.checkpoint,
                                  args.device, args.gpu_fraction)
    for parameter in agent.parameters():
        parameter.requires_grad_(False)
    mean_factors, mean_sha, mean_count = mean_adapter(
        agent, tokenizer, approved, args.device)
    latents = {}
    with torch.no_grad():
        for game, row in sources.items():
            fields = tokenize_records(tokenizer, row["source_records"], args.device)
            latents[game] = agent.encode(fields).detach().float().squeeze(0)
    mean_latent = torch.stack(list(latents.values())).mean(0)
    residual = CenteredResidual(agent, mean_factors, mean_latent).to(args.device)
    optimizer = torch.optim.AdamW(residual.heads.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    prefixes = {row["input_content_sha256"]: query_ids(
        tokenizer, row, history_turns=args.history_turns,
        max_prompt_tokens=args.max_prompt_tokens) for row in approved}
    order = list(range(len(train)))
    losses = []
    for step in range(args.steps):
        if step % len(order) == 0:
            rng.shuffle(order)
        row = train[order[step % len(order)]]
        key = row["input_content_sha256"]
        correct = latents[row["source_game"]]
        wrong = latents[wrong_source[key]]
        optimizer.zero_grad(set_to_none=True)
        penalty_correct = residual.mount(correct)
        correct_loss = target_loss(agent, tokenizer, row, prefixes[key], args.device)
        with torch.no_grad():
            residual.mount(wrong)
            wrong_reference = target_loss(agent, tokenizer, row, prefixes[key],
                                          args.device).detach()
        weight = torch.sigmoid(args.margin + correct_loss.detach() - wrong_reference)
        ((1 + args.rank_weight * weight) * correct_loss +
         .5 * args.residual_penalty * penalty_correct).backward()
        penalty_wrong = residual.mount(wrong)
        wrong_loss = target_loss(agent, tokenizer, row, prefixes[key], args.device)
        (-args.rank_weight * weight * wrong_loss +
         .5 * args.residual_penalty * penalty_wrong).backward()
        torch.nn.utils.clip_grad_norm_(residual.heads.parameters(), 1.0)
        optimizer.step()
        residual.clear()
        losses.append({"correct_ce": float(correct_loss.detach()),
                       "wrong_ce": float(wrong_loss.detach()),
                       "rank_gap": float((wrong_loss - correct_loss).detach())})
        if (step + 1) % 20 == 0:
            recent = losses[-20:]
            print(json.dumps({"step": step + 1,
                              "correct_ce": sum(x["correct_ce"] for x in recent) / 20,
                              "wrong_ce": sum(x["wrong_ce"] for x in recent) / 20,
                              "rank_gap": sum(x["rank_gap"] for x in recent) / 20}),
                  flush=True)
    evaluation_rows = dev[:args.dev_limit] if args.dev_limit else dev
    evaluation = evaluate(agent, residual, tokenizer, evaluation_rows, prefixes, latents,
                          wrong_source, args.device, args.max_new_tokens)
    result = {"protocol": "Train-only centered trajectory LoRA residual on top of frozen public mean; next-task reviewed action CE plus cross-family source ranking; dev offline imitation proxy, not environment reward or RL",
              "seed": args.seed, "steps": args.steps,
              "rank_weight": args.rank_weight, "margin": args.margin,
              "residual_penalty": args.residual_penalty,
              "lr": args.lr, "weight_decay": args.weight_decay,
              "train_targets": len(train), "dev_targets": len(dev),
              "evaluated_dev_targets": len(evaluation_rows),
              "mean_source_count": mean_count,
              "mean_adapter_sha256": mean_sha,
              "base_checkpoint_sha256": hashlib.sha256(
                  args.checkpoint.read_bytes()).hexdigest(),
              "candidate_manifest_sha256": hashlib.sha256(
                  args.candidates.read_bytes()).hexdigest(),
              "review_sha256": hashlib.sha256(args.review.read_bytes()).hexdigest(),
              "last_20": {key: sum(x[key] for x in losses[-20:]) /
                          min(20, len(losses)) for key in losses[0]},
              "evaluation": evaluation}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    if args.save_checkpoint:
        args.save_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"heads_state": residual.heads.state_dict(),
                    "mean_factors": [factor.cpu() for factor in residual.mean_factors()],
                    "mean_latent": residual.mean_latent.detach().cpu(),
                    "base_checkpoint_sha256": result["base_checkpoint_sha256"],
                    "review_sha256": result["review_sha256"]},
                   args.save_checkpoint)
    print(json.dumps({"evaluation": evaluation["summary"],
                      "output": str(args.output)}), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path(
        "current_work/delta-Mem/model/Qwen3-4B-Instruct-2507"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_next_task_candidates_20261005.json"))
    parser.add_argument("--review", type=Path, default=Path(
        "data/annotations/trajectory_hyperlora_alf_20261005_reviewed_v2.json"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpu-fraction", type=float, default=.40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--lr", type=float, default=.0002)
    parser.add_argument("--weight-decay", type=float, default=.01)
    parser.add_argument("--rank-weight", type=float, default=.5)
    parser.add_argument("--margin", type=float, default=.1)
    parser.add_argument("--residual-penalty", type=float, default=.01)
    parser.add_argument("--history-turns", type=int, default=2)
    parser.add_argument("--max-prompt-tokens", type=int, default=1800)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--dev-limit", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--save-checkpoint", type=Path)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
