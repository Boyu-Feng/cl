"""Read-only own/wrong effective LoRA separation for reviewed sibling sources."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics

import torch

from ttcl.trajectory_hyperlora.alfworld_same_game_retry import file_hash
from ttcl.trajectory_hyperlora.alfworld_zero_shot_probe import load_agent
from ttcl.trajectory_hyperlora.contextual_alf_source import contextual_source_fields
from ttcl.trajectory_hyperlora.diagnose_source_adapters import effective_gram
from ttcl.trajectory_hyperlora.relational_router_pilot import tokenize_records


def run(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    review = json.loads(args.source_review.read_text())
    if (review["candidates_sha256"] != file_hash(args.candidates) or
            len(review["annotations"]) != 42 or
            any(note["approved"] is not True for note in review["annotations"])):
        raise ValueError("Sibling source review changed")
    agent, tokenizer = load_agent(args.model, args.checkpoint,
                                  args.device, args.gpu_fraction)
    adapters = []
    with torch.no_grad():
        for note in review["annotations"]:
            pair = []
            for key in ("source_records", "wrong_records"):
                fields = (contextual_source_fields(agent, tokenizer,
                    note[key], args.device, args.contextual_source_max_tokens)
                    if agent.encoder_kind == "contextual" else
                    tokenize_records(tokenizer, note[key], args.device,
                        max_tokens=args.source_max_tokens,
                        truncation_mode="head_tail"))
                agent.set_source(fields)
                pair.append([layer.b.detach().cpu().squeeze(0).clone()
                             for layer in agent.adapters])
                agent.set_source(None)
            adapters.extend(pair)
    gram = effective_gram(adapters,
        [layer.a.detach().cpu() for layer in agent.adapters])
    norm = gram.diag().clamp_min(0).sqrt()
    cosine = gram / (norm[:, None] * norm[None, :]).clamp_min(1e-12)
    paired = [float(cosine[2 * i, 2 * i + 1]) for i in range(42)]
    result = {"protocol": "Read-only effective B@A comparison of correct sibling and wrong same-family reviewed train histories; no target action/reward used",
        "checkpoint_sha256": file_hash(args.checkpoint),
        "source_review_sha256": file_hash(args.source_review),
        "encoder_kind": agent.encoder_kind,
        "source_max_tokens": args.source_max_tokens,
        "contextual_source_max_tokens": args.contextual_source_max_tokens,
        "pairs": 42,
        "own_wrong_mean_cosine": statistics.mean(paired),
        "own_wrong_min_cosine": min(paired),
        "own_wrong_max_cosine": max(paired),
        "mean_effective_norm": float(norm.mean())}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False,
                                      indent=2) + "\n")
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path(
        "current_work/delta-Mem/model/Qwen3-4B-Instruct-2507"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_sibling_train42_candidates_20261006.json"))
    parser.add_argument("--source-review", type=Path, default=Path(
        "data/annotations/alf_sibling_train42_reviewed_20261006.json"))
    parser.add_argument("--source-max-tokens", type=int, default=40)
    parser.add_argument("--contextual-source-max-tokens", type=int,
                        default=2048)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpu-fraction", type=float, default=.6)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (args.source_max_tokens < 2 or
            args.contextual_source_max_tokens < 2 or
            not 0 < args.gpu_fraction <= 1):
        parser.error("Invalid source geometry budget")
    run(args)


if __name__ == "__main__":
    main()
