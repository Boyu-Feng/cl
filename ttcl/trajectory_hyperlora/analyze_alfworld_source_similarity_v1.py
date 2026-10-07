"""Test whether untrained contextual trajectory vectors retrieve related tasks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from ttcl.trajectory_hyperlora.alfworld_online_from_empty import (
    digest, file_hash, records_from_episode,
)
from ttcl.trajectory_hyperlora.alfworld_online_persistent_context_lora_v1 import (
    bounded_source_text,
)
from ttcl.trajectory_hyperlora.alfworld_zero_shot_probe import load_agent
from ttcl.trajectory_hyperlora.contextual_alf_source import (
    contextual_text_fields, task_context_text,
)


def analyze(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    report = json.loads(args.report.read_text())
    if (report["checkpoint_sha256"] != file_hash(args.checkpoint) or
            report["failures"] or len(report["games"]) != 36):
        raise ValueError("Changed or incomplete online source chain")
    agent, tokenizer = load_agent(args.model, args.checkpoint,
                                  args.device, args.gpu_fraction)
    sources = []
    prior = []
    rows = []
    for target in report["games"]:
        if (target["prior_episode_count"] != len(prior) or
                target["prior_episodes_sha256"] != digest(prior)):
            raise ValueError("Changed online history")
        context = task_context_text(target["online"]["initial_observation"],
                                    target["online"]["initial_observation"])
        query = contextual_text_fields(agent, tokenizer, context,
            args.device, report["context_tokens"], pooling="both")
        query_vector = F.normalize(query["pair_contextual"].float(), dim=-1)
        candidate = []
        for source in sources:
            score = float((query_vector * source["vector"]).sum().item())
            candidate.append({"source_index": source["index"],
                "source_family": source["family"],
                "same_family": source["family"] == target["family"],
                "cosine": score})
        rows.append({"game": target["game"], "family": target["family"],
                     "prior_successes": len(sources),
                     "candidates": candidate})
        episode = target["online"]
        records = records_from_episode(episode)
        prior.append({"game": target["game"],
                      "reward": episode["reward"], "records": records})
        if episode["reward"]:
            text, original, retained = bounded_source_text(
                tokenizer, records, report["context_tokens"])
            if (target["source_tokens_original"] != original or
                    target["source_tokens_retained"] != retained):
                raise ValueError("Source truncation changed")
            fields = contextual_text_fields(agent, tokenizer, text,
                args.device, report["context_tokens"], pooling="both")
            sources.append({"index": len(rows) - 1,
                "family": target["family"],
                "vector": F.normalize(fields["pair_contextual"].float(),
                                      dim=-1)})
        print(json.dumps({"n": len(rows), "sources": len(sources)}),
              flush=True)
    eligible = [row for row in rows if
        any(x["same_family"] for x in row["candidates"]) and
        any(not x["same_family"] for x in row["candidates"])]
    positive = [x["cosine"] for row in eligible for x in row["candidates"]
                if x["same_family"]]
    negative = [x["cosine"] for row in eligible for x in row["candidates"]
                if not x["same_family"]]
    wins = sum(a > b for a in positive for b in negative)
    ties = sum(a == b for a in positive for b in negative)
    top_correct = sum(max(row["candidates"], key=lambda x: x["cosine"])
                      ["same_family"] for row in eligible)
    summary = {"targets": len(rows), "success_sources": len(sources),
        "eligible_targets": len(eligible),
        "same_family_pairs": len(positive),
        "other_family_pairs": len(negative),
        "mean_same_family_cosine": sum(positive) / len(positive),
        "mean_other_family_cosine": sum(negative) / len(negative),
        "pooled_pair_auc": (wins + 0.5 * ties) /
                           (len(positive) * len(negative)),
        "top1_same_family": top_correct,
        "random_top1_same_family_expectation": sum(
            sum(x["same_family"] for x in row["candidates"]) /
            len(row["candidates"]) for row in eligible)}
    result = {"protocol": "Read-only causal-past trajectory/target vector similarity on frozen 36-game online run; family labels used only for analysis, never retrieval or policy",
        "report_sha256": file_hash(args.report),
        "checkpoint_sha256": file_hash(args.checkpoint),
        "summary": summary, "games": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path(
        "current_work/delta-Mem/model/Qwen3-4B-Instruct-2507"))
    parser.add_argument("--checkpoint", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_sibling_simple40_onpolicy600_20261006.pt"))
    parser.add_argument("--report", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_online_reward_gate_valid_unseen_36_20261007.json"))
    parser.add_argument("--output", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_online_source_similarity_valid_unseen36_20261007.json"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpu-fraction", type=float, default=.65)
    analyze(parser.parse_args())


if __name__ == "__main__":
    main()
