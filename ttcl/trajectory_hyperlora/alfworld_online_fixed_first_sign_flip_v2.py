"""Causal control: negate the first own-success LoRA while keeping its norm."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ttcl.trajectory_hyperlora.alfworld_online_context_vector_mean_v1 import vector_hash
from ttcl.trajectory_hyperlora.alfworld_online_from_empty import (
    digest, file_hash, records_from_episode,
)
from ttcl.trajectory_hyperlora.alfworld_online_persistent_context_lora_v1 import (
    bounded_source_text,
)
from ttcl.trajectory_hyperlora.alfworld_online_reward_gate_valid_unseen_v1 import (
    checked_targets,
)
from ttcl.trajectory_hyperlora.alfworld_zero_shot_probe import load_agent, run_episode
from ttcl.trajectory_hyperlora.contextual_alf_source import contextual_text_fields


def source_binding(args):
    reference = json.loads(args.reference.read_text())
    targets = checked_targets(argparse.Namespace(**{
        **vars(args), "review": args.target_review}))
    if (reference["review_sha256"] != file_hash(args.target_review) or
            reference["checkpoint_sha256"] != file_hash(args.checkpoint) or
            len(reference["games"]) != len(targets)):
        raise ValueError("Changed reference online trajectory")
    source = reference["games"][args.source_index]
    episode = source["online"]
    if (source["game"] != targets[args.source_index]["game"] or
            episode["status"] != "complete" or episode["reward"] != 1):
        raise ValueError("Selected original own source was not successful")
    records = records_from_episode(episode)
    if source["new_records_sha256"] != digest(records):
        raise ValueError("Changed original own source records")
    content = {"reference_sha256": file_hash(args.reference),
        "target_review_sha256": file_hash(args.target_review),
        "checkpoint_sha256": file_hash(args.checkpoint),
        "source_index": args.source_index,
        "source_game": source["game"],
        "source_records_sha256": digest(records),
        "method": "negate_first_successful_own_trajectory_lora"}
    return reference, targets, records, content


def prepare(args):
    if args.review.exists():
        raise FileExistsError(args.review)
    _, targets, _, content = source_binding(args)
    value = {"protocol": "Fresh content-bound first-own-success sign-flip LoRA control on frozen valid_unseen 36 targets",
        **content, "input_content_sha256": digest(content),
        "reviewed_source": True, "target_count": len(targets)}
    args.review.parent.mkdir(parents=True, exist_ok=True)
    args.review.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"reviewed_targets": len(targets),
                      "source_index": args.source_index}), flush=True)


def evaluate(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    reference, targets, records, content = source_binding(args)
    review = json.loads(args.review.read_text())
    if (not review["reviewed_source"] or
            review["target_count"] != len(targets) or
            review["input_content_sha256"] != digest(content) or
            any(review[key] != val for key, val in content.items())):
        raise ValueError("Unreviewed fixed own source")
    agent, tokenizer = load_agent(args.model, args.checkpoint,
                                  args.device, args.gpu_fraction)
    if (agent.encoder_kind != "contextual" or
            agent.task_context_scope != "current" or
            agent.task_pair_pooling != "mean"):
        raise ValueError("Expected frozen current-context hypernetwork")
    original_set_source = agent.set_source

    def signed_set_source(fields, oracle_bits=None, target_fields=None):
        original_set_source(fields, oracle_bits=oracle_bits,
                            target_fields=target_fields)
        if fields is not None:
            for adapter in agent.adapters:
                if adapter.b is not None:
                    adapter.b = -adapter.b

    agent.set_source = signed_set_source
    fields = None
    report = {"protocol": "Counterfactual same-source sign-flip control: replay first success from original from-empty chain, freeze its source vector, negate every generated B while keeping A and factor norm fixed; same 36 valid_unseen games and actor budget",
        "adapter_sign": -1,
        "review_sha256": file_hash(args.review),
        "reference_sha256": file_hash(args.reference),
        "target_review_sha256": file_hash(args.target_review),
        "checkpoint_sha256": file_hash(args.checkpoint),
        "source_index": args.source_index,
        "max_steps": 50, "max_new_tokens": 64,
        "actor_history_turns": 2, "loop_guard_max": 2,
        "context_tokens": args.context_tokens,
        "games": [], "failures": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for index, row in enumerate(targets):
        if index == args.source_index + 1:
            text, original, retained = bounded_source_text(
                tokenizer, records, args.context_tokens)
            source = reference["games"][args.source_index]
            if (source["source_tokens_original"] != original or
                    source["source_tokens_retained"] != retained):
                raise ValueError("Original source token binding changed")
            fields = contextual_text_fields(agent, tokenizer, text,
                args.device, args.context_tokens, pooling="both")
            if vector_hash(fields) != source["source_vector_sha256_after"]:
                raise ValueError("Fixed source does not match original first update")
        episode = run_episode(agent, tokenizer, args.data_root / row["game"],
            fields or {}, adapter=fields is not None,
            device=args.device, max_steps=50, max_new_tokens=64,
            constrain_actions=True, actor_history_turns=2, loop_guard_max=2)
        entry = {"game": row["game"], "family": row["family"],
            "input_content_sha256": row["input_content_sha256"],
            "fixed_source_vector_sha256": vector_hash(fields),
            "episode": episode}
        if episode["status"] != "complete":
            report["failures"].append({"game": row["game"],
                "error": episode.get("error", "incomplete episode")})
        if index <= args.source_index and episode["status"] == "complete":
            if episode["trajectory"] != reference["games"][index]["online"]["trajectory"]:
                report["failures"].append({"game": row["game"],
                    "error": "Before-first-source replay diverged"})
        report["games"].append(entry)
        report["summary"] = {"n": len(report["games"]),
            "success": sum(x["episode"].get("reward", 0)
                           for x in report["games"]),
            "failures": len(report["failures"])}
        temp = args.output.with_suffix(args.output.suffix + ".tmp")
        temp.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        temp.replace(args.output)
        print(json.dumps(report["summary"]), flush=True)
        if report["failures"]:
            raise RuntimeError("Fixed first-source control failure recorded")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "evaluate"))
    parser.add_argument("--model", type=Path, default=Path(
        "current_work/delta-Mem/model/Qwen3-4B-Instruct-2507"))
    parser.add_argument("--checkpoint", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_sibling_simple40_onpolicy600_20261006.pt"))
    parser.add_argument("--parent-review", type=Path, default=Path(
        "data/annotations/alf_memrl134_hyperlora_train_sources_reviewed_20261006.json"))
    parser.add_argument("--target-review", type=Path, default=Path(
        "data/annotations/alf_online_reward_gate_valid_unseen_36_reviewed_20261007.json"))
    parser.add_argument("--review", type=Path, default=Path(
        "data/annotations/alf_online_fixed_first_sign_flip_reviewed_20261007.json"))
    parser.add_argument("--reference", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_online_reward_gate_valid_unseen_36_20261007.json"))
    parser.add_argument("--output", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_online_fixed_first_sign_flip_valid_unseen36_20261007.json"))
    parser.add_argument("--data-root", type=Path, default=Path(
        "ttcl/data/alfworld_delta"))
    parser.add_argument("--source-index", type=int, default=1)
    parser.add_argument("--per-family", type=int, default=6)
    parser.add_argument("--context-tokens", type=int, default=2048)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpu-fraction", type=float, default=.65)
    args = parser.parse_args()
    if args.source_index < 0 or args.per_family < 1:
        parser.error("Invalid source index or family count")
    (prepare if args.command == "prepare" else evaluate)(args)


if __name__ == "__main__":
    main()
