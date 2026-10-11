"""ALFWorld: admit only own official wins to one persistent LoRA-B mean."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ttcl.trajectory_hyperlora.alfworld_online_accumulate_lora_v1 import (
    factors_hash, mean_factors,
)
from ttcl.trajectory_hyperlora.alfworld_online_from_empty import (
    digest, file_hash, records_from_episode,
)
from ttcl.trajectory_hyperlora.alfworld_zero_shot_probe import load_agent, run_episode
from ttcl.trajectory_hyperlora.contextual_alf_source import (
    contextual_text_fields, source_text, task_context_text,
)


def bounded_source_text(tokenizer, records, limit: int) -> tuple[str, int, int]:
    """Keep public beginning and ending evidence within the encoder budget."""
    ids = tokenizer(source_text(records), add_special_tokens=False).input_ids
    original = len(ids)
    if original > limit:
        head = limit // 2
        ids = ids[:head] + ids[-(limit - head):]
    text = tokenizer.decode(ids, skip_special_tokens=False)
    retained = len(tokenizer(text, add_special_tokens=False).input_ids)
    if retained > limit:
        raise ValueError("Decoded source text exceeds contextual encoder budget")
    return text, original, retained


def prepare(args):
    if args.review.exists():
        raise FileExistsError(args.review)
    source = json.loads(args.source_review.read_text())
    count = 36 if args.sequence == "valid_unseen36" else 18
    if (len(source["targets"]) != count or
            source["checkpoint_sha256"] != file_hash(args.checkpoint)):
        raise ValueError("Changed reviewed source target count or checkpoint")
    targets = []
    for index, old in enumerate(source["targets"]):
        game = old["game"]
        if (args.sequence == "valid_unseen36") != ("/valid_unseen/" in "/"+game):
            raise ValueError("Official ALFWorld split changed")
        game_hash = file_hash(args.data_root / game)
        if game_hash != old["game_sha256"]:
            raise ValueError("Changed reviewed game content")
        content = {"index": index, "game": game, "family": old.get("family"),
            "game_sha256": game_hash,
            "source_review_sha256": file_hash(args.source_review),
            "checkpoint_sha256": file_hash(args.checkpoint),
            "runner_sha256": file_hash(Path(__file__)),
            "actor_sha256": file_hash(Path(__file__).with_name(
                "alfworld_zero_shot_probe.py")),
            "factor_update_sha256": file_hash(Path(__file__).with_name(
                "alfworld_online_accumulate_lora_v1.py")),
            "model_config_sha256": file_hash(args.model / "config.json"),
            "sequence": args.sequence,
            "context_tokens": args.context_tokens}
        targets.append({**content, "input_content_sha256": digest(content),
            "reviewed_target": True,
            "review_basis": "Checked official game content, frozen model and checkpoint; only new own official wins update persistent B"})
    value = {"protocol": "Fresh content-bound ALFWorld target review for success-gated persistent LoRA-B mean",
        "sequence": args.sequence,
        "source_review_sha256": file_hash(args.source_review),
        "checkpoint_sha256": file_hash(args.checkpoint),
        "runner_sha256": file_hash(Path(__file__)),
        "actor_sha256": file_hash(Path(__file__).with_name(
            "alfworld_zero_shot_probe.py")),
        "factor_update_sha256": file_hash(Path(__file__).with_name(
            "alfworld_online_accumulate_lora_v1.py")),
        "targets": targets, "input_content_sha256": digest(targets)}
    args.review.parent.mkdir(parents=True, exist_ok=True)
    args.review.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"reviewed_targets": len(targets)}), flush=True)


def checked_targets(args):
    value = json.loads(args.review.read_text())
    source = json.loads(args.source_review.read_text())
    if (value["sequence"] != args.sequence or
            value["source_review_sha256"] != file_hash(args.source_review) or
            value["checkpoint_sha256"] != file_hash(args.checkpoint) or
            value["runner_sha256"] != file_hash(Path(__file__)) or
            value["actor_sha256"] != file_hash(Path(__file__).with_name(
                "alfworld_zero_shot_probe.py")) or
            value["factor_update_sha256"] != file_hash(Path(__file__).with_name(
                "alfworld_online_accumulate_lora_v1.py")) or
            value["input_content_sha256"] != digest(value["targets"]) or
            len(value["targets"]) != len(source["targets"])):
        raise ValueError("Persistent LoRA review lineage changed")
    for index, old in enumerate(source["targets"]):
        game = old["game"]
        content = {"index": index, "game": game, "family": old.get("family"),
            "game_sha256": file_hash(args.data_root / game),
            "source_review_sha256": file_hash(args.source_review),
            "checkpoint_sha256": file_hash(args.checkpoint),
            "runner_sha256": file_hash(Path(__file__)),
            "actor_sha256": file_hash(Path(__file__).with_name(
                "alfworld_zero_shot_probe.py")),
            "factor_update_sha256": file_hash(Path(__file__).with_name(
                "alfworld_online_accumulate_lora_v1.py")),
            "model_config_sha256": file_hash(args.model / "config.json"),
            "sequence": args.sequence,
            "context_tokens": args.context_tokens}
        row = value["targets"][index]
        if (not row["reviewed_target"] or
                row["game_sha256"] != old["game_sha256"] or
                row["input_content_sha256"] != digest(content) or
                any(row[key] != val for key, val in content.items())):
            raise ValueError("Unreviewed or changed persistent target")
    return value["targets"]


def episode_factor(agent, tokenizer, episode, args):
    records = records_from_episode(episode)
    text, original, retained = bounded_source_text(
        tokenizer, records, args.context_tokens)
    source = contextual_text_fields(agent, tokenizer, text, args.device,
                                    args.context_tokens, pooling="both")
    initial = episode["initial_observation"]
    target = contextual_text_fields(agent, tokenizer,
        task_context_text(initial, initial), args.device,
        args.context_tokens, pooling="both")
    with torch.no_grad():
        agent.set_source(source, target_fields=target)
        factors = [layer.b.detach().cpu().float().clone()
                   for layer in agent.adapters]
        agent.set_source(None)
    return factors, records, original, retained


def evaluate(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    targets = checked_targets(args)
    agent, tokenizer = load_agent(args.model, args.checkpoint,
                                  args.device, args.gpu_fraction)
    if (agent.encoder_kind != "contextual" or
            agent.task_conditioned is not True or
            agent.task_context_scope != "current" or
            agent.task_pair_pooling != "mean"):
        raise ValueError("Expected current-context ALFWorld hypernetwork")
    factors = None
    updates = 0
    prior_episodes = []
    report = {"protocol": "From-empty prequential ALFWorld: only own official wins generate and update one persistent running-mean LoRA B; frozen contextual hypernetwork and Qwen; paired base, 50 steps/64 tokens/2-turn history, constrained greedy, two-repeat loop guard",
        "review_sha256": file_hash(args.review),
        "source_review_sha256": file_hash(args.source_review),
        "runner_sha256": file_hash(Path(__file__)),
        "actor_sha256": file_hash(Path(__file__).with_name(
            "alfworld_zero_shot_probe.py")),
        "factor_update_sha256": file_hash(Path(__file__).with_name(
            "alfworld_online_accumulate_lora_v1.py")),
        "checkpoint_sha256": file_hash(args.checkpoint),
        "model_config_sha256": file_hash(args.model / "config.json"),
        "sequence": args.sequence,
        "max_steps": 50, "max_new_tokens": 64,
        "actor_history_turns": 2, "loop_guard_max": 2,
        "context_tokens": args.context_tokens,
        "update_rule": "if own official won: B_next = B_old + (B(new_own_trajectory, its_initial_goal) - B_old)/(accepted_updates+1); else B_next = B_old",
        "games": [], "failures": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for row in targets:
        game = args.data_root / row["game"]
        before = factors_hash(factors)
        base = run_episode(agent, tokenizer, game, {}, adapter=False,
            device=args.device, max_steps=50, max_new_tokens=64,
            constrain_actions=True, actor_history_turns=2,
            loop_guard_max=2)
        online = run_episode(agent, tokenizer, game, {}, adapter=False,
            fixed_adapter=factors, device=args.device, max_steps=50,
            max_new_tokens=64, constrain_actions=True,
            actor_history_turns=2, loop_guard_max=2)
        entry = {"game": row["game"], "family": row.get("family"),
            "input_content_sha256": row["input_content_sha256"],
            "prior_episode_count": len(prior_episodes),
            "prior_episodes_sha256": digest(prior_episodes),
            "factor_sha256_before": before,
            "factor_l2_before": [float(x.norm()) for x in factors] if factors else [],
            "base": base, "online": online, "memory_updated": False}
        if (base["status"] == "complete" and online["status"] == "complete" and
                base["initial_observation"] == online["initial_observation"]):
            try:
                records = records_from_episode(online)
                entry["new_records_sha256"] = digest(records)
                if bool(online["reward"]):
                    increment, _, original, retained = episode_factor(
                        agent, tokenizer, online, args)
                    factors = mean_factors(factors, increment, updates)
                    updates += 1
                    entry.update({"memory_updated": True,
                        "source_tokens_original": original,
                        "source_tokens_retained": retained,
                        "increment_l2": [float(x.norm()) for x in increment]})
                prior_episodes.append({"game": row["game"],
                    "reward": online["reward"], "records": records})
            except Exception as exc:
                report["failures"].append({"game": row["game"],
                    "error": f"{type(exc).__name__}: {exc}"})
        else:
            report["failures"].append({"game": row["game"],
                "error": "Incomplete paired episode or inconsistent reset"})
        entry["factor_sha256_after"] = factors_hash(factors)
        entry["factor_l2_after"] = [float(x.norm()) for x in factors] if factors else []
        entry["update_count_after"] = updates
        report["games"].append(entry)
        complete = [x for x in report["games"] if
                    x["base"]["status"] == x["online"]["status"] == "complete"]
        report["summary"] = {"n": len(complete),
            "base": sum(x["base"]["reward"] for x in complete),
            "online": sum(x["online"]["reward"] for x in complete),
            "updates": updates,
            "failures": len(report["failures"])}
        tmp = args.output.with_suffix(args.output.suffix + ".tmp")
        tmp.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        tmp.replace(args.output)
        print(json.dumps({"n": len(report["games"]),
            "summary": report["summary"],
            "factor_l2": entry["factor_l2_after"]}), flush=True)
        if report["failures"]:
            raise RuntimeError("Persistent contextual LoRA failure recorded")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "evaluate"))
    parser.add_argument("--model", type=Path, default=Path(
        "current_work/delta-Mem/model/Qwen3-4B-Instruct-2507"))
    parser.add_argument("--checkpoint", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_sibling_simple40_onpolicy600_20261006.pt"))
    parser.add_argument("--data-root", type=Path, default=Path(
        "ttcl/data/alfworld_delta"))
    parser.add_argument("--sequence", required=True,
        choices=("train_seq0_6", "train_seq6_6", "valid_unseen36"))
    parser.add_argument("--source-review", type=Path, required=True)
    parser.add_argument("--review", type=Path, default=Path(
        "data/annotations/alfworld_success_gated_factors_v27_reviewed_20261009.json"))
    parser.add_argument("--output", type=Path, default=Path(
        "results/trajectory_hyperlora/alfworld_success_gated_factors_v27_20261009.json"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpu-fraction", type=float, default=.65)
    parser.add_argument("--context-tokens", type=int, default=2048)
    args = parser.parse_args()
    if args.context_tokens < 2:
        parser.error("Invalid source token budget")
    (prepare if args.command == "prepare" else evaluate)(args)


if __name__ == "__main__":
    main()
