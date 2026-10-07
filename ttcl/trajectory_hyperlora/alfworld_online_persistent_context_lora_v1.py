"""From-empty online ALFWorld with persistent LoRA from a contextual hypernet.

After each own completed episode, encode only that trajectory once, generate
its LoRA using its public initial task context, and update a persistent mean
of B factors. Later tasks mount those factors without replaying source text.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ttcl.trajectory_hyperlora.alfworld_online_accumulate_lora_v1 import (
    factors_hash, mean_factors,
)
from ttcl.trajectory_hyperlora.alfworld_online_from_empty import (
    digest, file_hash, records_from_episode, select_games,
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
    parent = json.loads(args.parent_review.read_text())
    plan = json.loads(args.plan.read_text())
    games = select_games(plan, parent["first_sequence"], parent["sequences"])
    if (parent["plan_sha256"] != file_hash(args.plan) or
            [row["game"] for row in parent["targets"]] != games):
        raise ValueError("Online target sequence changed")
    targets = []
    for index, game in enumerate(games):
        game_hash = file_hash(args.data_root / game)
        if game_hash != parent["targets"][index]["game_sha256"]:
            raise ValueError("Online target content changed")
        content = {"index": index, "game": game,
            "game_sha256": game_hash,
            "parent_review_sha256": file_hash(args.parent_review),
            "checkpoint_sha256": file_hash(args.checkpoint)}
        targets.append({**content, "input_content_sha256": digest(content),
            "reviewed_target": True,
            "review_basis": "Fresh frozen train-plan target and game-content check; live own trajectory is the sole update source"})
    value = {"protocol": "Fresh content-bound target review for contextual hyper-LoRA persistent factor update from empty memory",
        "plan_sha256": file_hash(args.plan),
        "parent_review_sha256": file_hash(args.parent_review),
        "checkpoint_sha256": file_hash(args.checkpoint),
        "targets": targets}
    args.review.parent.mkdir(parents=True, exist_ok=True)
    args.review.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"reviewed_targets": len(targets)}), flush=True)


def checked_targets(args):
    value = json.loads(args.review.read_text())
    parent = json.loads(args.parent_review.read_text())
    plan = json.loads(args.plan.read_text())
    games = select_games(plan, parent["first_sequence"], parent["sequences"])
    if (value["plan_sha256"] != file_hash(args.plan) or
            value["parent_review_sha256"] != file_hash(args.parent_review) or
            value["checkpoint_sha256"] != file_hash(args.checkpoint) or
            len(value["targets"]) != len(games)):
        raise ValueError("Persistent LoRA review lineage changed")
    for index, game in enumerate(games):
        content = {"index": index, "game": game,
            "game_sha256": file_hash(args.data_root / game),
            "parent_review_sha256": file_hash(args.parent_review),
            "checkpoint_sha256": file_hash(args.checkpoint)}
        row = value["targets"][index]
        if (not row["reviewed_target"] or
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
    report = {"protocol": "From-empty prequential ALFWorld with frozen Qwen and contextual hypernetwork; after each own completed episode, produce one task-conditioned LoRA B and update persistent running mean; no historical source text at later action time; paired base, 50 steps/64 tokens/2-turn history, constrained greedy, same two-repeat loop guard",
        "review_sha256": file_hash(args.review),
        "checkpoint_sha256": file_hash(args.checkpoint),
        "model_config_sha256": file_hash(args.model / "config.json"),
        "max_steps": 50, "max_new_tokens": 64,
        "actor_history_turns": 2, "loop_guard_max": 2,
        "context_tokens": args.context_tokens,
        "update_rule": "B_next = B_old + (B(new_own_trajectory, its_initial_goal) - B_old)/(updates+1)",
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
        entry = {"game": row["game"],
            "input_content_sha256": row["input_content_sha256"],
            "prior_episode_count": len(prior_episodes),
            "prior_episodes_sha256": digest(prior_episodes),
            "factor_sha256_before": before,
            "factor_l2_before": [float(x.norm()) for x in factors] if factors else [],
            "base": base, "online": online}
        if (base["status"] == "complete" and online["status"] == "complete" and
                base["initial_observation"] == online["initial_observation"]):
            try:
                increment, records, original, retained = episode_factor(
                    agent, tokenizer, online, args)
                factors = mean_factors(factors, increment, updates)
                updates += 1
                entry.update({"new_records_sha256": digest(records),
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
    parser.add_argument("--plan", type=Path, default=Path(
        "ttcl/results/experience_evolution/alfworld_delta_20260922/plan.json"))
    parser.add_argument("--data-root", type=Path, default=Path(
        "ttcl/data/alfworld_delta"))
    parser.add_argument("--parent-review", type=Path, default=Path(
        "data/annotations/alfworld_online_from_empty_train_seq6_6_reviewed_20261007.json"))
    parser.add_argument("--review", type=Path, default=Path(
        "data/annotations/alfworld_online_context_persistent_train_seq6_6_reviewed_20261007.json"))
    parser.add_argument("--output", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_online_context_persistent_train_seq6_6_20261007.json"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpu-fraction", type=float, default=.65)
    parser.add_argument("--context-tokens", type=int, default=2048)
    args = parser.parse_args()
    if args.context_tokens < 2:
        parser.error("Invalid source token budget")
    (prepare if args.command == "prepare" else evaluate)(args)


if __name__ == "__main__":
    main()
