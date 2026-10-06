"""Online first attempt and trajectory-to-LoRA retry on ALFWorld valid_unseen.

The first attempt starts without history. Its complete public trajectory is
replayed and content-bound before the retry. A fixed train-game trajectory
from the same task family is the wrong-history control. No held-out outcome
is used to train or choose the hypernetwork.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

from ttcl.experience_evolution.environment import make_env
from ttcl.trajectory_hyperlora.alfworld_online_from_empty import records_from_episode
from ttcl.trajectory_hyperlora.alfworld_same_game_retry import digest, file_hash
from ttcl.trajectory_hyperlora.alfworld_zero_shot_probe import load_agent, run_episode
from ttcl.trajectory_hyperlora.contextual_alf_source import contextual_source_fields
from ttcl.trajectory_hyperlora.relational_router_pilot import tokenize_records


def prepare(args):
    if args.review.exists():
        raise FileExistsError(args.review)
    plan = json.loads(args.plan.read_text())
    first = json.loads(args.first_attempts.read_text())
    if (first["plan_sha256"] != file_hash(args.plan) or
            first["max_steps"] != 30 or first["max_new_tokens"] != 64):
        raise ValueError("Train source lineage or budget changed")
    training = {game for sequence in plan["training"]
                for game in sequence["games"]}
    groups = defaultdict(list)
    for sequence in plan["evaluation"]:
        for game in sequence["games"]:
            if "/valid_unseen/" not in "/" + game or game in training:
                raise ValueError("Evaluation target is not held out")
            groups[sequence["family"]].append(game)
    if len(groups) != 6 or any(len(games) != 6 for games in groups.values()):
        raise ValueError("Expected six official families with six held-out games each")
    sources = defaultdict(list)
    for item in first["games"]:
        if (item["game"] not in training or
                item["base"]["status"] != "complete"):
            continue
        sources[item["family"]].append(item)
    rows = []
    for position in range(6):
        for family in sorted(groups):
            game = sorted(groups[family], key=lambda value: digest(
                ["heldout_retry", value]))[position]
            available = sources[family] or [item for group in sources.values()
                                             for item in group]
            if not available:
                raise ValueError("No train-game wrong-history source")
            wrong = min(available, key=lambda value: (
                value["base"]["reward"] != 0,
                digest(["wrong_history", game, value["game"]])))
            wrong_episode = wrong["base"]
            wrong_records = records_from_episode(wrong_episode)
            row = {"game": game, "family": family,
                   "game_sha256": file_hash(args.data_root / game),
                   "wrong_source_game": wrong["game"],
                   "wrong_source_family": wrong["family"],
                   "wrong_source_game_sha256": file_hash(
                       args.data_root / wrong["game"]),
                   "wrong_source_episode_sha256": digest(wrong_episode),
                   "wrong_source_records": wrong_records}
            row["input_content_sha256"] = digest(row)
            rows.append(row)
    result = {"protocol": "Frozen official valid_unseen target order; one per family in each block of six; reviewed game content and train-only wrong-history binding (same-family where available); no trajectory labels inherited",
              "plan_sha256": file_hash(args.plan),
              "first_attempts_sha256": file_hash(args.first_attempts),
              "targets": rows}
    args.review.parent.mkdir(parents=True, exist_ok=True)
    args.review.write_text(json.dumps(result, ensure_ascii=False, indent=2)
                           + "\n")
    print(json.dumps({"reviewed_heldout_games": len(rows)}), flush=True)


def checked_targets(args):
    review = json.loads(args.review.read_text())
    first = json.loads(args.first_attempts.read_text())
    if (review["plan_sha256"] != file_hash(args.plan) or
            review["first_attempts_sha256"] != file_hash(args.first_attempts)):
        raise ValueError("Held-out review lineage changed")
    by_game = {item["game"]: item for item in first["games"]}
    for row in review["targets"]:
        content = {key: value for key, value in row.items()
                   if key != "input_content_sha256"}
        wrong = by_game[row["wrong_source_game"]]
        if (digest(content) != row["input_content_sha256"] or
                file_hash(args.data_root / row["game"]) !=
                row["game_sha256"] or
                file_hash(args.data_root / row["wrong_source_game"]) !=
                row["wrong_source_game_sha256"] or
                digest(wrong["base"]) != row["wrong_source_episode_sha256"] or
                records_from_episode(wrong["base"]) !=
                row["wrong_source_records"] or
                wrong["family"] != row["wrong_source_family"]):
            raise ValueError("Held-out target or wrong history changed")
    return review["targets"]


def replay_episode(game: Path, episode: dict):
    if (episode["status"] != "complete" or
            episode["steps"] != len(episode["trajectory"])):
        raise ValueError("Incomplete online first attempt")
    env = make_env(game)
    try:
        state = env.reset()
        if str(state["feedback"]) != episode["initial_observation"]:
            raise ValueError("Online first observation changed")
        for index, step in enumerate(episode["trajectory"]):
            if (step["command"] not in state["admissible_commands"] or
                    step["valid"] is not True):
                raise ValueError("Online first action was inadmissible")
            state, _, done = env.step(step["command"])
            if (str(state["feedback"]) != step["observation"] or
                    bool(state["won"]) != bool(step["won"]) or
                    (done and index != len(episode["trajectory"]) - 1)):
                raise ValueError("Online first transition changed")
        if bool(state["won"]) != bool(episode["reward"]):
            raise ValueError("Online first official won changed")
    finally:
        env.close()


def evaluate(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    targets = checked_targets(args)
    if args.family:
        targets = [row for row in targets if row["family"] == args.family]
    targets = targets[args.offset:args.offset + args.limit]
    if not targets:
        raise ValueError("Empty held-out target block")
    agent, tokenizer = load_agent(args.model, args.checkpoint,
                                  args.device, args.gpu_fraction)
    for adapter in agent.adapters:
        adapter.scale = args.adapter_scale
    def source_fields(records):
        if agent.encoder_kind == "contextual":
            return contextual_source_fields(agent, tokenizer, records,
                args.device, args.contextual_source_max_tokens,
                pooling="both" if agent.task_conditioned and
                    agent.task_pair_pooling == "mean" else "last")
        return tokenize_records(tokenizer, records, args.device,
            max_tokens=args.source_max_tokens,
            truncation_mode=args.source_truncation,
            repeat_initial_observation=args.repeat_initial_observation)
    result = {"protocol": "Online official valid_unseen: no-history first attempt, then same-game retry with own trajectory LoRA, wrong train trajectory LoRA, or no LoRA; wrong is same-family where available; full first attempt replayed and source-content-bound; won reward",
              "review_sha256": file_hash(args.review),
              "checkpoint_sha256": file_hash(args.checkpoint),
              "offset": args.offset, "limit": args.limit,
              "family": args.family,
              "actor_history_turns": args.actor_history_turns,
              "source_encoder": agent.encoder_kind,
              "contextual_source_max_tokens": args.contextual_source_max_tokens,
              "source_max_tokens": args.source_max_tokens,
              "source_truncation": args.source_truncation,
              "repeat_initial_observation": args.repeat_initial_observation,
              "adapter_scale": args.adapter_scale,
              "max_steps": 30, "max_new_tokens": 64,
              "games": [], "failures": []}
    for row in targets:
        try:
            game = args.data_root / row["game"]
            base = run_episode(agent, tokenizer, game, {}, adapter=False,
                device=args.device, max_steps=30, max_new_tokens=64,
                constrain_actions=True,
                actor_history_turns=args.actor_history_turns)
            replay_episode(game, base)
            own_records = records_from_episode(base)
            own = source_fields(own_records)
            wrong = source_fields(row["wrong_source_records"])
            arms = {"base": base}
            for name, fields in (("own", own), ("wrong", wrong)):
                arms[name] = run_episode(agent, tokenizer, game, fields,
                    adapter=True, device=args.device, max_steps=30,
                    max_new_tokens=64, constrain_actions=True,
                    actor_history_turns=args.actor_history_turns)
                if (arms[name]["status"] != "complete" or
                        arms[name]["initial_observation"] !=
                        base["initial_observation"]):
                    raise ValueError("Retry rollout or reset failed")
            result["games"].append({"game": row["game"],
                "family": row["family"],
                "target_input_content_sha256": row["input_content_sha256"],
                "own_source_episode_sha256": digest(base),
                "own_source_records_sha256": digest(own_records),
                "wrong_source_episode_sha256":
                    row["wrong_source_episode_sha256"],
                "arms": arms})
        except Exception as exc:
            result["failures"].append({"game": row["game"],
                                       "error": f"{type(exc).__name__}: {exc}"})
        result["summary"] = {"n": len(result["games"]),
            "failures": len(result["failures"]),
            **{name: sum(item["arms"][name]["reward"]
                         for item in result["games"])
               for name in ("base", "own", "wrong")}}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(json.dumps(result, ensure_ascii=False,
                                        indent=2) + "\n")
        temporary.replace(args.output)
        print(json.dumps(result["summary"]), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "evaluate"))
    parser.add_argument("--model", type=Path, default=Path(
        "current_work/delta-Mem/model/Qwen3-4B-Instruct-2507"))
    parser.add_argument("--checkpoint", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_next_task_warmstart_20261005/seed42_v2.pt"))
    parser.add_argument("--plan", type=Path, default=Path(
        "ttcl/results/experience_evolution/alfworld_delta_20260922/plan.json"))
    parser.add_argument("--first-attempts", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_rl_20261005/train_rewards_complete.json"))
    parser.add_argument("--data-root", type=Path, default=Path(
        "ttcl/data/alfworld_delta"))
    parser.add_argument("--review", type=Path, default=Path(
        "data/annotations/alf_heldout_retry_reviewed_20261006.json"))
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=6)
    parser.add_argument("--family", type=str)
    parser.add_argument("--actor-history-turns", type=int)
    parser.add_argument("--contextual-source-max-tokens", type=int,
                        default=2048)
    parser.add_argument("--source-max-tokens", type=int, default=40)
    parser.add_argument("--source-truncation", choices=("head", "head_tail"),
                        default="head_tail")
    parser.add_argument("--repeat-initial-observation", action="store_true")
    parser.add_argument("--adapter-scale", type=float, default=1.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpu-fraction", type=float, default=.6)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if (args.offset < 0 or args.limit < 1 or args.source_max_tokens < 2 or
            args.contextual_source_max_tokens < 2 or
            (args.actor_history_turns is not None and
             args.actor_history_turns < 1) or
            (args.repeat_initial_observation and
             args.source_truncation != "head_tail") or
            not 0 < args.adapter_scale <= 1 or
            not 0 < args.gpu_fraction <= 1):
        parser.error("Invalid held-out retry budget")
    if args.command == "prepare":
        prepare(args)
    else:
        if args.output is None:
            parser.error("evaluate needs --output")
        evaluate(args)


if __name__ == "__main__":
    main()
