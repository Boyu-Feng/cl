"""Prequential ALFWorld probe: empty memory, then own trajectories to LoRA.

The hypernetwork is frozen. Every target is selected from the reviewed train
plan before any rollout; only completed earlier online episodes enter memory.
This is an exploratory online protocol, not a new trained online learner.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from ttcl.trajectory_hyperlora.alfworld_zero_shot_probe import load_agent, run_episode
from ttcl.trajectory_hyperlora.relational_router_pilot import tokenize_records


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True,
                                    ensure_ascii=False).encode()).hexdigest()


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select_games(plan: dict, first_sequence: int, sequences: int) -> list[str]:
    train = plan["training"]
    chosen = train[first_sequence:first_sequence + sequences]
    if len(chosen) != sequences:
        raise ValueError("Requested training sequences are unavailable")
    games = [game for sequence in chosen for game in sequence["games"]]
    if len(games) != len(set(games)) or not games:
        raise ValueError("Online sequence repeats a game")
    evaluation = {game for sequence in plan["evaluation"]
                  for game in sequence["games"]}
    if any("/train/" not in game or game in evaluation for game in games):
        raise ValueError("Online sequence contains a non-training game")
    return games


def prepare(args: argparse.Namespace) -> dict:
    if args.review.exists():
        raise FileExistsError(args.review)
    plan = json.loads(args.plan.read_text())
    games = select_games(plan, args.first_sequence, args.sequences)
    rows = []
    for game in games:
        path = args.data_root / game
        if not path.is_file():
            raise FileNotFoundError(path)
        rows.append({"game": game, "game_sha256": file_hash(path),
                     "reviewed_target": True,
                     "review_basis": "Frozen train-plan game; official environment won at rollout"})
    review = {"protocol": "Fresh content-bound online ALFWorld train target order; no historical trajectory preloaded",
              "plan_sha256": file_hash(args.plan),
              "first_sequence": args.first_sequence,
              "sequences": args.sequences,
              "targets": rows}
    args.review.parent.mkdir(parents=True, exist_ok=True)
    args.review.write_text(json.dumps(review, ensure_ascii=False,
                                      indent=2) + "\n")
    return review


def records_from_episode(episode: dict) -> list[dict[str, str]]:
    if episode.get("status") != "complete" or not episode.get("trajectory"):
        raise ValueError("Only complete, nonempty own episodes can enter memory")
    before = episode["initial_observation"]
    records = []
    steps = episode["trajectory"]
    for index, step in enumerate(steps):
        if not step["valid"]:
            status = "invalid command"
        elif index == len(steps) - 1 and episode["reward"]:
            status = "task completed"
        elif index == len(steps) - 1:
            status = "task not completed"
        else:
            status = "valid command"
        # Keep the feedback category first: tokenizer fields are bounded and
        # the historical ALFWorld hypernetwork was trained with these tags.
        feedback = status + ". " + step["observation"]
        records.append({"observation": before,
                        "action": step["command"],
                        "feedback": feedback})
        before = step["observation"]
    return records


def summarize(rows: list[dict]) -> dict:
    complete = [row for row in rows if row["base"]["status"] == "complete"
                and row["online"]["status"] == "complete"]
    return {"completed": len(complete),
            "base_success": sum(row["base"]["reward"] for row in complete),
            "online_success": sum(row["online"]["reward"] for row in complete),
            "online_only": sum(row["online"]["reward"] == 1 and
                               row["base"]["reward"] == 0 for row in complete),
            "base_only": sum(row["base"]["reward"] == 1 and
                             row["online"]["reward"] == 0 for row in complete)}


def evaluate(args: argparse.Namespace) -> dict:
    if args.output.exists():
        raise FileExistsError(args.output)
    review = json.loads(args.review.read_text())
    plan = json.loads(args.plan.read_text())
    games = select_games(plan, review["first_sequence"], review["sequences"])
    if review["plan_sha256"] != file_hash(args.plan) or \
            [row["game"] for row in review["targets"]] != games:
        raise ValueError("Reviewed target order or plan changed")
    for row in review["targets"]:
        if not row["reviewed_target"] or \
                file_hash(args.data_root / row["game"]) != row["game_sha256"]:
            raise ValueError("Reviewed game content changed")
    agent, tokenizer = load_agent(args.model, args.checkpoint,
                                  args.device, args.gpu_fraction)
    report = {"protocol": "Exploratory prequential ALFWorld: empty memory on first game; frozen hypernetwork regenerates LoRA from all earlier own episodes; matched no-LoRA arm; official won reward",
              "review_sha256": file_hash(args.review),
              "checkpoint_sha256": file_hash(args.checkpoint),
              "model_config_sha256": file_hash(args.model / "config.json"),
              "max_steps": args.max_steps,
              "max_new_tokens": args.max_new_tokens,
              "source_field_token_limit": args.field_tokens,
              "games": []}
    memory: list[dict[str, str]] = []
    prior_episodes: list[dict] = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for row in review["targets"]:
        game = row["game"]
        memory_hash = digest(prior_episodes)
        input_hash = digest({"target_game": game,
                             "prior_episodes": prior_episodes})
        # One observed transition cannot form the encoder's covariance.
        # This matters only for pathological one-step initial episodes.
        adapter_on = len(memory) >= 2
        fields = tokenize_records(tokenizer, memory, args.device,
                                  max_tokens=args.field_tokens) if adapter_on else {}
        base = run_episode(agent, tokenizer, args.data_root / game, {},
                           adapter=False, device=args.device,
                           max_steps=args.max_steps,
                           max_new_tokens=args.max_new_tokens,
                           constrain_actions=True)
        online = run_episode(agent, tokenizer, args.data_root / game, fields,
                             adapter=adapter_on, device=args.device,
                             max_steps=args.max_steps,
                             max_new_tokens=args.max_new_tokens,
                             constrain_actions=True)
        entry = {"game": game, "game_sha256": row["game_sha256"],
                 "memory_episode_count_before": len(prior_episodes),
                 "memory_step_count_before": len(memory),
                 "memory_sha256_before": memory_hash,
                 "input_content_sha256": input_hash,
                 "adapter_mounted": adapter_on,
                 "base": base, "online": online}
        if (base["status"] == "complete" and online["status"] == "complete"
                and base["initial_observation"] != online["initial_observation"]):
            entry["status"] = "binding_failure"
        report["games"].append(entry)
        if online["status"] == "complete":
            records = records_from_episode(online)
            prior_episodes.append({"game": game, "reward": online["reward"],
                                   "records": records})
            memory.extend(records)
        report["summary"] = summarize(report["games"])
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(json.dumps(report, ensure_ascii=False,
                                        indent=2) + "\n")
        temporary.replace(args.output)
        print(json.dumps({"game": game,
                          "base": base.get("reward"),
                          "online": online.get("reward"),
                          "adapter_mounted": adapter_on,
                          "memory_episodes": len(prior_episodes),
                          "summary": report["summary"]}), flush=True)
        if base["status"] != "complete" or online["status"] != "complete" or \
                entry.get("status") == "binding_failure":
            raise RuntimeError("Online experiment failure recorded")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "evaluate"))
    parser.add_argument("--model", type=Path, default=Path(
        "current_work/delta-Mem/model/Qwen3-4B-Instruct-2507"))
    parser.add_argument("--checkpoint", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_next_task_warmstart_20261005/seed42_v2.pt"))
    parser.add_argument("--plan", type=Path, default=Path(
        "ttcl/results/experience_evolution/alfworld_delta_20260922/plan.json"))
    parser.add_argument("--data-root", type=Path, default=Path(
        "ttcl/data/alfworld_delta"))
    parser.add_argument("--first-sequence", type=int, default=0)
    parser.add_argument("--sequences", type=int, default=1)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpu-fraction", type=float, default=.40)
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--field-tokens", type=int, default=40)
    args = parser.parse_args()
    if (args.first_sequence < 0 or args.sequences < 1 or
            args.max_steps < 1 or args.max_new_tokens < 1 or
            args.field_tokens < 1 or not 0 < args.gpu_fraction <= 1):
        parser.error("Invalid online experiment budget")
    if args.command == "prepare":
        prepare(args)
    else:
        if args.output is None:
            parser.error("evaluate requires --output")
        evaluate(args)


if __name__ == "__main__":
    main()
