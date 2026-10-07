"""Fresh content-bound action targets from paired ALFWorld training wins."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ttcl.experience_evolution.environment import ACTOR_SYSTEM, make_env
from ttcl.trajectory_hyperlora.alfworld_same_game_retry import digest, file_hash
from ttcl.trajectory_hyperlora.alfworld_reward_router_data_v1 import checked_pairs
from ttcl.trajectory_hyperlora.train_alfworld_retry_reward import label_content


def checked_rollouts(args):
    pairs = checked_pairs(args)
    by_game = {}
    for part in args.parts:
        report = json.loads(part.read_text())
        if (report["reviewed_pairs_sha256"] != file_hash(args.reviewed_pairs) or
                report["checkpoint_sha256"] != file_hash(args.checkpoint) or
                report["max_steps"] != 50 or report["max_new_tokens"] != 64 or
                report["actor_history_turns"] != 2 or report["failures"]):
            raise ValueError("Paired rollout lineage or budget changed")
        for row in report["games"]:
            if row["game"] in by_game:
                raise ValueError("Duplicate reward game")
            by_game[row["game"]] = row
    if set(by_game) != {pair["target_game"] for pair in pairs}:
        raise ValueError("Missing paired rollout")
    for pair in pairs:
        row = by_game[pair["target_game"]]
        if (row["input_content_sha256"] != pair["input_content_sha256"] or
                row["source_records_sha256"] != pair["source_records_sha256"] or
                row["split"] != pair["split"] or
                any(row["arms"][arm]["status"] != "complete" or
                    row["arms"][arm]["steps"] > 50 or
                    row["arms"][arm]["invalid_commands"] != 0
                    for arm in ("base", "lora"))):
            raise ValueError("Changed or incomplete reward pair")
    return pairs, by_game


def build(args):
    pairs, by_game = checked_rollouts(args)
    tasks = []
    for pair in pairs:
        if pair["split"] != "train":
            continue
        row = by_game[pair["target_game"]]
        winners = [arm for arm in ("lora", "base")
                   if row["arms"][arm]["reward"] == 1.0]
        if not winners:
            continue
        arm = min(winners, key=lambda name: (row["arms"][name]["steps"],
                                             name != "lora"))
        episode = row["arms"][arm]
        game = args.data_root / pair["target_game"]
        if file_hash(game) != pair["target_game_sha256"]:
            raise ValueError("Reward target game content changed")
        env = make_env(game)
        try:
            state = env.reset()
            if str(state["feedback"]) != episode["initial_observation"]:
                raise ValueError("Reward teacher reset changed")
            messages = [{"role": "system", "content": ACTOR_SYSTEM}]
            labels = []
            for index, step in enumerate(episode["trajectory"]):
                command = step["command"]
                available = list(state["admissible_commands"])
                if not step["valid"] or command not in available or \
                        step["response"] != command:
                    raise ValueError("Reward teacher action inadmissible")
                messages.append({"role": "user", "content":
                    str(state["feedback"]) + "\nAvailable commands:\n" +
                    "\n".join(available)})
                content = {"target_game": pair["target_game"],
                    "target_game_sha256": pair["target_game_sha256"],
                    "source_episode_sha256": pair["source_records_sha256"],
                    "source_records": pair["source_records"],
                    "teacher_arm": arm,
                    "target_messages": list(messages),
                    "target_action": command}
                labels.append({**content, "input_content_sha256": digest(content)})
                state, _, done = env.step(command)
                if (str(state["feedback"]) != step["observation"] or
                        bool(state["won"]) != bool(step["won"]) or
                        done and index != len(episode["trajectory"]) - 1):
                    raise ValueError("Reward teacher transition changed")
                messages.append({"role": "assistant", "content": command})
            if not state["won"] or len(labels) != episode["steps"]:
                raise ValueError("Reward-selected route did not replay to won")
        finally:
            env.close()
        tasks.append({"target_game": pair["target_game"],
            "pair_input_content_sha256": pair["input_content_sha256"],
            "target_game_sha256": pair["target_game_sha256"],
            "source_episode_sha256": pair["source_records_sha256"],
            "teacher_arm": arm,
            "base_reward": row["arms"]["base"]["reward"],
            "old_lora_reward": row["arms"]["lora"]["reward"],
            "teacher_steps": episode["steps"],
            "labels": labels})
    if len(tasks) < 12:
        raise ValueError("Too few official training successes")
    return {"protocol": "Train-only reward-selected successful base or LoRA route; exact admissible actions and feedback replayed to official won; each new target prompt bound to reviewed source and target content; no dev reward labels",
        "reviewed_pairs_sha256": file_hash(args.reviewed_pairs),
        "checkpoint_sha256": file_hash(args.checkpoint),
        "parts_sha256": {str(path): file_hash(path) for path in args.parts},
        "tasks": tasks}


def prepare(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    value = build(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"tasks": len(value["tasks"]),
        "actions": sum(len(task["labels"]) for task in value["tasks"]),
        "teacher_arms": {arm: sum(task["teacher_arm"] == arm
            for task in value["tasks"]) for arm in ("base", "lora")}}), flush=True)


def review(args):
    if args.output_review.exists():
        raise FileExistsError(args.output_review)
    value = json.loads(args.output.read_text())
    if value != build(args):
        raise ValueError("Reward-selected labels changed before review")
    notes = []
    for task in value["tasks"]:
        for label in task["labels"]:
            if label["input_content_sha256"] != digest(label_content(label)):
                raise ValueError("Action input-content binding mismatch")
            notes.append({"target_game": task["target_game"],
                "input_content_sha256": label["input_content_sha256"],
                "source_episode_sha256": task["source_episode_sha256"],
                "approved": True,
                "review_basis": "Train-only official-won route; original reset, admissible command, every transition and terminal win freshly replayed; source and full action prompt content bound"})
    result = {"protocol": "Fresh reviewed reward-selected ALFWorld action targets",
        "labels_sha256": file_hash(args.output),
        "annotations": notes}
    args.output_review.parent.mkdir(parents=True, exist_ok=True)
    args.output_review.write_text(json.dumps(result, ensure_ascii=False,
                                              indent=2) + "\n")
    print(json.dumps({"reviewed_actions": len(notes)}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "review"))
    parser.add_argument("--parts", nargs="+", type=Path, required=True)
    parser.add_argument("--reviewed-pairs", type=Path, default=Path(
        "data/annotations/alf_reward_router_train72_reviewed_20261007.json"))
    parser.add_argument("--candidates", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_sibling_train600_candidates_20261006.json"))
    parser.add_argument("--source-review", type=Path, default=Path(
        "data/annotations/alf_sibling_train600_reviewed_20261006.json"))
    parser.add_argument("--retry-candidates", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_same_game_retry_candidates_20261006.json"))
    parser.add_argument("--retry-review", type=Path, default=Path(
        "data/annotations/alf_same_game_retry_reviewed_20261006.json"))
    parser.add_argument("--all-source-review", type=Path, default=Path(
        "data/annotations/alf_expert_first_attempts_reviewed_20261006.json"))
    parser.add_argument("--checkpoint", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_sibling_simple40_onpolicy600_20261006.pt"))
    parser.add_argument("--data-root", type=Path, default=Path("ttcl/data/alfworld_delta"))
    parser.add_argument("--output", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_reward_selected_train48_v1_20261007.json"))
    parser.add_argument("--output-review", type=Path, default=Path(
        "data/annotations/alf_reward_selected_train48_v1_reviewed_20261007.json"))
    args = parser.parse_args()
    (prepare if args.command == "prepare" else review)(args)


if __name__ == "__main__":
    main()
