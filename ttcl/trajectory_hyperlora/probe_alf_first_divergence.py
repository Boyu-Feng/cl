"""Intervene on one reviewed ALFWorld action under a fixed source LoRA.

The common prefix is replayed verbatim. At its first own/wrong action split,
swap only that action and let the *same* source-conditioned policy continue.
The unmodified policy must first reproduce its recorded complete rollout.
This isolates an action intervention for one frozen policy and task reset;
it does not establish a generalizable reward label by itself.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ttcl.experience_evolution.environment import ACTOR_SYSTEM, make_env
from ttcl.trajectory_hyperlora.alfworld_same_game_retry import digest, file_hash
from ttcl.trajectory_hyperlora.alfworld_zero_shot_probe import (
    clean_command, generate, load_agent, run_episode,
)
from ttcl.trajectory_hyperlora.contextual_alf_source import (
    contextual_source_fields, contextual_text_fields, task_context_text,
)
from ttcl.trajectory_hyperlora.review_alf_first_divergence import (
    candidate_content, candidates,
)


def swapped_episode(agent, tokenizer, game: Path, fields: dict,
                    reference: dict, split: int, alternate: str,
                    device: str, max_source_tokens: int) -> dict:
    env = make_env(game)
    trajectory = []
    try:
        state = env.reset()
        initial = str(state["feedback"])
        if initial != reference["initial_observation"]:
            raise ValueError("Counterfactual reset differs from reference")
        messages = [{"role": "system", "content": ACTOR_SYSTEM}]
        for turn in range(30):
            if agent.task_conditioned:
                if agent.task_context_scope == "current" or turn == 0:
                    target = contextual_text_fields(agent, tokenizer,
                        task_context_text(initial, str(state["feedback"])),
                        device, max_source_tokens,
                        pooling="both" if agent.task_pair_pooling == "mean"
                            else "last")
                    with torch.no_grad():
                        agent.set_source(fields, target_fields=target)
            elif turn == 0:
                with torch.no_grad():
                    agent.set_source(fields)
            available = list(state["admissible_commands"])
            messages.append({"role": "user", "content":
                str(state["feedback"]) + "\nAvailable commands:\n" +
                "\n".join(available)})
            if turn < split:
                original = reference["trajectory"][turn]
                response = original["response"]
                command = original["command"]
                if command not in available:
                    raise ValueError("Counterfactual shared prefix changed")
            elif turn == split:
                response = command = alternate
                if command not in available:
                    raise ValueError("Counterfactual alternate inadmissible")
            else:
                response = generate(agent, tokenizer, messages, device, 64,
                                    available)
                command = clean_command(response, available)
            state, _, done = env.step(command)
            messages.append({"role": "assistant", "content": response})
            step = {"turn": turn, "response": response, "command": command,
                    "valid": command in available,
                    "observation": str(state["feedback"]),
                    "won": bool(state["won"])}
            if turn < split and step != reference["trajectory"][turn]:
                raise ValueError("Counterfactual shared prefix transition changed")
            trajectory.append(step)
            if done or state["won"]:
                break
        return {"status": "complete", "reward": float(bool(state["won"])),
                "steps": len(trajectory),
                "invalid_commands": sum(not step["valid"] for step in trajectory),
                "trajectory": trajectory}
    finally:
        env.close()
        agent.set_source(None)


def run(args) -> None:
    if args.output.exists() and not args.validate_only:
        raise FileExistsError(args.output)
    selected = json.loads(args.selection_candidates.read_text())
    reviewed = json.loads(args.selection_review.read_text())
    rows, rollouts, by_game = candidates(args)
    if (selected["rows"] != rows or
            selected["rollouts_sha256"] != file_hash(args.rollouts) or
            reviewed["candidates_sha256"] !=
                file_hash(args.selection_candidates) or
            rollouts["checkpoint_sha256"] != file_hash(args.checkpoint)):
        raise ValueError("Preference or policy content changed")
    notes = {note["selection_input_content_sha256"]: note
             for note in reviewed["annotations"]}
    if len(notes) != len(rows):
        raise ValueError("Preference review is incomplete")
    games = {game["game"]: game for game in rollouts["games"]}
    for row in rows:
        note = notes.get(row["input_content_sha256"])
        source_row, source_note = by_game[row["target_game"]]
        note_fields = ("target_game", "selection_input_content_sha256",
            "source_records_sha256", "wrong_records_sha256",
            "target_messages", "preferred_action", "rejected_action",
            "preferred_arm")
        if (note is None or note["approved"] is not True or
                digest({key: note[key] for key in note_fields}) !=
                    note["input_content_sha256"] or
                note["target_game"] != row["target_game"] or
                note["preferred_action"] != row["preferred_action"] or
                note["rejected_action"] != row["rejected_action"] or
                note["preferred_arm"] != row["preferred_arm"] or
                digest(candidate_content(row)) != row["input_content_sha256"] or
                note["source_records_sha256"] !=
                    source_note["source_records_sha256"] or
                note["wrong_records_sha256"] !=
                    source_note["wrong_records_sha256"]):
            raise ValueError("Counterfactual input review changed")
    if args.validate_only:
        print(json.dumps({"validated_preferences": len(rows),
            "checkpoint_sha256": rollouts["checkpoint_sha256"]}), flush=True)
        return
    agent, tokenizer = load_agent(args.model, args.checkpoint,
                                  args.device, args.gpu_fraction)
    if agent.encoder_kind != "contextual":
        raise ValueError("Counterfactual probe expects contextual sources")
    results = []
    for row in rows:
        source_row, source_note = by_game[row["target_game"]]
        game_path = args.data_root / source_row["target_game"]
        arms = games[row["target_game"]]["arms"]
        branch = {}
        for arm, records, alternate in (
                ("own", source_note["source_records"],
                 row["rejected_action"] if row["preferred_arm"] == "own"
                 else row["preferred_action"]),
                ("wrong", source_note["wrong_records"],
                 row["preferred_action"] if row["preferred_arm"] == "own"
                 else row["rejected_action"])):
            fields = contextual_source_fields(agent, tokenizer, records,
                args.device, args.max_source_tokens,
                pooling="both" if agent.task_conditioned and
                    agent.task_pair_pooling == "mean" else "last")
            reference = run_episode(agent, tokenizer, game_path, fields,
                adapter=True, device=args.device, max_steps=30,
                max_new_tokens=64, constrain_actions=True)
            original = arms[arm]
            if (reference["status"] != "complete" or
                    reference["reward"] != original["reward"] or
                    reference["steps"] != original["steps"] or
                    digest(reference["trajectory"]) !=
                        digest(original["trajectory"])):
                raise ValueError("Frozen policy did not reproduce original rollout")
            changed = swapped_episode(agent, tokenizer, game_path, fields,
                original, row["first_divergence"], alternate, args.device,
                args.max_source_tokens)
            if changed["invalid_commands"]:
                raise ValueError("Counterfactual generated invalid command")
            branch[arm] = {"original_reward": original["reward"],
                "original_steps": original["steps"],
                "alternate_action": alternate,
                "swapped_reward": changed["reward"],
                "swapped_steps": changed["steps"],
                "swapped_trajectory_sha256": digest(changed["trajectory"])}
        results.append({"target_game": row["target_game"],
            "selection_input_content_sha256": row["input_content_sha256"],
            "preferred_arm": row["preferred_arm"],
            "first_divergence": row["first_divergence"],
            "branches": branch})
        print(json.dumps({"n": len(results), "game": row["target_game"],
                          "branches": branch}), flush=True)
    output = {"protocol": "Train-only first-divergence action intervention; frozen source-conditioned policy and shared reset; original complete rollout reproduced before alternate action, same source LoRA continued thereafter; not a general causal skill proof",
        "rollouts_sha256": file_hash(args.rollouts),
        "selection_candidates_sha256": file_hash(args.selection_candidates),
        "selection_review_sha256": file_hash(args.selection_review),
        "checkpoint_sha256": file_hash(args.checkpoint),
        "results": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False,
                                      indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path(
        "current_work/delta-Mem/model/Qwen3-4B-Instruct-2507"))
    parser.add_argument("--checkpoint", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_sibling_taskpair_current1000_20261006.pt"))
    parser.add_argument("--rollouts", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_sibling_train60_taskpair_current1000_20261006.json"))
    parser.add_argument("--expected-games", type=int, default=60)
    parser.add_argument("--source-candidates", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_sibling_train120_candidates_20261006.json"))
    parser.add_argument("--source-review", type=Path, default=Path(
        "data/annotations/alf_sibling_train120_reviewed_20261006.json"))
    parser.add_argument("--retry-candidates", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_same_game_retry_candidates_20261006.json"))
    parser.add_argument("--retry-review", type=Path, default=Path(
        "data/annotations/alf_same_game_retry_reviewed_20261006.json"))
    parser.add_argument("--all-source-review", type=Path, default=Path(
        "data/annotations/alf_expert_first_attempts_reviewed_20261006.json"))
    parser.add_argument("--data-root", type=Path, default=Path(
        "ttcl/data/alfworld_delta"))
    parser.add_argument("--selection-candidates", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_first_divergence_train60_candidates_20261006.json"))
    parser.add_argument("--selection-review", type=Path, default=Path(
        "data/annotations/alf_first_divergence_train60_reviewed_20261006.json"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--gpu-fraction", type=float, default=.6)
    parser.add_argument("--max-source-tokens", type=int, default=2048)
    parser.add_argument("--output", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_first_divergence_intervention_train60_20261006.json"))
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
