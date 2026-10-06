"""Cross-task ALFWorld history probe using a completed sibling train game.

Each development target is a distinct game file. Its positive source is an
officially replayed expert trajectory from another trial of the same task
directory; the wrong source is an equally successful train trajectory from a
different task directory of the same family. Target walkthroughs are never
read. This diagnostic tests cross-game history rather than same-game retry.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import torch

from ttcl.experience_evolution.environment import make_env
from ttcl.trajectory_hyperlora.alfworld_expert_first_attempts import checked_all_sources
from ttcl.trajectory_hyperlora.alfworld_same_game_retry import (
    checked_rows, digest, file_hash,
)
from ttcl.trajectory_hyperlora.alfworld_zero_shot_probe import load_agent, run_episode
from ttcl.trajectory_hyperlora.contextual_alf_source import (
    contextual_source_fields, contextual_text_fields,
)
from ttcl.trajectory_hyperlora.relational_router_pilot import tokenize_records


def sibling(game: str, root: Path) -> str:
    path = root / game
    others = [item for item in path.parent.parent.glob("trial_*/game.tw-pddl")
              if item != path]
    if not others:
        raise ValueError(f"No distinct sibling game for {game}")
    selected = min(others, key=lambda item: digest(
        ["sibling_expert_source", game, item.relative_to(root).as_posix()]))
    result = selected.relative_to(root).as_posix()
    if "/train/" not in "/" + result:
        raise ValueError("Sibling source escaped training split")
    return result


def row_content(row):
    return {key: value for key, value in row.items()
            if key != "input_content_sha256"}


def large_train_targets(args):
    """One frozen game per task directory, excluding development directories."""
    development = checked_rows(argparse.Namespace(
        candidates=args.retry_candidates, review=args.review,
        checkpoint=args.checkpoint, first_attempts=args.first_attempts,
        plan=args.plan, data_root=args.data_root, split="dev"))
    excluded = {Path(row["target_game"]).parent.parent
                for row in development}
    root = args.data_root / "json_2.1.1" / "train"
    groups = defaultdict(list)
    for directory in root.iterdir():
        if not directory.is_dir() or directory.relative_to(args.data_root) in excluded:
            continue
        games = list(directory.glob("trial_*/game.tw-pddl"))
        if len(games) < 2:
            continue
        game = min(games, key=lambda item: digest([
            "large_train_target_trial", directory.name,
            item.relative_to(args.data_root).as_posix()]))
        relative = game.relative_to(args.data_root).as_posix()
        family = directory.name.split("-", 1)[0]
        sha = file_hash(game)
        groups[family].append({"target_game": relative,
            "target_game_sha256": sha, "family": family,
            "input_content_sha256": digest([
                "large_train_target", relative, sha])})
    chosen = []
    for family in sorted(groups):
        ordered = sorted(groups[family], key=lambda row: digest([
            "large_train_directory", row["target_game"]]))
        if len(ordered) < args.per_family:
            raise ValueError(f"Too few isolated train directories: {family}")
        chosen.extend(ordered[:args.per_family])
    if len(chosen) != 6 * args.per_family:
        raise ValueError("Expanded sibling training needs all six families")
    return chosen


def prepare(args):
    if args.candidates.exists():
        raise FileExistsError(args.candidates)
    sources = checked_all_sources(argparse.Namespace(
        candidates=args.all_candidates,
        source_review=args.all_source_review,
        first_attempts=args.first_attempts,
        retry_candidates=args.retry_candidates,
        plan=args.plan, data_root=args.data_root))
    targets = (large_train_targets(args) if args.split == "train_large" else
        sources if args.split == "train" else checked_rows(
            argparse.Namespace(candidates=args.retry_candidates,
                review=args.review, checkpoint=args.checkpoint,
                first_attempts=args.first_attempts, plan=args.plan,
                data_root=args.data_root, split="dev")))
    groups = defaultdict(list)
    for row in (targets if args.split == "train_large" else sources):
        groups[row["family"]].append(row)
    rows = []
    for target in targets:
        game = target["target_game"]
        source = sibling(game, args.data_root)
        other = min((item for item in groups[target["family"]]
                     if Path(item["target_game"]).parent.parent !=
                        Path(game).parent.parent),
                    key=lambda item: digest([
                        "wrong_sibling_expert", game, item["target_game"]]))
        wrong = sibling(other["target_game"], args.data_root)
        if wrong == source or wrong == game or source == game:
            raise ValueError("Sibling source/target isolation failed")
        row = {"target_game": game,
            "target_game_sha256": target["target_game_sha256"],
            "family": target["family"],
            "source_game": source,
            "source_game_sha256": file_hash(args.data_root / source),
            "wrong_source_game": wrong,
            "wrong_source_game_sha256": file_hash(args.data_root / wrong),
            "target_review_input_sha256": target["input_content_sha256"]}
        row["input_content_sha256"] = digest(row_content(row))
        rows.append(row)
    result = {"protocol": "Frozen ALFWorld train or train-domain development targets; expert source is another trial of the same task directory, wrong expert source is another directory of the same family; development target expert content never read",
        "split": args.split,
        "retry_candidates_sha256": file_hash(args.retry_candidates),
        "retry_review_sha256": file_hash(args.review),
        "all_source_review_sha256": file_hash(args.all_source_review),
        "rows": rows}
    args.candidates.parent.mkdir(parents=True, exist_ok=True)
    args.candidates.write_text(json.dumps(result, ensure_ascii=False,
                                           indent=2) + "\n")
    print(json.dumps({"split": args.split,
                      "targets": len(rows)}), flush=True)


def expert_records(game: Path):
    commands = json.loads(game.read_text())["walkthrough"]
    if not commands or len(commands) > 30:
        raise ValueError("Sibling expert exceeds 30-step source budget")
    env = make_env(game)
    try:
        state = env.reset()
        result = []
        for index, command in enumerate(commands):
            if command not in state["admissible_commands"]:
                raise ValueError("Sibling expert command inadmissible")
            before = str(state["feedback"])
            state, _, done = env.step(command)
            if done and index != len(commands) - 1:
                raise ValueError("Sibling expert ended before final command")
            status = "task completed" if index == len(commands) - 1 else "valid command"
            result.append({"observation": before,
                           "action": command,
                           "feedback": status + ". " + str(state["feedback"])})
        if not state["won"]:
            raise ValueError("Sibling expert did not win")
        return result
    finally:
        env.close()


def checked_review(args):
    candidates = json.loads(args.candidates.read_text())
    review = json.loads(args.source_review.read_text())
    if (candidates.get("split", "dev") != args.split or
            candidates["retry_candidates_sha256"] !=
            file_hash(args.retry_candidates) or
            candidates["retry_review_sha256"] != file_hash(args.review) or
            candidates["all_source_review_sha256"] !=
            file_hash(args.all_source_review) or
            review["candidates_sha256"] != file_hash(args.candidates)):
        raise ValueError("Sibling transfer lineage changed")
    approved = {note["input_content_sha256"]: note
                for note in review["annotations"]}
    if len(approved) != len(candidates["rows"]):
        raise ValueError("Sibling sources lack reviewed bindings")
    for row in candidates["rows"]:
        note = approved.get(row["input_content_sha256"])
        if (digest(row_content(row)) != row["input_content_sha256"] or
                file_hash(args.data_root / row["target_game"]) !=
                row["target_game_sha256"] or
                file_hash(args.data_root / row["source_game"]) !=
                row["source_game_sha256"] or
                file_hash(args.data_root / row["wrong_source_game"]) !=
                row["wrong_source_game_sha256"] or
                note is None or note["approved"] is not True or
                note["source_records_sha256"] !=
                    digest(note["source_records"]) or
                note["wrong_records_sha256"] !=
                    digest(note["wrong_records"])):
            raise ValueError("Sibling reviewed source content changed")
    return [(row, approved[row["input_content_sha256"]])
            for row in candidates["rows"]]


def review(args):
    if args.source_review.exists():
        raise FileExistsError(args.source_review)
    candidates = json.loads(args.candidates.read_text())
    notes = []
    for row in candidates["rows"]:
        if (digest(row_content(row)) != row["input_content_sha256"] or
                file_hash(args.data_root / row["target_game"]) !=
                row["target_game_sha256"] or
                file_hash(args.data_root / row["source_game"]) !=
                row["source_game_sha256"] or
                file_hash(args.data_root / row["wrong_source_game"]) !=
                row["wrong_source_game_sha256"]):
            raise ValueError("Sibling candidate content changed")
        own = expert_records(args.data_root / row["source_game"])
        wrong = expert_records(args.data_root / row["wrong_source_game"])
        notes.append({"target_game": row["target_game"],
            "input_content_sha256": row["input_content_sha256"],
            "source_records": own,
            "source_records_sha256": digest(own),
            "wrong_records": wrong,
            "wrong_records_sha256": digest(wrong),
            "approved": True,
            "review_basis": "Both distinct train-game sibling histories replayed with admissible commands and terminal won; target walkthrough never read"})
    result = {"protocol": "Fresh content-bound successful sibling histories for cross-game ALFWorld development transfer",
        "candidates_sha256": file_hash(args.candidates),
        "annotations": notes}
    args.source_review.parent.mkdir(parents=True, exist_ok=True)
    args.source_review.write_text(json.dumps(result, ensure_ascii=False,
                                       indent=2) + "\n")
    print(json.dumps({"reviewed": len(notes)}), flush=True)


def evaluate(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    rows = checked_review(args)
    if args.per_family_limit:
        selected, counts = [], defaultdict(int)
        for row, note in rows:
            if counts[row["family"]] < args.per_family_limit:
                selected.append((row, note))
                counts[row["family"]] += 1
        rows = selected
    if args.limit:
        rows = rows[:args.limit]
    agent, tokenizer = load_agent(args.model, args.checkpoint,
                                  args.device, args.gpu_fraction)
    result = {"protocol": f"Cross-game ALFWorld {args.split} transfer; task-sibling expert source versus another same-family expert source versus no LoRA; target rollout has no expert leakage; 30 steps, 64 tokens, official won",
        "candidates_sha256": file_hash(args.candidates),
        "source_review_sha256": file_hash(args.source_review),
        "checkpoint_sha256": file_hash(args.checkpoint),
        "source_truncation": args.source_truncation,
        "source_max_tokens": args.source_max_tokens,
        "per_family_limit": args.per_family_limit,
        "games": [], "failures": []}
    for row, note in rows:
        try:
            arms = {}
            for arm, records in (("base", None),
                ("own", note["source_records"]),
                ("wrong", note["wrong_records"])):
                fields = (
                    contextual_source_fields(agent, tokenizer, records,
                        args.device, args.contextual_source_max_tokens)
                    if agent.encoder_kind == "contextual" else
                    tokenize_records(tokenizer, records, args.device,
                        max_tokens=args.source_max_tokens,
                        truncation_mode=args.source_truncation)) \
                    if records is not None else {}
                arms[arm] = run_episode(agent, tokenizer,
                    args.data_root / row["target_game"], fields,
                    adapter=records is not None, device=args.device,
                    max_steps=30, max_new_tokens=64,
                    constrain_actions=True)
                if arms[arm]["status"] != "complete":
                    raise ValueError("Sibling target rollout failed")
            if len({arm["initial_observation"] for arm in arms.values()}) != 1:
                raise ValueError("Sibling target reset mismatch")
            result["games"].append({"game": row["target_game"],
                "family": row["family"],
                "source_game": row["source_game"],
                "wrong_source_game": row["wrong_source_game"],
                "input_content_sha256": row["input_content_sha256"],
                "arms": arms})
        except Exception as exc:
            result["failures"].append({"game": row["target_game"],
                "error": f"{type(exc).__name__}: {exc}"})
        result["summary"] = {"n": len(result["games"]),
            "failures": len(result["failures"]),
            **{arm: sum(item["arms"][arm]["reward"]
                for item in result["games"])
                for arm in ("base", "own", "wrong")}}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(json.dumps(result, ensure_ascii=False,
                                        indent=2) + "\n")
        temporary.replace(args.output)
        print(json.dumps(result["summary"]), flush=True)


def evaluate_text(args):
    """Check whether the same reviewed source helps when shown as text."""
    if args.output.exists():
        raise FileExistsError(args.output)
    rows = checked_review(args)
    if args.limit:
        rows = rows[:args.limit]
    agent, tokenizer = load_agent(args.model, args.checkpoint,
                                  args.device, args.gpu_fraction)
    result = {"protocol": "ALFWorld sibling expert text diagnostic; reviewed source and wrong source with full initial observation, actions, and clipped later observations/feedback in actor system context; no LoRA; 30 steps, 64 tokens, official won",
        "candidates_sha256": file_hash(args.candidates),
        "source_review_sha256": file_hash(args.source_review),
        "checkpoint_sha256": file_hash(args.checkpoint),
        "games": [], "failures": []}
    for row, note in rows:
        try:
            arms = {}
            for arm, records in (("own_text", note["source_records"]),
                                 ("wrong_text", note["wrong_records"])):
                arms[arm] = run_episode(agent, tokenizer,
                    args.data_root / row["target_game"], {}, adapter=False,
                    device=args.device, max_steps=30, max_new_tokens=64,
                    constrain_actions=True,
                    memory_text=sibling_memory_text(records))
                if arms[arm]["status"] != "complete":
                    raise ValueError("Sibling text target rollout failed")
            result["games"].append({"game": row["target_game"],
                "input_content_sha256": row["input_content_sha256"],
                "arms": arms})
        except Exception as exc:
            result["failures"].append({"game": row["target_game"],
                                       "error": f"{type(exc).__name__}: {exc}"})
        result["summary"] = {"n": len(result["games"]),
            "failures": len(result["failures"]),
            **{arm: sum(item["arms"][arm]["reward"]
                for item in result["games"])
                for arm in ("own_text", "wrong_text")}}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(json.dumps(result, ensure_ascii=False,
                                        indent=2) + "\n")
        temporary.replace(args.output)
        print(json.dumps(result["summary"]), flush=True)


def sibling_memory_text(records):
    """Retain the initial goal in full; cap later observations uniformly."""
    return "\n".join(
        f"{index + 1}. Observation: "
        f"{step['observation'] if index == 0 else step['observation'][:180]} "
        f"Action: {step['action']} Feedback: {step['feedback'][:180]}"
        for index, step in enumerate(records))


def evaluate_centered(args):
    """Remove the train-source common LoRA component before target rollout."""
    if args.output.exists():
        raise FileExistsError(args.output)
    rows = checked_review(args)
    if args.limit:
        rows = rows[:args.limit]
    training = checked_review(argparse.Namespace(**{
        **vars(args), "candidates": args.train_sibling_candidates,
        "source_review": args.train_sibling_source_review, "split": "train"}))
    agent, tokenizer = load_agent(args.model, args.checkpoint,
                                  args.device, args.gpu_fraction)

    def generated(records):
        fields = (contextual_source_fields(agent, tokenizer, records,
            args.device, args.contextual_source_max_tokens)
            if agent.encoder_kind == "contextual" else
            tokenize_records(tokenizer, records, args.device,
                max_tokens=args.source_max_tokens,
                truncation_mode=args.source_truncation))
        with torch.no_grad():
            agent.set_source(fields)
            factors = [layer.b.detach().clone() for layer in agent.adapters]
            agent.set_source(None)
        return factors

    centers = None
    for _, note in training:
        factors = generated(note["source_records"])
        if centers is None:
            centers = [value.clone() for value in factors]
        else:
            for center, value in zip(centers, factors, strict=True):
                center.add_(value)
    centers = [center / len(training) for center in centers]
    result = {"protocol": "ALFWorld sibling residual-LoRA diagnostic; subtract mean B generated by 42 reviewed train sibling expert histories, then run correct and wrong development sources; fixed scale one; no target labels",
        "candidates_sha256": file_hash(args.candidates),
        "source_review_sha256": file_hash(args.source_review),
        "train_source_review_sha256": file_hash(args.train_sibling_source_review),
        "checkpoint_sha256": file_hash(args.checkpoint),
        "source_truncation": args.source_truncation,
        "source_max_tokens": args.source_max_tokens,
        "games": [], "failures": []}
    for row, note in rows:
        try:
            arms = {}
            for arm, records in (("own_centered", note["source_records"]),
                                 ("wrong_centered", note["wrong_records"])):
                factors = generated(records)
                residual = [value - center for value, center in
                            zip(factors, centers, strict=True)]
                arms[arm] = run_episode(agent, tokenizer,
                    args.data_root / row["target_game"], {}, adapter=False,
                    fixed_adapter=residual, device=args.device,
                    max_steps=30, max_new_tokens=64,
                    constrain_actions=True)
                if arms[arm]["status"] != "complete":
                    raise ValueError("Centered sibling target rollout failed")
            result["games"].append({"game": row["target_game"],
                "input_content_sha256": row["input_content_sha256"],
                "arms": arms})
        except Exception as exc:
            result["failures"].append({"game": row["target_game"],
                                       "error": f"{type(exc).__name__}: {exc}"})
        result["summary"] = {"n": len(result["games"]),
            "failures": len(result["failures"]),
            **{arm: sum(item["arms"][arm]["reward"]
                for item in result["games"])
                for arm in ("own_centered", "wrong_centered")}}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(json.dumps(result, ensure_ascii=False,
                                        indent=2) + "\n")
        temporary.replace(args.output)
        print(json.dumps(result["summary"]), flush=True)


def evaluate_gated(args):
    """Mount a trajectory LoRA only when a train-calibrated match score passes."""
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.gate_calibration is None:
        raise ValueError("Gated evaluation needs a train-only calibration")
    calibration = json.loads(args.gate_calibration.read_text())
    if (calibration["checkpoint_sha256"] != file_hash(args.checkpoint) or
            calibration["labels_sha256"] != file_hash(args.gate_train_labels) or
            calibration["label_review_sha256"] !=
                file_hash(args.gate_train_label_review) or
            calibration["source_review_sha256"] !=
                file_hash(args.train_sibling_source_review) or
            calibration["max_source_tokens"] !=
                args.contextual_source_max_tokens):
        raise ValueError("Gate calibration or train lineage changed")
    rows = checked_review(args)
    if args.limit:
        rows = rows[:args.limit]
    agent, tokenizer = load_agent(args.model, args.checkpoint,
                                  args.device, args.gpu_fraction)
    if agent.encoder_kind != "contextual":
        raise ValueError("Gated evaluation needs contextual trajectory LoRA")
    result = {"protocol": "Train-calibrated source/target latent compatibility gate for sibling trajectory LoRA; no development labels/rewards used for threshold; 30 steps, 64 tokens, official won",
        "candidates_sha256": file_hash(args.candidates),
        "source_review_sha256": file_hash(args.source_review),
        "checkpoint_sha256": file_hash(args.checkpoint),
        "gate_calibration_sha256": file_hash(args.gate_calibration),
        "threshold": calibration["threshold"],
        "games": [], "failures": []}
    for row, note in rows:
        try:
            game = args.data_root / row["target_game"]
            env = make_env(game)
            try:
                initial_observation = str(env.reset()["feedback"])
            finally:
                env.close()
            target_fields = contextual_text_fields(agent, tokenizer,
                "Current task observation:\n" + initial_observation,
                args.device, args.contextual_source_max_tokens)
            target_latent = agent.encode(target_fields).detach()
            arms, matches = {}, {}
            for arm, records in (("own_gated", note["source_records"]),
                                 ("wrong_gated", note["wrong_records"])):
                fields = contextual_source_fields(agent, tokenizer, records,
                    args.device, args.contextual_source_max_tokens)
                with torch.no_grad():
                    score = float((agent.encode(fields) * target_latent).mean())
                enabled = score >= calibration["threshold"]
                matches[arm] = {"score": score, "mounted": enabled}
                arms[arm] = run_episode(agent, tokenizer, game,
                    fields if enabled else {}, adapter=enabled,
                    device=args.device, max_steps=30, max_new_tokens=64,
                    constrain_actions=True)
                if (arms[arm]["status"] != "complete" or
                        arms[arm]["initial_observation"] !=
                        initial_observation):
                    raise ValueError("Gated target rollout failed or reset changed")
            result["games"].append({"game": row["target_game"],
                "input_content_sha256": row["input_content_sha256"],
                "matches": matches, "arms": arms})
        except Exception as exc:
            result["failures"].append({"game": row["target_game"],
                                       "error": f"{type(exc).__name__}: {exc}"})
        result["summary"] = {"n": len(result["games"]),
            "failures": len(result["failures"]),
            **{arm: sum(item["arms"][arm]["reward"]
                for item in result["games"])
                for arm in ("own_gated", "wrong_gated")},
            **{arm + "_mounted": sum(item["matches"][arm]["mounted"]
                for item in result["games"])
                for arm in ("own_gated", "wrong_gated")}}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(json.dumps(result, ensure_ascii=False,
                                        indent=2) + "\n")
        temporary.replace(args.output)
        print(json.dumps(result["summary"]), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "review", "evaluate",
                                            "evaluate-text", "evaluate-centered",
                                            "evaluate-gated"))
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
    parser.add_argument("--retry-candidates", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_same_game_retry_candidates_20261006.json"))
    parser.add_argument("--review", type=Path, default=Path(
        "data/annotations/alf_same_game_retry_reviewed_20261006.json"))
    parser.add_argument("--candidates", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_sibling_dev_candidates_20261006.json"))
    parser.add_argument("--source-review", type=Path, default=Path(
        "data/annotations/alf_sibling_dev_reviewed_20261006.json"))
    parser.add_argument("--all-candidates", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_expert_first_attempts_candidates_20261006.json"))
    parser.add_argument("--all-source-review", type=Path, default=Path(
        "data/annotations/alf_expert_first_attempts_reviewed_20261006.json"))
    parser.add_argument("--train-sibling-candidates", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_sibling_train42_candidates_20261006.json"))
    parser.add_argument("--train-sibling-source-review", type=Path, default=Path(
        "data/annotations/alf_sibling_train42_reviewed_20261006.json"))
    parser.add_argument("--gate-calibration", type=Path)
    parser.add_argument("--gate-train-labels", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_sibling_train120_labels_20261006.json"))
    parser.add_argument("--gate-train-label-review", type=Path, default=Path(
        "data/annotations/alf_sibling_train120_labels_reviewed_20261006.json"))
    parser.add_argument("--split", choices=("train", "train_large", "dev"),
                        default="dev")
    parser.add_argument("--per-family", type=int, default=20)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--per-family-limit", type=int, default=0)
    parser.add_argument("--source-max-tokens", type=int, default=40)
    parser.add_argument("--contextual-source-max-tokens", type=int,
                        default=2048)
    parser.add_argument("--source-truncation", choices=("head", "head_tail"),
                        default="head_tail")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpu-fraction", type=float, default=.6)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if (args.limit < 0 or args.per_family_limit < 0 or args.per_family < 1 or
            args.source_max_tokens < 2 or
            args.contextual_source_max_tokens < 2 or
            not 0 < args.gpu_fraction <= 1):
        parser.error("Invalid sibling transfer budget")
    if args.command == "prepare":
        prepare(args)
    elif args.command == "review":
        review(args)
    else:
        if args.output is None:
            parser.error("evaluate needs --output")
        if args.command == "evaluate-text":
            evaluate_text(args)
        elif args.command == "evaluate-centered":
            evaluate_centered(args)
        elif args.command == "evaluate-gated":
            evaluate_gated(args)
        else:
            evaluate(args)


if __name__ == "__main__":
    main()
