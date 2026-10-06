"""Reviewed ALFWorld first-failure -> same-game retry source specificity pilot.

The first attempt is a frozen, model-generated, failed train-game rollout.
The next attempt receives that attempt only through a generated LoRA. Wrong
source and no-LoRA controls receive the same game reset and actor budget.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path

from ttcl.experience_evolution.environment import make_env
from ttcl.trajectory_hyperlora.alfworld_online_from_empty import records_from_episode
from ttcl.trajectory_hyperlora.alfworld_zero_shot_probe import load_agent, run_episode
from ttcl.trajectory_hyperlora.relational_router_pilot import tokenize_records


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True,
        ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def source_input(row: dict) -> dict:
    return {key: row[key] for key in ("target_game", "target_game_sha256",
        "own_source_sha256", "own_records", "wrong_source_game",
        "wrong_source_sha256", "wrong_records")}


def prepare(args):
    if args.candidates.exists():
        raise FileExistsError(args.candidates)
    plan = json.loads(args.plan.read_text())
    report = json.loads(args.first_attempts.read_text())
    if (report["plan_sha256"] != file_hash(args.plan) or
            report["checkpoint_sha256"] != file_hash(args.checkpoint) or
            report["max_steps"] != 30 or report["max_new_tokens"] != 64):
        raise ValueError("First-attempt frozen lineage/budget mismatch")
    train = {game for sequence in plan["training"] for game in sequence["games"]}
    evaluation = {game for sequence in plan["evaluation"]
                  for game in sequence["games"]}
    if train & evaluation or plan["max_steps"] != 30 or \
            plan["actor_max_tokens"] != 64:
        raise ValueError("Frozen train/test plan changed")
    by_family = defaultdict(list)
    for item in report["games"]:
        episode = item["base"]
        game = item["game"]
        if (game not in train or game in evaluation or
                file_hash(args.data_root / game) != item["game_sha256"] or
                episode["status"] != "complete" or
                episode["reward"] != 0 or episode["steps"] != 30 or
                len(episode["trajectory"]) != 30):
            continue
        row = {"target_game": game,
               "target_game_sha256": item["game_sha256"],
               "family": item["family"],
               "own_source_sha256": digest(episode),
               "own_records": records_from_episode(episode),
               "reference_input_sha256": item["input_content_sha256"]}
        by_family[row["family"]].append(row)
    selected = []
    for family, rows in sorted(by_family.items()):
        if len(rows) < 3:
            continue  # A dev target needs distinct reviewed train wrong sources.
        rows.sort(key=lambda row: digest(["retry_split", row["target_game"]]))
        count = max(2, math.floor(.7 * len(rows)))
        for index, row in enumerate(rows):
            selected.append({**row, "split": "train" if index < count else "dev"})
    train_by_family = defaultdict(list)
    for row in selected:
        if row["split"] == "train":
            train_by_family[row["family"]].append(row)
    for row in selected:
        wrong = min((candidate for candidate in train_by_family[row["family"]]
                     if candidate["target_game"] != row["target_game"]),
                    key=lambda candidate: digest(["retry_wrong",
                        row["target_game"], candidate["target_game"]]))
        row["wrong_source_game"] = wrong["target_game"]
        row["wrong_source_sha256"] = wrong["own_source_sha256"]
        row["wrong_records"] = wrong["own_records"]
        row["input_content_sha256"] = digest(source_input(row))
    selected.sort(key=lambda row: (row["split"], row["target_game"]))
    if len({row["target_game"] for row in selected}) != len(selected):
        raise ValueError("Duplicate retry target")
    result = {"protocol": "Frozen ALFWorld train-game first failed attempts for same-game retry; exact source and target content binding; dev split distinct from reward training",
              "plan_sha256": file_hash(args.plan),
              "first_attempts_sha256": file_hash(args.first_attempts),
              "checkpoint_sha256": file_hash(args.checkpoint),
              "budget": {"steps": 30, "new_tokens": 64},
              "rows": selected}
    args.candidates.parent.mkdir(parents=True, exist_ok=True)
    args.candidates.write_text(json.dumps(result, ensure_ascii=False,
                                           indent=2) + "\n")
    print(json.dumps({split: sum(row["split"] == split for row in selected)
                      for split in ("train", "dev")}), flush=True)
    return result


def review(args):
    if args.review.exists():
        raise FileExistsError(args.review)
    candidates = json.loads(args.candidates.read_text())
    report = json.loads(args.first_attempts.read_text())
    original = {row["game"]: row["base"] for row in report["games"]}
    if candidates["first_attempts_sha256"] != file_hash(args.first_attempts):
        raise ValueError("Frozen source report changed")
    notes = []
    for row in candidates["rows"]:
        if (file_hash(args.data_root / row["target_game"]) !=
                row["target_game_sha256"] or
                digest(source_input(row)) != row["input_content_sha256"] or
                digest(original[row["target_game"]]) !=
                row["own_source_sha256"] or
                records_from_episode(original[row["target_game"]]) !=
                row["own_records"] or
                digest(original[row["wrong_source_game"]]) !=
                row["wrong_source_sha256"] or
                records_from_episode(original[row["wrong_source_game"]]) !=
                row["wrong_records"]):
            raise ValueError("New retry source or target binding changed")
        episode = original[row["target_game"]]
        env = make_env(args.data_root / row["target_game"])
        try:
            state = env.reset()
            if str(state["feedback"]) != episode["initial_observation"]:
                raise ValueError("First observation mismatch on replay")
            for step in episode["trajectory"]:
                if step["command"] not in state["admissible_commands"]:
                    raise ValueError("First attempt used an inadmissible action")
                state, _, done = env.step(step["command"])
                if (str(state["feedback"]) != step["observation"] or
                        bool(state["won"]) != bool(step["won"])):
                    raise ValueError("First-attempt transition mismatch")
                if done and step is not episode["trajectory"][-1]:
                    raise ValueError("Episode continued after environment done")
            if state["won"]:
                raise ValueError("Reviewed failure actually won")
        finally:
            env.close()
        notes.append({"target_game": row["target_game"],
                      "input_content_sha256": row["input_content_sha256"],
                      "own_source_sha256": row["own_source_sha256"],
                      "wrong_source_sha256": row["wrong_source_sha256"],
                      "approved": True,
                      "review_basis": "Official train game; model's failed 30-step first attempt replayed exactly; public observations, admissible commands, action and won checked; new source/target content hash"})
    result = {"protocol": "Fresh reviewed same-game retry targets; no old action labels inherited",
              "candidates_sha256": file_hash(args.candidates),
              "annotations": notes}
    args.review.parent.mkdir(parents=True, exist_ok=True)
    args.review.write_text(json.dumps(result, ensure_ascii=False,
                                       indent=2) + "\n")
    print(json.dumps({"reviewed": len(notes)}), flush=True)
    return result


def checked_rows(args):
    candidates = json.loads(args.candidates.read_text())
    review = json.loads(args.review.read_text())
    if (review["candidates_sha256"] != file_hash(args.candidates) or
            candidates["plan_sha256"] != file_hash(args.plan) or
            candidates["checkpoint_sha256"] != file_hash(args.checkpoint) or
            candidates["first_attempts_sha256"] != file_hash(args.first_attempts)):
        raise ValueError("Reviewed retry lineage changed")
    approved = {note["input_content_sha256"]: note
                for note in review["annotations"]}
    if len(approved) != len(candidates["rows"]):
        raise ValueError("Every retry needs a unique new annotation")
    for row in candidates["rows"]:
        note = approved.get(row["input_content_sha256"])
        if (digest(source_input(row)) != row["input_content_sha256"] or
                note is None or note["approved"] is not True or
                note["target_game"] != row["target_game"] or
                note["own_source_sha256"] != row["own_source_sha256"] or
                note["wrong_source_sha256"] != row["wrong_source_sha256"] or
                file_hash(args.data_root / row["target_game"]) !=
                row["target_game_sha256"]):
            raise ValueError("Unreviewed or changed retry target")
    return [row for row in candidates["rows"] if row["split"] == args.split]


def evaluate(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    rows = checked_rows(args)
    if args.limit:
        rows = rows[:args.limit]
    actor_checkpoint = args.actor_checkpoint or args.checkpoint
    agent, tokenizer = load_agent(args.model, actor_checkpoint,
                                  args.device, args.gpu_fraction)
    for adapter in agent.adapters:
        adapter.scale = args.adapter_scale
    original = {row["game"]: row["base"] for row in
                json.loads(args.first_attempts.read_text())["games"]}
    output = {"protocol": "Paired same-game retry on official ALFWorld train games; frozen hypernetwork; own failed first attempt versus wrong same-family failed first attempt versus none; won reward",
              "candidates_sha256": file_hash(args.candidates),
              "review_sha256": file_hash(args.review),
              "checkpoint_sha256": file_hash(args.checkpoint),
              "actor_checkpoint_sha256": file_hash(actor_checkpoint),
              "source_max_tokens": args.source_max_tokens,
              "source_truncation": args.source_truncation,
              "repeat_initial_observation": args.repeat_initial_observation,
              "adapter_scale": args.adapter_scale,
              "arms_requested": args.arms,
              "split": args.split, "max_steps": 30, "max_new_tokens": 64,
              "games": [], "failures": []}
    selected_arms = (("base", "own", "wrong") if args.arms == "all"
                     else ("own",))
    for row in rows:
        try:
            own = tokenize_records(tokenizer, row["own_records"], args.device,
                max_tokens=args.source_max_tokens,
                truncation_mode=args.source_truncation,
                repeat_initial_observation=args.repeat_initial_observation)
            wrong = tokenize_records(tokenizer, row["wrong_records"], args.device,
                max_tokens=args.source_max_tokens,
                truncation_mode=args.source_truncation,
                repeat_initial_observation=args.repeat_initial_observation)
            arms = {}
            for arm, fields in (("base", None), ("own", own), ("wrong", wrong)):
                if arm not in selected_arms:
                    continue
                arms[arm] = run_episode(agent, tokenizer,
                    args.data_root / row["target_game"], fields or {},
                    adapter=fields is not None, device=args.device,
                    max_steps=30, max_new_tokens=64,
                    constrain_actions=True)
                if arms[arm]["status"] != "complete":
                    raise RuntimeError(f"{arm}: {arms[arm].get('error')}")
            reference = original[row["target_game"]]
            if (any(episode["initial_observation"] !=
                    reference["initial_observation"]
                    for episode in arms.values()) or
                    ("base" in arms and
                     arms["base"]["reward"] != reference["reward"])):
                raise ValueError("Retry reset or frozen baseline mismatch")
            output["games"].append({"game": row["target_game"],
                "family": row["family"],
                "input_content_sha256": row["input_content_sha256"],
                "own_source_sha256": row["own_source_sha256"],
                "wrong_source_game": row["wrong_source_game"],
                "wrong_source_sha256": row["wrong_source_sha256"],
                "arms": arms})
        except Exception as exc:
            output["failures"].append({"game": row["target_game"],
                "error": f"{type(exc).__name__}: {exc}"})
        output["summary"] = {"n": len(output["games"]),
            "failures": len(output["failures"]),
            **{arm: sum(item["arms"][arm]["reward"]
                        for item in output["games"])
                for arm in selected_arms}}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(json.dumps(output, ensure_ascii=False,
                                        indent=2) + "\n")
        temporary.replace(args.output)
        print(json.dumps(output["summary"]), flush=True)
    return output


def memory_text(records: list[dict[str, str]]) -> str:
    """Serialize every public step without task-specific parsing or labels."""
    return "\n".join(f"{index + 1}. Observation: {step['observation'][:180]} "
        f"Action: {step['action']} Feedback: {step['feedback'][:180]}"
        for index, step in enumerate(records))


def evaluate_text(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    rows = checked_rows(args)
    if args.limit:
        rows = rows[:args.limit]
    agent, tokenizer = load_agent(args.model, args.checkpoint,
                                  args.device, args.gpu_fraction)
    output = {"protocol": "Same-game retry information probe: frozen actor sees entire public first attempt as text; own versus same-family wrong source; no LoRA",
              "candidates_sha256": file_hash(args.candidates),
              "review_sha256": file_hash(args.review),
              "checkpoint_sha256": file_hash(args.checkpoint),
              "split": args.split, "max_steps": 30, "max_new_tokens": 64,
              "games": [], "failures": []}
    for row in rows:
        try:
            arms = {}
            for arm, records in (("own_text", row["own_records"]),
                                 ("wrong_text", row["wrong_records"])):
                text = memory_text(records)
                arms[arm] = run_episode(agent, tokenizer,
                    args.data_root / row["target_game"], {},
                    adapter=False, device=args.device,
                    max_steps=30, max_new_tokens=64,
                    constrain_actions=True, memory_text=text)
                if arms[arm]["status"] != "complete":
                    raise RuntimeError(f"{arm}: {arms[arm].get('error')}")
            if (arms["own_text"]["initial_observation"] !=
                    arms["wrong_text"]["initial_observation"]):
                raise ValueError("Text retry reset mismatch")
            output["games"].append({"game": row["target_game"],
                "input_content_sha256": row["input_content_sha256"],
                "own_memory_sha256": hashlib.sha256(memory_text(
                    row["own_records"]).encode()).hexdigest(),
                "wrong_memory_sha256": hashlib.sha256(memory_text(
                    row["wrong_records"]).encode()).hexdigest(),
                "arms": arms})
        except Exception as exc:
            output["failures"].append({"game": row["target_game"],
                "error": f"{type(exc).__name__}: {exc}"})
        output["summary"] = {"n": len(output["games"]),
            "failures": len(output["failures"]),
            **{arm: sum(item["arms"][arm]["reward"]
                        for item in output["games"])
                for arm in ("own_text", "wrong_text")}}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(json.dumps(output, ensure_ascii=False,
                                        indent=2) + "\n")
        temporary.replace(args.output)
        print(json.dumps(output["summary"]), flush=True)
    return output


def evaluate_cross(args):
    """Diagnostic: use a reviewed train source from another task family."""
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.all_candidates is None or args.all_source_review is None:
        raise ValueError("Cross-family diagnostic needs reviewed train sources")
    from ttcl.trajectory_hyperlora.alfworld_expert_first_attempts import checked_all_sources
    sources = checked_all_sources(argparse.Namespace(
        candidates=args.all_candidates,
        source_review=args.all_source_review,
        first_attempts=args.first_attempts,
        retry_candidates=args.candidates,
        plan=args.plan, data_root=args.data_root))
    rows = checked_rows(args)
    if args.limit:
        rows = rows[:args.limit]
    actor_checkpoint = args.actor_checkpoint or args.checkpoint
    agent, tokenizer = load_agent(args.model, actor_checkpoint,
                                  args.device, args.gpu_fraction)
    for adapter in agent.adapters:
        adapter.scale = args.adapter_scale
    original = {row["game"]: row["base"] for row in
                json.loads(args.first_attempts.read_text())["games"]}
    output = {"protocol": "ALFWorld same-game retry cross-family diagnostic; different reviewed train-game history; same frozen actor, reset, 30-step and 64-token budget",
              "candidates_sha256": file_hash(args.candidates),
              "review_sha256": file_hash(args.review),
              "all_candidates_sha256": file_hash(args.all_candidates),
              "all_source_review_sha256": file_hash(args.all_source_review),
              "actor_checkpoint_sha256": file_hash(actor_checkpoint),
              "source_max_tokens": args.source_max_tokens,
              "source_truncation": args.source_truncation,
              "repeat_initial_observation": args.repeat_initial_observation,
              "adapter_scale": args.adapter_scale,
              "split": args.split, "max_steps": 30, "max_new_tokens": 64,
              "games": [], "failures": []}
    for row in rows:
        try:
            source = min((candidate for candidate in sources
                          if candidate["family"] != row["family"]),
                         key=lambda candidate: digest(["cross_family",
                             row["target_game"], candidate["target_game"]]))
            fields = tokenize_records(tokenizer, source["source_records"],
                args.device, max_tokens=args.source_max_tokens,
                truncation_mode=args.source_truncation,
                repeat_initial_observation=args.repeat_initial_observation)
            episode = run_episode(agent, tokenizer,
                args.data_root / row["target_game"], fields, adapter=True,
                device=args.device, max_steps=30, max_new_tokens=64,
                constrain_actions=True)
            if (episode["status"] != "complete" or
                    episode["initial_observation"] != original[
                        row["target_game"]]["initial_observation"]):
                raise ValueError("Cross-family reset or rollout mismatch")
            output["games"].append({"game": row["target_game"],
                "family": row["family"],
                "input_content_sha256": row["input_content_sha256"],
                "cross_source_game": source["target_game"],
                "cross_source_family": source["family"],
                "cross_source_content_sha256": source["input_content_sha256"],
                "cross": episode})
        except Exception as exc:
            output["failures"].append({"game": row["target_game"],
                "error": f"{type(exc).__name__}: {exc}"})
        output["summary"] = {"n": len(output["games"]),
            "failures": len(output["failures"]),
            "cross": sum(item["cross"]["reward"]
                         for item in output["games"])}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(json.dumps(output, ensure_ascii=False,
                                        indent=2) + "\n")
        temporary.replace(args.output)
        print(json.dumps(output["summary"]), flush=True)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "review", "evaluate",
                                            "evaluate-text", "evaluate-cross"))
    parser.add_argument("--model", type=Path, default=Path(
        "current_work/delta-Mem/model/Qwen3-4B-Instruct-2507"))
    parser.add_argument("--checkpoint", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_next_task_warmstart_20261005/seed42_v2.pt"))
    parser.add_argument("--actor-checkpoint", type=Path,
                        help="Evaluation-only trained hypernetwork; --checkpoint remains the frozen source lineage")
    parser.add_argument("--plan", type=Path, default=Path(
        "ttcl/results/experience_evolution/alfworld_delta_20260922/plan.json"))
    parser.add_argument("--first-attempts", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_rl_20261005/train_rewards_complete.json"))
    parser.add_argument("--data-root", type=Path, default=Path(
        "ttcl/data/alfworld_delta"))
    parser.add_argument("--candidates", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_same_game_retry_candidates_20261006.json"))
    parser.add_argument("--review", type=Path, default=Path(
        "data/annotations/alf_same_game_retry_reviewed_20261006.json"))
    parser.add_argument("--all-candidates", type=Path)
    parser.add_argument("--all-source-review", type=Path)
    parser.add_argument("--split", choices=("train", "dev"), default="train")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpu-fraction", type=float, default=.6)
    parser.add_argument("--source-max-tokens", type=int, default=40)
    parser.add_argument("--source-truncation", choices=("head", "head_tail"),
                        default="head")
    parser.add_argument("--repeat-initial-observation", action="store_true")
    parser.add_argument("--adapter-scale", type=float, default=1.0)
    parser.add_argument("--arms", choices=("all", "own"), default="all")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if (args.limit < 0 or args.source_max_tokens < 2 or
            (args.repeat_initial_observation and
             args.source_truncation != "head_tail") or
            not 0 < args.adapter_scale <= 1 or
            not 0 < args.gpu_fraction <= 1):
        parser.error("Invalid retry budget")
    if args.command == "prepare":
        prepare(args)
    elif args.command == "review":
        review(args)
    elif args.command == "evaluate":
        if args.output is None:
            parser.error("evaluate needs --output")
        evaluate(args)
    elif args.command == "evaluate-text":
        if args.output is None:
            parser.error("evaluate-text needs --output")
        evaluate_text(args)
    else:
        if args.output is None:
            parser.error("evaluate-cross needs --output")
        evaluate_cross(args)


if __name__ == "__main__":
    main()
