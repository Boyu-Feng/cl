"""Exploratory same-source ALFWorld development comparison for v28.

This twelve-game matrix was exposed in earlier research. It is model
development, never an independent final test or an online-own-history run.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from types import SimpleNamespace

import torch

from ttcl.trajectory_hyperlora.alfworld_actor_generic_v28 import (
    load_agent, run_episode,
)
from ttcl.trajectory_hyperlora.alfworld_online_from_empty import digest, file_hash
from ttcl.trajectory_hyperlora.alfworld_online_persistent_context_lora_v1 import (
    bounded_source_text,
)
from ttcl.trajectory_hyperlora.alfworld_source_utility_dataset_v2 import checked, save
from ttcl.trajectory_hyperlora.my_online_update_v28_model import OnlineUpdateModel
from ttcl.trajectory_hyperlora.train_alf_feedback_residual_v1 import (
    ResidualInjection, source_fields,
)
from ttcl.trajectory_hyperlora.train_my_online_update_v28_reward import (
    effective_residual_norm,
)


def inputs(args):
    parent = SimpleNamespace(**vars(args))
    parent.review = args.source_review
    sources, targets, _ = checked(parent)
    features = torch.load(args.features, map_location="cpu", weights_only=True)
    trained = torch.load(args.trained_model, map_location="cpu", weights_only=True)
    audit = json.loads(args.source_audit.read_text())
    matrix = json.loads(args.source_matrix.read_text())
    if (len(sources) != 8 or len(targets) != 12 or
            audit["review_sha256"] != file_hash(args.source_review) or
            audit["raw_report_sha256"] != file_hash(args.source_matrix) or
            audit["checkpoint_sha256"] != file_hash(args.checkpoint) or
            features["review_sha256"] != audit["review_sha256"] or
            features["checkpoint_sha256"] != audit["checkpoint_sha256"] or
            trained["checkpoint_sha256"] != audit["checkpoint_sha256"] or
            trained["stage"] != "reward_trained" or
            matrix["failures"] or len(matrix["base"]) != 12 or
            len(matrix["pairs"]) != 96 or
            digest([row["episode"]["initial_observation"]
                    for row in matrix["base"]]) !=
            features["base_observations_sha256"]):
        raise ValueError("Changed or incomplete development lineage")
    fixed = {"method": "我的在线更新 v28 exposed development comparison",
        "source_review_sha256": file_hash(args.source_review),
        "source_audit_sha256": file_hash(args.source_audit),
        "source_matrix_sha256": file_hash(args.source_matrix),
        "features_sha256": file_hash(args.features),
        "trained_model_sha256": file_hash(args.trained_model),
        "checkpoint_sha256": file_hash(args.checkpoint),
        "runner_sha256": file_hash(Path(__file__)),
        "actor_sha256": file_hash(Path(__file__).with_name(
            "alfworld_actor_generic_v28.py")),
        "action_guard_sha256": file_hash(Path(__file__).with_name(
            "action_feedback_guard_v28.py")),
        "seed": args.seed, "max_steps": 50,
        "max_new_tokens": 64, "history_turns": 2,
        "repeat_limit": 2, "max_effective_norm": args.max_effective_norm}
    target_rows = []
    for i, target in enumerate(targets):
        content = {"index": i, "game": target["game"],
            "game_sha256": file_hash(args.data_root / target["game"]),
            "source_review_sha256": fixed["source_review_sha256"],
            "trained_model_sha256": fixed["trained_model_sha256"],
            "runner_sha256": fixed["runner_sha256"]}
        if content["game_sha256"] != target["game_sha256"]:
            raise ValueError("Changed official development target")
        target_rows.append({**content,
            "input_content_sha256": digest(content),
            "reviewed_target": True})
    return sources, targets, features, trained, matrix, fixed, target_rows


def prepare(args):
    if args.review.exists():
        raise FileExistsError(args.review)
    *_, fixed, target_rows = inputs(args)
    args.review.parent.mkdir(parents=True, exist_ok=True)
    args.review.write_text(json.dumps({"protocol":
        "Fresh method-input binding for exposed 12-game development matrix; six same-source controls; not independent final test",
        **fixed, "targets": target_rows,
        "targets_sha256": digest(target_rows)},
        ensure_ascii=False, indent=2)+"\n")
    print(json.dumps({"reviewed_targets": len(target_rows)}), flush=True)


def evaluate(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    (sources, targets, features, trained, matrix, fixed,
     target_rows) = inputs(args)
    review = json.loads(args.review.read_text())
    if (review["targets"] != target_rows or
            review["targets_sha256"] != digest(target_rows) or
            any(review[key] != val for key, val in fixed.items())):
        raise ValueError("Changed development review")
    model = OnlineUpdateModel(trained["width"], trained["hidden"],
                              trained["code_dim"])
    model.load_state_dict(trained["model"])
    model.eval()
    with torch.no_grad():
        utility, codes = model(features["global"], features["events"],
            features["deltas"], features["event_counts"],
            features["targets"])
    agent, tokenizer = load_agent(args.model, args.checkpoint,
                                  args.device, args.gpu_fraction)
    agent.eval()
    for parameter in agent.parameters():
        parameter.requires_grad_(False)
    fields = source_fields(agent, tokenizer, sources, args)
    basis = trained["basis"].to(args.device)
    rng = random.Random(args.seed)
    report = {"protocol": "我的在线更新 v28 exposed development, same reviewed source bank and domain-neutral action guard for all six arms",
        "review_sha256": file_hash(args.review),
        "trained_model_sha256": file_hash(args.trained_model),
        "checkpoint_sha256": file_hash(args.checkpoint),
        "actor_sha256": fixed["actor_sha256"],
        "action_guard_sha256": fixed["action_guard_sha256"],
        "max_steps": 50, "max_new_tokens": 64,
        "history_turns": 2, "repeat_limit": 2,
        "games": [], "failures": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for i, (target, binding) in enumerate(zip(targets, target_rows,
                                              strict=True)):
        scores = utility[i]
        selected = int(scores.argmax()) if scores.max() > 0 else -1
        random_source = rng.randrange(len(sources))
        permuted = (selected + 3) % len(sources) if selected >= 0 else -1
        commands = {"no_lora": -1, "fixed_first": 0,
            "random_same_reward": random_source,
            "source_permuted": permuted,
            "selected_text": selected,
            "joint_lora": selected}
        game = args.data_root / target["game"]
        row = {"index": i, "game": target["game"],
            "input_content_sha256": binding["input_content_sha256"],
            "scores": scores.tolist(), "selected_source": selected,
            "arm_sources": commands, "arms": {}}
        for arm, source_id in commands.items():
            kwargs = {"adapter": source_id >= 0 and arm not in
                      ("no_lora", "selected_text"),
                "device": args.device, "max_steps": 50,
                "max_new_tokens": 64, "constrain_actions": True,
                "actor_history_turns": 2, "loop_guard_max": 2}
            memory_text = None
            if source_id >= 0 and arm == "selected_text":
                memory_text, _, _ = bounded_source_text(tokenizer,
                    sources[source_id]["records"], 2048)
                kwargs["memory_text"] = memory_text
            source_field = fields[source_id] if kwargs["adapter"] else {}
            if source_id >= 0 and arm == "joint_lora":
                mean = codes[i, source_id].to(args.device)
                norm = effective_residual_norm(agent, basis, mean)
                scale = min(1.0, args.max_effective_norm /
                            max(float(norm), 1e-12))
                row["joint_residual_norm"] = float(norm)
                row["joint_residual_scale"] = scale
                with ResidualInjection(agent, (mean * scale) @ basis.T):
                    episode = run_episode(agent, tokenizer, game,
                        source_field, **kwargs)
            else:
                episode = run_episode(agent, tokenizer, game,
                    source_field, **kwargs)
            row["arms"][arm] = episode
            if (episode["status"] != "complete" or
                    episode["initial_observation"] !=
                    matrix["base"][i]["episode"]["initial_observation"]):
                report["failures"].append({"index": i, "arm": arm,
                                           "episode": episode})
                break
        report["games"].append(row)
        save(args.output, report)
        print(json.dumps({"game": i + 1,
            "wins": {arm: sum(r["arms"].get(arm, {}).get("reward", 0)
                      for r in report["games"]) for arm in commands},
            "failures": len(report["failures"])}), flush=True)
        if report["failures"]:
            raise RuntimeError("Development episode failure recorded")
    report["summary"] = {arm: sum(row["arms"][arm]["reward"]
        for row in report["games"]) for arm in commands}
    save(args.output, report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "evaluate"))
    parser.add_argument("--model", type=Path, default=Path(
        "current_work/delta-Mem/model/Qwen3-4B-Instruct-2507"))
    parser.add_argument("--checkpoint", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_sibling_simple40_onpolicy600_20261006.pt"))
    parser.add_argument("--trained-model", type=Path, default=Path(
        "results/trajectory_hyperlora/my_online_update_v28_reward_train_20261009.pt"))
    parser.add_argument("--source-report-seq0", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_online_context_vector_reward_gate_train_seq0_6_20261007.json"))
    parser.add_argument("--source-report-seq6", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_online_context_vector_reward_gate_train_seq6_6_20261007.json"))
    parser.add_argument("--plan", type=Path, default=Path(
        "ttcl/results/experience_evolution/alfworld_delta_20260922/plan.json"))
    parser.add_argument("--data-root", type=Path, default=Path(
        "ttcl/data/alfworld_delta"))
    parser.add_argument("--target-policy", choices=("independent12",),
                        default="independent12")
    parser.add_argument("--checkpoint-training-candidates", type=Path,
        default=Path("results/trajectory_hyperlora/alf_sibling_train600_candidates_20261006.json"))
    parser.add_argument("--warmstart-candidates", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_next_task_candidates_20261005.json"))
    parser.add_argument("--prior-train-review", type=Path, default=Path(
        "data/annotations/alf_own_success_future_reward_8x18_reviewed_20261007.json"))
    parser.add_argument("--source-review", type=Path, default=Path(
        "data/annotations/alf_own_success_future_reward_holdout_8x12_v3c_reviewed_20261007.json"))
    parser.add_argument("--source-audit", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_own_success_future_reward_holdout_8x12_v3c_merged_audited_20261007.json"))
    parser.add_argument("--source-matrix", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_own_success_future_reward_holdout_8x12_v3c_merged_20261007.json"))
    parser.add_argument("--features", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_future_reward_holdout8x12_v3c_features_20261007.pt"))
    parser.add_argument("--review", type=Path, default=Path(
        "data/annotations/my_online_update_v28_dev12_20261009.json"))
    parser.add_argument("--output", type=Path, default=Path(
        "results/trajectory_hyperlora/my_online_update_v28_dev12_20261009.json"))
    parser.add_argument("--seed", type=int, default=311)
    parser.add_argument("--max-effective-norm", type=float, default=1.5)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--gpu-fraction", type=float, default=.65)
    args = parser.parse_args()
    (prepare if args.command == "prepare" else evaluate)(args)
