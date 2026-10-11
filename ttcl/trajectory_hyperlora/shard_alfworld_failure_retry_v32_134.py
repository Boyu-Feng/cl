"""Independent disjoint v32 retry shard; merge only after bound-row audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ttcl.trajectory_hyperlora import evaluate_alfworld_failure_retry_v32_134 as v32
from ttcl.trajectory_hyperlora.alfworld_online_from_empty import file_hash


def frozen_args(args):
    root = Path("results/trajectory_hyperlora")
    notes = Path("data/annotations")
    return argparse.Namespace(
        memrl_plan=Path("results/memrl_comparison/20260928_budgeted/plan.json"),
        source_candidates=root / "alf_sibling_train600_candidates_20261006.json",
        source_review=notes / "alf_sibling_train600_reviewed_20261006.json",
        retry_candidates=root / "alf_same_game_retry_candidates_20261006.json",
        retry_review=notes / "alf_same_game_retry_reviewed_20261006.json",
        all_source_review=notes / "alf_expert_first_attempts_reviewed_20261006.json",
        target_review=notes / "alf_memrl134_hyperlora_train_sources_reviewed_20261006.json",
        first_parts=[root / f"alf_parent_loop2_official_{start}_{count}_20261007.json"
                     for start, count in ((0, 34), (34, 34), (68, 33), (101, 33))],
        old_audit=root / "alf_parent_loop2_official_audited_20261007.json",
        dev_report=root / "alfworld_failure_retry_v31b_dev24_20261009.json",
        dev_audit=root / "alfworld_failure_retry_v31b_dev24_audit_20261009.json",
        placebo_report=root / "alfworld_failure_retry_placebo_v31c_dev24_20261009.json",
        placebo_audit=root / "alfworld_failure_retry_placebo_v31c_dev24_audit_20261009.json",
        checkpoint=root / "alf_sibling_simple40_onpolicy600_20261006.pt",
        model=Path("current_work/delta-Mem/model/Qwen3-4B-Instruct-2507"),
        data_root=Path("ttcl/data/alfworld_delta"),
        review=notes / "alfworld_failure_retry_v32_134_20261009.json",
        device=args.device, gpu_fraction=args.gpu_fraction)


def run(args):
    if args.output.exists() or not (0 <= args.start < args.end <= 134):
        raise ValueError("Changed or existing retry shard")
    frozen = frozen_args(args)
    targets, first, fixed, bindings = v32.checked_input(frozen)
    review = json.loads(frozen.review.read_text())
    if (review["targets"] != bindings or
            review["targets_sha256"] != v32.digest(bindings) or
            any(review[k] != value for k, value in fixed.items())):
        raise ValueError("Changed reviewed v32 target")
    agent, tokenizer = v32.load_agent(frozen.model, frozen.checkpoint,
                                      frozen.device, frozen.gpu_fraction)
    if (agent.encoder_kind != "contextual" or not agent.task_conditioned or
            agent.task_context_scope != "current" or
            agent.task_pair_pooling != "mean"):
        raise ValueError("Changed frozen old hypernetwork")
    result = {"start": args.start, "end": args.end,
        "review_sha256": file_hash(frozen.review),
        "original_runner_sha256": file_hash(Path(v32.__file__)),
        "shard_runner_sha256": file_hash(Path(__file__)),
        "checkpoint_sha256": fixed["checkpoint_sha256"],
        "rows": [], "failures": []}
    for target, first_row, binding in zip(targets[args.start:args.end],
            first[args.start:args.end], bindings[args.start:args.end], strict=True):
        old = first_row["lora"]
        row = {"index": binding["index"], "game": target["game"],
            "input_content_sha256": binding["input_content_sha256"],
            "first_episode_sha256": binding["first_episode_sha256"],
            "first_reward": old["reward"], "retries": {}}
        if not bool(old["reward"]):
            fields = v32.contextual_source_fields(agent, tokenizer,
                target["source_records"], frozen.device, 2048, pooling="both")
            expert = v32.source_factors(agent, tokenizer, fields,
                                        old["initial_observation"], frozen.device)
            failed, _, original, retained = v32.episode_factor(
                agent, tokenizer, old, argparse.Namespace(
                    device=frozen.device, context_tokens=2048))
            residual, info = v32.bounded_residual(agent, expert, failed,
                                                  max_relative=.25)
            seed = int(v32.digest(["matched_random_v32", target["game"]])[:16], 16)
            random_residual = v32.matched_random_residual(agent, expert,
                info["residual_effective_norm"], seed)
            row.update(expert_factor_sha256=v32.factors_hash(expert),
                failed_factor_sha256=v32.factors_hash(failed),
                residual_factor_sha256=v32.factors_hash(residual),
                random_residual_sha256=v32.factors_hash(random_residual),
                random_seed=seed, residual_info=info,
                failed_source_tokens_original=original,
                failed_source_tokens_retained=retained)
            adjustments = {"static_retry": None,
                "away_from_failure": [-x for x in residual],
                "matched_random": random_residual}
            for arm in v32.ARMS:
                episode = v32.retry(agent, tokenizer,
                    frozen.data_root / target["game"], fields, frozen.device,
                    adjustments[arm])
                row["retries"][arm] = episode
                if (episode["status"] != "complete" or
                        episode["initial_observation"] != old["initial_observation"]):
                    result["failures"].append({"index": binding["index"],
                        "arm": arm, "episode": episode})
                    break
        result["rows"].append(row)
        v32.save(args.output, result)
        print(json.dumps({"start": args.start, "end": args.end,
                          "completed": len(result["rows"])}), flush=True)
        if result["failures"]:
            raise RuntimeError("Retry shard episode failed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--gpu-fraction", type=float, default=.65)
    run(parser.parse_args())
