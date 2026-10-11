"""Resume the content-bound v32 report without replacing completed rows."""

from __future__ import annotations

import json

from ttcl.trajectory_hyperlora import evaluate_alfworld_failure_retry_v32_134 as v32
from ttcl.trajectory_hyperlora.alfworld_failure_retry_placebo_v31c import matched_random_residual
from ttcl.trajectory_hyperlora.alfworld_failure_retry_residual_v31b import bounded_residual
from ttcl.trajectory_hyperlora.alfworld_online_accumulate_lora_v1 import factors_hash
from ttcl.trajectory_hyperlora.alfworld_online_from_empty import digest, file_hash
from ttcl.trajectory_hyperlora.alfworld_success_gated_factors_v27 import episode_factor
from ttcl.trajectory_hyperlora.alfworld_zero_shot_probe import load_agent
from ttcl.trajectory_hyperlora.contextual_alf_source import contextual_source_fields
from ttcl.trajectory_hyperlora.evaluate_alfworld_failure_retry_v31b import retry, source_factors


def main():
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--gpu-fraction", type=float, default=.65)
    parser.add_argument("--output", type=Path, default=Path(
        "results/trajectory_hyperlora/alfworld_failure_retry_v32_134_20261009.json"))
    args = parser.parse_args()
    root = Path("results/trajectory_hyperlora")
    annotations = Path("data/annotations")
    frozen = argparse.Namespace(
        memrl_plan=Path("results/memrl_comparison/20260928_budgeted/plan.json"),
        source_candidates=root / "alf_sibling_train600_candidates_20261006.json",
        source_review=annotations / "alf_sibling_train600_reviewed_20261006.json",
        retry_candidates=root / "alf_same_game_retry_candidates_20261006.json",
        retry_review=annotations / "alf_same_game_retry_reviewed_20261006.json",
        all_source_review=annotations / "alf_expert_first_attempts_reviewed_20261006.json",
        target_review=annotations / "alf_memrl134_hyperlora_train_sources_reviewed_20261006.json",
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
        review=annotations / "alfworld_failure_retry_v32_134_20261009.json",
        output=args.output, device=args.device, gpu_fraction=args.gpu_fraction)
    targets, first, fixed, bindings = v32.checked_input(frozen)
    review = json.loads(frozen.review.read_text())
    report = json.loads(args.output.read_text())
    if (review["targets"] != bindings or
            review["targets_sha256"] != digest(bindings) or
            report["review_sha256"] != file_hash(frozen.review) or
            any(report[k] != value for k, value in fixed.items()) or
            tuple(report["arms"]) != v32.ARMS or report["failures"]):
        raise ValueError("Existing v32 report changed")
    completed = len(report["games"])
    for index, row in enumerate(report["games"]):
        if (row["index"] != index or row["game"] != bindings[index]["game"] or
                row["input_content_sha256"] != bindings[index]["input_content_sha256"] or
                row["first_episode_sha256"] != digest(first[index]["lora"]) or
                row["first_reward"] != first[index]["lora"]["reward"] or
                (not row["first_reward"] and tuple(row["retries"]) != v32.ARMS)):
            raise ValueError(f"Existing row {index} changed")
    if completed == 134:
        print(json.dumps(report["summary"]), flush=True)
        return
    agent, tokenizer = load_agent(frozen.model, frozen.checkpoint,
                                  frozen.device, frozen.gpu_fraction)
    for target, first_row, binding in zip(
            targets[completed:], first[completed:], bindings[completed:], strict=True):
        old = first_row["lora"]
        row = {"index": binding["index"], "game": target["game"],
            "input_content_sha256": binding["input_content_sha256"],
            "first_episode_sha256": binding["first_episode_sha256"],
            "first_reward": old["reward"], "retries": {}}
        if not bool(old["reward"]):
            fields = contextual_source_fields(agent, tokenizer,
                target["source_records"], frozen.device, 2048, pooling="both")
            expert = source_factors(agent, tokenizer, fields,
                                    old["initial_observation"], frozen.device)
            failed, _, original, retained = episode_factor(
                agent, tokenizer, old, argparse.Namespace(
                    device=frozen.device, context_tokens=2048))
            residual, info = bounded_residual(agent, expert, failed,
                                              max_relative=.25)
            seed = int(digest(["matched_random_v32", target["game"]])[:16], 16)
            random_residual = matched_random_residual(agent, expert,
                info["residual_effective_norm"], seed)
            row.update(expert_factor_sha256=factors_hash(expert),
                failed_factor_sha256=factors_hash(failed),
                residual_factor_sha256=factors_hash(residual),
                random_residual_sha256=factors_hash(random_residual),
                random_seed=seed, residual_info=info,
                failed_source_tokens_original=original,
                failed_source_tokens_retained=retained)
            adjustments = {"static_retry": None,
                "away_from_failure": [-value for value in residual],
                "matched_random": random_residual}
            for arm in v32.ARMS:
                episode = retry(agent, tokenizer,
                    frozen.data_root / target["game"], fields, frozen.device,
                    adjustments[arm])
                row["retries"][arm] = episode
                if (episode["status"] != "complete" or
                        episode["initial_observation"] != old["initial_observation"]):
                    report["failures"].append({"index": binding["index"],
                        "arm": arm, "episode": episode})
                    break
        report["games"].append(row)
        report["summary"] = {"n": len(report["games"]),
            "first_wins": sum(x["first_reward"] for x in report["games"]),
            "first_failures": sum(not x["first_reward"] for x in report["games"]),
            "two_attempt_wins": {arm: sum(x["first_reward"] or
                x["retries"].get(arm, {}).get("reward", 0)
                for x in report["games"]) for arm in v32.ARMS},
            "failures": len(report["failures"])}
        v32.save(args.output, report)
        print(json.dumps(report["summary"]), flush=True)
        if report["failures"]:
            raise RuntimeError("Official retry episode failure recorded")


if __name__ == "__main__":
    main()
