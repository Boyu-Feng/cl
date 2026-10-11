"""Static full LoRA exposed to the old hypernetwork's reviewed action lineage.

This control matches the four historical data stages and their optimizer-step
counts, while giving direct LoRA its own learning rates. It never reads a
validation trajectory or reward during training.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from types import SimpleNamespace

import torch

from ttcl.trajectory_hyperlora.alfworld_online_from_empty import digest, file_hash
from ttcl.trajectory_hyperlora.alfworld_persistent_failure_delta_v36 import save
from ttcl.trajectory_hyperlora.alfworld_bootstrap_own_success_v43 import new_agent
from ttcl.trajectory_hyperlora.prepare_alf_next_task import validate_review
from ttcl.trajectory_hyperlora.train_alf_next_task import query_ids, target_loss
from ttcl.trajectory_hyperlora.train_alfworld_static_lora_v41 import checked_labels


def inputs(args):
    warm_report = json.loads(args.warm_report.read_text())
    warm_manifest = json.loads(args.warm_candidates.read_text())
    warm_review = json.loads(args.warm_review.read_text())
    if (warm_report["steps"] != 240 or warm_report["seed"] != 42 or
            warm_report["candidate_manifest_sha256"] != file_hash(args.warm_candidates) or
            warm_report["reviewed_annotations_sha256"] != file_hash(args.warm_review)):
        raise ValueError("Changed historical warmstart")
    warm = [r for r in validate_review(warm_manifest, warm_review)
            if r["split"] == "train"]
    if len(warm) != warm_report["train_targets"] or len({r["target_game"] for r in warm}) != 31:
        raise ValueError("Changed warmstart target games")
    labels120 = json.loads(args.labels120.read_text())
    review120 = json.loads(args.review120.read_text())
    if (review120["labels_sha256"] != file_hash(args.labels120) or
            len(labels120["tasks"]) != 120 or
            len(review120["annotations"]) != 698):
        raise ValueError("Changed 120-task labels or annotations")
    reviewed120 = {(r["target_game"], r["input_content_sha256"]): r
                   for r in review120["annotations"]}
    if len(reviewed120) != 698:
        raise ValueError("Duplicate 120-task annotations")
    for task in labels120["tasks"]:
        if ("/train/" not in "/" + task["target_game"] or
                file_hash(args.data_root / task["target_game"]) != task["target_game_sha256"]):
            raise ValueError("Changed 120-task official train game")
        for row in task["labels"]:
            ann = reviewed120.get((row["target_game"], row["input_content_sha256"]))
            if (ann is None or ann.get("approved") is not True or
                    row["target_game"] != task["target_game"] or
                    row["source_episode_sha256"] != task["source_episode_sha256"]):
                raise ValueError("Changed 120-task reviewed action")
    labels33 = checked_labels(SimpleNamespace(labels=args.labels33,
        source_review=args.review33, source_training=args.stage600,
        data_root=args.data_root))
    lineage = []
    for path, steps, label_path, review_path in (
            (args.stage1800, 1800, args.labels120, args.review120),
            (args.stage800, 800, args.labels120, args.review120),
            (args.stage1000, 1000, args.labels120, args.review120),
            (args.stage600, 600, args.labels33, args.review33)):
        report = json.loads(path.read_text())
        if (report["steps"] != steps or report["labels_sha256"] != file_hash(label_path) or
                report["label_review_sha256"] != file_hash(review_path)):
            raise ValueError(f"Changed old hypernetwork stage {path}")
        lineage.append({"report_sha256": file_hash(path), "steps": steps,
                        "labels_sha256": file_hash(label_path),
                        "review_sha256": file_hash(review_path)})
    if sum(len(t["labels"]) for t in labels120["tasks"]) != 698 or sum(
            len(t["labels"]) for t in labels33["tasks"]) != 269:
        raise ValueError("Changed action counts")
    fixed = {"protocol": "Static full rank-4 A+B LoRA; same reviewed action sets and stage step counts as old hypernetwork; no valid reward used to fit",
             "warm_report_sha256": file_hash(args.warm_report),
             "warm_candidates_sha256": file_hash(args.warm_candidates),
             "warm_review_sha256": file_hash(args.warm_review),
             "lineage": lineage, "lineage_sha256": digest(lineage),
             "model_config_sha256": file_hash(args.model / "config.json"),
             "trainer_sha256": file_hash(Path(__file__)),
             "seed": args.seed, "rank": 4, "layers": 2,
             "module": "last_two_mlp.down_proj", "trainable_factors": "A+B",
             "target_counts": [len(warm), 698, 698, 698, 269],
             "step_counts": [240, 1800, 800, 1000, 600],
             "a_lrs": args.a_lrs, "b_lrs": args.b_lrs,
             "history_turns": 2, "max_prompt_tokens": 3000}
    return warm, labels120["tasks"], labels33["tasks"], fixed


def prepare(args):
    if args.review.exists():
        raise FileExistsError(args.review)
    warm, tasks120, tasks33, fixed = inputs(args)
    targets = []
    for stage, rows in (("warm", warm), ("train120", [r for t in tasks120 for r in t["labels"]]),
                        ("train33", [r for t in tasks33 for r in t["labels"]])):
        for row in rows:
            targets.append({"stage": stage, "target_game": row["target_game"],
                            "input_content_sha256": row["input_content_sha256"],
                            "target_action": row["target_action"],
                            "reviewed_target": True})
    save(args.review, {**fixed, "targets": targets, "targets_sha256": digest(targets)})
    print(json.dumps({"targets": len(targets), "steps": sum(fixed["step_counts"])}), flush=True)


def train(args):
    if args.model_output.exists() or args.report_output.exists():
        raise FileExistsError("Use fresh v51 outputs")
    warm, tasks120, tasks33, fixed = inputs(args)
    review = json.loads(args.review.read_text())
    if any(review.get(k) != v for k, v in fixed.items()) or review.get(
            "targets_sha256") != digest(review.get("targets")):
        raise ValueError("Changed v51 frozen review")
    if args.device.startswith("cuda"):
        torch.cuda.set_per_process_memory_fraction(args.gpu_fraction, device=args.device)
    agent, tokenizer = new_agent(args)
    agent.requires_grad_(False)
    a_params = [layer.a for layer in agent.adapters]
    for a in a_params:
        a.requires_grad_(True)
    b_params = [torch.nn.Parameter(torch.zeros(1, layer.base.out_features,
                 layer.rank, device=args.device)) for layer in agent.adapters]
    for layer, b in zip(agent.adapters, b_params, strict=True):
        layer.b = b
    pools = [warm, tasks120, tasks120, tasks120, tasks33]
    prefixes = {}
    for pool in pools:
        rows = pool if pool is warm else [r for task in pool for r in task["labels"]]
        for row in rows:
            key = row["input_content_sha256"]
            if key not in prefixes:
                prefixes[key] = query_ids(tokenizer, row, history_turns=2,
                                          max_prompt_tokens=3000)
    rng = random.Random(args.seed)
    stage_losses = []
    for stage_idx, (pool, count) in enumerate(zip(pools, fixed["step_counts"], strict=True)):
        optimizer = torch.optim.AdamW([
            {"params": a_params, "lr": args.a_lrs[stage_idx]},
            {"params": b_params, "lr": args.b_lrs[stage_idx]}], weight_decay=.01)
        losses = []
        order = list(range(len(warm)))
        for step in range(count):
            if stage_idx == 0:
                if step % len(order) == 0:
                    rng.shuffle(order)
                row = warm[order[step % len(order)]]
            else:
                task = rng.choice(pool)
                row = rng.choice(task["labels"])
            loss = target_loss(agent, tokenizer, row,
                               prefixes[row["input_content_sha256"]], args.device)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(a_params + b_params, 1., error_if_nonfinite=True)
            optimizer.step()
            value = float(loss.detach())
            if not torch.isfinite(loss) or not (value < float("inf")):
                raise ValueError("Nonfinite LoRA loss")
            losses.append(value)
            if (step + 1) % 50 == 0 or step + 1 == count:
                print(json.dumps({"stage": stage_idx, "step": step + 1,
                                  "last50_loss": sum(losses[-50:]) / len(losses[-50:])}), flush=True)
        stage_losses.append({"stage": stage_idx, "steps": count,
                             "first50_loss": sum(losses[:50]) / min(50, len(losses)),
                             "last50_loss": sum(losses[-50:]) / min(50, len(losses))})
        checkpoint = {"rank": 4, "layers": 2,
                      "A": [a.detach().cpu().float().clone() for a in a_params],
                      "factors": [b.detach().cpu().float().clone() for b in b_params],
                      "review_sha256": file_hash(args.review),
                      "trainer_sha256": fixed["trainer_sha256"],
                      "stage_completed": stage_idx,
                      "stage_losses": stage_losses}
        stage_path = args.model_output.with_name(args.model_output.stem +
                                                 f"_stage{stage_idx}" + args.model_output.suffix)
        stage_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, stage_path)
    torch.save(checkpoint, args.model_output)
    save(args.report_output, {**fixed, "review_sha256": file_hash(args.review),
        "model_sha256": file_hash(args.model_output), "stage_losses": stage_losses})


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=("prepare", "train"))
    paths = {
        "warm_candidates": "results/trajectory_hyperlora/alf_next_task_candidates_20261005.json",
        "warm_review": "data/annotations/trajectory_hyperlora_alf_20261005_reviewed_v2.json",
        "warm_report": "results/trajectory_hyperlora/alf_next_task_warmstart_20261005/seed42_v2.json",
        "labels120": "results/trajectory_hyperlora/alf_sibling_train120_labels_20261006.json",
        "review120": "data/annotations/alf_sibling_train120_labels_reviewed_20261006.json",
        "labels33": "results/trajectory_hyperlora/alf_sibling_simple_train40_onpolicy_best_labels_20261006.json",
        "review33": "data/annotations/alf_sibling_simple_train40_onpolicy_best_reviewed_20261006.json",
        "stage1800": "results/trajectory_hyperlora/alf_sibling_contextual120_1800_20261006.json",
        "stage800": "results/trajectory_hyperlora/alf_sibling_contextual120_match800_20261006.json",
        "stage1000": "results/trajectory_hyperlora/alf_sibling_taskpair_current1000_20261006.json",
        "stage600": "results/trajectory_hyperlora/alf_sibling_simple40_onpolicy600_20261006.json",
        "model": "current_work/delta-Mem/model/Qwen3-4B-Instruct-2507",
        "data_root": "ttcl/data/alfworld_delta",
        "review": "data/annotations/alfworld_full_lineage_static_v51_20261010.json",
        "model_output": "results/trajectory_hyperlora/alfworld_full_lineage_static_v51_20261010.pt",
        "report_output": "results/trajectory_hyperlora/alfworld_full_lineage_static_v51_20261010.json",
    }
    for k, value in paths.items():
        p.add_argument("--" + k.replace("_", "-"), type=Path, default=Path(value))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--gpu-fraction", type=float, default=.85)
    p.add_argument("--a-lrs", type=float, nargs=5,
                   default=[1e-4, 5e-5, 3e-5, 3e-5, 2e-5])
    p.add_argument("--b-lrs", type=float, nargs=5,
                   default=[1e-3, 5e-4, 3e-4, 3e-4, 2e-4])
    return p


if __name__ == "__main__":
    args = parser().parse_args()
    (prepare if args.command == "prepare" else train)(args)
