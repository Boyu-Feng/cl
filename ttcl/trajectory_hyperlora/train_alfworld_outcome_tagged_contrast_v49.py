"""Train an own-trajectory hypernetwork with explicit observed outcomes."""

from __future__ import annotations

import json
from pathlib import Path
import random

import torch
import torch.nn.functional as F

from ttcl.trajectory_hyperlora.alfworld_online_from_empty import digest, file_hash
from ttcl.trajectory_hyperlora.alfworld_persistent_failure_delta_v36 import save
from ttcl.trajectory_hyperlora.alfworld_zero_shot_probe import load_agent
from ttcl.trajectory_hyperlora.contextual_alf_source import (
    contextual_text_fields, source_text, task_context_text,
)
from ttcl.trajectory_hyperlora.alfworld_success_gated_factors_v27 import bounded_source_text
from ttcl.trajectory_hyperlora.train_alf_next_task import query_ids, target_loss
from ttcl.trajectory_hyperlora.alfworld_bootstrap_own_success_v43 import examples
from ttcl.trajectory_hyperlora.train_alfworld_success_failure_contrast_v44 import (
    checked as contrast_checked, parser,
)


def outcome_text(tokenizer, records, succeeded: bool, limit: int = 2048):
    tag = "\nEnvironment verified outcome: " + ("SUCCESS" if succeeded else "FAILURE")
    source, _, _ = bounded_source_text(tokenizer, records, limit - 24)
    result = source + tag
    if len(tokenizer(result, add_special_tokens=False).input_ids) > limit:
        raise ValueError("Tagged trajectory exceeds encoder budget")
    return result


def bound(args):
    successes, failures, suffix, parent_fixed, parent_targets = contrast_checked(args)
    parent_audit = json.loads(args.parent_audit.read_text())
    if (parent_audit.get("audit") != "passed" or
            parent_audit.get("review_sha256") != file_hash(args.contrast_review) or
            parent_audit.get("model_sha256") != file_hash(args.contrast_model)):
        raise ValueError("Changed parent contrast audit")
    fixed = {"method": "v49 outcome-tagged own-success/failure contrast",
        "parent_contrast_review_sha256": file_hash(args.contrast_review),
        "parent_contrast_audit_sha256": file_hash(args.parent_audit),
        "bootstrap_hyper_model_sha256": file_hash(args.hyper_model),
        "parent_targets_sha256": digest(parent_targets),
        "trainer_sha256": file_hash(Path(__file__)),
        "steps": args.tag_steps, "lr": args.tag_lr,
        "margin": args.margin, "contrast_weight": args.contrast_weight,
        "seed": args.seed, "source_token_limit": 2048}
    targets = []
    for i, parent in enumerate(parent_targets):
        content = {"index": i,
            "parent_input_content_sha256": parent["input_content_sha256"],
            "good_source_index": parent["good_source_index"],
            "bad_source_index": parent["bad_source_index"],
            "target_success_index": parent["target_success_index"],
            "trainer_sha256": fixed["trainer_sha256"]}
        targets.append({**content, "input_content_sha256": digest(content),
                        "reviewed_target": True})
    return successes, failures, suffix, fixed, targets


def prepare(args):
    if args.tag_review.exists():
        raise FileExistsError(args.tag_review)
    successes, failures, _, fixed, targets = bound(args)
    save(args.tag_review, {"protocol": "First120 official train tasks only: 31 own successes and 89 own failures, both outcomes environment confirmed. Add an explicit outcome tag to the source trajectory text and train 366 cross-task action contrasts. No suffix outcome used.",
        **fixed, "own_successes": len(successes),
        "own_failures": len(failures),
        "targets": targets, "targets_sha256": digest(targets)})
    print(json.dumps({"targets": len(targets)}), flush=True)


def checked(args):
    successes, failures, suffix, fixed, targets = bound(args)
    review = json.loads(args.tag_review.read_text())
    if (review.get("targets") != targets or
            review.get("targets_sha256") != digest(targets) or
            any(review.get(k) != v for k, v in fixed.items())):
        raise ValueError("Changed v49 reviewed inputs")
    return successes, failures, suffix, fixed, targets


def train(args):
    if args.tag_model.exists() or args.tag_report.exists():
        raise FileExistsError("Use fresh v49 output paths")
    successes, failures, _, fixed, targets = checked(args)
    agent, tokenizer = load_agent(args.model, args.hyper_model,
                                  args.device, args.gpu_fraction)
    agent.model.config.use_cache = False
    positive = [contextual_text_fields(agent, tokenizer,
        outcome_text(tokenizer, row["source_records"], True),
        args.device, 2048, pooling="both") for row in successes]
    negative = [contextual_text_fields(agent, tokenizer,
        outcome_text(tokenizer, row["source_records"], False),
        args.device, 2048, pooling="both") for row in failures]
    items = examples(successes)
    keys = {}
    for _, k, action in items:
        key = digest([k, action["target_messages"]])
        if key not in keys:
            initial = successes[k]["actions"][0]["current_observation"]
            target = contextual_text_fields(agent, tokenizer,
                task_context_text(initial, action["current_observation"]),
                args.device, 2048, pooling="both")
            prefix = query_ids(tokenizer, action, history_turns=2,
                               max_prompt_tokens=3000)
            keys[key] = (target, prefix)
    optimizer = torch.optim.AdamW((p for p in agent.parameters()
                                   if p.requires_grad),
                                  lr=args.tag_lr, weight_decay=.01)
    order = list(range(len(items)))
    rng = random.Random(args.seed)
    good_losses, bad_losses, contrasts = [], [], []
    for step in range(args.tag_steps):
        if step % len(order) == 0:
            rng.shuffle(order)
        idx = order[step % len(order)]
        j, k, action = items[idx]
        target, prefix = keys[digest([k, action["target_messages"]])]
        bad = targets[idx]["bad_source_index"]
        agent.set_source(positive[j], target_fields=target)
        good_loss = target_loss(agent, tokenizer, action, prefix, args.device)
        agent.set_source(negative[bad], target_fields=target)
        bad_loss = target_loss(agent, tokenizer, action, prefix, args.device)
        contrast = F.softplus((args.margin + good_loss - bad_loss) * 5.) / 5.
        total = good_loss + args.contrast_weight * contrast
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_((p for p in agent.parameters()
                                         if p.requires_grad), 1.)
        optimizer.step()
        agent.set_source(None)
        good_losses.append(float(good_loss.detach()))
        bad_losses.append(float(bad_loss.detach()))
        contrasts.append(float(contrast.detach()))
        if (step + 1) % 50 == 0:
            print(json.dumps({"step": step + 1,
                "good_loss": sum(good_losses[-50:]) / 50,
                "bad_loss": sum(bad_losses[-50:]) / 50,
                "contrast": sum(contrasts[-50:]) / 50}), flush=True)
    checkpoint = torch.load(args.hyper_model, map_location="cpu", weights_only=True)
    checkpoint["trainable_state"] = {name: p.detach().cpu().clone()
        for name, p in agent.named_parameters() if p.requires_grad}
    checkpoint.update(tag_review_sha256=file_hash(args.tag_review),
        tag_trainer_sha256=fixed["trainer_sha256"],
        parent_hyper_sha256=file_hash(args.hyper_model))
    args.tag_model.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.tag_model)
    save(args.tag_report, {"protocol": "Own-prefix outcome-tagged success/failure action contrast on fresh H",
        **fixed, "review_sha256": file_hash(args.tag_review),
        "model_sha256": file_hash(args.tag_model),
        "own_successes": 31, "own_failures": 89,
        "action_targets": len(targets),
        "last50_good_loss": sum(good_losses[-50:]) / 50,
        "last50_bad_loss": sum(bad_losses[-50:]) / 50,
        "last50_contrast": sum(contrasts[-50:]) / 50})


if __name__ == "__main__":
    p = parser()
    paths = {"parent_audit": "results/trajectory_hyperlora/alfworld_success_failure_contrast_v44_train_audit_20261010.json",
        "tag_review": "data/annotations/alfworld_outcome_tagged_contrast_v49_train120_20261010.json",
        "tag_model": "results/trajectory_hyperlora/alfworld_outcome_tagged_contrast_v49_20261010.pt",
        "tag_report": "results/trajectory_hyperlora/alfworld_outcome_tagged_contrast_v49_20261010.json"}
    for name, value in paths.items():
        p.add_argument("--" + name.replace("_", "-"), type=Path,
                       default=Path(value))
    p.add_argument("--tag-steps", type=int, default=300)
    p.add_argument("--tag-lr", type=float, default=1e-5)
    args = p.parse_args()
    (prepare if args.command == "prepare" else train)(args)
