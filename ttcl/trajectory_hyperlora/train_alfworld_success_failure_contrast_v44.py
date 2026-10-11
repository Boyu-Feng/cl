"""Use own successful and failed prefix trajectories to refine fresh H.

On a different successful target's reviewed action, require the successful
source to give lower action loss than a same-family failed source. Both source
episodes precede the evaluation suffix. This is a train-stream mechanism test.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

import torch
import torch.nn.functional as F

from ttcl.trajectory_hyperlora.alfworld_online_from_empty import (
    digest, file_hash, records_from_episode,
)
from ttcl.trajectory_hyperlora.alfworld_persistent_failure_delta_v36 import save
from ttcl.trajectory_hyperlora.alfworld_zero_shot_probe import load_agent
from ttcl.trajectory_hyperlora.contextual_alf_source import (
    contextual_source_fields, contextual_text_fields, task_context_text,
)
from ttcl.trajectory_hyperlora.alfworld_success_gated_factors_v27 import (
    bounded_source_text,
)
from ttcl.trajectory_hyperlora.train_alf_next_task import query_ids, target_loss
from ttcl.trajectory_hyperlora.audit_my_online_update_v28_reward import replay
from ttcl.trajectory_hyperlora.alfworld_bootstrap_own_success_v43 import (
    checked as bootstrap_checked, examples, parser as bootstrap_parser,
)


def inputs(args):
    rows, successes, suffix, fixed, _, _ = bootstrap_checked(args)
    train = json.loads(args.train_report.read_text())
    if (train.get("review_sha256") != file_hash(args.review) or
            train.get("hyper_model_sha256") != file_hash(args.hyper_model) or
            train.get("hyper_cross_task_actions") != 366 or
            any(train.get(k) != v for k, v in fixed.items())):
        raise ValueError("Changed audited success bootstrap")
    failures = []
    for i, row in enumerate(rows[:120]):
        ep = row["arms"]["base"]
        if ep["reward"]:
            continue
        replay(args.data_root / row["game"], ep)
        failures.append({"stream_index": i, "game": row["game"],
            "family": row["family"],
            "game_sha256": file_hash(args.data_root / row["game"]),
            "episode_sha256": digest(ep),
            "source_records": records_from_episode(ep)})
    if len(failures) != 89:
        raise ValueError("Changed base-own-failure prefix")
    return successes, failures, suffix


def bad_source_for(target, failures):
    candidates = [(i, row) for i, row in enumerate(failures)
                  if row["family"] == target["family"]]
    if not candidates:
        raise ValueError("No same-family failed source")
    return min(candidates, key=lambda pair: digest([
        "v44_failure_source", target["game"], pair[1]["game"]]))[0]


def bindings(args, successes, failures):
    fixed = {"method": "v44 own-success vs own-failure trajectory contrast",
        "bootstrap_review_sha256": file_hash(args.review),
        "bootstrap_train_report_sha256": file_hash(args.train_report),
        "bootstrap_hyper_model_sha256": file_hash(args.hyper_model),
        "source_report_sha256": file_hash(args.source_report),
        "trainer_sha256": file_hash(Path(__file__)),
        "model_config_sha256": file_hash(args.model / "config.json"),
        "steps": args.contrast_steps, "lr": args.contrast_lr,
        "margin": args.margin, "contrast_weight": args.contrast_weight,
        "seed": args.seed, "context_tokens": 2048}
    failure_bindings = []
    for i, item in enumerate(failures):
        content = {"index": i, "stream_index": item["stream_index"],
            "game": item["game"], "family": item["family"],
            "game_sha256": item["game_sha256"],
            "episode_sha256": item["episode_sha256"],
            "source_records_sha256": digest(item["source_records"]),
            "trainer_sha256": fixed["trainer_sha256"]}
        failure_bindings.append({**content,
            "input_content_sha256": digest(content), "reviewed_target": True})
    targets = []
    for i, (j, k, action) in enumerate(examples(successes)):
        bad = bad_source_for(successes[k], failures)
        content = {"index": i, "good_source_index": j,
            "target_success_index": k,
            "bad_source_index": bad,
            "target_game": successes[k]["game"],
            "target_action": action["target_action"],
            "target_messages_sha256": digest(action["target_messages"]),
            "good_records_sha256": digest(successes[j]["source_records"]),
            "bad_records_sha256": digest(failures[bad]["source_records"]),
            "trainer_sha256": fixed["trainer_sha256"]}
        targets.append({**content, "input_content_sha256": digest(content),
                        "reviewed_target": True})
    return fixed, failure_bindings, targets


def prepare(args):
    if args.contrast_review.exists():
        raise FileExistsError(args.contrast_review)
    successes, failures, _ = inputs(args)
    fixed, sources, targets = bindings(args, successes, failures)
    save(args.contrast_review, {"protocol": "Only first120 official train base episodes: 31 rechecked own successes and 89 rechecked own failures. For 366 successful cross-task action targets, bind same-family failed source as negative. No suffix task content or outcome used in contrast training.",
        **fixed, "failure_sources": sources,
        "failure_sources_sha256": digest(sources),
        "targets": targets, "targets_sha256": digest(targets)})
    print(json.dumps({"own_successes": len(successes),
        "own_failures": len(failures), "contrast_actions": len(targets)}),
        flush=True)


def checked(args):
    successes, failures, suffix = inputs(args)
    fixed, sources, targets = bindings(args, successes, failures)
    review = json.loads(args.contrast_review.read_text())
    if (review.get("failure_sources") != sources or
            review.get("failure_sources_sha256") != digest(sources) or
            review.get("targets") != targets or
            review.get("targets_sha256") != digest(targets) or
            any(review.get(k) != v for k, v in fixed.items())):
        raise ValueError("Changed reviewed contrast input")
    return successes, failures, suffix, fixed, targets


def train(args):
    if args.contrast_model.exists() or args.contrast_report.exists():
        raise FileExistsError("Use fresh contrast outputs")
    successes, failures, _, fixed, targets = checked(args)
    agent, tokenizer = load_agent(args.model, args.hyper_model,
                                  args.device, args.gpu_fraction)
    agent.model.config.use_cache = False
    positive_fields = [contextual_source_fields(agent, tokenizer,
        x["source_records"], args.device, 2048, pooling="both")
        for x in successes]
    negative_fields = []
    for row in failures:
        text, _, _ = bounded_source_text(tokenizer,
                                         row["source_records"], 2048)
        negative_fields.append(contextual_text_fields(agent, tokenizer,
            text, args.device, 2048, pooling="both"))
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
                                  lr=args.contrast_lr, weight_decay=.01)
    order = list(range(len(items)))
    rng = random.Random(args.seed)
    good_losses, bad_losses, contrasts = [], [], []
    for step in range(args.contrast_steps):
        if step % len(order) == 0:
            rng.shuffle(order)
        idx = order[step % len(order)]
        j, k, action = items[idx]
        target, prefix = keys[digest([k, action["target_messages"]])]
        bad = targets[idx]["bad_source_index"]
        agent.set_source(positive_fields[j], target_fields=target)
        good_loss = target_loss(agent, tokenizer, action, prefix, args.device)
        agent.set_source(negative_fields[bad], target_fields=target)
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
    checkpoint.update(contrast_review_sha256=file_hash(args.contrast_review),
        contrast_trainer_sha256=fixed["trainer_sha256"],
        parent_hyper_sha256=file_hash(args.hyper_model))
    args.contrast_model.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.contrast_model)
    save(args.contrast_report, {"protocol": "Additional train-prefix success-versus-failure action contrast on fresh hypernetwork; no evaluation suffix used",
        **fixed, "review_sha256": file_hash(args.contrast_review),
        "model_sha256": file_hash(args.contrast_model),
        "own_successes": 31, "own_failures": 89,
        "action_targets": len(targets),
        "last50_good_loss": sum(good_losses[-50:]) / 50,
        "last50_bad_loss": sum(bad_losses[-50:]) / 50,
        "last50_contrast": sum(contrasts[-50:]) / 50})


def parser():
    p = bootstrap_parser()
    defaults = {"contrast_review": "data/annotations/alfworld_success_failure_contrast_v44_train120_20261010.json",
        "contrast_model": "results/trajectory_hyperlora/alfworld_success_failure_contrast_v44_20261010.pt",
        "contrast_report": "results/trajectory_hyperlora/alfworld_success_failure_contrast_v44_20261010.json"}
    for name, value in defaults.items():
        p.add_argument("--" + name.replace("_", "-"), type=Path,
                       default=Path(value))
    p.add_argument("--contrast-steps", type=int, default=300)
    p.add_argument("--contrast-lr", type=float, default=1e-5)
    p.add_argument("--margin", type=float, default=.1)
    p.add_argument("--contrast-weight", type=float, default=1.)
    return p


if __name__ == "__main__":
    args = parser().parse_args()
    (prepare if args.command == "prepare" else train)(args)
