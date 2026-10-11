"""Strong direct LoRA control: train both A and B on identical own-win targets."""

from __future__ import annotations

import json
from pathlib import Path
import random

import torch

from ttcl.trajectory_hyperlora.alfworld_online_from_empty import file_hash
from ttcl.trajectory_hyperlora.alfworld_persistent_failure_delta_v36 import save
from ttcl.trajectory_hyperlora.train_alf_next_task import query_ids, target_loss
from ttcl.trajectory_hyperlora.alfworld_bootstrap_own_success_v43 import (
    checked, examples, new_agent, parser,
)


def train(args):
    if args.full_static_model.exists() or args.full_static_report.exists():
        raise FileExistsError("Use fresh v43c paths")
    _, successes, _, fixed, _, _ = checked(args)
    if args.device.startswith("cuda"):
        torch.cuda.set_per_process_memory_fraction(args.gpu_fraction, device=args.device)
    agent, tokenizer = new_agent(args)
    represented = {i for _, i, _ in examples(successes)}
    items = [(i, action) for i, row in enumerate(successes)
             if i in represented for action in row["actions"]]
    if len(items) != len(examples(successes)):
        raise ValueError("Changed matched action targets")
    prefixes = { (i, j): query_ids(tokenizer, action, history_turns=2,
                                   max_prompt_tokens=3000)
        for i, row in enumerate(successes) if i in represented
        for j, action in enumerate(row["actions"]) }
    # Preserve action order as a transparent enumeration of the same targets.
    indexed = [(i, j, action) for i, row in enumerate(successes)
               if i in represented for j, action in enumerate(row["actions"])]
    factors = torch.nn.ParameterList([
        torch.nn.Parameter(torch.zeros(1, layer.base.out_features, layer.rank,
                                       device=args.device))
        for layer in agent.adapters])
    a_params = [layer.a for layer in agent.adapters]
    optimizer = torch.optim.AdamW([
        {"params": a_params, "lr": args.a_lr},
        {"params": list(factors), "lr": args.static_lr}],
        weight_decay=.01)
    order = list(range(len(indexed)))
    rng = random.Random(args.seed)
    losses = []
    for step in range(args.steps):
        if step % len(order) == 0:
            rng.shuffle(order)
        i, j, action = indexed[order[step % len(order)]]
        for layer, factor in zip(agent.adapters, factors, strict=True):
            layer.b = factor
        loss = target_loss(agent, tokenizer, action,
                           prefixes[(i, j)], args.device)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(a_params + list(factors), 1.)
        optimizer.step()
        agent.set_source(None)
        losses.append(float(loss.detach()))
        if (step + 1) % 50 == 0:
            print(json.dumps({"arm": "full_static", "step": step + 1,
                "last50_loss": sum(losses[-50:]) / 50}), flush=True)
    model = {"rank": 4, "layers": 2,
        "A": [x.detach().cpu().float().clone() for x in a_params],
        "factors": [x.detach().cpu().float().clone() for x in factors],
        "review_sha256": file_hash(args.review),
        "trainer_sha256": file_hash(Path(__file__)),
        "bootstrap_trainer_sha256": fixed["runner_sha256"],
        "seed": args.seed, "steps": args.steps,
        "a_lr": args.a_lr, "b_lr": args.static_lr}
    args.full_static_model.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model, args.full_static_model)
    save(args.full_static_report, {"protocol": "Fresh full A+B rank4 LoRA from same random seed and same own-success action targets as v43b hypernetwork; official train prefix only",
        "bootstrap_review_sha256": file_hash(args.review),
        "bootstrap_trainer_sha256": fixed["runner_sha256"],
        "trainer_sha256": file_hash(Path(__file__)),
        "model_sha256": file_hash(args.full_static_model),
        "action_targets": len(items), "steps": args.steps,
        "seed": args.seed, "a_lr": args.a_lr, "b_lr": args.static_lr,
        "last50_loss": sum(losses[-50:]) / 50})


if __name__ == "__main__":
    p = parser()
    p.add_argument("--full-static-model", type=Path, default=Path(
        "results/trajectory_hyperlora/alfworld_bootstrap_own_success_v43c_full_static_20261010.pt"))
    p.add_argument("--full-static-report", type=Path, default=Path(
        "results/trajectory_hyperlora/alfworld_bootstrap_own_success_v43c_full_static_20261010.json"))
    p.add_argument("--a-lr", type=float, default=1e-4)
    args = p.parse_args()
    if args.command != "train":
        raise ValueError("v43c is a train-only stronger direct LoRA control")
    train(args)
