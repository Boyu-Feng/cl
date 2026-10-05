"""Fine-tune official-history hyper-LoRA on paired opposing-action queries.

The target observation and prompt are identical within each pair. Only the
same-task source history differs. Training cannot solve this objective with a
single task-independent adapter. This is an exploratory dev diagnostic, not
an online-reward or official-return experiment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import torch
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from ttcl.trajectory_hyperlora.prepare_xland_conflict_pairs import digest
from ttcl.trajectory_hyperlora.train_xland_official_history_hyperlora import (
    QwenRawHyperLoRA, prompt_ids, source_tensor,
)
from ttcl.trajectory_hyperlora.train_xland_raw_hyperlora import checked_query


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(rows, tokenizer, device):
    prepared = []
    for row in rows:
        if not row["reviewed_target"] or row["pair_sha256"] != digest({
                key: row[key] for key in
                ("observation_sha256", "left", "right")}):
            raise ValueError("Unreviewed or changed opposing pair")
        left, y_left = checked_query(row["left"]["query"])
        right, y_right = checked_query(row["right"]["query"])
        if (y_left != row["left"]["label"] or
                y_right != row["right"]["label"] or
                y_left == y_right or
                row["left"]["ruleset_id"] == row["right"]["ruleset_id"] or
                left["target_initial_state"] != right["target_initial_state"] or
                digest(left["target_initial_state"]) != row[
                    "observation_sha256"] or
                prompt_ids(tokenizer, left) != prompt_ids(tokenizer, right)):
            raise ValueError("Pair no longer has identical query/opposing labels")
        prepared.append({"prompt": prompt_ids(tokenizer, left),
                         "sources": (source_tensor(left, device),
                                     source_tensor(right, device)),
                         "labels": (y_left, y_right),
                         "rulesets": (row["left"]["ruleset_id"],
                                      row["right"]["ruleset_id"])})
    return prepared


def pair_logits(agent, pair, choice_ids, device):
    factors = [agent.compile_adapters(source) for source in pair["sources"]]
    agent.mount([torch.cat([left, right], dim=0)
                 for left, right in zip(*factors, strict=True)])
    output = agent.choice_logits(pair["prompt"], choice_ids, 2, device)
    agent.mount(None)
    return output


def evaluate(agent, pairs, choice_ids, device):
    agent.eval()
    own = swapped = changes = both = 0
    with torch.no_grad():
        for pair in pairs:
            output = pair_logits(agent, pair, choice_ids, device)
            predicted = output.argmax(dim=-1).tolist()
            y0, y1 = pair["labels"]
            own += int(predicted[0] == y0) + int(predicted[1] == y1)
            swapped += int(predicted[0] == y1) + int(predicted[1] == y0)
            changes += int(predicted[0] != predicted[1])
            both += int(predicted == [y0, y1])
    return {"pairs": len(pairs), "own_correct": own,
            "swapped_correct": swapped, "pair_action_changes": changes,
            "both_correct": both}


def run(args):
    if args.output.exists() or args.checkpoint.exists():
        raise FileExistsError("Fresh output and checkpoint paths required")
    raw = args.pairs.read_bytes()
    reviewed = json.loads(raw)
    if reviewed["source_annotations_sha256"] != sha256(args.annotations):
        raise ValueError("Pair manifest source changed")
    saved = torch.load(args.init_checkpoint, map_location="cpu",
                       weights_only=True)
    initial = json.loads(args.init_result.read_text())
    if (saved["annotations_sha256"] != sha256(args.annotations) or
            initial["annotations_sha256"] != saved["annotations_sha256"] or
            initial["seed"] != saved["seed"]):
        raise ValueError("Initial checkpoint lineage mismatch")
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    torch.cuda.set_per_process_memory_fraction(args.gpu_fraction,
                                               device=args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    choices = [tokenizer(str(i), add_special_tokens=False).input_ids
               for i in range(5)]
    if any(len(ids) != 1 for ids in choices):
        raise ValueError("Actions must each use one token")
    choice_ids = [ids[0] for ids in choices]
    split = {name: prepare(reviewed["split"][name], tokenizer, args.device)
             for name in ("train", "dev")}
    base = AutoModelForCausalLM.from_pretrained(args.model,
        torch_dtype=torch.bfloat16, local_files_only=True,
        attn_implementation="sdpa").to(args.device)
    base.config.use_cache = False
    agent = QwenRawHyperLoRA(base, initial["rank"], initial["layers"],
                             initial.get("width", 64), False).to(args.device)
    params = dict(agent.named_parameters())
    with torch.no_grad():
        for name, value in saved["trainable_state"].items():
            if name not in params or params[name].shape != value.shape:
                raise ValueError("Initial architecture changed")
            params[name].copy_(value.to(params[name].device))
    optimizer = torch.optim.AdamW(
        [p for p in agent.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=0.)
    initial_dev = evaluate(agent, split["dev"], choice_ids, args.device)
    history = []
    best_score = (initial_dev["both_correct"],
                  initial_dev["own_correct"] - initial_dev["swapped_correct"])
    best_step = 0
    best_state = {name: p.detach().cpu().clone() for name, p in
                  agent.named_parameters() if p.requires_grad}
    for step in range(1, args.steps + 1):
        agent.train()
        pair = rng.choice(split["train"])
        output = pair_logits(agent, pair, choice_ids, args.device)
        labels = torch.tensor(pair["labels"], device=args.device)
        logp = F.log_softmax(output, dim=-1)
        own = logp.gather(1, labels[:, None]).squeeze(-1)
        other = logp[[1, 0], labels]
        loss = -own.mean() + args.margin_weight * F.softplus(
            args.margin - own + other).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in agent.parameters() if p.requires_grad], 1.)
        optimizer.step()
        if step % args.eval_every == 0 or step == args.steps:
            dev = evaluate(agent, split["dev"], choice_ids, args.device)
            history.append({"step": step, "loss": float(loss.detach()),
                            "dev": dev})
            print(json.dumps(history[-1]), flush=True)
            score = (dev["both_correct"],
                     dev["own_correct"] - dev["swapped_correct"])
            if score > best_score:
                best_score, best_step = score, step
                best_state = {name: p.detach().cpu().clone() for name, p in
                              agent.named_parameters() if p.requires_grad}
    final = {"protocol": "Exploratory official XLand same-observation opposing-action source contrast; initialized from frozen offline-history hyper-LoRA; no environment reward optimization; test split untouched",
             "pairs_sha256": hashlib.sha256(raw).hexdigest(),
             "annotations_sha256": sha256(args.annotations),
             "init_checkpoint_sha256": sha256(args.init_checkpoint),
             "model_config_sha256": sha256(args.model / "config.json"),
             "seed": args.seed, "steps": args.steps, "lr": args.lr,
             "margin": args.margin, "margin_weight": args.margin_weight,
             "initial_dev": initial_dev, "best_step": best_step,
             "best_dev": {"both_correct": best_score[0],
                          "correct_minus_swapped": best_score[1]},
             "history": history}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(final, indent=2) + "\n")
    torch.save({"trainable_state": best_state,
                "pairs_sha256": final["pairs_sha256"],
                "init_checkpoint_sha256": final["init_checkpoint_sha256"],
                "seed": args.seed, "best_step": best_step}, args.checkpoint)
    return final


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, default=Path(
        "data/annotations/xland_official_history_reviewed_64_v1_20261005.json"))
    parser.add_argument("--init-checkpoint", type=Path, required=True)
    parser.add_argument("--init-result", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path(
        "current_work/delta-Mem/model/Qwen3-4B-Instruct-2507"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpu-fraction", type=float, default=.6)
    parser.add_argument("--seed", type=int, default=20261006)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--margin", type=float, default=1.)
    parser.add_argument("--margin-weight", type=float, default=1.)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
