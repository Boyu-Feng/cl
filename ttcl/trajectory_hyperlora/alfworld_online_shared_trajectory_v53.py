"""Chronological own-trajectory replay: H-only updates versus online LoRA.

The teacher stream is v52's live, environment-replayed own wins. Every arm
receives those same successes only after the corresponding task. The first
120 train tasks are a causal warm-up; tasks 120..239 are prequentially scored.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from types import SimpleNamespace

import torch
from transformers import AutoTokenizer

from ttcl.trajectory_hyperlora.alfworld_online_from_empty import (
    digest, file_hash, records_from_episode,
)
from ttcl.trajectory_hyperlora.alfworld_persistent_failure_delta_v36 import save
from ttcl.trajectory_hyperlora.alfworld_bootstrap_own_success_v43 import (
    action_rows, examples, new_agent,
)
from ttcl.trajectory_hyperlora.alfworld_online_own_trajectory_v52b import (
    bound as v52_bound, bounded_source_fields, fit_examples,
    parser as v52_parser, trainable_digest,
)
from ttcl.trajectory_hyperlora.evaluate_alfworld_bootstrap_own_success_v43 import (
    rotated_source_for, source_for,
)
from ttcl.trajectory_hyperlora.alfworld_zero_shot_probe import run_episode
from ttcl.trajectory_hyperlora.contextual_alf_source import source_text, task_context_text
from ttcl.trajectory_hyperlora.train_alf_next_task import query_ids, target_loss
from ttcl.trajectory_hyperlora.audit_my_online_update_v28_reward import replay


def shifted_actions(records):
    """Break action/state alignment while preserving every action string."""
    if len(records) < 2:
        raise ValueError("Action alignment control needs at least two records")
    actions = [row["action"] for row in records]
    return [{**row, "action": actions[(i + 1) % len(actions)]}
            for i, row in enumerate(records)]


def a_digest(agent):
    h = hashlib.sha256()
    for layer in agent.adapters:
        h.update(layer.a.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def alignment_pair(records, tokenizer):
    """Fit both controls using identical retained observations and actions."""
    chosen = [dict(row) for row in records]
    while len(chosen) >= 2:
        shifted = shifted_actions(chosen)
        lengths = [len(tokenizer(source_text(x), add_special_tokens=False).input_ids)
                   for x in (chosen, shifted)]
        if max(lengths) <= 2048:
            return chosen, shifted
        chosen.pop(1)
    raise ValueError("Cannot fit aligned and shifted sources into the source budget")


def inputs(args):
    v52_args = v52_parser().parse_args(["evaluate"])
    rows, fixed, targets = v52_bound(v52_args)
    report = json.loads(args.teacher_report.read_text())
    notes = json.loads(args.teacher_annotations.read_text())
    if (len(report["rows"]) != len(rows) or len(rows) != 240 or
            report["failures"] or report["review_sha256"] != file_hash(v52_args.review) or
            any(report.get(k) != value for k, value in fixed.items()) or
            not args.teacher_model.exists() or
            notes["eval_review_sha256"] != file_hash(v52_args.review)):
        raise ValueError("v52 teacher must be a completed and unchanged 240-task run")
    wins = {}
    reviewed_notes = iter(notes["annotations"])
    for i, (row, target, result) in enumerate(zip(rows, targets, report["rows"], strict=True)):
        if (result["index"] != i or result["game"] != target["game"] or
                result["target_input_content_sha256"] != target["input_content_sha256"] or
                result["online"]["status"] != "complete" or
                result["frozen_same_sources"]["status"] != "complete"):
            raise ValueError("Changed teacher stream row")
        if result["online"]["reward"] != 1:
            if result["annotation_input_content_sha256"] is not None:
                raise ValueError("Failure used as positive target")
            continue
        note = next(reviewed_notes)
        episode = result["online"]
        game = args.data_root / target["game"]
        replay(game, episode)
        actions = action_rows(game, episode)
        records = records_from_episode(episode)
        content = {"stream_index": i, "game": target["game"],
                   "family": target["family"], "game_sha256": target["game_sha256"],
                   "episode_sha256": digest(episode),
                   "source_records_sha256": digest(records),
                   "actions_sha256": digest(actions),
                   "runner_sha256": fixed["runner_sha256"]}
        if (any(note.get(k) != v for k, v in content.items()) or
                note.get("input_content_sha256") != digest(content) or
                note.get("reviewed_target") is not True or
                result["annotation_input_content_sha256"] != note["input_content_sha256"]):
            raise ValueError("Changed teacher success annotation")
        wins[i] = {"stream_index": i, "game": target["game"],
                   "family": target["family"], "episode_sha256": digest(episode),
                   "game_sha256": target["game_sha256"],
                   "source_records": records, "actions": actions,
                   "annotation_input_content_sha256": note["input_content_sha256"]}
    if next(reviewed_notes, None) is not None or len(wins) != report["summary"]["own_successes"]:
        raise ValueError("Changed teacher win count")
    return rows, targets, report, wins


def fixed(args):
    return {"method": "v53 chronological H-only online versus direct online LoRA",
        "teacher_report_sha256": file_hash(args.teacher_report),
        "teacher_annotations_sha256": file_hash(args.teacher_annotations),
        "teacher_model_sha256": file_hash(args.teacher_model),
        "model_config_sha256": file_hash(args.model / "config.json"),
        "runner_sha256": file_hash(Path(__file__)),
        "actor_sha256": file_hash(Path(__file__).with_name("alfworld_zero_shot_probe.py")),
        "seed": args.seed, "bootstrap_steps": 300, "online_steps": 20,
        "hyper_bootstrap_lr": 1e-4, "static_bootstrap_lr": 1e-4,
        "hyper_online_lr": 1e-5, "static_online_lr": 1e-5,
        "rank": 4, "layers": 2, "score_start": 120,
        "max_steps": 30, "max_new_tokens": 64,
        "actor_history_turns": 2, "loop_guard_max": 2,
        "constrain_actions": True,
        "teacher_success_only": True,
        "same_family_min_count": 2,
        "hyper_A_frozen_after_bootstrap": True,
        "action_alignment_control": "circular shift actions by one record; preserve observations and feedback",
        "source_helper_sha256": file_hash(Path(__file__).with_name("alfworld_online_own_trajectory_v52b.py"))}


def budget_preflight(args, wins):
    """Check every teacher action and content-control input before GPU work."""
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    max_prompt = max_target = max_source = 0
    actions = 0
    for item in wins.values():
        coherent, shifted = alignment_pair(item["source_records"], tokenizer)
        max_source = max(max_source, *(len(tokenizer(source_text(records),
            add_special_tokens=False).input_ids) for records in (coherent, shifted)))
        initial = item["actions"][0]["current_observation"]
        for action in item["actions"]:
            max_prompt = max(max_prompt, len(query_ids(tokenizer, action,
                history_turns=2, max_prompt_tokens=3000)))
            n = len(tokenizer(task_context_text(initial, action["current_observation"]),
                              add_special_tokens=False).input_ids)
            if n > 2048:
                raise ValueError(f"Teacher target context exceeds budget: {n}")
            max_target = max(max_target, n)
            actions += 1
    return {"successes_checked": len(wins), "actions_checked": actions,
            "max_action_prompt_tokens": max_prompt,
            "max_target_context_tokens": max_target,
            "max_paired_source_tokens": max_source}


def prepare(args):
    if args.review.exists():
        raise FileExistsError(args.review)
    rows, targets, report, wins = inputs(args)
    config = fixed(args)
    budgets = budget_preflight(args, wins)
    reviewed_targets = []
    for i, (row, target) in enumerate(zip(rows, targets, strict=True)):
        content = {"index": i, "game": target["game"],
                   "game_sha256": target["game_sha256"],
                   "v52_target_input_content_sha256": target["input_content_sha256"],
                   "teacher_episode_sha256": digest(report["rows"][i]["online"]),
                   "own_win_annotation_input_content_sha256":
                       wins[i]["annotation_input_content_sha256"] if i in wins else None,
                   "runner_sha256": config["runner_sha256"]}
        reviewed_targets.append({**content, "input_content_sha256": digest(content),
                                 "reviewed_target": True})
    save(args.review, {"protocol": "Previously exposed official train240 tasks; v52 own wins re-reviewed by game content and replay; all arms receive the same teacher wins only after each task; no teacher target labels from future tasks; score suffix120 before current update",
        **config, "budget_preflight": budgets, "targets": reviewed_targets,
        "targets_sha256": digest(reviewed_targets)})
    print(json.dumps({"stream_tasks": len(rows), "teacher_wins": len(wins),
                      "scored_suffix": len(rows)-120}), flush=True)


def checked(args):
    rows, targets, report, wins = inputs(args)
    config = fixed(args)
    review = json.loads(args.review.read_text())
    if any(review.get(k) != v for k, v in config.items()):
        raise ValueError("Changed v53 protocol")
    expected = []
    for i, target in enumerate(targets):
        content = {"index": i, "game": target["game"],
                   "game_sha256": target["game_sha256"],
                   "v52_target_input_content_sha256": target["input_content_sha256"],
                   "teacher_episode_sha256": digest(report["rows"][i]["online"]),
                   "own_win_annotation_input_content_sha256":
                       wins[i]["annotation_input_content_sha256"] if i in wins else None,
                   "runner_sha256": config["runner_sha256"]}
        expected.append({**content, "input_content_sha256": digest(content),
                         "reviewed_target": True})
    if review.get("targets") != expected or review.get("targets_sha256") != digest(expected):
        raise ValueError("Changed v53 input-content bindings")
    if review.get("budget_preflight") != budget_preflight(args, wins):
        raise ValueError("Changed v53 input budget validation")
    return rows, report, wins, expected, config


def static_train(agent, tokenizer, factors, optimizer, items, steps, rng, device):
    if not items:
        return {"steps": 0, "loss_mean": None}
    agent.train()
    prefixes = [query_ids(tokenizer, action, history_turns=2,
                          max_prompt_tokens=3000) for action in items]
    order = list(range(len(items)))
    losses = []
    for step in range(steps):
        if step % len(order) == 0:
            rng.shuffle(order)
        k = order[step % len(order)]
        for layer, factor in zip(agent.adapters, factors, strict=True):
            layer.b = factor
        loss = target_loss(agent, tokenizer, items[k], prefixes[k], device)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        params = [p for p in agent.parameters() if p.requires_grad] + list(factors)
        torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True)
        optimizer.step()
        agent.set_source(None)
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite static LoRA loss")
        losses.append(float(loss.detach()))
    agent.eval()
    return {"steps": steps, "loss_mean": sum(losses) / len(losses)}


def evaluate(args):
    if any(p.exists() for p in (args.output, args.hyper_model, args.static_model)):
        raise FileExistsError("Use fresh v53 outputs")
    rows, teacher, wins, bindings, config = checked(args)
    for device in (args.hyper_device, args.static_device, args.frozen_device):
        torch.cuda.set_per_process_memory_fraction(.8, device=device)
    def make(device):
        return new_agent(SimpleNamespace(seed=args.seed, model=args.model, device=device))
    hyper, hyper_tok = make(args.hyper_device)
    static, static_tok = make(args.static_device)
    hyper.eval()
    static.eval()
    initial_a_equal = all(torch.equal(a.a.detach().cpu(), b.a.detach().cpu())
                          for a, b in zip(hyper.adapters, static.adapters, strict=True))
    if not initial_a_equal:
        raise ValueError("H and static A initialization differs")
    factors = torch.nn.ParameterList([torch.nn.Parameter(torch.zeros(
        1, layer.base.out_features, layer.rank, device=args.static_device))
        for layer in static.adapters])
    h_boot_opt = torch.optim.AdamW((p for p in hyper.parameters() if p.requires_grad),
        lr=1e-4, weight_decay=.01)
    static_boot_opt = torch.optim.AdamW([
        {"params": [layer.a for layer in static.adapters], "lr": 1e-4},
        {"params": list(factors), "lr": 1e-4}], weight_decay=.01)
    h_rng = random.Random(args.seed)
    s_rng = random.Random(args.seed)
    memory = []
    frozen = None
    frozen_tok = None
    h_opt = None
    s_opt = None
    result = {"protocol": "v53 chronological shared own-win stream; each scored target precedes its own teacher feedback; H online updates have frozen shared A; source substitution tests content dependence",
              "review_sha256": file_hash(args.review), **config,
              "initial_A_equal": initial_a_equal,
              "bootstrap": None, "rows": [], "failures": []}
    for i, (old, binding) in enumerate(zip(rows, bindings, strict=True)):
        target = {"game": binding["game"], "family": old["family"]}
        game = args.data_root / binding["game"]
        before = len(memory)
        same = sum(x["family"] == target["family"] for x in memory)
        active = frozen is not None and same >= 2
        j = source_for(target, memory) if same else None
        entry = {"index": i, "game": binding["game"],
            "target_input_content_sha256": binding["input_content_sha256"],
            "memory_before": before,
            "source_index": j,
            "source_episode_sha256": memory[j]["episode_sha256"] if j is not None else None,
            "adapter_active": active, "teacher_reward": teacher["rows"][i]["online"]["reward"],
            "teacher_episode_sha256": binding["teacher_episode_sha256"],
            "hyper_update": {"steps": 0, "loss_mean": None},
            "static_update": {"steps": 0, "loss_mean": None}}
        entry["hyper_A_before_sha256"] = a_digest(hyper)
        if i >= 120:
            if active:
                coherent_records, corrupted_records = alignment_pair(
                    memory[j]["source_records"], hyper_tok)
                entry["coherent_source_records_sha256"] = digest(coherent_records)
                h_fields = bounded_source_fields(hyper, hyper_tok,
                    coherent_records, args.hyper_device)
                h_ep = run_episode(hyper, hyper_tok, game, h_fields,
                    adapter=True, device=args.hyper_device, max_steps=30,
                    max_new_tokens=64, constrain_actions=True,
                    actor_history_turns=2, loop_guard_max=2)
                f_fields = bounded_source_fields(frozen, frozen_tok,
                    coherent_records, args.frozen_device)
                f_ep = run_episode(frozen, frozen_tok, game, f_fields,
                    adapter=True, device=args.frozen_device, max_steps=30,
                    max_new_tokens=64, constrain_actions=True,
                    actor_history_turns=2, loop_guard_max=2)
                k = rotated_source_for(target, memory, j)
                if memory[k]["family"] != target["family"]:
                    raise ValueError("Rotated source left family")
                rotated_fields = bounded_source_fields(hyper, hyper_tok,
                    memory[k]["source_records"], args.hyper_device)
                rotated_ep = run_episode(hyper, hyper_tok, game, rotated_fields,
                    adapter=True, device=args.hyper_device, max_steps=30,
                    max_new_tokens=64, constrain_actions=True,
                    actor_history_turns=2, loop_guard_max=2)
                entry["rotated_source_index"] = k
                entry["rotated_source_episode_sha256"] = memory[k]["episode_sha256"]
                entry["shifted_source_records_sha256"] = digest(corrupted_records)
                shifted_fields = bounded_source_fields(hyper, hyper_tok,
                    corrupted_records, args.hyper_device)
                shifted_ep = run_episode(hyper, hyper_tok, game, shifted_fields,
                    adapter=True, device=args.hyper_device, max_steps=30,
                    max_new_tokens=64, constrain_actions=True,
                    actor_history_turns=2, loop_guard_max=2)
            else:
                h_ep = run_episode(hyper, hyper_tok, game, {}, adapter=False,
                    device=args.hyper_device, max_steps=30,
                    max_new_tokens=64, constrain_actions=True,
                    actor_history_turns=2, loop_guard_max=2)
                f_ep = h_ep
                rotated_ep = None
                shifted_ep = None
            s_ep = run_episode(static, static_tok, game, {}, adapter=False,
                fixed_adapter=list(factors), device=args.static_device,
                max_steps=30, max_new_tokens=64, constrain_actions=True,
                actor_history_turns=2, loop_guard_max=2)
            arms = {"hyper_online": h_ep, "hyper_frozen": f_ep,
                    "hyper_rotated": rotated_ep, "hyper_shifted_actions": shifted_ep,
                    "online_static_lora": s_ep}
            if any(ep is not None and (ep["status"] != "complete" or
                    ep["initial_observation"] != old["arms"]["base"]["initial_observation"])
                    for ep in arms.values()):
                result["failures"].append({"index": i, "status": {
                    name: ep["status"] if ep is not None else None
                    for name, ep in arms.items()}})
                save(args.output, result)
                raise RuntimeError(f"Incomplete v53 task {i}")
            entry.update(arms)
        if i in wins:
            item = wins[i]
            if item["stream_index"] != i:
                raise ValueError("Future teacher trajectory")
            if frozen is None:
                memory.append(item)
                if len(memory) == 8:
                    pairs = examples(memory)
                    h_before = trainable_digest(hyper)
                    h_fit = fit_examples(hyper, hyper_tok, h_boot_opt,
                        memory, pairs, 300, h_rng, args.hyper_device)
                    s_fit = static_train(static, static_tok, factors, s_boot_opt,
                        [action for _, _, action in pairs], 300, s_rng,
                        args.static_device)
                    result["bootstrap"] = {"last_index": i,
                        "own_wins": len(memory), "paired_action_targets": len(pairs),
                        "hyper_before_sha256": h_before,
                        "hyper_after_sha256": trainable_digest(hyper),
                        "hyper": h_fit, "static": s_fit}
                    result["bootstrap"]["hyper_A_sha256"] = a_digest(hyper)
                    frozen, frozen_tok = make(args.frozen_device)
                    current = {name: p.detach().cpu() for name, p in hyper.named_parameters()
                               if p.requires_grad}
                    with torch.no_grad():
                        for name, p in frozen.named_parameters():
                            if p.requires_grad:
                                p.copy_(current[name].to(p.device, p.dtype))
                    frozen.eval()
                    for layer in hyper.adapters:
                        layer.a.requires_grad_(False)
                    h_opt = torch.optim.AdamW(
                        (p for p in hyper.parameters() if p.requires_grad),
                        lr=1e-5, weight_decay=.01)
                    s_opt = torch.optim.AdamW([
                        {"params": [layer.a for layer in static.adapters], "lr": 1e-5},
                        {"params": list(factors), "lr": 1e-5}], weight_decay=.01)
            else:
                if j is not None:
                    pool = memory + [item]
                    h_before = trainable_digest(hyper)
                    entry["hyper_update"] = fit_examples(hyper, hyper_tok, h_opt,
                        pool, [(j, len(memory), action) for action in item["actions"]],
                        20, h_rng, args.hyper_device)
                    entry["static_update"] = static_train(static, static_tok,
                        factors, s_opt, item["actions"], 20, s_rng,
                        args.static_device)
                    entry["hyper_before_update_sha256"] = h_before
                    entry["hyper_after_update_sha256"] = trainable_digest(hyper)
                memory.append(item)
        entry["hyper_A_after_sha256"] = a_digest(hyper)
        if result["bootstrap"] is not None and i > result["bootstrap"]["last_index"]:
            if entry["hyper_A_before_sha256"] != entry["hyper_A_after_sha256"] or entry["hyper_A_after_sha256"] != result["bootstrap"]["hyper_A_sha256"]:
                raise ValueError("Frozen H A changed during online updates")
        result["rows"].append(entry)
        scored = [x for x in result["rows"] if x["index"] >= 120]
        result["summary"] = {"processed": len(result["rows"]),
            "scored": len(scored),
            "own_success_memory": len(memory),
            "hyper_updates": sum(x["hyper_update"]["steps"] for x in result["rows"]),
            "static_updates": sum(x["static_update"]["steps"] for x in result["rows"]),
            **{name: sum(x[name]["reward"] for x in scored if x[name] is not None)
               for name in ("hyper_online", "hyper_frozen", "hyper_rotated", "hyper_shifted_actions",
                            "online_static_lora")},
            "rotated_n": sum(x["hyper_rotated"] is not None for x in scored),
            "failures": len(result["failures"])}
        save(args.output, result)
        if i >= 120 or (i+1) % 20 == 0:
            print(json.dumps(result["summary"]), flush=True)
    common = {"review_sha256": file_hash(args.review),
              "report_sha256": file_hash(args.output),
              "runner_sha256": config["runner_sha256"], "rank": 4, "layers": 2}
    torch.save({**common, "trainable_state": {
        name: p.detach().cpu().clone() for name, p in hyper.named_parameters()
        if p.requires_grad},
        "A": [layer.a.detach().cpu().clone() for layer in hyper.adapters]},
        args.hyper_model)
    torch.save({**common,
        "A": [layer.a.detach().cpu().clone() for layer in static.adapters],
        "B": [p.detach().cpu().clone() for p in factors]}, args.static_model)


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=("prepare", "evaluate"))
    defaults = {"teacher_report": "results/trajectory_hyperlora/alfworld_online_own_trajectory_v52b_stream240_20261010.json",
        "teacher_annotations": "data/annotations/alfworld_online_own_trajectory_v52b_own_wins_20261010.json",
        "teacher_model": "results/trajectory_hyperlora/alfworld_online_own_trajectory_v52b_20261010.pt",
        "review": "data/annotations/alfworld_online_shared_trajectory_v53_20261010.json",
        "output": "results/trajectory_hyperlora/alfworld_online_shared_trajectory_v53_20261010.json",
        "hyper_model": "results/trajectory_hyperlora/alfworld_online_shared_trajectory_v53_hyper_20261010.pt",
        "static_model": "results/trajectory_hyperlora/alfworld_online_shared_trajectory_v53_static_20261010.pt",
        "model": "current_work/delta-Mem/model/Qwen3-4B-Instruct-2507",
        "data_root": "ttcl/data/alfworld_delta"}
    for name, value in defaults.items():
        p.add_argument("--" + name.replace("_", "-"), type=Path, default=Path(value))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--hyper-device", default="cuda:0")
    p.add_argument("--static-device", default="cuda:1")
    p.add_argument("--frozen-device", default="cuda:2")
    return p


if __name__ == "__main__":
    args = parser().parse_args()
    (prepare if args.command == "prepare" else evaluate)(args)
