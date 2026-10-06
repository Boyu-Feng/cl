"""Reward-filtered ALFWorld retry distillation into the LoRA generator.

The environment's terminal won reward selects successful retry trajectories.
Their public state/action sequences become newly reviewed, content-bound
training targets. The frozen Qwen actor receives no source text at inference;
the trainable source encoder and LoRA generator read the first failed attempt.
This is off-policy reward-filtered distillation, not policy-gradient RL.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import torch
from torch import nn

from ttcl.experience_evolution.environment import ACTOR_SYSTEM, make_env
from ttcl.trajectory_hyperlora.alfworld_same_game_retry import (
    checked_rows, digest, file_hash,
)
from ttcl.trajectory_hyperlora.alfworld_zero_shot_probe import load_agent
from ttcl.trajectory_hyperlora.contextual_alf_source import (
    contextual_source_fields, contextual_text_fields, task_context_text,
)
from ttcl.trajectory_hyperlora.relational_router_pilot import tokenize_records
from ttcl.trajectory_hyperlora.train_alf_next_task import query_ids, target_loss


def label_content(row: dict) -> dict:
    return {key: row[key] for key in ("target_game", "target_game_sha256",
        "source_episode_sha256", "source_records", "teacher_arm",
        "target_messages", "target_action")}


def utility(episode: dict) -> float:
    if episode["status"] != "complete":
        raise ValueError("Incomplete reward episode")
    return float(episode["reward"]) * (1.0 - .25 *
        (episode["steps"] - 1) / 29.0)


def checked_reward_rows(args):
    reviewed = checked_rows(args)
    by_game = {row["target_game"]: row for row in reviewed}
    report = json.loads(args.retries.read_text())
    if (report["candidates_sha256"] != file_hash(args.candidates) or
            report["review_sha256"] != file_hash(args.review) or
            report["split"] != "train" or
            report["max_steps"] != 30 or report["max_new_tokens"] != 64 or
            report["failures"]):
        raise ValueError("Retry reward lineage/budget/failure mismatch")
    games = report["games"]
    if len(games) != len(by_game) or len({row["game"] for row in games}) != len(games):
        raise ValueError("Incomplete or duplicate train reward rollouts")
    for item in games:
        row = by_game.get(item["game"])
        if (row is None or item["input_content_sha256"] !=
                row["input_content_sha256"] or
                item["own_source_sha256"] != row["own_source_sha256"] or
                item["wrong_source_sha256"] != row["wrong_source_sha256"] or
                any(item["arms"][arm]["status"] != "complete"
                    for arm in ("base", "own", "wrong"))):
            raise ValueError("Retry reward source/target binding mismatch")
    return [(by_game[item["game"]], item) for item in games]


def build_labels(args):
    tasks = []
    first_attempts = {row["game"]: row["base"] for row in
                      json.loads(args.first_attempts.read_text())["games"]}
    for source, item in checked_reward_rows(args):
        own, wrong = item["arms"]["own"], item["arms"]["wrong"]
        if args.teacher_mode == "walkthrough":
            teacher_arm = "walkthrough"
            commands = json.loads((args.data_root /
                source["target_game"]).read_text())["walkthrough"]
            if not commands or len(commands) > 30:
                raise ValueError("Invalid train-game walkthrough budget")
            teacher = {"status": "complete", "reward": 1,
                       "steps": len(commands),
                       "trajectory": [{"command": command, "valid": True,
                                       "response": command}
                                      for command in commands]}
        elif own["reward"] > 0:
            teacher_arm, teacher = "own", own
        elif wrong["reward"] > 0:
            teacher_arm, teacher = "wrong", wrong
        else:
            continue
        env = make_env(args.data_root / source["target_game"])
        try:
            state = env.reset()
            if ("initial_observation" in teacher and
                    str(state["feedback"]) != teacher["initial_observation"]):
                raise ValueError("Reward teacher reset mismatch")
            messages = [{"role": "system", "content": ACTOR_SYSTEM}]
            labels = []
            for step in teacher["trajectory"]:
                available = list(state["admissible_commands"])
                if (step["command"] not in available or
                        not step["valid"]):
                    raise ValueError("Reward teacher action is inadmissible")
                messages.append({"role": "user", "content":
                    str(state["feedback"]) + "\nAvailable commands:\n" +
                    "\n".join(available)})
                content = {"target_game": source["target_game"],
                    "target_game_sha256": source["target_game_sha256"],
                    "source_episode_sha256": source["own_source_sha256"],
                    "source_records": source["own_records"],
                    "teacher_arm": teacher_arm,
                    "target_messages": list(messages),
                    "target_action": step["command"]}
                labels.append({**content,
                    "input_content_sha256": digest(content)})
                state, _, done = env.step(step["command"])
                if ("observation" in step and
                    (str(state["feedback"]) != step["observation"] or
                     bool(state["won"]) != bool(step["won"]))):
                    raise ValueError("Reward teacher transition mismatch")
                messages.append({"role": "assistant",
                                 "content": step["response"]})
                if done and step is not teacher["trajectory"][-1]:
                    raise ValueError("Reward teacher continued after done")
            if not state["won"] or len(labels) != teacher["steps"]:
                raise ValueError("Selected reward teacher did not win")
        finally:
            env.close()
        failed = first_attempts[source["target_game"]]
        divergence = next((index for index, (success_step, failed_step)
            in enumerate(zip(teacher["trajectory"], failed["trajectory"]))
            if success_step["command"] != failed_step["command"]), None)
        if divergence is None or divergence >= len(labels):
            raise ValueError("Winning retry has no comparable decision change")
        if (teacher["trajectory"][divergence]["command"] ==
                failed["trajectory"][divergence]["command"]):
            raise ValueError("Divergence target mismatch")
        tasks.append({"target_game": source["target_game"],
            "target_game_sha256": source["target_game_sha256"],
            "source_episode_sha256": source["own_source_sha256"],
            "teacher_arm": teacher_arm,
            "own_reward": own["reward"], "wrong_reward": wrong["reward"],
            "own_utility": utility(own), "wrong_utility": utility(wrong),
            "teacher_utility": utility(teacher),
            "first_changed_action": divergence,
            "failed_action_at_change": failed["trajectory"][divergence]["command"],
            "labels": labels})
    return tasks


def prepare(args):
    if args.labels.exists():
        raise FileExistsError(args.labels)
    tasks = build_labels(args)
    if len(tasks) < 2:
        raise ValueError("Too few environment-confirmed retry successes")
    result = {"protocol": "ALFWorld train-only environment-confirmed successful actions; own failed first attempt is the sole LoRA source; no dev labels",
              "teacher_mode": args.teacher_mode,
              "candidates_sha256": file_hash(args.candidates),
              "retry_review_sha256": file_hash(args.review),
              "retry_reward_sha256": file_hash(args.retries),
              "tasks": tasks}
    args.labels.parent.mkdir(parents=True, exist_ok=True)
    args.labels.write_text(json.dumps(result, ensure_ascii=False,
                                      indent=2) + "\n")
    print(json.dumps({"tasks": len(tasks),
        "actions": sum(len(task["labels"]) for task in tasks),
        "own_only": sum(task["own_reward"] > task["wrong_reward"]
                        for task in tasks)}), flush=True)
    return result


def review(args):
    if args.label_review.exists():
        raise FileExistsError(args.label_review)
    candidate = json.loads(args.labels.read_text())
    if (candidate["retry_reward_sha256"] != file_hash(args.retries) or
            candidate["teacher_mode"] != args.teacher_mode or
            candidate["tasks"] != build_labels(args)):
        raise ValueError("Successful target labels no longer replay exactly")
    notes = [{"target_game": task["target_game"],
              "input_content_sha256": label["input_content_sha256"],
              "source_episode_sha256": task["source_episode_sha256"],
              "approved": True,
              "review_basis": f"New {args.teacher_mode} action target; initial state, admissible command, every transition and terminal won replayed"}
             for task in candidate["tasks"] for label in task["labels"]]
    result = {"protocol": "Fresh action-level reviewed bindings for reward-filtered ALFWorld distillation",
              "labels_sha256": file_hash(args.labels),
              "annotations": notes}
    args.label_review.parent.mkdir(parents=True, exist_ok=True)
    args.label_review.write_text(json.dumps(result, ensure_ascii=False,
                                            indent=2) + "\n")
    print(json.dumps({"approved_actions": len(notes)}), flush=True)
    return result


def train(args):
    if args.output.exists() or args.save_checkpoint.exists():
        raise FileExistsError("Fresh result/checkpoint paths required")
    dataset = json.loads(args.labels.read_text())
    review = json.loads(args.label_review.read_text())
    source_scope = dataset.get("source_scope", "same_game_retry")
    if source_scope == "same_game_retry":
        reward_rows = checked_reward_rows(args)
        wrong_by_game = {source["target_game"]: source["wrong_records"]
                         for source, _ in reward_rows}
        source_lineage_ok = (dataset["retry_reward_sha256"] ==
            file_hash(args.retries) and dataset["retry_review_sha256"] ==
            file_hash(args.review))
    elif source_scope == "all_first_attempts":
        if args.source_review is None or args.all_candidates is None:
            raise ValueError("All-first-attempt training needs --source-review and --all-candidates")
        from ttcl.trajectory_hyperlora.alfworld_expert_first_attempts import checked_all_sources
        sources = checked_all_sources(argparse.Namespace(
            candidates=args.all_candidates,
            source_review=args.source_review,
            first_attempts=args.first_attempts,
            retry_candidates=args.candidates,
            plan=args.plan, data_root=args.data_root))
        source_by_game = {row["target_game"]: row for row in sources}
        if (len(dataset["tasks"]) != len(sources) or
                len({task["target_game"] for task in dataset["tasks"]}) !=
                    len(sources) or
                any(task["target_game"] not in source_by_game or
                    task["target_game_sha256"] !=
                        source_by_game[task["target_game"]]["target_game_sha256"] or
                    task["source_episode_sha256"] !=
                        source_by_game[task["target_game"]]["source_episode_sha256"] or
                    task["labels"][0]["source_records"] !=
                        source_by_game[task["target_game"]]["source_records"] or
                    task["wrong_records"] !=
                        source_by_game[task["target_game"]]["wrong_records"] or
                    task["own_utility"] != float(
                        source_by_game[task["target_game"]]["source_reward"]) *
                        (1.0 - .25 * (source_by_game[
                            task["target_game"]]["source_steps"] - 1) / 29.0) or
                    task["teacher_utility"] != (1.0 - .25 *
                        (len(task["labels"]) - 1) / 29.0)
                    for task in dataset["tasks"])):
            raise ValueError("Expanded train task/source bindings changed")
        source_lineage_ok = (dataset["source_review_sha256"] ==
            file_hash(args.source_review))
        wrong_by_game = {task["target_game"]: task["wrong_records"]
                         for task in dataset["tasks"]}
    elif source_scope == "sibling_expert":
        if args.sibling_candidates is None or args.sibling_source_review is None:
            raise ValueError("Sibling training needs reviewed source paths")
        from ttcl.trajectory_hyperlora.alfworld_sibling_transfer import checked_review
        source_split = dataset.get("split", "train")
        if source_split not in ("train", "train_large"):
            raise ValueError("Sibling training must stay in the train split")
        sources = checked_review(argparse.Namespace(
            candidates=args.sibling_candidates,
            source_review=args.sibling_source_review,
            retry_candidates=args.candidates,
            review=args.review,
            all_source_review=args.all_source_review,
            data_root=args.data_root, split=source_split))
        source_rows_by_game = {row["target_game"]: row for row, _ in sources}
        source_by_game = {row["target_game"]: note for row, note in sources}
        if (dataset["sibling_candidates_sha256"] !=
                file_hash(args.sibling_candidates) or
                dataset["sibling_source_review_sha256"] !=
                file_hash(args.sibling_source_review) or
                dataset["first_attempts_sha256"] !=
                (file_hash(args.first_attempts)
                 if source_split == "train" else None) or
                len(dataset["tasks"]) != len(sources) or
                len({task["target_game"] for task in dataset["tasks"]}) !=
                    len(sources) or
                any(task["target_game"] not in source_by_game or
                    task["target_game_sha256"] !=
                        source_rows_by_game[task["target_game"]]["target_game_sha256"] or
                    task["source_episode_sha256"] !=
                        source_by_game[task["target_game"]]["source_records_sha256"] or
                    task["labels"][0]["source_records"] !=
                        source_by_game[task["target_game"]]["source_records"] or
                    task["wrong_records"] !=
                        source_by_game[task["target_game"]]["wrong_records"] or
                    task["teacher_utility"] != (1.0 - .25 *
                        (len(task["labels"]) - 1) / 29.0)
                    for task in dataset["tasks"])):
            raise ValueError("Sibling train task/source bindings changed")
        source_lineage_ok = True
        if source_split == "train_large" and args.advantage_bonus:
            raise ValueError("Large sibling targets lack own-attempt reward; set --advantage-bonus 0")
        wrong_by_game = {task["target_game"]: task["wrong_records"]
                         for task in dataset["tasks"]}
    else:
        raise ValueError("Unknown reward-distillation source scope")
    if (not source_lineage_ok or
            dataset["teacher_mode"] != args.teacher_mode or
            review["labels_sha256"] != file_hash(args.labels)):
        raise ValueError("Reward-distillation lineage changed")
    approved = {note["input_content_sha256"]: note
                for note in review["annotations"]}
    examples = [label for task in dataset["tasks"]
                for label in task["labels"]]
    if len(approved) != len(examples):
        raise ValueError("Duplicate or missing reviewed reward labels")
    for task in dataset["tasks"]:
        for label in task["labels"]:
            note = approved.get(label["input_content_sha256"])
            if (digest(label_content(label)) != label["input_content_sha256"]
                    or note is None or note["approved"] is not True or
                    note["target_game"] != task["target_game"] or
                    note["source_episode_sha256"] !=
                    task["source_episode_sha256"]):
                raise ValueError("Unreviewed or changed reward action")
    teacher_advantage = {}
    train_tasks = dataset["tasks"]
    discriminative_by_game = {}
    discriminative_meta = {}
    if args.discriminative_candidates is not None or args.discriminative_review is not None:
        if (source_scope != "sibling_expert" or
                args.discriminative_candidates is None or
                args.discriminative_review is None):
            raise ValueError("Discriminative action selection needs both reviewed paths")
        from ttcl.trajectory_hyperlora.select_alf_sibling_action_evidence import (
            content as selection_content,
        )
        selected = json.loads(args.discriminative_candidates.read_text())
        selected_review = json.loads(args.discriminative_review.read_text())
        if (selected["labels_sha256"] != file_hash(args.labels) or
                selected["label_review_sha256"] != file_hash(args.label_review) or
                selected["source_review_sha256"] !=
                    dataset["sibling_source_review_sha256"] or
                selected_review["candidates_sha256"] !=
                    file_hash(args.discriminative_candidates)):
            raise ValueError("Discriminative action lineage changed")
        selected_notes = {note["selection_content_sha256"]: note
            for note in selected_review["annotations"]}
        if len(selected_notes) != len(selected["rows"]):
            raise ValueError("Discriminative annotations incomplete")
        tasks_by_game = {task["target_game"]: task
                         for task in dataset["tasks"]}
        for item in selected["rows"]:
            task = tasks_by_game.get(item["target_game"])
            note = selected_notes.get(item["selection_content_sha256"])
            if (task is None or note is None or note["approved"] is not True or
                    digest(selection_content(item)) !=
                        item["selection_content_sha256"] or
                    note["action_input_sha256"] != item["action_input_sha256"] or
                    item["target_game_sha256"] != task["target_game_sha256"] or
                    item["source_records_sha256"] !=
                        digest(task["labels"][0]["source_records"]) or
                    item["wrong_records_sha256"] !=
                        digest(task["wrong_records"]) or
                    item["action_index"] >= len(task["labels"])):
                raise ValueError("Discriminative selection content changed")
            label = task["labels"][item["action_index"]]
            if (label["input_content_sha256"] != item["action_input_sha256"] or
                    label["target_action"] != item["expert_action"]):
                raise ValueError("Discriminative target action changed")
            discriminative_by_game.setdefault(item["target_game"], []).append(label)
            discriminative_meta[item["action_input_sha256"]] = item
        train_tasks = [task for task in dataset["tasks"]
                       if task["target_game"] in discriminative_by_game]
        if len(train_tasks) < 2 or len(discriminative_meta) != len(selected["rows"]):
            raise ValueError("Too few unique discriminative training actions")
    if args.text_teacher_scores is not None:
        if source_scope != "sibling_expert":
            raise ValueError("Text-teacher scores require sibling expert sources")
        teacher = json.loads(args.text_teacher_scores.read_text())
        if (teacher["labels_sha256"] != file_hash(args.labels) or
                teacher["label_review_sha256"] !=
                    file_hash(args.label_review) or
                teacher["source_review_sha256"] !=
                    dataset["sibling_source_review_sha256"]):
            raise ValueError("Text-teacher source/action binding changed")
        scored = {row["input_content_sha256"]: row
                  for row in teacher["rows"]}
        if len(scored) != len(teacher["rows"]):
            raise ValueError("Duplicate text-teacher action score")
        for task in dataset["tasks"]:
            for label in task["labels"]:
                score = scored.get(label["input_content_sha256"])
                if score is None:
                    continue
                if (score["target_game"] != task["target_game"] or
                        score["action"] != label["target_action"] or
                        any(not isinstance(score["ce"][arm], (int, float))
                            for arm in ("base", "own_text", "wrong_text"))):
                    raise ValueError("Text-teacher scored action changed")
                teacher_advantage[label["input_content_sha256"]] = (
                    min(score["ce"]["base"], score["ce"]["wrong_text"]) -
                    score["ce"]["own_text"])
        if len(teacher_advantage) != len(scored):
            raise ValueError("Text-teacher scores include unknown actions")
        if args.only_text_positive:
            train_tasks = [task for task in dataset["tasks"]
                if teacher_advantage.get(task["labels"][0][
                    "input_content_sha256"], float("-inf")) >
                    args.text_advantage_min]
            if len(train_tasks) < 2:
                raise ValueError("Too few positive text-teacher tasks")
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    agent, tokenizer = load_agent(args.model, args.checkpoint,
                                  args.device, args.gpu_fraction,
                                  context_mode_override=args.context_mode)
    if args.contextual_source:
        if source_scope != "sibling_expert":
            raise ValueError("Contextual source training requires sibling trajectories")
        if agent.encoder_kind == "covariance":
            old_latents = []
            with torch.no_grad():
                for task in dataset["tasks"]:
                    fields = tokenize_records(tokenizer,
                        task["labels"][0]["source_records"], args.device,
                        max_tokens=args.source_max_tokens,
                        truncation_mode=args.source_truncation)
                    old_latents.append(agent.encode(fields).detach())
            latent_mean = torch.cat(old_latents).mean(0)
            width = agent.model.get_input_embeddings().embedding_dim
            agent.contextual_latent = nn.Sequential(
                nn.LayerNorm(width), nn.Linear(width, 128)).to(args.device)
            nn.init.zeros_(agent.contextual_latent[1].weight)
            agent.contextual_latent[1].bias.data.copy_(latent_mean)
            del agent.latent
            agent.encoder_kind = "contextual"
            for module in (agent.encoder, agent.oracle_latent,
                           agent.relation_head):
                for parameter in module.parameters():
                    parameter.requires_grad_(False)
        elif agent.encoder_kind != "contextual":
            raise ValueError("Contextual source warm-start needs a compatible checkpoint")
    elif agent.encoder_kind == "contextual":
        raise ValueError("Contextual checkpoint requires --contextual-source")
    existing_task_conditioned = agent.task_conditioned
    if args.task_conditioned:
        if source_scope != "sibling_expert" or not args.contextual_source:
            raise ValueError("Task conditioning needs reviewed sibling contextual sources")
        if agent.task_conditioned and agent.task_pair_pooling != args.task_pair_pooling:
            raise ValueError("Task-pair pooling changed across checkpoints")
        if not agent.task_conditioned:
            width = agent.model.get_input_embeddings().embedding_dim
            agent.task_pair_latent = nn.Sequential(
                nn.LayerNorm(2 * width), nn.Linear(2 * width, 128)).to(args.device)
            nn.init.zeros_(agent.task_pair_latent[1].weight)
            nn.init.zeros_(agent.task_pair_latent[1].bias)
            agent.task_conditioned = True
        agent.task_pair_pooling = args.task_pair_pooling
    elif agent.task_conditioned:
        raise ValueError("Task-conditioned checkpoint requires --task-conditioned")
    if (args.task_conditioned and existing_task_conditioned and
            agent.task_context_scope != args.task_context_scope):
        raise ValueError("Task-context scope changed across checkpoints")
    agent.task_context_scope = args.task_context_scope
    checkpoint_parameter_names = {name for name, parameter in
        agent.named_parameters() if parameter.requires_grad}
    if args.train_task_pair_only:
        if not args.task_conditioned:
            raise ValueError("Pair-only training needs task conditioning")
        for name, parameter in agent.named_parameters():
            parameter.requires_grad_(name.startswith("task_pair_latent."))
    source_fields = {}
    wrong_fields = {}
    target_fields = {}
    step_target_fields = {}
    prefixes = {}
    for task in dataset["tasks"]:
        row = task["labels"][0]
        source_fields[task["target_game"]] = (
            contextual_source_fields(agent, tokenizer,
                row["source_records"], args.device,
                args.contextual_source_max_tokens,
                pooling="both" if args.task_conditioned and
                    args.task_pair_pooling == "mean" else "last")
            if args.contextual_source else tokenize_records(
                tokenizer, row["source_records"], args.device,
                max_tokens=args.source_max_tokens,
                truncation_mode=args.source_truncation,
                repeat_initial_observation=args.repeat_initial_observation))
        if args.source_contrast_weight:
            wrong_fields[task["target_game"]] = (
                contextual_source_fields(agent, tokenizer,
                    wrong_by_game[task["target_game"]], args.device,
                    args.contextual_source_max_tokens,
                    pooling="both" if args.task_conditioned and
                        args.task_pair_pooling == "mean" else "last")
                if args.contextual_source else tokenize_records(
                    tokenizer, wrong_by_game[task["target_game"]], args.device,
                    max_tokens=args.source_max_tokens,
                    truncation_mode=args.source_truncation,
                    repeat_initial_observation=args.repeat_initial_observation))
        if args.match_contrast_weight or args.task_conditioned:
            if not args.contextual_source:
                raise ValueError("Task matching needs contextual source vectors")
            first_observation = row["target_messages"][1]["content"].split(
                "\nAvailable commands:\n", 1)[0]
            target_fields[task["target_game"]] = contextual_text_fields(
                agent, tokenizer,
                task_context_text(first_observation, first_observation),
                args.device, args.contextual_source_max_tokens,
                pooling="both" if args.task_conditioned and
                    args.task_pair_pooling == "mean" else "last")
        for label in task["labels"]:
            if args.task_conditioned and args.task_context_scope == "current":
                current_observation = label["target_messages"][-1][
                    "content"].split("\nAvailable commands:\n", 1)[0]
                step_target_fields[label["input_content_sha256"]] = (
                    target_fields[task["target_game"]]
                    if current_observation == first_observation else
                    contextual_text_fields(agent, tokenizer,
                        task_context_text(first_observation,
                                          current_observation),
                        args.device, args.contextual_source_max_tokens,
                        pooling="both" if args.task_pair_pooling == "mean"
                            else "last"))
            prefixes[label["input_content_sha256"]] = query_ids(
                tokenizer, label, history_turns=args.history_turns,
                max_prompt_tokens=args.max_prompt_tokens)
    source_factors = {}
    with torch.no_grad():
        for task in dataset["tasks"]:
            game = task["target_game"]
            agent.set_source(source_fields[game],
                target_fields=target_fields.get(game))
            source_factors[game] = [adapter.b.detach().clone()
                for adapter in agent.adapters]
            agent.set_source(None)
    parameters = [param for param in agent.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.lr,
                                  weight_decay=args.weight_decay)
    losses = []
    for step in range(args.steps):
        task = rng.choice(train_tasks)
        row = (rng.choice(discriminative_by_game[task["target_game"]])
               if discriminative_by_game else
               task["labels"][0] if args.only_text_positive else
               task["labels"][task["first_changed_action"]]
               if rng.random() < args.focus_probability
               else rng.choice(task["labels"]))
        game = task["target_game"]
        current_target = step_target_fields.get(
            row["input_content_sha256"], target_fields.get(game))
        optimizer.zero_grad(set_to_none=True)
        agent.set_source(source_fields[game],
                         target_fields=current_target)
        ce = target_loss(agent, tokenizer, row,
            prefixes[row["input_content_sha256"]], args.device)
        anchor = sum((adapter.b - reference).float().square().mean()
                     for adapter, reference in zip(agent.adapters,
                         source_factors[game], strict=True))
        if args.source_contrast_weight:
            agent.set_source(wrong_fields[game],
                             target_fields=current_target)
            wrong_ce = target_loss(agent, tokenizer, row,
                prefixes[row["input_content_sha256"]], args.device)
            contrast = torch.relu(args.source_contrast_margin + ce - wrong_ce)
        else:
            contrast = ce.new_zeros(())
        if args.match_contrast_weight:
            own_latent = agent.encode(source_fields[game])
            wrong_latent = agent.encode(wrong_fields[game])
            target_latent = agent.encode(target_fields[game])
            # A dot score keeps a gradient when a warm-started projection
            # initially maps every source to the same latent vector.
            own_match = (own_latent * target_latent).mean()
            wrong_match = (wrong_latent * target_latent).mean()
            match_contrast = torch.relu(args.match_contrast_margin +
                wrong_match - own_match)
        else:
            match_contrast = ce.new_zeros(())
        if source_scope in ("all_first_attempts", "sibling_expert"):
            advantage = (max(0.0, task["teacher_utility"] -
                             task["own_utility"])
                if task["own_utility"] is not None else 0.0)
        elif args.teacher_mode == "retry":
            advantage = max(0.0, task["own_utility"] -
                            task["wrong_utility"])
        else:
            advantage = max(0.0, task["teacher_utility"] -
                            max(task["own_utility"], task["wrong_utility"]))
        text_gain = max(0.0, min(2.0, teacher_advantage.get(
            row["input_content_sha256"], 0.0)))
        weight = (1.0 + args.advantage_bonus * advantage) * (
            1.0 + args.text_teacher_weight * text_gain)
        if discriminative_by_game:
            weight *= (1.0 + args.admissible_wrong_bonus *
                float(discriminative_meta[row["input_content_sha256"]][
                    "wrong_action_admissible"]))
        loss = (weight * ce + args.anchor_weight * anchor +
                args.source_contrast_weight * contrast +
                args.match_contrast_weight * match_contrast)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.)
        optimizer.step()
        agent.set_source(None)
        losses.append({"ce": float(ce.detach()),
                       "anchor": float(anchor.detach()),
                       "contrast": float(contrast.detach()),
                       "match_contrast": float(match_contrast.detach())})
        if (step + 1) % args.log_every == 0 or step + 1 == args.steps:
            recent = losses[-args.log_every:]
            print(json.dumps({"step": step + 1,
                "ce": sum(x["ce"] for x in recent) / len(recent),
                "anchor": sum(x["anchor"] for x in recent) / len(recent),
                "contrast": sum(x["contrast"] for x in recent) / len(recent),
                "match_contrast": sum(x["match_contrast"] for x in recent) / len(recent)}),
                flush=True)
    result = {"protocol": "Train-only environment-confirmed ALFWorld expert or winning-retry action distillation into trajectory-generated LoRA; reward-gap weighting applies only when recorded own/wrong rollouts and a nonzero advantage bonus are available; frozen actor; no development reward in training",
              "teacher_mode": args.teacher_mode,
              "source_scope": source_scope,
              "source_checkpoint_sha256": file_hash(args.checkpoint),
              "labels_sha256": file_hash(args.labels),
              "label_review_sha256": file_hash(args.label_review),
              "source_lineage_sha256": (file_hash(args.retries)
                  if source_scope == "same_game_retry" else
                  file_hash(args.sibling_source_review)
                  if source_scope == "sibling_expert" else
                  file_hash(args.source_review)),
              "seed": args.seed, "steps": args.steps,
              "lr": args.lr, "weight_decay": args.weight_decay,
              "anchor_weight": args.anchor_weight,
              "advantage_bonus": args.advantage_bonus,
              "focus_probability": args.focus_probability,
              "source_contrast_weight": args.source_contrast_weight,
              "source_contrast_margin": args.source_contrast_margin,
              "match_contrast_weight": args.match_contrast_weight,
              "match_contrast_margin": args.match_contrast_margin,
              "text_teacher_scores_sha256": (file_hash(args.text_teacher_scores)
                  if args.text_teacher_scores is not None else None),
              "text_teacher_weight": args.text_teacher_weight,
              "text_advantage_min": args.text_advantage_min,
              "only_text_positive": args.only_text_positive,
              "selected_train_tasks": len(train_tasks),
              "discriminative_candidates_sha256": (
                  file_hash(args.discriminative_candidates)
                  if args.discriminative_candidates is not None else None),
              "discriminative_review_sha256": (
                  file_hash(args.discriminative_review)
                  if args.discriminative_review is not None else None),
              "admissible_wrong_bonus": args.admissible_wrong_bonus,
              "history_turns": args.history_turns,
              "max_prompt_tokens": args.max_prompt_tokens,
              "source_max_tokens": args.source_max_tokens,
              "source_truncation": args.source_truncation,
              "repeat_initial_observation": args.repeat_initial_observation,
              "context_mode": args.context_mode,
              "contextual_source": args.contextual_source,
              "contextual_source_max_tokens": args.contextual_source_max_tokens,
              "task_conditioned": args.task_conditioned,
              "train_task_pair_only": args.train_task_pair_only,
              "task_pair_pooling": args.task_pair_pooling,
              "task_context_scope": args.task_context_scope,
              "train_tasks": len(dataset["tasks"]),
              "train_actions": len(examples),
              "selected_training_action_candidates": (
                  len(discriminative_meta) if discriminative_by_game else
                  len(train_tasks) if args.only_text_positive else len(examples)),
              "last_losses": losses[-args.log_every:]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.save_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False,
                                       indent=2) + "\n")
    torch.save({"trainable_state": {name: param.detach().cpu()
                                    for name, param in agent.named_parameters()
                                    if name in checkpoint_parameter_names},
                "rank": len(agent.adapters[0].a),
                "layers": len(agent.adapters),
                "encoder_kind": agent.encoder_kind,
                "relation_bottleneck": agent.relation_bottleneck,
                "source_encoder": ("contextual_lm" if args.contextual_source
                                   else "raw"),
                "training_domain": "alfworld_retry_reward",
                "teacher_mode": args.teacher_mode,
                "source_scope": source_scope,
                "source_max_tokens": args.source_max_tokens,
                "source_truncation": args.source_truncation,
                "repeat_initial_observation": args.repeat_initial_observation,
                "context_mode": args.context_mode,
                "contextual_source": args.contextual_source,
                "contextual_source_max_tokens": args.contextual_source_max_tokens,
                "task_conditioned": args.task_conditioned,
                "train_task_pair_only": args.train_task_pair_only,
                "task_pair_pooling": args.task_pair_pooling,
                "task_context_scope": args.task_context_scope,
                "context_strength": agent.context_strength,
                "source_contrast_weight": args.source_contrast_weight,
                "match_contrast_weight": args.match_contrast_weight,
                "match_contrast_margin": args.match_contrast_margin,
                "discriminative_candidates_sha256": result[
                    "discriminative_candidates_sha256"],
                "labels_sha256": result["labels_sha256"]},
               args.save_checkpoint)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "review", "train"))
    parser.add_argument("--model", type=Path, default=Path(
        "current_work/delta-Mem/model/Qwen3-4B-Instruct-2507"))
    parser.add_argument("--checkpoint", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_next_task_warmstart_20261005/seed42_v2.pt"))
    parser.add_argument("--plan", type=Path, default=Path(
        "ttcl/results/experience_evolution/alfworld_delta_20260922/plan.json"))
    parser.add_argument("--first-attempts", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_rl_20261005/train_rewards_complete.json"))
    parser.add_argument("--data-root", type=Path, default=Path(
        "ttcl/data/alfworld_delta"))
    parser.add_argument("--split", choices=("train",), default="train")
    parser.add_argument("--candidates", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_same_game_retry_candidates_20261006.json"))
    parser.add_argument("--review", type=Path, default=Path(
        "data/annotations/alf_same_game_retry_reviewed_20261006.json"))
    parser.add_argument("--retries", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_same_game_retry_train17_20261006.json"))
    parser.add_argument("--source-review", type=Path)
    parser.add_argument("--all-candidates", type=Path)
    parser.add_argument("--all-source-review", type=Path)
    parser.add_argument("--sibling-candidates", type=Path)
    parser.add_argument("--sibling-source-review", type=Path)
    parser.add_argument("--labels", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_retry_reward_labels_20261006.json"))
    parser.add_argument("--label-review", type=Path, default=Path(
        "data/annotations/alf_retry_reward_labels_reviewed_20261006.json"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpu-fraction", type=float, default=.65)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--teacher-mode", choices=("retry", "walkthrough"),
                        default="retry")
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--lr", type=float, default=.00003)
    parser.add_argument("--weight-decay", type=float, default=.01)
    parser.add_argument("--anchor-weight", type=float, default=.1)
    parser.add_argument("--advantage-bonus", type=float, default=2.)
    parser.add_argument("--focus-probability", type=float, default=.5,
                        help="Chance to train the earliest action changed between failed first attempt and winning retry")
    parser.add_argument("--source-contrast-weight", type=float, default=0.)
    parser.add_argument("--source-contrast-margin", type=float, default=.2)
    parser.add_argument("--match-contrast-weight", type=float, default=0.)
    parser.add_argument("--match-contrast-margin", type=float, default=.1)
    parser.add_argument("--task-conditioned", action="store_true")
    parser.add_argument("--train-task-pair-only", action="store_true")
    parser.add_argument("--task-pair-pooling", choices=("last", "mean"),
                        default="last")
    parser.add_argument("--task-context-scope", choices=("initial", "current"),
                        default="initial")
    parser.add_argument("--discriminative-candidates", type=Path)
    parser.add_argument("--discriminative-review", type=Path)
    parser.add_argument("--admissible-wrong-bonus", type=float, default=0.)
    parser.add_argument("--text-teacher-scores", type=Path)
    parser.add_argument("--text-teacher-weight", type=float, default=0.)
    parser.add_argument("--text-advantage-min", type=float, default=.1)
    parser.add_argument("--only-text-positive", action="store_true")
    parser.add_argument("--history-turns", type=int, default=2)
    parser.add_argument("--max-prompt-tokens", type=int, default=3000)
    parser.add_argument("--source-max-tokens", type=int, default=40)
    parser.add_argument("--source-truncation", choices=("head", "head_tail"),
                        default="head_tail")
    parser.add_argument("--repeat-initial-observation", action="store_true")
    parser.add_argument("--context-mode", choices=("none", "initial"),
                        default="none")
    parser.add_argument("--contextual-source", action="store_true")
    parser.add_argument("--contextual-source-max-tokens", type=int, default=2048)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--save-checkpoint", type=Path)
    args = parser.parse_args()
    if (args.steps < 1 or args.log_every < 1 or args.lr <= 0 or
            args.anchor_weight < 0 or args.advantage_bonus < 0 or
            args.source_contrast_weight < 0 or args.source_contrast_margin < 0 or
            args.match_contrast_weight < 0 or args.match_contrast_margin < 0 or
            (args.match_contrast_weight and not args.source_contrast_weight) or
            args.admissible_wrong_bonus < 0 or
            (args.task_context_scope == "current" and
             not args.task_conditioned) or
            (args.task_pair_pooling == "mean" and
             not args.task_conditioned) or
            args.text_teacher_weight < 0 or
            (args.only_text_positive and args.text_teacher_scores is None) or
            not 0 <= args.focus_probability <= 1 or
            args.history_turns < 0 or args.max_prompt_tokens < 1 or
            args.source_max_tokens < 2 or
            args.contextual_source_max_tokens < 2 or
            (args.repeat_initial_observation and
             args.source_truncation != "head_tail") or
            not 0 < args.gpu_fraction <= 1):
        parser.error("Invalid reward-distillation budget")
    if args.command == "prepare":
        prepare(args)
    elif args.command == "review":
        review(args)
    else:
        if args.output is None or args.save_checkpoint is None:
            parser.error("train needs --output and --save-checkpoint")
        train(args)


if __name__ == "__main__":
    main()
