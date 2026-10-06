"""Cross-episode reward pilot for mixed-domain reviewed XLand training.

Source episodes are collected by the frozen actor from empty memory. At each
prefix, three policies face the same new reset: no adapter, an adapter made
from observed transitions, and an adapter gated on observed positive reward.
This is a small environment-reward pilot, not the 500-episode paper protocol.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch

from ttcl.trajectory_hyperlora.evaluate_xland_official_live import (
    Worker, digest, load_actor, select, sha256, state,
)
from ttcl.trajectory_hyperlora.train_xland_official_history_hyperlora import source_tensor


def rollout(worker, agent, tokenizer, choices, rule_id, seed, budget, factors,
            device, epsilon, rng_seed, feedback_window, expected_observation=None,
            action_novelty=False):
    current = worker.send({"command": "reset", "ruleset_id": rule_id, "seed": seed})
    if expected_observation is not None and current["observation"] != expected_observation:
        raise ValueError("Target reset changed after adapter compilation")
    if current["num_actions"] != 6:
        raise ValueError("Official environment did not expose six actions")
    rng = random.Random(rng_seed)
    rows = []
    tried = {}
    for _ in range(min(budget, current["max_steps"])):
        before = state(current["observation"])
        model_action, prompt_hash, logits = select(
            agent, tokenizer, choices, current["observation"], factors,
            device, rows[-feedback_window:] if feedback_window else None)
        action = model_action
        view_hash = digest(before)
        if action_novelty:
            available = [candidate for candidate in range(6)
                         if candidate not in tried.get(view_hash, set())]
            if available:
                action = max(available, key=lambda candidate: logits[candidate])
        if rng.random() < epsilon:
            action = rng.randrange(6)
        tried.setdefault(view_hash, set()).add(action)
        current = worker.send({"command": "step", "action": action})
        rows.append({"state": before, "action": action,
                     "model_action": model_action,
                     "next_state": state(current["observation"]),
                     "reward": current["reward"], "done": current["done"],
                     "prompt_sha256": prompt_hash,
                     "choice_logits": logits})
        if current["done"]:
            break
    return {"steps": rows, "return": sum(x["reward"] for x in rows),
            "positive": any(x["reward"] > 0 for x in rows),
            "ended": bool(rows and rows[-1]["done"])}


def compile_history(agent, episodes, target_state, device, positive_only=False):
    if not episodes:
        return None, None
    candidates = [ep for ep in episodes if ep["positive"]] if positive_only else episodes
    if not candidates:
        return None, None
    limit = agent.encoder.max_source_length
    selected = []
    for episode in candidates:
        steps = episode["steps"]
        if positive_only:
            # Keep the successful action and the preceding local context.
            end = max(i for i, x in enumerate(steps) if x["reward"] > 0) + 1
            steps = steps[:end]
        selected.append({"goal": [0, 0], "steps": [{key: step[key]
            for key in ("state", "action", "next_state", "reward", "done")}
            for step in steps]})
    # The encoder has a bounded context; select the last complete observed
    # transition window, without accessing future actions or rule internals.
    flat = [step for episode in selected for step in episode["steps"]]
    flat = flat[-limit:]
    content = {"source_episodes": [{"goal": [0, 0], "steps": flat}],
        "target_initial_state": target_state, "goal": [0, 0]}
    with torch.no_grad():
        factors = agent.compile_adapters(source_tensor(content, device, limit))
    return factors, digest(content)


def summarize(rows):
    output = {}
    for arm in ("none", "all_history", "reward_gated"):
        episodes = [row["targets"][arm] for row in rows]
        output[arm] = {"n": len(episodes),
                       "positive": sum(ep["positive"] for ep in episodes),
                       "return": sum(ep["return"] for ep in episodes)}
    return output


def run(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    if len(set(args.rule_ids)) != len(args.rule_ids) or min(args.rule_ids) < 0:
        raise ValueError("Distinct nonnegative ruleset IDs required")
    if args.source_episodes < 1 or args.budget < 1 or not 0 <= args.epsilon <= 1:
        raise ValueError("Invalid episode budget")
    training = json.loads(args.training_result.read_text())
    reviewed = json.loads(args.annotations.read_text())
    # Mixed annotations namespace medium rules as "medium:N". They belong to
    # a different benchmark and must not be parsed as trivial rule IDs.
    used = {int(item["ruleset_id"])
            for group in reviewed["split"].values() for item in group
            if isinstance(item["ruleset_id"], int)}
    if used.intersection(args.rule_ids):
        raise ValueError("Evaluation ruleset occurs in existing train/dev/test")
    if args.device.startswith("cuda"):
        torch.cuda.set_per_process_memory_fraction(args.gpu_fraction, device=args.device)
    agent, tokenizer, _ = load_actor(args)
    # The environment has six actions. The earlier offline checkpoint was
    # trained on five expert labels; all three arms get the same six-action
    # inference interface here, including the previously omitted toggle.
    tokens = [tokenizer(str(x), add_special_tokens=False).input_ids for x in range(6)]
    if any(len(ids) != 1 for ids in tokens):
        raise ValueError("Six action labels must each be atomic")
    choices = [ids[0] for ids in tokens]
    worker = Worker(args.xland_python, args.data_dir, args.benchmark,
                    args.environment, args.benchmark_path)
    result = {"protocol": "Exploratory official trivial cross-episode reward pilot for mixed-domain XLand hypernetwork; source from empty frozen actor; matched target resets and six-action interface; not full XLand-100B 500-episode evaluation",
              "benchmark": args.benchmark, "benchmark_sha256": sha256(args.benchmark_path),
              "annotations_sha256": sha256(args.annotations),
              "checkpoint_sha256": sha256(args.checkpoint),
              "training_result_sha256": sha256(args.training_result),
              "model_config_sha256": sha256(args.model / "config.json"),
              "rule_ids": args.rule_ids, "source_episodes": args.source_episodes,
              "only_final_target": args.only_final_target,
              "budget": args.budget, "epsilon": args.epsilon,
              "feedback_window": args.feedback_window,
              "action_novelty": args.action_novelty,
              "trained_action_count": training.get("action_count", 5),
              "source_window": agent.encoder.max_source_length,
              "evaluation_action_count": 6, "rows": [], "failures": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        for rule_id in args.rule_ids:
            sources = []
            try:
                for prefix in range(args.source_episodes + 1):
                    if args.only_final_target and prefix < args.source_episodes:
                        sources.append(rollout(worker, agent, tokenizer, choices,
                            rule_id, args.source_seed + 1009 * rule_id + prefix,
                            args.budget, None, args.device, args.epsilon,
                            args.action_seed + 100000 + 1009 * rule_id + prefix,
                            args.feedback_window,
                            action_novelty=args.action_novelty))
                        continue
                    target_seed = args.target_seed + 1009 * rule_id + prefix
                    target_initial = worker.send({"command": "reset",
                        "ruleset_id": rule_id, "seed": target_seed})
                    target_state = state(target_initial["observation"])
                    all_adapter, all_hash = compile_history(
                        agent, sources, target_state, args.device)
                    gated_adapter, gated_hash = compile_history(
                        agent, sources, target_state, args.device,
                        positive_only=True)
                    targets = {}
                    for arm, factors in (("none", None),
                                         ("all_history", all_adapter),
                                         ("reward_gated", gated_adapter)):
                        targets[arm] = rollout(worker, agent, tokenizer, choices,
                            rule_id, target_seed,
                            args.budget, factors, args.device, args.epsilon,
                            args.action_seed + 1009 * rule_id + prefix,
                            args.feedback_window, target_initial["observation"],
                            args.action_novelty)
                    result["rows"].append({"ruleset_id": rule_id,
                        "source_prefix": prefix,
                        "source_episodes": list(sources),
                        "source_sha256": digest(sources),
                        "target_initial_sha256": digest(target_state),
                        "all_history_input_sha256": all_hash,
                        "reward_gated_input_sha256": gated_hash,
                        "source_positive_so_far": sum(ep["positive"] for ep in sources),
                        "targets": targets})
                    if prefix < args.source_episodes:
                        sources.append(rollout(worker, agent, tokenizer, choices,
                            rule_id, args.source_seed + 1009 * rule_id + prefix,
                            args.budget, None, args.device, args.epsilon,
                            args.action_seed + 100000 + 1009 * rule_id + prefix,
                            args.feedback_window, action_novelty=args.action_novelty))
            except Exception as exc:
                result["failures"].append({"ruleset_id": rule_id,
                    "error": f"{type(exc).__name__}: {exc}",
                    "source_prefix_complete": len(sources)})
            result["summary"] = summarize(result["rows"])
            temporary = args.output.with_suffix(args.output.suffix + ".tmp")
            temporary.write_text(json.dumps(result, indent=2) + "\n")
            temporary.replace(args.output)
            print(json.dumps({"complete_rule_ids": len(set(row["ruleset_id"]
                for row in result["rows"])), "failures": len(result["failures"]),
                "summary": result["summary"]}), flush=True)
    finally:
        worker.close()
        agent.mount(None)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--training-result", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--benchmark-path", type=Path, required=True)
    parser.add_argument("--environment", default="XLand-MiniGrid-R1-9x9")
    parser.add_argument("--rule-ids", nargs="+", type=int, required=True)
    parser.add_argument("--model", type=Path, default=Path(
        "current_work/delta-Mem/model/Qwen3-4B-Instruct-2507"))
    parser.add_argument("--xland-python", type=Path, default=Path(
        "ttcl/.runtime/xland_env/bin/python"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/xland_minigrid"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpu-fraction", type=float, default=.6)
    parser.add_argument("--source-episodes", type=int, default=2)
    parser.add_argument("--only-final-target", action="store_true")
    parser.add_argument("--budget", type=int, default=100)
    parser.add_argument("--epsilon", type=float, default=.2)
    parser.add_argument("--feedback-window", type=int, default=4)
    parser.add_argument("--action-novelty", action="store_true")
    parser.add_argument("--source-seed", type=int, default=20261007)
    parser.add_argument("--target-seed", type=int, default=30261007)
    parser.add_argument("--action-seed", type=int, default=40261007)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
