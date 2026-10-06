"""Fresh-reset official reward pilot with observed no-change avoidance.

Each source is an independently reviewed positive-reward history from the
published HDF5. The source is rebound to the actual target reset observation
before adapter generation; no target action or environment internals enter the
hypernetwork. This is not the paper's 500-episode online evaluation.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ttcl.trajectory_hyperlora.evaluate_xland_official_live import (
    Worker, digest, load_actor, select, sha256, state,
)
from ttcl.trajectory_hyperlora.train_xland_official_history_hyperlora import source_tensor
from ttcl.trajectory_hyperlora.train_xland_qwen_hyperlora import QwenRawHyperLoRA
from ttcl.trajectory_hyperlora.train_xland_raw_hyperlora import checked_query


def rollout(worker, agent, tokenizer, choices, rule_id, seed, budget, factors,
            device, epsilon, rng_seed, feedback_window, expected_observation=None,
            action_novelty=False):
    current = worker.send({"command": "reset", "ruleset_id": rule_id,
                           "seed": seed})
    if (expected_observation is not None and
            current["observation"] != expected_observation):
        raise ValueError("Target reset changed after adapter compilation")
    if current["num_actions"] != 6:
        raise ValueError("Official environment did not expose six actions")
    rng = random.Random(rng_seed)
    rows = []
    observed_no_change: dict[str, set[int]] = {}
    for _ in range(min(budget, current["max_steps"])):
        before = state(current["observation"])
        model_action, prompt_hash, action_logits = select(
            agent, tokenizer, choices, current["observation"], factors,
            device, rows[-feedback_window:] if feedback_window else None)
        view_hash = digest(before)
        available = [action for action in range(6)
                     if action not in observed_no_change.get(view_hash, set())]
        if not available:
            available = list(range(6))
        action = max(available, key=lambda candidate: action_logits[candidate])
        if rng.random() < epsilon:
            action = rng.choice(available)
        current = worker.send({"command": "step", "action": action})
        after = state(current["observation"])
        if after == before and current["reward"] <= 0:
            observed_no_change.setdefault(view_hash, set()).add(action)
        rows.append({"state": before, "action": action,
                     "model_action": model_action, "next_state": after,
                     "reward": current["reward"], "done": current["done"],
                     "prompt_sha256": prompt_hash,
                     "choice_logits": action_logits})
        if current["done"]:
            break
    return {"steps": rows, "return": sum(row["reward"] for row in rows),
            "positive": any(row["reward"] > 0 for row in rows),
            "ended": bool(rows and rows[-1]["done"])}


def load_controlled_v3(args):
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if not saved.get("order_invariant_source") or not saved.get("prefix_review_sha256"):
        raise ValueError("Checkpoint is not a controlled v3 prefix model")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    base = AutoModelForCausalLM.from_pretrained(args.model,
        torch_dtype=torch.bfloat16, local_files_only=True,
        attn_implementation="sdpa").to(args.device)
    base.config.use_cache = False
    agent = QwenRawHyperLoRA(base, order_invariant_source=True).to(args.device)
    parameters = dict(agent.named_parameters())
    with torch.no_grad():
        for name, value in saved["trainable_state"].items():
            if name not in parameters or parameters[name].shape != value.shape:
                raise ValueError("Controlled v3 architecture changed")
            parameters[name].copy_(value.to(parameters[name].device))
    agent.eval()
    return agent, tokenizer


def run(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    review = json.loads(args.annotations.read_text())
    items = review["split"]["test"][:args.limit]
    if len(items) < 2 or len({item["ruleset_id"] for item in items}) != len(items):
        raise ValueError("Need distinct reviewed test rulesets")
    if args.device.startswith("cuda"):
        torch.cuda.set_per_process_memory_fraction(args.gpu_fraction,
                                               device=args.device)
    if args.controlled_v3:
        agent, tokenizer = load_controlled_v3(args)
    else:
        agent, tokenizer, _ = load_actor(args)
    tokens = [tokenizer(str(x), add_special_tokens=False).input_ids
              for x in range(6)]
    if any(len(ids) != 1 for ids in tokens):
        raise ValueError("Six action labels must each be atomic")
    choices = [ids[0] for ids in tokens]
    sources = []
    for item in items:
        content, _ = checked_query(item["queries"][0])
        if not any(step["reward"] > 0 for episode in content["source_episodes"]
                   for step in episode["steps"]):
            raise ValueError("Reviewed source lacks positive reward")
        sources.append(content["source_episodes"])
    worker = Worker(args.xland_python, args.data_dir, args.benchmark,
                    args.environment, args.benchmark_path)
    result = {"protocol": "Exploratory XLand-100B reviewed positive source to fresh official reward; six actions; avoid repeating source-independent observed no-change actions at same local view; paired none/correct/wrong; not full 500-episode evaluation",
              "annotations_sha256": sha256(args.annotations),
              "checkpoint_sha256": sha256(args.checkpoint),
              "checkpoint_kind": ("controlled_v3_zero_shot_transfer" if
                  args.controlled_v3 else "official_history_trained"),
              "training_result_sha256": (sha256(args.training_result) if
                  args.training_result else None),
              "benchmark_sha256": sha256(args.benchmark_path),
              "model_config_sha256": sha256(args.model / "config.json"),
              "test_task_ids": [item["task_id"] for item in items],
              "test_ruleset_ids": [item["ruleset_id"] for item in items],
              "target_episodes": args.target_episodes,
              "budget": args.budget, "epsilon": args.epsilon,
              "feedback_window": args.feedback_window,
              "observed_no_change_avoidance": True,
              "action_novelty": False,
              "rows": [], "failures": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        for index, item in enumerate(items):
            rid = item["ruleset_id"]
            for target_index in range(args.target_episodes):
                try:
                    seed = args.seed + rid * 1009 + target_index
                    initial = worker.send({"command": "reset",
                        "ruleset_id": rid, "seed": seed})
                    target_state = state(initial["observation"])
                    adapters = {}
                    source_hashes = {}
                    for name, source in (("correct", sources[index]),
                                         ("wrong", sources[(index + 1) % len(items)])):
                        content = {"source_episodes": source,
                            "target_initial_state": target_state,
                            "goal": [0, 0]}
                        source_hashes[name] = digest(content)
                        with torch.no_grad():
                            adapters[name] = agent.compile_adapters(source_tensor(
                                content, args.device,
                                agent.encoder.max_source_length))
                    arms = {}
                    for name, factors in (("none", None),
                                          ("correct", adapters["correct"]),
                                          ("wrong", adapters["wrong"])):
                        arms[name] = rollout(worker, agent, tokenizer, choices,
                            rid, seed, args.budget, factors, args.device,
                            args.epsilon, args.action_seed + rid * 1009 +
                            target_index, args.feedback_window,
                            initial["observation"], args.action_novelty)
                    result["rows"].append({"task_id": item["task_id"],
                        "ruleset_id": rid, "target_episode": target_index,
                        "target_initial_sha256": digest(target_state),
                        "source_input_sha256": source_hashes,
                        "source_history_sha256": {
                            "correct": digest(sources[index]),
                            "wrong": digest(sources[(index + 1) % len(items)])},
                        "arms": arms})
                except Exception as exc:
                    result["failures"].append({"task_id": item["task_id"],
                        "ruleset_id": rid, "target_episode": target_index,
                        "error": f"{type(exc).__name__}: {exc}"})
                result["summary"] = {name: {"n": len(result["rows"]),
                    "positive": sum(row["arms"][name]["positive"]
                                    for row in result["rows"]),
                    "return": sum(row["arms"][name]["return"]
                                  for row in result["rows"])}
                    for name in ("none", "correct", "wrong")}
                temporary = args.output.with_suffix(args.output.suffix + ".tmp")
                temporary.write_text(json.dumps(result, indent=2) + "\n")
                temporary.replace(args.output)
                print(json.dumps({"rows": len(result["rows"]),
                    "failures": len(result["failures"]),
                    "summary": result["summary"]}), flush=True)
    finally:
        worker.close()
        agent.mount(None)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--training-result", type=Path)
    parser.add_argument("--controlled-v3", action="store_true")
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--benchmark-path", type=Path, required=True)
    parser.add_argument("--environment", required=True)
    parser.add_argument("--model", type=Path, default=Path(
        "current_work/delta-Mem/model/Qwen3-4B-Instruct-2507"))
    parser.add_argument("--xland-python", type=Path, default=Path(
        "ttcl/.runtime/xland_env/bin/python"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/xland_minigrid"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpu-fraction", type=float, default=.6)
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument("--target-episodes", type=int, default=2)
    parser.add_argument("--budget", type=int, default=200)
    parser.add_argument("--epsilon", type=float, default=.2)
    parser.add_argument("--feedback-window", type=int, default=4)
    parser.add_argument("--action-novelty", action="store_true")
    parser.add_argument("--seed", type=int, default=20261007)
    parser.add_argument("--action-seed", type=int, default=30261007)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.action_novelty:
        parser.error("This evaluator uses only observed no-change feedback")
    if not args.controlled_v3 and args.training_result is None:
        parser.error("Official-history checkpoint requires --training-result")
    run(args)


if __name__ == "__main__":
    main()
