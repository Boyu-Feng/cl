"""Exploratory official XLand-100B live rollout from empty history.

The actor chooses every source action. Only the official environment returns
transitions. A frozen hypernetwork can compile each source prefix into LoRA;
the target episode never sees source text. All policies use the same target
reset seed and action budget. This does not train or select a checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import subprocess
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ttcl.trajectory_hyperlora.train_xland_official_history_hyperlora import (
    QwenRawHyperLoRA, question, source_tensor,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True,
        separators=(",", ":")).encode()).hexdigest()


def state(observation: list) -> dict:
    if len(observation) != 5 or any(len(row) != 5 for row in observation):
        raise ValueError("Unexpected official observation")
    return {"observation": observation, "pocket": [0, 0]}


class Worker:
    def __init__(self, python: Path, data_dir: Path, benchmark: str,
                 environment_name: str, benchmark_path: Path | None) -> None:
        environment = os.environ.copy()
        environment["XLAND_MINIGRID_DATA"] = str(data_dir.resolve())
        environment["JAX_PLATFORMS"] = "cpu"
        command = [str(python), "-m",
            "ttcl.trajectory_hyperlora.xland_official_live_worker",
            "--benchmark", benchmark, "--environment", environment_name]
        if benchmark_path:
            command.extend(("--benchmark-path", str(benchmark_path.resolve())))
        self.process = subprocess.Popen(command,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
            bufsize=1, env=environment)

    def send(self, request: dict) -> dict:
        assert self.process.stdin is not None
        assert self.process.stdout is not None
        self.process.stdin.write(json.dumps(request) + "\n")
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError(f"XLand worker exited: {self.process.poll()}")
        response = json.loads(line)
        if "error" in response:
            raise RuntimeError(response["error"])
        return response

    def close(self) -> None:
        if self.process.poll() is None:
            self.process.terminate()
            self.process.wait(timeout=10)


def load_actor(args):
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if saved["annotations_sha256"] != sha256(args.annotations):
        raise ValueError("Checkpoint annotations lineage changed")
    result = json.loads(args.training_result.read_text())
    if (result["annotations_sha256"] != saved["annotations_sha256"] or
            result["seed"] != saved["seed"]):
        raise ValueError("Training result and checkpoint mismatch")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    labels = tuple(range(result.get("action_count", 5)))
    tokens = [tokenizer(str(x), add_special_tokens=False).input_ids
              for x in labels]
    if any(len(x) != 1 for x in tokens):
        raise ValueError("Action tokens must be atomic")
    base = AutoModelForCausalLM.from_pretrained(args.model,
        torch_dtype=torch.bfloat16, local_files_only=True,
        attn_implementation="sdpa").to(args.device)
    base.config.use_cache = False
    agent = QwenRawHyperLoRA(base, result["rank"], result["layers"],
                             result.get("width", 64), False,
                             max_source_length=result.get(
                                 "max_source_length", 16)).to(args.device)
    parameters = dict(agent.named_parameters())
    with torch.no_grad():
        for name, value in saved["trainable_state"].items():
            if name not in parameters or parameters[name].shape != value.shape:
                raise ValueError("Checkpoint architecture mismatch")
            parameters[name].copy_(value.to(parameters[name].device))
    agent.eval()
    return agent, tokenizer, [x[0] for x in tokens]


def selection_prompt(tokenizer, choice_count, observation,
                     recent_feedback: list[dict] | None = None):
    content = {"target_initial_state": state(observation), "goal": [0, 0]}
    query = question(content, choice_count)
    if recent_feedback:
        feedback = "\nRecent attempts (action, reward, local view change):\n"
        for step in recent_feedback:
            feedback += (f"{step['action']}, {step['reward']}, " +
                         ("changed" if step["state"]["observation"] !=
                          step["next_state"]["observation"] else
                          "unchanged") + "\n")
        feedback += ("Use this feedback to choose the next action. A zero "
                     "reward can still mean useful movement. Reply with one "
                     "digit only.\n")
        query = query.replace("\nAction:", feedback + "\nAction:")
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": query}],
        tokenize=False, add_generation_prompt=True)
    return prompt


def select(agent, tokenizer, choices, observation, factors, device,
           recent_feedback: list[dict] | None = None):
    prompt = selection_prompt(tokenizer, len(choices), observation,
                              recent_feedback)
    ids = tokenizer(prompt, add_special_tokens=False).input_ids
    agent.mount(factors)
    with torch.no_grad():
        logits = agent.choice_logits(ids, choices, 1, device)[0]
    agent.mount(None)
    return int(logits.argmax().item()), digest(prompt), logits.tolist()


def selected_source_steps(rows: list[dict], limit: int = 16):
    """Select the latest observed success window, otherwise recent events."""
    if not rows:
        return [], (0, 0)
    positives = [i for i, step in enumerate(rows) if step["reward"] > 0]
    end = positives[-1] + 1 if positives else len(rows)
    start = max(0, end - limit)
    public = [{key: step[key] for key in
              ("state", "action", "next_state", "reward", "done")}
              for step in rows[start:end]]
    return public, (start, end)


def episode(worker, agent, tokenizer, choices, rule_id, seed, budget,
            factors, device, sample_temperature=0., sample_seed=None,
            stream_updates=False, epsilon=0., feedback_window=0):
    response = worker.send({"command": "reset", "ruleset_id": rule_id,
                            "seed": seed})
    if response["num_actions"] != 6:
        raise ValueError("Unexpected official action space")
    rng = random.Random(sample_seed) if sample_temperature > 0 or epsilon > 0 else None
    rows = []
    for _ in range(budget):
        before = state(response["observation"])
        history_hash = None
        if stream_updates and rows:
            public_steps, selected = selected_source_steps(
                rows, agent.encoder.max_source_length)
            source_input = {"source_episodes": [{"goal": [0, 0],
                "steps": public_steps}],
                "target_initial_state": before, "goal": [0, 0]}
            history_hash = digest(source_input)
            with torch.no_grad():
                factors = agent.compile_adapters(
                    source_tensor(source_input, device,
                                  agent.encoder.max_source_length))
        action, input_hash, logits = select(agent, tokenizer, choices,
                                            response["observation"], factors,
                                            device, rows[-feedback_window:]
                                            if feedback_window else None)
        if rng is not None and rng.random() < epsilon:
            action = rng.randrange(6)
        elif rng is not None and sample_temperature > 0:
            peak = max(logits)
            weights = [math.exp((value - peak) / sample_temperature)
                       for value in logits]
            action = rng.choices(range(len(choices)), weights=weights, k=1)[0]
        response = worker.send({"command": "step", "action": action})
        rows.append({"state": before, "action": action,
                     "next_state": state(response["observation"]),
                     "reward": response["reward"], "done": response["done"],
                     "input_sha256": input_hash,
                     "history_input_sha256": history_hash,
                     "history_window": list(selected) if history_hash else None,
                     "choice_logits": logits})
        if response["done"]:
            break
    return {"steps": rows, "return": sum(row["reward"] for row in rows),
            "positive_rewards": sum(row["reward"] > 0 for row in rows),
            "ended": bool(rows and rows[-1]["done"])}


def run(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    source_budget = args.source_steps or args.steps
    target_budget = args.target_steps or args.steps
    if source_budget < 1 or target_budget < 1 or \
            args.source_temperature < 0 or not 0 <= args.source_epsilon <= 1 or \
            not 0 <= args.target_epsilon <= 1 or \
            args.source_feedback_window < 0 or args.target_feedback_window < 0:
        raise ValueError("Episode budgets must be positive")
    annotations = json.loads(args.annotations.read_text())
    if args.rule_ids:
        test = [{"task_id": rule_id, "ruleset_id": rule_id}
                for rule_id in args.rule_ids]
    else:
        test = annotations["split"]["test"][:args.limit]
        if args.task_ids:
            test = [task for task in test if task["task_id"] in args.task_ids]
    if not test:
        raise ValueError("No official test tasks selected")
    torch.cuda.set_per_process_memory_fraction(args.gpu_fraction,
                                               device=args.device)
    agent, tokenizer, choices = load_actor(args)
    worker = Worker(args.xland_python, args.data_dir, args.benchmark,
                    args.environment, args.benchmark_path)
    result = {"protocol": "Version-bound XLand live environment; frozen offline-trained hypernetwork; model-selected source episode from empty history; paired held-out target rollouts with generated LoRA versus no LoRA",
              "annotations_sha256": sha256(args.annotations),
              "checkpoint_sha256": sha256(args.checkpoint),
              "training_result_sha256": sha256(args.training_result),
              "model_config_sha256": sha256(args.model / "config.json"),
              "ruleset_split": ("explicit exploratory ruleset IDs" if
                  args.rule_ids else "offline pilot test rulesets only"),
              "benchmark": args.benchmark,
              "benchmark_path": (str(args.benchmark_path.resolve()) if
                                  args.benchmark_path else None),
              "benchmark_sha256": (sha256(args.benchmark_path) if
                                   args.benchmark_path else None),
              "environment": args.environment,
              "source_seed": args.source_seed,
              "target_seed": args.target_seed,
              "steps_per_episode": args.steps,
              "source_steps": source_budget,
              "target_steps": target_budget,
              "source_temperature": args.source_temperature,
              "source_epsilon": args.source_epsilon,
              "target_epsilon": args.target_epsilon,
              "target_sampling_seed": args.target_sampling_seed,
              "source_feedback_window": args.source_feedback_window,
              "target_feedback_window": args.target_feedback_window,
              "source_sampling_seed": args.source_sampling_seed,
              "stream_source_updates": args.stream_source_updates,
              "choice_set": list(range(len(choices))),
              "environment_choice_set": list(range(6)),
              "rows": [], "failures": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        for task in test:
            rule_id = task["ruleset_id"]
            try:
                source = episode(worker, agent, tokenizer, choices,
                    rule_id, args.source_seed + rule_id, source_budget, None,
                    args.device, args.source_temperature,
                    args.source_sampling_seed + rule_id,
                    args.stream_source_updates, args.source_epsilon,
                    args.source_feedback_window)
                public_steps, selected_window = selected_source_steps(
                    source["steps"], agent.encoder.max_source_length)
                model_input = {"source_episodes": [{"goal": [0, 0],
                    "steps": public_steps}],
                    "target_initial_state": source["steps"][-1]["next_state"],
                    "goal": [0, 0]}
                with torch.no_grad():
                    factors = agent.compile_adapters(
                        source_tensor(model_input, args.device,
                                      agent.encoder.max_source_length))
                with_memory = episode(worker, agent, tokenizer, choices,
                    rule_id, args.target_seed + rule_id, target_budget,
                    factors, args.device,
                    sample_seed=args.target_sampling_seed + rule_id,
                    epsilon=args.target_epsilon,
                    feedback_window=args.target_feedback_window)
                baseline = episode(worker, agent, tokenizer, choices,
                    rule_id, args.target_seed + rule_id, target_budget,
                    None, args.device,
                    sample_seed=args.target_sampling_seed + rule_id,
                    epsilon=args.target_epsilon,
                    feedback_window=args.target_feedback_window)
                result["rows"].append({"task_id": task["task_id"],
                    "ruleset_id": rule_id,
                    "selected_source_window": list(selected_window),
                    "source_input_sha256": digest(model_input),
                    "source": source, "with_memory": with_memory,
                    "baseline": baseline})
            except Exception as exc:
                result["failures"].append({"task_id": task["task_id"],
                    "ruleset_id": rule_id,
                    "error": f"{type(exc).__name__}: {exc}"})
            result["summary"] = summarize(result["rows"])
            temporary = args.output.with_suffix(args.output.suffix + ".tmp")
            temporary.write_text(json.dumps(result, indent=2) + "\n")
            temporary.replace(args.output)
            print(json.dumps({"finished": len(result["rows"]),
                              "failures": len(result["failures"]),
                              "summary": result["summary"]}), flush=True)
    finally:
        worker.close()
    return result


def summarize(rows):
    return {"n": len(rows),
            "source_positive": sum(row["source"]["positive_rewards"] > 0
                                   for row in rows),
            "with_memory_positive": sum(row["with_memory"][
                "positive_rewards"] > 0 for row in rows),
            "baseline_positive": sum(row["baseline"][
                "positive_rewards"] > 0 for row in rows),
            "with_memory_return": sum(row["with_memory"]["return"]
                                      for row in rows),
            "baseline_return": sum(row["baseline"]["return"]
                                   for row in rows),
            "first_action_changed": sum(row["with_memory"]["steps"][0][
                "action"] != row["baseline"]["steps"][0]["action"]
                for row in rows)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, default=Path(
        "data/annotations/xland_official_history_reviewed_64_v1_20261005.json"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--training-result", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path(
        "current_work/delta-Mem/model/Qwen3-4B-Instruct-2507"))
    parser.add_argument("--xland-python", type=Path, default=Path(
        "ttcl/.runtime/xland_env/bin/python"))
    parser.add_argument("--data-dir", type=Path,
                        default=Path("data/xland_minigrid"))
    parser.add_argument("--benchmark", default="medium-1m")
    parser.add_argument("--benchmark-path", type=Path)
    parser.add_argument("--environment", default="XLand-MiniGrid-R1-13x13")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpu-fraction", type=float, default=.6)
    parser.add_argument("--source-seed", type=int, default=61005)
    parser.add_argument("--target-seed", type=int, default=71005)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--source-steps", type=int)
    parser.add_argument("--target-steps", type=int)
    parser.add_argument("--source-temperature", type=float, default=0.)
    parser.add_argument("--source-epsilon", type=float, default=0.)
    parser.add_argument("--target-epsilon", type=float, default=0.)
    parser.add_argument("--target-sampling-seed", type=int, default=91005)
    parser.add_argument("--source-feedback-window", type=int, default=0)
    parser.add_argument("--target-feedback-window", type=int, default=0)
    parser.add_argument("--source-sampling-seed", type=int, default=81005)
    parser.add_argument("--stream-source-updates", action="store_true")
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--task-ids", type=int, nargs="*")
    parser.add_argument("--rule-ids", type=int, nargs="*")
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
