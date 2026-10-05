"""Exploratory official XLand-100B live rollout from empty history.

The actor chooses every source action. Only the official environment returns
transitions. A frozen hypernetwork compiles the source prefix once into LoRA;
the target episode never sees source text. All policies use the same target
reset seed and action budget. This does not train or select a checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
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
    def __init__(self, python: Path, data_dir: Path) -> None:
        environment = os.environ.copy()
        environment["XLAND_MINIGRID_DATA"] = str(data_dir.resolve())
        environment["JAX_PLATFORMS"] = "cpu"
        self.process = subprocess.Popen([str(python), "-m",
            "ttcl.trajectory_hyperlora.xland_official_live_worker"],
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
    labels = tuple(range(6))
    tokens = [tokenizer(str(x), add_special_tokens=False).input_ids
              for x in labels]
    if any(len(x) != 1 for x in tokens):
        raise ValueError("Action tokens must be atomic")
    base = AutoModelForCausalLM.from_pretrained(args.model,
        torch_dtype=torch.bfloat16, local_files_only=True,
        attn_implementation="sdpa").to(args.device)
    base.config.use_cache = False
    agent = QwenRawHyperLoRA(base, result["rank"], result["layers"],
                             result.get("width", 64), False).to(args.device)
    parameters = dict(agent.named_parameters())
    with torch.no_grad():
        for name, value in saved["trainable_state"].items():
            if name not in parameters or parameters[name].shape != value.shape:
                raise ValueError("Checkpoint architecture mismatch")
            parameters[name].copy_(value.to(parameters[name].device))
    agent.eval()
    return agent, tokenizer, [x[0] for x in tokens]


def select(agent, tokenizer, choices, observation, factors, device):
    content = {"target_initial_state": state(observation), "goal": [0, 0]}
    # Retain the exact training prompt for the five trained choices; action 5
    # is nevertheless allowed by the official environment and scored here.
    query = question(content).replace("4=put down. ",
                                      "4=put down, 5=toggle. ")
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": query}],
        tokenize=False, add_generation_prompt=True)
    ids = tokenizer(prompt, add_special_tokens=False).input_ids
    agent.mount(factors)
    with torch.no_grad():
        logits = agent.choice_logits(ids, choices, 1, device)[0]
    agent.mount(None)
    return int(logits.argmax().item()), digest(content), logits.tolist()


def episode(worker, agent, tokenizer, choices, rule_id, seed, budget,
            factors, device):
    response = worker.send({"command": "reset", "ruleset_id": rule_id,
                            "seed": seed})
    if response["num_actions"] != 6:
        raise ValueError("Unexpected official action space")
    rows = []
    for _ in range(budget):
        before = state(response["observation"])
        action, input_hash, logits = select(agent, tokenizer, choices,
                                            response["observation"], factors,
                                            device)
        response = worker.send({"command": "step", "action": action})
        rows.append({"state": before, "action": action,
                     "next_state": state(response["observation"]),
                     "reward": response["reward"], "done": response["done"],
                     "input_sha256": input_hash, "choice_logits": logits})
        if response["done"]:
            break
    return {"steps": rows, "return": sum(row["reward"] for row in rows),
            "positive_rewards": sum(row["reward"] > 0 for row in rows),
            "ended": bool(rows and rows[-1]["done"])}


def run(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    if not 1 <= args.steps <= 16:
        raise ValueError("Source encoder supports at most 16 steps")
    annotations = json.loads(args.annotations.read_text())
    test = annotations["split"]["test"][:args.limit]
    if args.task_ids:
        test = [task for task in test if task["task_id"] in args.task_ids]
    if not test:
        raise ValueError("No official test tasks selected")
    torch.cuda.set_per_process_memory_fraction(args.gpu_fraction,
                                               device=args.device)
    agent, tokenizer, choices = load_actor(args)
    worker = Worker(args.xland_python, args.data_dir)
    result = {"protocol": "Official medium-1m XLand live environment; frozen offline-trained hypernetwork; model-selected source episode from empty history; paired held-out target rollouts with generated LoRA versus no LoRA",
              "annotations_sha256": sha256(args.annotations),
              "checkpoint_sha256": sha256(args.checkpoint),
              "training_result_sha256": sha256(args.training_result),
              "model_config_sha256": sha256(args.model / "config.json"),
              "ruleset_split": "offline pilot test rulesets only",
              "source_seed": args.source_seed,
              "target_seed": args.target_seed,
              "steps_per_episode": args.steps,
              "choice_set": list(range(6)),
              "trained_choice_set": list(range(5)),
              "rows": [], "failures": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        for task in test:
            rule_id = task["ruleset_id"]
            try:
                source = episode(worker, agent, tokenizer, choices,
                    rule_id, args.source_seed + rule_id, args.steps, None,
                    args.device)
                public_steps = [{key: step[key] for key in
                    ("state", "action", "next_state", "reward", "done")}
                    for step in source["steps"]]
                model_input = {"source_episodes": [{"goal": [0, 0],
                    "steps": public_steps}],
                    "target_initial_state": source["steps"][-1]["next_state"],
                    "goal": [0, 0]}
                factors = agent.compile_adapters(
                    source_tensor(model_input, args.device))
                with_memory = episode(worker, agent, tokenizer, choices,
                    rule_id, args.target_seed + rule_id, args.steps,
                    factors, args.device)
                baseline = episode(worker, agent, tokenizer, choices,
                    rule_id, args.target_seed + rule_id, args.steps,
                    None, args.device)
                result["rows"].append({"task_id": task["task_id"],
                    "ruleset_id": rule_id,
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
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpu-fraction", type=float, default=.6)
    parser.add_argument("--source-seed", type=int, default=61005)
    parser.add_argument("--target-seed", type=int, default=71005)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--task-ids", type=int, nargs="*")
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
