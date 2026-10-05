"""Evaluate frozen XLand hyper-LoRA from zero history through own trials.

Each selected action receives one audited deterministic environment transition.
Unchosen transitions are held only by the evaluator and never enter model input.
This is the previous controlled one-step XLand setting, not official full-task
rollout. Trial choices are made by the frozen actor, not pre-scripted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ttcl.trajectory_hyperlora.collect_xland_multi_mechanism import ACTIONS, KINDS
from ttcl.trajectory_hyperlora.review_xland_multi_mechanism import validate
from ttcl.trajectory_hyperlora.review_xland_rule_pairs import has_product
from ttcl.trajectory_hyperlora.train_xland_official_history_hyperlora import source_tensor
from ttcl.trajectory_hyperlora.train_xland_qwen_hyperlora import (
    QwenRawHyperLoRA, target_prompt,
)


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True,
                                    separators=(",", ":")).encode()).hexdigest()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(args: argparse.Namespace) -> dict:
    if args.review.exists():
        raise FileExistsError(args.review)
    raw = args.candidates.read_bytes()
    candidate = json.loads(raw)
    independently_reviewed = validate(candidate)
    fixed = json.loads(args.reference_annotations.read_text())
    if fixed["candidates_sha256"] != hashlib.sha256(raw).hexdigest() or \
            [x["item_id"] for x in fixed["split"]["test"]] != \
            [x["item_id"] for x in independently_reviewed["test"]]:
        raise ValueError("Existing split lineage changed")
    targets = []
    for item in candidate["split"]["test"]:
        arms = {}
        for kind in KINDS:
            arm = item["arms"][kind]
            history = arm["source_episodes"]
            trials = {str(action): history[i]["steps"][0]
                      for i, action in enumerate(ACTIONS)}
            target = {str(action): arm["target_transitions"][i]
                      for i, action in enumerate(ACTIONS)}
            product = item["product_tile"]
            labels = [action for action in ACTIONS
                      if has_product(target[str(action)]["next_state"], product)]
            if len(labels) != 1 or labels[0] != arm["target_action"]:
                raise ValueError("Online target not independently rederived")
            source_state = trials[str(ACTIONS[0])]["state"]
            if any(step["state"] != source_state for step in trials.values()):
                raise ValueError("Source actions do not start identically")
            arms[kind] = {"goal": product, "source_state": source_state,
                          "target_state": target[str(ACTIONS[0])]["state"],
                          "source_trials": trials,
                          "target_transitions": target,
                          "target_action": labels[0],
                          "reviewed_target": True,
                          "review_basis": "Independent deterministic environment transition/effect audit; each online prefix bound at use time"}
        targets.append({"item_id": item["item_id"],
                        "rule_sha256": item["rule_sha256"],
                        "arms": arms})
    result = {"protocol": "New reviewed targets for from-empty adaptive source trials; model-selected actions only, exact dynamic input bindings",
              "candidates_sha256": hashlib.sha256(raw).hexdigest(),
              "reference_annotations_sha256": sha256(args.reference_annotations),
              "target_count": len(targets) * len(KINDS),
              "trial_action_set": ACTIONS,
              "targets": targets}
    args.review.parent.mkdir(parents=True, exist_ok=True)
    args.review.write_text(json.dumps(result, ensure_ascii=False,
                                      indent=2) + "\n")
    return result


def load_agent(args: argparse.Namespace):
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if (saved["annotations_sha256"] != sha256(args.reference_annotations) or
            saved.get("order_invariant_source") is not True):
        raise ValueError("Expected frozen order-invariant XLand checkpoint")
    if args.device.startswith("cuda"):
        torch.cuda.set_per_process_memory_fraction(args.gpu_fraction,
                                                   device=args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    choices = [tokenizer(str(action), add_special_tokens=False).input_ids
               for action in ACTIONS]
    if any(len(ids) != 1 for ids in choices):
        raise ValueError("Each action must use one output token")
    choice_ids = [ids[0] for ids in choices]
    base = AutoModelForCausalLM.from_pretrained(args.model,
        torch_dtype=torch.bfloat16, local_files_only=True,
        attn_implementation="sdpa").to(args.device)
    base.config.use_cache = False
    agent = QwenRawHyperLoRA(base, order_invariant_source=True).to(args.device)
    params = dict(agent.named_parameters())
    with torch.no_grad():
        for name, value in saved["trainable_state"].items():
            if name not in params or params[name].shape != value.shape:
                raise ValueError("Checkpoint architecture changed")
            params[name].copy_(value.to(params[name].device))
    agent.eval()
    return agent, tokenizer, choice_ids


def predict(agent, tokenizer, choice_ids: list[int], state: dict,
            goal: list[int], history: list[dict], device: str) -> tuple[int, str]:
    model_input = {"source_episodes": history,
                   "target_initial_state": state, "goal": goal}
    prompt = target_prompt(tokenizer, model_input)
    if history:
        factors = agent.compile_adapters(source_tensor(model_input, device))
        agent.mount(factors)
    else:
        agent.mount(None)
    with torch.no_grad():
        prediction = int(agent.choice_logits(prompt, choice_ids, 1,
                                              device).argmax(-1)[0])
    agent.mount(None)
    return ACTIONS[prediction], digest(model_input)


def source_effect(step: dict, product: list[int]) -> bool:
    return bool(has_product(step["next_state"], product))


def evaluate(args: argparse.Namespace) -> dict:
    if args.output.exists():
        raise FileExistsError(args.output)
    review = json.loads(args.review.read_text())
    if (review["candidates_sha256"] != sha256(args.candidates) or
            review["reference_annotations_sha256"] !=
            sha256(args.reference_annotations)):
        raise ValueError("Online review lineage changed")
    agent, tokenizer, choice_ids = load_agent(args)
    rows = review["targets"][:args.limit] if args.limit else review["targets"]
    report = {"protocol": "Frozen Qwen hyper-LoRA in controlled XLand one-step tasks: zero source at round 0, then own model-selected audited source trial after each round; future target probed without updating memory",
              "review_sha256": sha256(args.review),
              "checkpoint_sha256": sha256(args.checkpoint),
              "model_config_sha256": sha256(args.model / "config.json"),
              "seed": torch.load(args.checkpoint, map_location="cpu",
                                 weights_only=True)["seed"],
              "trial_budget": args.trials,
              "item_count": len(rows), "rows": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for item in rows:
        for kind in KINDS:
            arm = item["arms"][kind]
            history = []
            baseline, baseline_hash = predict(agent, tokenizer, choice_ids,
                arm["target_state"], arm["goal"], [], args.device)
            rounds = []
            for t in range(args.trials + 1):
                action, target_input_hash = predict(agent, tokenizer, choice_ids,
                    arm["target_state"], arm["goal"], history, args.device)
                target_step = arm["target_transitions"][str(action)]
                entry = {"round": t, "history_length": len(history),
                         "history_sha256": digest(history),
                         "target_input_sha256": target_input_hash,
                         "target_action": action,
                         "target_product_effect": source_effect(target_step,
                                                                arm["goal"]),
                         "target_official_reward": target_step["reward"],
                         "correct_action": arm["target_action"]}
                if t < args.trials:
                    trial_action, source_input_hash = predict(
                        agent, tokenizer, choice_ids, arm["source_state"],
                        arm["goal"], history, args.device)
                    step = arm["source_trials"][str(trial_action)]
                    # Only this chosen environment transition becomes visible.
                    history.append({"goal": arm["goal"],
                                    "steps": [step]})
                    entry["own_trial_action"] = trial_action
                    entry["own_trial_input_sha256"] = source_input_hash
                    entry["own_trial_effect"] = source_effect(step,
                                                              arm["goal"])
                    entry["own_trial_official_reward"] = step["reward"]
                    entry["own_trial_step_sha256"] = digest(step)
                rounds.append(entry)
            if rounds[0]["target_action"] != baseline or \
                    rounds[0]["target_input_sha256"] != baseline_hash:
                raise RuntimeError("Empty-history policy is not reproducible")
            report["rows"].append({"item_id": item["item_id"],
                "rule_sha256": item["rule_sha256"], "kind": kind,
                "baseline_action": baseline,
                "baseline_correct": baseline == arm["target_action"],
                "rounds": rounds})
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        report["summary"] = summarize(report["rows"], args.trials)
        temporary.write_text(json.dumps(report, ensure_ascii=False,
                                        indent=2) + "\n")
        temporary.replace(args.output)
        print(json.dumps({"items_done": len(report["rows"]) // len(KINDS),
                          "summary": report["summary"]}), flush=True)
    return report


def summarize(rows: list[dict], trials: int) -> dict:
    n = len(rows)
    return {"n": n, "by_round": [{"round": t,
        "correct": sum(row["rounds"][t]["target_product_effect"] for row in rows),
        "official_reward_sum": sum(row["rounds"][t]["target_official_reward"]
                                   for row in rows),
        "source_actions_distinct_mean": (
            sum(len({entry.get("own_trial_action") for entry in
                     row["rounds"][:t]}) for row in rows) / n if n else 0)}
        for t in range(trials + 1)],
        "final_vs_empty": {"gains": sum(not row["rounds"][0][
            "target_product_effect"] and row["rounds"][-1][
                "target_product_effect"] for row in rows),
            "losses": sum(row["rounds"][0]["target_product_effect"] and
                          not row["rounds"][-1]["target_product_effect"]
                          for row in rows)}}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "evaluate"))
    parser.add_argument("--candidates", type=Path, default=Path(
        "results/trajectory_hyperlora/xland_multi_mechanism_candidates_v1_20261005.json"))
    parser.add_argument("--reference-annotations", type=Path, default=Path(
        "data/annotations/xland_multi_mechanism_reviewed_v1_20261005.json"))
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--model", type=Path, default=Path(
        "current_work/delta-Mem/model/Qwen3-4B-Instruct-2507"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpu-fraction", type=float, default=.55)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.trials < 1 or args.trials > 16 or args.limit < 0 or \
            not 0 < args.gpu_fraction <= 1:
        parser.error("Invalid online trial budget")
    if args.command == "prepare":
        prepare(args)
    else:
        if args.checkpoint is None or args.output is None:
            parser.error("evaluate requires --checkpoint and --output")
        evaluate(args)


if __name__ == "__main__":
    main()
