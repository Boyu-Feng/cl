"""Text controls on the reviewed XLand three-mechanism LoRA test set.

Both controls derive solely from the same public source transitions as the
hypernetwork; the second compresses them into action/effect evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ttcl.trajectory_hyperlora.evaluate_xland_memrl import public_trace
from ttcl.trajectory_hyperlora.train_xland_qwen_hyperlora import (
    ACTIONS, ARMS, target_question,
)
from ttcl.trajectory_hyperlora.train_xland_raw_hyperlora import checked_query


def count_goal(state: dict, goal: list[int]) -> int:
    tile = tuple(goal)
    return sum(tuple(value) == tile for row in state["observation"]
               for value in row) + int(tuple(state["pocket"]) == tile)


def evidence_text(content: dict) -> str:
    lines = ["Previous one-step trials in another layout, from the same hidden rule. "
             "Each effect below was calculated from the observed before/after state, "
             "not from the target answer:"]
    for episode in content["source_episodes"]:
        for step in episode["steps"]:
            goal = episode["goal"]
            before, after = step["state"], step["next_state"]
            delta = count_goal(after, goal) - count_goal(before, goal)
            pocket = tuple(after["pocket"]) == tuple(goal)
            lines.append(f"Action {step['action']}: target tile count changed by "
                         f"{delta:+d}; target in pocket afterward: "
                         f"{'yes' if pocket else 'no'}; environment reward "
                         f"{step['reward']:.3f}.")
    lines.append("Use these observed action effects to choose the target action "
                 "in the new layout. A target tile in the pocket also counts as produced.")
    return "\n".join(lines)


def explicit_effect_text(content: dict) -> str:
    """Strong source-only text control: name the positive observed action."""
    effects = []
    for episode in content["source_episodes"]:
        for step in episode["steps"]:
            delta = count_goal(step["next_state"], episode["goal"]) - \
                count_goal(step["state"], episode["goal"])
            effects.append((step["action"], delta))
    positive = [action for action, delta in effects if delta > 0]
    if len(positive) != 1:
        raise ValueError("Source trials do not identify one observed positive effect")
    return ("Observed in previous trials under the same hidden rule: "
            f"action {positive[0]} increased the number of target tiles; "
            "the other actions did not. For the new layout, choose the action "
            "that had the positive observed effect. This conclusion uses only "
            "source transitions, not any target answer.")


def choose(model, tokenizer, content: dict, history: str,
           choice_ids: list[int], device: str) -> tuple[int, int]:
    prompt = ("Prior trials:\n" + history + "\n\n" if history else "") + \
        target_question(content)
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False,
        add_generation_prompt=True)
    ids = tokenizer(text, add_special_tokens=False,
                    return_tensors="pt").input_ids.to(device)
    with torch.inference_mode():
        logits = model(input_ids=ids, use_cache=False).logits[0, -1, choice_ids]
    return ACTIONS[int(logits.argmax())], ids.shape[1]


def run(args: argparse.Namespace) -> dict:
    if args.output.exists():
        raise FileExistsError(args.output)
    raw = args.annotations.read_bytes()
    reviewed = json.loads(raw)
    if len(reviewed["split"]["test"]) != 30:
        raise ValueError("Unexpected reviewed test group count")
    if args.device.startswith("cuda"):
        torch.cuda.set_per_process_memory_fraction(args.gpu_fraction,
                                                   device=args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    choices = [tokenizer(str(action), add_special_tokens=False).input_ids
               for action in ACTIONS]
    if any(len(tokens) != 1 for tokens in choices):
        raise ValueError("Action labels are not single tokens")
    choice_ids = [tokens[0] for tokens in choices]
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, local_files_only=True,
        attn_implementation="sdpa").to(args.device).eval()
    output = {
        "protocol": "Reviewed controlled XLand one-step target action; frozen Qwen no LoRA; same public source transitions as LoRA; direct JSON trace versus deterministic public action/effect text and explicit observed positive effect; target labels never enter text; three seen mechanisms, held-out rule content",
        "annotations_sha256": hashlib.sha256(raw).hexdigest(),
        "model_config_sha256": hashlib.sha256(
            (args.model / "config.json").read_bytes()).hexdigest(),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "rows": [],
    }
    for item in reviewed["split"]["test"]:
        for arm in ARMS:
            content, target = checked_query(item["arms"][arm]["queries"][0])
            arms = {}
            for name, history in (("none", ""),
                                  ("raw", public_trace(content)),
                                  ("evidence", evidence_text(content)),
                                  ("explicit_effect", explicit_effect_text(content))):
                action, tokens = choose(model, tokenizer, content, history,
                                        choice_ids, args.device)
                arms[name] = {"action": action, "prompt_tokens": tokens,
                              "text_sha256": hashlib.sha256(
                                  history.encode()).hexdigest()}
            output["rows"].append({"item_id": item["item_id"], "arm": arm,
                                   "target_action": target, "arms": arms})
        output["summary"] = {
            "n": len(output["rows"]),
            **{name: sum(row["arms"][name]["action"] == row["target_action"]
                         for row in output["rows"])
               for name in ("none", "raw", "evidence", "explicit_effect")},
            "mean_prompt_tokens": {
                name: sum(row["arms"][name]["prompt_tokens"]
                          for row in output["rows"]) / len(output["rows"])
                for name in ("none", "raw", "evidence", "explicit_effect")},
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temp = args.output.with_suffix(args.output.suffix + ".tmp")
        temp.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
        temp.replace(args.output)
        print(json.dumps(output["summary"]), flush=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, default=Path(
        "data/annotations/xland_multi_mechanism_reviewed_v1_20261005.json"))
    parser.add_argument("--model", type=Path, default=Path(
        "current_work/delta-Mem/model/Qwen3-4B-Instruct-2507"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpu-fraction", type=float, default=.55)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 0 < args.gpu_fraction <= 1:
        parser.error("Invalid GPU memory fraction")
    run(args)


if __name__ == "__main__":
    main()
