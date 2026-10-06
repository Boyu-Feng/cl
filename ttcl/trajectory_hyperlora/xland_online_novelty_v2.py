"""Generic untried-action exploration for the frozen online XLand LoRA.

This is a new version of the collection policy, not a modification of the
2026-10-05 hypernetwork or its frozen online results. It masks already tried
discrete actions while choosing the next source trial; no hidden rule or
unobserved transition is read by the policy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ttcl.trajectory_hyperlora.collect_xland_multi_mechanism import ACTIONS, KINDS
from ttcl.trajectory_hyperlora.train_xland_official_history_hyperlora import source_tensor
from ttcl.trajectory_hyperlora.train_xland_qwen_hyperlora import target_prompt
from ttcl.trajectory_hyperlora.xland_online_from_empty import (
    digest, load_agent, predict, prepare, sha256, source_effect, summarize,
)


def select_untried(agent, tokenizer, choice_ids, state, goal, history,
                   used: set[int], device: str) -> tuple[int, str]:
    content = {"source_episodes": history,
               "target_initial_state": state, "goal": goal}
    prompt = target_prompt(tokenizer, content)
    if history:
        agent.mount(agent.compile_adapters(source_tensor(content, device)))
    else:
        agent.mount(None)
    with torch.no_grad():
        logits = agent.choice_logits(prompt, choice_ids, 1, device)[0].clone()
    agent.mount(None)
    allowed = [index for index, action in enumerate(ACTIONS) if action not in used]
    if not allowed:
        allowed = list(range(len(ACTIONS)))
    action = ACTIONS[max(allowed, key=lambda index: float(logits[index]))]
    return action, digest(content)


def evaluate(args: argparse.Namespace) -> dict:
    if args.output.exists():
        raise FileExistsError(args.output)
    review = json.loads(args.review.read_text())
    if (review["candidates_sha256"] != sha256(args.candidates) or
            review["reference_annotations_sha256"] !=
            sha256(args.reference_annotations) or
            review["target_count"] != 90):
        raise ValueError("Independent new online review lineage changed")
    agent, tokenizer, choice_ids = load_agent(args)
    rows = review["targets"][:args.limit] if args.limit else review["targets"]
    output = {
        "protocol": "v2 generic untried-action source exploration; frozen Qwen and hypernetwork, same independently rederived controlled XLand target labels, only chosen audited source transitions visible; target probe one-step tile effect, not official full task success",
        "review_sha256": sha256(args.review),
        "checkpoint_sha256": sha256(args.checkpoint),
        "model_config_sha256": sha256(args.model / "config.json"),
        "runner_sha256": sha256(Path(__file__)),
        "trial_budget": args.trials, "item_count": len(rows), "rows": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for item in rows:
        for kind in KINDS:
            arm = item["arms"][kind]
            history = []
            used = set()
            rounds = []
            for turn in range(args.trials + 1):
                action, target_hash = predict(agent, tokenizer, choice_ids,
                    arm["target_state"], arm["goal"], history, args.device)
                target_step = arm["target_transitions"][str(action)]
                entry = {"round": turn, "history_length": len(history),
                         "history_sha256": digest(history),
                         "target_input_sha256": target_hash,
                         "target_action": action,
                         "target_product_effect": source_effect(target_step,
                                                                arm["goal"]),
                         "target_official_reward": target_step["reward"],
                         "correct_action": arm["target_action"]}
                if turn < args.trials:
                    trial_action, source_hash = select_untried(
                        agent, tokenizer, choice_ids, arm["source_state"],
                        arm["goal"], history, used, args.device)
                    if trial_action in used:
                        raise ValueError("Novelty policy repeated an action")
                    step = arm["source_trials"][str(trial_action)]
                    used.add(trial_action)
                    history.append({"goal": arm["goal"], "steps": [step]})
                    entry.update({"own_trial_action": trial_action,
                                  "own_trial_input_sha256": source_hash,
                                  "own_trial_effect": source_effect(step,
                                                                   arm["goal"]),
                                  "own_trial_official_reward": step["reward"],
                                  "own_trial_step_sha256": digest(step)})
                rounds.append(entry)
            output["rows"].append({"item_id": item["item_id"],
                "rule_sha256": item["rule_sha256"], "kind": kind,
                "baseline_action": rounds[0]["target_action"],
                "baseline_correct": rounds[0]["target_product_effect"],
                "rounds": rounds})
        output["summary"] = summarize(output["rows"], args.trials)
        temp = args.output.with_suffix(args.output.suffix + ".tmp")
        temp.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
        temp.replace(args.output)
        print(json.dumps({"items_done": len(output["rows"]) // len(KINDS),
                          "summary": output["summary"]}), flush=True)
    return output


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
    if (args.trials != 3 or args.limit < 0 or
            not 0 < args.gpu_fraction <= 1):
        parser.error("v2 requires three trials and a positive memory budget")
    if args.command == "prepare":
        prepare(args)
    else:
        if args.checkpoint is None or args.output is None:
            parser.error("evaluate needs --checkpoint and --output")
        evaluate(args)


if __name__ == "__main__":
    main()
