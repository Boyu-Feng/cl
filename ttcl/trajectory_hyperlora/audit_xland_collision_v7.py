"""Frozen same-observation source-swap diagnostic for six-action models."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ttcl.trajectory_hyperlora.train_xland_collision_v7 import collision_pairs
from ttcl.trajectory_hyperlora.train_xland_official_history_hyperlora import (
    logits, prepare,
)
from ttcl.trajectory_hyperlora.train_xland_qwen_hyperlora import QwenRawHyperLoRA


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--model", type=Path, default=Path(
        "current_work/delta-Mem/model/Qwen3-4B-Instruct-2507"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    raw = args.annotations.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    annotations = json.loads(raw)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    choice_ids = [tokenizer(str(i), add_special_tokens=False).input_ids[0]
                  for i in range(6)]
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    split = {name: collision_pairs(prepare(annotations["split"][name],
                                     tokenizer, args.device, action_count=6))
             for name in ("dev", "test")}
    base = AutoModelForCausalLM.from_pretrained(args.model,
        torch_dtype=torch.bfloat16, local_files_only=True,
        attn_implementation="sdpa").to(args.device)
    base.config.use_cache = False
    agent = QwenRawHyperLoRA(base, 8, 2, 64,
        order_invariant_source=False, max_source_length=16).to(args.device)
    params = dict(agent.named_parameters())
    result = {"protocol": "Frozen, same-target-observation conflicting-label source swap; diagnostic only",
              "annotations_sha256": digest,
              "pairs": {name: len(rows) for name, rows in split.items()},
              "checkpoints": {}}
    for path in args.checkpoint:
        saved = torch.load(path, map_location="cpu", weights_only=True)
        if saved["annotations_sha256"] != digest:
            raise ValueError("Checkpoint and annotation mismatch")
        with torch.no_grad():
            for name, value in saved["trainable_state"].items():
                params[name].copy_(value.to(params[name].device))
        agent.eval()
        outcome = {}
        with torch.no_grad():
            for name, pairs in split.items():
                both_correct = 0
                own_correct = 0
                swapped_correct = 0
                own_margin = 0.
                swapped_margin = 0.
                for left_item, left_target, right_item, right_target in pairs:
                    # The two prompts are exactly identical; vary only source.
                    target = [left_target]
                    left = logits(agent, target,
                        agent.compile_adapters(left_item["source"]),
                        choice_ids, pad_id, args.device)[0]
                    right = logits(agent, target,
                        agent.compile_adapters(right_item["source"]),
                        choice_ids, pad_id, args.device)[0]
                    a, b = left_target["label"], right_target["label"]
                    ca = int(left.argmax().item() == a)
                    cb = int(right.argmax().item() == b)
                    own_correct += ca + cb
                    both_correct += ca * cb
                    swapped_correct += int(right.argmax().item() == a)
                    swapped_correct += int(left.argmax().item() == b)
                    own_margin += float((left[a] - left[b]).item() +
                                        (right[b] - right[a]).item())
                    swapped_margin += float((right[a] - right[b]).item() +
                                            (left[b] - left[a]).item())
                outcome[name] = {"n_pairs": len(pairs),
                    "both_correct": both_correct,
                    "own_correct_of_two": own_correct,
                    "swapped_correct_of_two": swapped_correct,
                    "own_pair_margin_sum": own_margin,
                    "swapped_pair_margin_sum": swapped_margin}
        agent.mount(None)
        result["checkpoints"][path.name] = {
            "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            **outcome}
        print(json.dumps({path.name: outcome}), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
