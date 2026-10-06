"""Frozen medium-domain expert-action check for mixed-training checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ttcl.trajectory_hyperlora.train_xland_official_history_hyperlora import (
    evaluate, prepare,
)
from ttcl.trajectory_hyperlora.train_xland_qwen_hyperlora import QwenRawHyperLoRA


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--medium-annotations", type=Path, required=True)
    parser.add_argument("--training-annotations", type=Path, action="append",
                        required=True)
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--model", type=Path, default=Path(
        "current_work/delta-Mem/model/Qwen3-4B-Instruct-2507"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or len(args.checkpoint) != len(args.training_annotations):
        raise ValueError("Fresh output and paired checkpoints/annotations required")
    reviewed = json.loads(args.medium_annotations.read_text())
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    choices = [tokenizer(str(i), add_special_tokens=False).input_ids
               for i in range(6)]
    if any(len(tokens) != 1 for tokens in choices):
        raise ValueError("Action digits are not atomic")
    choice_ids = [tokens[0] for tokens in choices]
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    split = {name: prepare(reviewed["split"][name], tokenizer,
                           args.device, action_count=6)
             for name in ("dev", "test")}
    base = AutoModelForCausalLM.from_pretrained(args.model,
        torch_dtype=torch.bfloat16, local_files_only=True,
        attn_implementation="sdpa").to(args.device)
    base.config.use_cache = False
    agent = QwenRawHyperLoRA(base, 8, 2, 64,
        order_invariant_source=False, max_source_length=16).to(args.device)
    parameters = dict(agent.named_parameters())
    result = {"protocol": "Frozen official XLand-100B medium reviewed dev/test expert-action agreement; no medium held-out task used in mixed training",
              "medium_annotations_sha256": sha(args.medium_annotations),
              "checkpoints": {}}
    for checkpoint, training_annotations in zip(args.checkpoint,
                                                args.training_annotations):
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if saved["annotations_sha256"] != sha(training_annotations):
            raise ValueError("Checkpoint training annotation mismatch")
        with torch.no_grad():
            for name, value in saved["trainable_state"].items():
                parameters[name].copy_(value.to(parameters[name].device))
        result["checkpoints"][checkpoint.name] = {
            "checkpoint_sha256": sha(checkpoint),
            "training_annotations_sha256": sha(training_annotations),
            **{name: evaluate(agent, split[name], choice_ids, pad_id,
                              args.device, 8)
               for name in ("dev", "test")}}
        print(json.dumps({checkpoint.name: {name: {
            key: result["checkpoints"][checkpoint.name][name][key]
            for key in ("n", "correct", "wrong", "none")}
            for name in ("dev", "test")}}), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
