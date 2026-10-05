"""One-shot exploratory evaluation of dev-selected conflict checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ttcl.trajectory_hyperlora.train_xland_conflict_hyperlora import (
    evaluate as evaluate_pairs, prepare as prepare_pairs, sha256,
)
from ttcl.trajectory_hyperlora.train_xland_official_history_hyperlora import (
    QwenRawHyperLoRA, evaluate as evaluate_all, prepare as prepare_all,
)


def run(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    pairs_raw = args.pairs.read_bytes()
    pairs = json.loads(pairs_raw)
    annotations = json.loads(args.annotations.read_text())
    result = json.loads(args.training_result.read_text())
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    initial = json.loads(args.init_result.read_text())
    if (saved["pairs_sha256"] != sha256(args.pairs) or
            result["pairs_sha256"] != saved["pairs_sha256"] or
            result["best_step"] != saved["best_step"] or
            pairs["source_annotations_sha256"] != sha256(args.annotations)):
        raise ValueError("Frozen result lineage changed")
    torch.cuda.set_per_process_memory_fraction(args.gpu_fraction,
                                               device=args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    choices = [tokenizer(str(i), add_special_tokens=False).input_ids
               for i in range(5)]
    if any(len(ids) != 1 for ids in choices):
        raise ValueError("Non-atomic action token")
    choice_ids = [ids[0] for ids in choices]
    conflict = prepare_pairs(pairs["split"]["test"], tokenizer, args.device)
    full = prepare_all(annotations["split"]["test"], tokenizer, args.device)
    base = AutoModelForCausalLM.from_pretrained(args.model,
        torch_dtype=torch.bfloat16, local_files_only=True,
        attn_implementation="sdpa").to(args.device)
    base.config.use_cache = False
    agent = QwenRawHyperLoRA(base, initial["rank"], initial["layers"],
                             initial.get("width", 64), False).to(args.device)
    params = dict(agent.named_parameters())
    with torch.no_grad():
        for name, value in saved["trainable_state"].items():
            if name not in params or params[name].shape != value.shape:
                raise ValueError("Checkpoint architecture changed")
            params[name].copy_(value.to(params[name].device))
    output = {"protocol": "Dev-selected conflict checkpoint, evaluated once on previously exposed official pilot test rulesets; exploratory only",
              "checkpoint_sha256": sha256(args.checkpoint),
              "pairs_sha256": sha256(args.pairs),
              "annotations_sha256": sha256(args.annotations),
              "best_step": saved["best_step"],
              "conflict_test": evaluate_pairs(agent, conflict, choice_ids,
                                              args.device),
              "full_test": evaluate_all(agent, full, choice_ids,
                                        tokenizer.pad_token_id or
                                        tokenizer.eos_token_id,
                                        args.device, args.batch_size)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({key: output[key] for key in
                      ("best_step", "conflict_test", "full_test") if
                      key != "full_test"} | {"full_test": {key:
                      output["full_test"][key] for key in
                      ("n", "correct", "wrong", "none",
                       "source_swap_changes")}}), flush=True)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, default=Path(
        "data/annotations/xland_official_history_reviewed_64_v1_20261005.json"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--training-result", type=Path, required=True)
    parser.add_argument("--init-result", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path(
        "current_work/delta-Mem/model/Qwen3-4B-Instruct-2507"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpu-fraction", type=float, default=.6)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
