"""Read-only evaluation of a frozen trajectory-generated Qwen LoRA."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ttcl.trajectory_hyperlora.train_xland_qwen_hyperlora import (
    QwenRawHyperLoRA, evaluate, queries,
)
from ttcl.trajectory_hyperlora.train_xland_raw_hyperlora import make_split


def run(args: argparse.Namespace) -> dict:
    if args.output.exists():
        raise FileExistsError(args.output)
    reference = args.reference_annotations.read_bytes()
    probe = args.probe_annotations.read_bytes()
    ref_data, probe_data = json.loads(reference), json.loads(probe)
    for name in ("dev", "test"):
        old, new = ref_data["split"][name], probe_data["split"][name]
        if ([item["item_id"] for item in old] !=
                [item["item_id"] for item in new]):
            raise ValueError("Probe item identities changed")
        for left, right in zip(old, new, strict=True):
            for arm in ("hold", "near", "tile_near"):
                before = left["arms"][arm]["queries"][0]
                after = right["arms"][arm]["queries"][0]
                if (before["target_action"] != after["target_action"] or
                        before["model_input"]["goal"] != after["model_input"]["goal"] or
                        before["model_input"]["target_initial_state"] !=
                        after["model_input"]["target_initial_state"]):
                    raise ValueError("Probe changed target public task")
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if saved["annotations_sha256"] != hashlib.sha256(reference).hexdigest():
        raise ValueError("Checkpoint training lineage changed")
    if args.device.startswith("cuda"):
        torch.cuda.set_per_process_memory_fraction(.55, device=args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    action_ids = [tokenizer(str(action), add_special_tokens=False).input_ids[0]
                  for action in (0, 3, 4)]
    base = AutoModelForCausalLM.from_pretrained(args.model,
        torch_dtype=torch.bfloat16, local_files_only=True,
        attn_implementation="sdpa").to(args.device)
    agent = QwenRawHyperLoRA(base, order_invariant_source=saved.get(
        "order_invariant_source", False)).to(args.device)
    parameters = dict(agent.named_parameters())
    with torch.no_grad():
        for name, value in saved["trainable_state"].items():
            parameters[name].copy_(value.to(parameters[name].device))
    scores = {}
    for title, data in (("original", ref_data), ("trial_order_probe", probe_data)):
        items = data["split"]["test"]
        scores[title] = evaluate(agent, make_split(items, args.device),
                                 queries(items, tokenizer), action_ids,
                                 args.device)
    result = {"checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
              "reference_annotations_sha256": hashlib.sha256(reference).hexdigest(),
              "probe_annotations_sha256": hashlib.sha256(probe).hexdigest(),
              "scores": scores}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(scores), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reference-annotations", type=Path, required=True)
    parser.add_argument("--probe-annotations", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path(
        "current_work/delta-Mem/model/Qwen3-4B-Instruct-2507"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
