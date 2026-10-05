"""Export and independently verify one generic trajectory-generated PEFT LoRA."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random

import torch
from peft import LoraConfig, PeftModel, TaskType
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from ttcl.trajectory_hyperlora.general_relation_hyperlora import first_logits
from ttcl.trajectory_hyperlora.pretrained_relation_hyperlora import PretrainedRelationHyperLoRA
from ttcl.trajectory_hyperlora.slot_lora_oracle_pilot import (
    CUES, EVAL_NUMBERS, EVAL_QUERY, action, source_records,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export(args: argparse.Namespace) -> dict:
    if args.adapter.exists():
        raise FileExistsError(args.adapter)
    if args.device.startswith("cuda"):
        torch.cuda.set_per_process_memory_fraction(args.gpu_fraction,
                                                   device=args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    base = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, local_files_only=True,
        attn_implementation="sdpa").to(args.device)
    relation_checkpoint = torch.load(args.relation_checkpoint,
                                     map_location="cpu", weights_only=True)
    head_checkpoint = torch.load(args.head_checkpoint,
                                 map_location="cpu", weights_only=True)
    if head_checkpoint["relation_checkpoint_sha256"] != sha256(args.relation_checkpoint):
        raise ValueError("Head and relation checkpoints have different lineage")
    agent = PretrainedRelationHyperLoRA(
        base, relation_checkpoint, head_checkpoint["rank"],
        head_checkpoint["layers"], head_checkpoint["latent_width"]).to(args.device)
    parameters = dict(agent.named_parameters())
    with torch.no_grad():
        for name, value in head_checkpoint["trainable_state"].items():
            if name not in parameters or parameters[name].shape != value.shape:
                raise ValueError(f"Head checkpoint mismatch: {name}")
            parameters[name].copy_(value.to(args.device))
    agent.eval()
    rule = args.rule
    source = source_records(rule, random.Random(args.source_seed), test=True)
    source_hash = hashlib.sha256(json.dumps(source, sort_keys=True).encode()).hexdigest()
    with torch.no_grad():
        factors = agent.source_factors((tokenizer, source, args.device))
        agent.mount(factors)
        rows = []
        for cue_index, cue in enumerate(CUES):
            for number in EVAL_NUMBERS["test"]:
                question = EVAL_QUERY["test"].format(number=number, cue=cue)
                logits = first_logits(agent, tokenizer, question, args.device)
                choice = int(logits.argmax(-1)[0])
                rows.append({"cue": cue, "number": number,
                             "query_sha256": hashlib.sha256(question.encode()).hexdigest(),
                             "expected": action(rule, cue_index),
                             "in_memory_first_token_id": choice,
                             "in_memory_first_token": tokenizer.decode(choice).strip()})
        agent.mount(None)
    tensors = {}
    targets = []
    first_layer = len(base.model.layers) - len(agent.adapters)
    for offset, (adapter, factor) in enumerate(zip(
            agent.adapters, factors, strict=True)):
        layer = first_layer + offset
        name = f"model.layers.{layer}.mlp.down_proj"
        targets.append(name)
        prefix = f"base_model.model.{name}"
        tensors[f"{prefix}.lora_A.weight"] = adapter.a.detach().float().cpu().contiguous()
        tensors[f"{prefix}.lora_B.weight"] = factor.squeeze(0).detach().float().cpu().contiguous()
    args.adapter.mkdir(parents=True)
    config = LoraConfig(r=head_checkpoint["rank"],
                        lora_alpha=head_checkpoint["rank"] / 2,
                        lora_dropout=0.0, target_modules=targets,
                        bias="none", task_type=TaskType.CAUSAL_LM,
                        inference_mode=True,
                        base_model_name_or_path=str(args.model.resolve()))
    config.save_pretrained(args.adapter)
    save_file(tensors, args.adapter / "adapter_model.safetensors")
    manifest = {"protocol": "One raw trajectory compiled into a standalone PEFT LoRA; test queries contain no source history",
                "rule": rule, "source_seed": args.source_seed,
                "source_sha256": source_hash,
                "relation_checkpoint_sha256": sha256(args.relation_checkpoint),
                "head_checkpoint_sha256": sha256(args.head_checkpoint),
                "rank": head_checkpoint["rank"], "targets": targets,
                "rows": rows}
    (args.adapter / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"adapter": str(args.adapter), "source_sha256": source_hash,
                      "correct": sum(row["in_memory_first_token"] == row["expected"]
                                     for row in rows), "n": len(rows)}), flush=True)
    return manifest


def verify(args: argparse.Namespace) -> dict:
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.device.startswith("cuda"):
        torch.cuda.set_per_process_memory_fraction(args.gpu_fraction,
                                                   device=args.device)
    manifest = json.loads((args.adapter / "manifest.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    base = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, local_files_only=True,
        attn_implementation="sdpa").to(args.device)
    agent = PeftModel.from_pretrained(base, args.adapter, is_trainable=False)
    agent.eval()
    rows = []
    with torch.no_grad():
        for original in manifest["rows"]:
            question = EVAL_QUERY["test"].format(number=original["number"],
                                                  cue=original["cue"])
            if hashlib.sha256(question.encode()).hexdigest() != original["query_sha256"]:
                raise ValueError("Export query content changed")
            logits = first_logits(agent, tokenizer, question, args.device)
            choice = int(logits.argmax(-1)[0])
            rows.append({"cue": original["cue"], "number": original["number"],
                         "same_first_token": choice == original["in_memory_first_token_id"],
                         "reloaded_first_token": tokenizer.decode(choice).strip(),
                         "expected": original["expected"]})
    result = {"adapter_model_sha256": sha256(args.adapter / "adapter_model.safetensors"),
              "source_sha256": manifest["source_sha256"],
              "n": len(rows),
              "same_first_token": sum(row["same_first_token"] for row in rows),
              "correct": sum(row["reloaded_first_token"] == row["expected"]
                             for row in rows),
              "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("n", "same_first_token", "correct")}),
          flush=True)
    if result["same_first_token"] != result["n"]:
        raise AssertionError("PEFT reload changed the first action token")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("export", "verify"))
    parser.add_argument("--model", type=Path, default=Path(
        "current_work/delta-Mem/model/Qwen3-4B-Instruct-2507"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpu-fraction", type=float, default=.35)
    parser.add_argument("--relation-checkpoint", type=Path)
    parser.add_argument("--head-checkpoint", type=Path)
    parser.add_argument("--rule", type=int, default=8)
    parser.add_argument("--source-seed", type=int, default=32001)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.mode == "export":
        if not args.relation_checkpoint or not args.head_checkpoint:
            parser.error("Export needs both checkpoint paths")
        export(args)
    else:
        if not args.output:
            parser.error("Verification needs --output")
        verify(args)


if __name__ == "__main__":
    main()
