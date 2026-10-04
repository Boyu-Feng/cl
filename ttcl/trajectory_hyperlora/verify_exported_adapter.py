"""Verify a generated PEFT LoRA independently of its hypernetwork process."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from ttcl.trajectory_hyperlora.experience_pilot import parity
from ttcl.trajectory_hyperlora.pilot import prompt
from ttcl.trajectory_hyperlora.verify_relational_router import (
    FRESH_NUMBERS, FRESH_QUERY,
)


def run(args: argparse.Namespace) -> dict:
    if args.device.startswith("cuda"):
        torch.cuda.set_per_process_memory_fraction(args.gpu_fraction,
                                                  device=args.device)
    reference = json.loads(args.reference.read_text())
    manifest = json.loads((args.adapter / "manifest.json").read_text())
    source_rows = [row for row in reference["rows"]
                   if not row["corrected"] and row["policy"] == 0
                   and row["source_seed"] == 3001]
    if len(source_rows) != len(FRESH_NUMBERS):
        raise ValueError("Reference must contain the complete exported source probe")
    if {row["source_sha256"] for row in source_rows} != {manifest["source_sha256"]}:
        raise ValueError("Exported adapter is bound to a different source")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    base = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, local_files_only=True,
        attn_implementation="sdpa").to(args.device)
    model = PeftModel.from_pretrained(base, args.adapter, is_trainable=False)
    model.eval()
    model.generation_config.temperature = 1.0
    model.generation_config.top_p = 1.0
    model.generation_config.top_k = 50
    rows = []
    for number in FRESH_NUMBERS:
        question = FRESH_QUERY.format(number=number, parity=parity(number))
        prior = next(row for row in source_rows if row["number"] == number)
        if hashlib.sha256(question.encode()).hexdigest() != prior["query_sha256"]:
            raise ValueError("Reference query content binding failed")
        inputs = tokenizer(prompt(tokenizer, question), return_tensors="pt").to(args.device)
        with torch.no_grad():
            output = model.generate(**inputs, do_sample=False, max_new_tokens=5,
                                    pad_token_id=tokenizer.eos_token_id)
        answer = tokenizer.decode(output[0, inputs.input_ids.shape[1]:],
                                  skip_special_tokens=True).strip()
        rows.append({"number": number, "expected": prior["expected"],
                     "reloaded": answer, "in_memory": prior["generated"]["answer"],
                     "same_output": answer == prior["generated"]["answer"],
                     "correct": answer == prior["expected"]})
    result = {"adapter": str(args.adapter),
              "source_sha256": manifest["source_sha256"],
              "n": len(rows),
              "same_output": sum(row["same_output"] for row in rows),
              "correct": sum(row["correct"] for row in rows),
              "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"same_output": result["same_output"],
                      "correct": result["correct"],
                      "n": result["n"], "output": str(args.output)}), flush=True)
    if result["same_output"] != result["n"]:
        raise AssertionError("Exported adapter changed the answer")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="current_work/delta-Mem/model/Qwen3-4B-Instruct-2507")
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpu-fraction", type=float, default=0.40)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
