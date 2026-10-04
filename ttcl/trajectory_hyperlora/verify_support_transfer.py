"""Reload local trajectory LoRAs and score entirely new readings/wording."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from ttcl.trajectory_hyperlora.experience_pilot import action, parity
from ttcl.trajectory_hyperlora.pilot import prompt
from ttcl.trajectory_hyperlora.support_transfer_pilot import source_text


FRESH_NUMBERS = (121, 122, 137, 138)


def fresh_query(number: int) -> str:
    return (f"Sensor reading {number} is {parity(number)}. Under this panel's "
            "learned convention, which lever should be pressed? "
            "Reply with exactly LEFT or RIGHT.")


def run(args: argparse.Namespace) -> dict:
    torch.cuda.set_per_process_memory_fraction(args.gpu_fraction, device=args.device)
    old_metrics = json.loads(Path(args.training_metrics).read_text())
    for policy in (0, 1):
        assert not set(FRESH_NUMBERS).intersection(
            old_metrics["support"][str(policy)]["numbers"])
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    base = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, local_files_only=True,
        attn_implementation="sdpa"
    ).to(args.device)
    base.generation_config.temperature = 1.0
    base.generation_config.top_p = 1.0
    base.generation_config.top_k = 50
    root = Path(args.adapter_root)
    model = PeftModel.from_pretrained(base, root / "trajectory_adapter_0",
                                      adapter_name="trajectory_0", is_trainable=False)
    model.load_adapter(root / "trajectory_adapter_1", adapter_name="trajectory_1",
                       is_trainable=False)
    rows = []
    for policy in (0, 1):
        numbers = old_metrics["support"][str(policy)]["numbers"]
        history = source_text(policy, numbers)
        assert hashlib.sha256(history.encode()).hexdigest() == old_metrics["support"][str(policy)]["source_sha256"]
        for number in FRESH_NUMBERS:
            question = fresh_query(number)
            expected = action(policy, number)
            row = {"policy": policy, "number": number, "expected": expected,
                   "question_sha256": hashlib.sha256(question.encode()).hexdigest()}
            for arm in ("base", "text", "trajectory_lora", "wrong_lora"):
                if arm == "base" or arm == "text":
                    model.disable_adapter_layers()
                else:
                    model.enable_adapter_layers()
                    selected = policy if arm == "trajectory_lora" else 1 - policy
                    model.set_adapter(f"trajectory_{selected}")
                rendered = prompt(tokenizer, question,
                                  history if arm == "text" else None)
                inputs = tokenizer(rendered, return_tensors="pt").to(args.device)
                with torch.no_grad():
                    output = model.generate(**inputs, do_sample=False, max_new_tokens=5,
                                            pad_token_id=tokenizer.eos_token_id)
                answer = tokenizer.decode(output[0, inputs.input_ids.shape[1]:],
                                          skip_special_tokens=True).strip()
                first = answer.upper().split()[0].strip(".,:;!") if answer else ""
                row[arm] = {"answer": answer, "correct": first == expected}
            rows.append(row)
    arms = ("base", "text", "trajectory_lora", "wrong_lora")
    metrics = {"n": len(rows),
               "accuracy": {arm: sum(row[arm]["correct"] for row in rows) / len(rows)
                            for arm in arms},
               "rows": rows,
               "protocol": "independent PEFT reload; fresh numbers and query wording"}
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps({"accuracy": metrics["accuracy"], "output": str(output_path)}), flush=True)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="current_work/delta-Mem/model/Qwen3-4B-Instruct-2507")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpu-fraction", type=float, default=0.40)
    parser.add_argument("--adapter-root", default="results/trajectory_hyperlora/experience_pilot_20261004")
    parser.add_argument("--training-metrics", default="results/trajectory_hyperlora/experience_pilot_20261004/support_seed42.json")
    parser.add_argument("--output", default="results/trajectory_hyperlora/experience_pilot_20261004/fresh_reload_seed42.json")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
