"""Independent reload with unseen source, feedback, query wording and numbers."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ttcl.trajectory_hyperlora.experience_pilot import (
    TRAIN_EVEN, TRAIN_ODD, action, generate, parity,
)
from ttcl.trajectory_hyperlora.pilot import TrajectoryHyperLoRA
from ttcl.trajectory_hyperlora.relational_router_pilot import (
    RelationalEncoder, export_generated_adapter, tokenize_records,
)
from ttcl.trajectory_hyperlora.stepwise_router_pilot import load_expert_bank, mount


FRESH_NUMBERS = (121, 122, 137, 138)
FRESH_OBSERVATION = (
    "Archived panel display {number}, with marker {parity}.",
    "The old gauge recorded {number}; its status was {parity}.",
)
FRESH_ACTION = ("Lever moved to {action}.", "The chosen command was {action}.")
FRESH_QUERY = (
    "The panel now reads {number} and is marked {parity}. "
    "Under its existing convention, reply with exactly LEFT or RIGHT."
)


def source_records(policy: int, rng: random.Random,
                   corrected: bool) -> list[dict[str, str]]:
    numbers = rng.sample(TRAIN_ODD, 8) + rng.sample(TRAIN_EVEN, 8)
    rng.shuffle(numbers)
    output = []
    for number in numbers:
        observation = rng.choice(FRESH_OBSERVATION).format(
            number=number, parity=parity(number))
        action_template = rng.choice(FRESH_ACTION)
        if corrected:
            output.append({"observation": observation,
                           "action": action_template.format(
                               action=action(1 - policy, number)),
                           "feedback": "invalid attempt"})
        output.append({"observation": observation,
                       "action": action_template.format(
                           action=action(policy, number)),
                       "feedback": "confirmed successful"})
    return output


def run(args: argparse.Namespace) -> dict:
    torch.manual_seed(args.seed)
    if args.device.startswith("cuda"):
        torch.cuda.set_per_process_memory_fraction(args.gpu_fraction,
                                                  device=args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, local_files_only=True,
        attn_implementation="sdpa").to(args.device)
    model.config.use_cache = False
    model.generation_config.temperature = 1.0
    model.generation_config.top_p = 1.0
    model.generation_config.top_k = 50
    agent = TrajectoryHyperLoRA(model, experts=2, rank=4, layers=2).to(args.device)
    load_expert_bank(agent, args.bank)
    encoder = RelationalEncoder(agent.model.get_input_embeddings()).to(args.device)
    weights = torch.load(args.weights, map_location="cpu", weights_only=True)
    state = encoder.load_state_dict(weights, strict=False)
    if state.unexpected_keys or state.missing_keys != ["embedding.weight"]:
        raise ValueError(f"Unexpected encoder checkpoint keys: {state}")
    encoder.eval()
    agent.eval()
    rows = []
    for corrected in (False, True):
        for policy in (0, 1):
            for source_seed in (3001, 3002):
                source = source_records(policy,
                                        random.Random(source_seed + 23 * policy),
                                        corrected)
                wrong = source_records(1 - policy,
                                       random.Random(source_seed + 23 * policy),
                                       corrected)
                with torch.no_grad():
                    coefficients = encoder(tokenize_records(tokenizer, source,
                                                            args.device))
                    wrong_coefficients = encoder(tokenize_records(tokenizer, wrong,
                                                                  args.device))
                history = "\n".join(
                    f"Observation: {step['observation']} Action: {step['action']} "
                    f"Feedback: {step['feedback']}" for step in source)
                raw = json.dumps(source, ensure_ascii=False, sort_keys=True)
                if args.export is not None and not corrected and policy == 0 and source_seed == 3001:
                    export_generated_adapter(
                        agent, coefficients, hashlib.sha256(raw.encode()).hexdigest(),
                        args.model, args.export)
                for number in FRESH_NUMBERS:
                    question = FRESH_QUERY.format(number=number,
                                                  parity=parity(number))
                    expected = action(policy, number)
                    row = {"corrected": corrected, "policy": policy,
                           "source_seed": source_seed, "number": number,
                           "expected": expected,
                           "source_sha256": hashlib.sha256(raw.encode()).hexdigest(),
                           "query_sha256": hashlib.sha256(question.encode()).hexdigest(),
                           "coefficients": coefficients.detach().float().cpu().tolist()[0]}
                    for arm in ("base", "text", "generated", "wrong_source", "expert"):
                        selected = (coefficients if arm == "generated" else
                                    wrong_coefficients if arm == "wrong_source" else
                                    torch.tensor([[float(i == policy) for i in (0, 1)]],
                                                 device=args.device) if arm == "expert" else None)
                        mount(agent, selected)
                        answer = generate(agent, tokenizer, question, args.device,
                                          history=history if arm == "text" else None)
                        first = answer.upper().split()[0].strip(".,:;!") if answer else ""
                        row[arm] = {"answer": answer, "correct": first == expected}
                    mount(agent, None)
                    rows.append(row)
    arms = ("base", "text", "generated", "wrong_source", "expert")
    result = {"n": len(rows), "seed": args.seed,
              "weights_path": str(args.weights),
              "exported_adapter": str(args.export) if args.export is not None else None,
              "source_type": "new source numbers/templates; corrected feedback uses unseen wording",
              "accuracy": {str(corrected): {
                  arm: sum(row[arm]["correct"] for row in rows
                           if row["corrected"] == corrected) / 16 for arm in arms}
                  for corrected in (False, True)},
              "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"accuracy": result["accuracy"],
                      "output": str(args.output)}), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="current_work/delta-Mem/model/Qwen3-4B-Instruct-2507")
    parser.add_argument("--bank", type=Path, default=Path(
        "results/trajectory_hyperlora/experience_pilot_20261004"))
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpu-fraction", type=float, default=0.40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--export", type=Path)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
