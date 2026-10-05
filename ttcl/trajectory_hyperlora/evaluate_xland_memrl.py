"""Adapt the repository's native MemRL memory service to XLand source traces.

The frozen Qwen reader sees only MemRL-retrieved abstracts and the target
public state. Each test arm starts from the same train-memory snapshot, then
observes its own source trajectory before the target query.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ttcl.memrl_comparison.memory import Memory
from ttcl.trajectory_hyperlora.train_xland_qwen_hyperlora import (
    ACTIONS, ARMS, target_question,
)
from ttcl.trajectory_hyperlora.train_xland_raw_hyperlora import checked_query


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True,
                                    separators=(",", ":")).encode()).hexdigest()


def memory_query(content: dict) -> str:
    return ("XLand: choose the first action that creates target tile "
            f"({content['goal'][0]},{content['goal'][1]}).")


def public_trace(content: dict) -> str:
    lines = []
    for i, episode in enumerate(content["source_episodes"]):
        for step in episode["steps"]:
            lines.append(json.dumps({"trial": i, "goal": episode["goal"],
                "observation": step["state"]["observation"],
                "pocket": step["state"]["pocket"],
                "action": step["action"],
                "next_observation": step["next_state"]["observation"],
                "next_pocket": step["next_state"]["pocket"],
                "environment_reward": step["reward"], "done": step["done"]},
                separators=(",", ":")))
    return "\n".join(lines)


def source_reward(content: dict) -> float:
    return sum(step["reward"] for episode in content["source_episodes"]
               for step in episode["steps"])


class LocalQwenClient:
    def __init__(self, model, tokenizer, device: str, repeat: int) -> None:
        self.model, self.tokenizer = model, tokenizer
        self.device, self.repeat = device, repeat
        self.writer_calls = 0

    def complete(self, messages, seed, *, tokens, temperature, top_p):
        # MemRL's upstream script prompt is preserved by memory.py/upstream.py.
        self.writer_calls += 1
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        ids = self.tokenizer(text, add_special_tokens=False,
                             return_tensors="pt").input_ids.to(self.device)
        with torch.inference_mode():
            generated = self.model.generate(ids, max_new_tokens=tokens,
                attention_mask=torch.ones_like(ids), do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id)
        new = generated[0, ids.shape[1]:]
        response = self.tokenizer.decode(new, skip_special_tokens=True)
        return {"raw_response": response, "input_tokens": ids.shape[1],
                "output_tokens": len(new), "finish_reason":
                "length" if len(new) >= tokens else "stop"}


def plan(args: argparse.Namespace) -> dict:
    return {"upstream": str(args.upstream.resolve()),
            "model": str(args.model.resolve()),
            "embedding": str(args.embedding.resolve()),
            "memory_tokens": args.memory_tokens,
            "writer_max_tokens": args.writer_tokens,
            "q_min_threshold": -10,
            "rl_config": {"epsilon": 0, "tau": .62, "alpha": .3,
                "gamma": 0., "q_init_pos": 0., "q_init_neg": 0.,
                "success_reward": 1., "failure_reward": -1.,
                "sim_threshold": .62, "topk": 3,
                "novelty_threshold": .85, "recency_boost": 0.,
                "reward_merge_gain": .1, "weight_sim": .5, "weight_q": .5}}


def reader_logits(model, tokenizer, content: dict, context: str,
                  choice_ids: list[int], device: str) -> torch.Tensor:
    question = target_question(content)
    if context:
        question = "[Retrieved Memory Context]\n" + context + "\n\n" + question
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": question}], tokenize=False,
        add_generation_prompt=True)
    ids = tokenizer(text, add_special_tokens=False,
                    return_tensors="pt").input_ids.to(device)
    with torch.inference_mode():
        return model(input_ids=ids, use_cache=False).logits[0, -1,
                                                            choice_ids].float().cpu()


def run(args: argparse.Namespace) -> dict:
    if args.output.exists() or args.directory.exists():
        raise FileExistsError("Fresh output and memory directory required")
    annotation_bytes = args.annotations.read_bytes()
    annotations = json.loads(annotation_bytes)
    if args.device.startswith("cuda"):
        torch.cuda.set_per_process_memory_fraction(args.gpu_fraction,
                                                   device=args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    choice_tokens = [tokenizer(str(action), add_special_tokens=False).input_ids
                     for action in ACTIONS]
    if any(len(tokens) != 1 for tokens in choice_tokens):
        raise ValueError("Choice labels are not single Qwen tokens")
    choice_ids = [tokens[0] for tokens in choice_tokens]
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16,
              local_files_only=True, attn_implementation="sdpa").to(args.device).eval()
    client = LocalQwenClient(model, tokenizer, args.device, args.seed)
    memory = Memory(plan(args), client, args.directory / "memory",
                    {"threshold": args.retrieval_threshold,
                     "mean": 0., "std": 1.})

    def observe(content: dict) -> dict:
        query = memory_query(content)
        prior = memory.retrieve(query)
        reward = source_reward(content)
        update = memory.update(query, public_trace(content), reward,
                               reward > 0, prior, digest(content))
        retrieved = memory.retrieve(query)
        return {"context": retrieved["context"],
                "new_memory_id": update["new_memory_id"],
                "retrieved_ids": retrieved["ids"],
                "retrieval_candidates": retrieved["candidates"],
                "q_updates": update["q_updates"],
                "official_source_reward": reward,
                "memory_tokens": retrieved["tokens"]}

    if args.train_snapshot is None:
        train_items = annotations["split"]["train"]
        for index, item in enumerate(train_items[:args.preload]):
            arm = ARMS[index % len(ARMS)]
            content, _ = checked_query(item["arms"][arm]["queries"][0])
            observe(content)
            if (index + 1) % 10 == 0:
                print(json.dumps({"preloaded": index + 1,
                                  "writer_calls": client.writer_calls}), flush=True)
        snapshot = args.directory / "train_memory_snapshot.json"
        memory.snapshot(snapshot)
    else:
        snapshot = args.train_snapshot
        memory.restore(snapshot)
        print(json.dumps({"restored_train_snapshot": str(snapshot)}), flush=True)
    snapshot_state = json.loads(snapshot.read_text())
    preload_count = snapshot_state["calls"]
    rows = []
    test_items = annotations["split"]["test"]
    for item_index, item in enumerate(test_items[:args.test_limit]):
        contexts, inputs, labels, details = [], [], [], []
        for arm in ARMS:
            memory.restore(snapshot)
            content, target = checked_query(item["arms"][arm]["queries"][0])
            detail = observe(content)
            contexts.append(detail["context"])
            inputs.append(content)
            labels.append(target)
            details.append({k: v for k, v in detail.items() if k != "context"})
        no_source = int(ACTIONS[reader_logits(model, tokenizer, inputs[0], "",
                             choice_ids, args.device).argmax()])
        raw_contexts = [public_trace(content) for content in inputs]
        for arm_index, arm in enumerate(ARMS):
            correct = int(ACTIONS[reader_logits(model, tokenizer, inputs[arm_index],
                contexts[arm_index], choice_ids, args.device).argmax()])
            wrong = int(ACTIONS[reader_logits(model, tokenizer, inputs[arm_index],
                contexts[(arm_index + 1) % len(ARMS)], choice_ids,
                args.device).argmax()])
            raw_action = int(ACTIONS[reader_logits(model, tokenizer, inputs[arm_index],
                raw_contexts[arm_index], choice_ids, args.device).argmax()])
            wrong_raw = int(ACTIONS[reader_logits(model, tokenizer, inputs[arm_index],
                raw_contexts[(arm_index + 1) % len(ARMS)], choice_ids,
                args.device).argmax()])
            rows.append({"item_id": item["item_id"], "arm": arm,
                         "target_action": labels[arm_index],
                         "correct_memory_action": correct,
                         "wrong_memory_action": wrong,
                         "no_memory_action": no_source,
                         "raw_history_action": raw_action,
                         "wrong_raw_history_action": wrong_raw,
                         "retrieval": details[arm_index],
                         "context_sha256": hashlib.sha256(
                             contexts[arm_index].encode()).hexdigest()})
        print(json.dumps({"test_item": item_index + 1,
                          "correct": sum(r["correct_memory_action"] ==
                                         r["target_action"] for r in rows),
                          "writer_calls": client.writer_calls}), flush=True)
        (args.directory / "test_progress.json").write_text(json.dumps({
            "annotations_sha256": hashlib.sha256(annotation_bytes).hexdigest(),
            "train_snapshot_sha256": hashlib.sha256(snapshot.read_bytes()).hexdigest(),
            "completed_groups": item_index + 1, "rows": rows},
            ensure_ascii=False, indent=2) + "\n")
    counts = {"n": len(rows),
              "correct": sum(r["correct_memory_action"] == r["target_action"]
                             for r in rows),
              "wrong": sum(r["wrong_memory_action"] == r["target_action"]
                           for r in rows),
              "none": sum(r["no_memory_action"] == r["target_action"]
                          for r in rows),
              "raw_history": sum(r["raw_history_action"] == r["target_action"]
                                 for r in rows),
              "wrong_raw_history": sum(r["wrong_raw_history_action"] ==
                                       r["target_action"] for r in rows),
              "retrieved_current": sum(r["retrieval"]["new_memory_id"] in
                                       r["retrieval"]["retrieved_ids"] for r in rows),
              "empty_contexts": sum(not r["retrieval"]["retrieved_ids"] for r in rows),
              "writer_calls": client.writer_calls}
    result = {"protocol": "Native upstream MemRL procedural writer, query retrieval and Q updates adapted to public XLand source traces; same frozen Qwen actor and target prompt as LoRA arm; train-only preload, independently restored test arms; raw-history ICL diagnostic without writer",
              "annotations_sha256": hashlib.sha256(annotation_bytes).hexdigest(),
              "train_snapshot_sha256": hashlib.sha256(snapshot.read_bytes()).hexdigest(),
              "model_config_sha256": hashlib.sha256((args.model / "config.json").read_bytes()).hexdigest(),
              "plan": plan(args), "preload": preload_count,
              "test_limit": args.test_limit, "seed": args.seed,
              "counts": counts, "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(counts), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path(
        "current_work/delta-Mem/model/Qwen3-4B-Instruct-2507"))
    parser.add_argument("--upstream", type=Path, default=Path("current_work/MemRL"))
    parser.add_argument("--embedding", type=Path, default=Path("models/embedding/bge-m3"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpu-fraction", type=float, default=.55)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--preload", type=int, default=24)
    parser.add_argument("--train-snapshot", type=Path)
    parser.add_argument("--test-limit", type=int, default=12)
    parser.add_argument("--writer-tokens", type=int, default=256)
    parser.add_argument("--memory-tokens", type=int, default=2048)
    parser.add_argument("--retrieval-threshold", type=float, default=.5)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (args.preload < 0 or args.test_limit < 1 or
            not 0 < args.gpu_fraction <= 1):
        parser.error("Invalid budget")
    run(args)


if __name__ == "__main__":
    main()
