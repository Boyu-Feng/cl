"""Reuse each prefilling cache across the questions sharing one context."""

import argparse
import contextlib
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from kv_cache_bench_20261010 import run

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache


@torch.inference_mode()
def evaluate_group(model, tokenizer, cases, method, eviction, max_context_tokens, on_row=None):
    context_ids, _ = run.tokenize_case(tokenizer, cases[0])
    question_ids = [run.tokenize_case(tokenizer, case)[1] for case in cases]
    context_length = context_ids.shape[1]
    if any(context_length + q.shape[1] + c["max_new_tokens"] > max_context_tokens
           for q, c in zip(question_ids, cases)):
        raise ValueError("group contains question exceeding native context limit")
    cache = DynamicCache()
    context_ids = context_ids.to(model.device)
    press = (run.SnapKVPress(compression_ratio=eviction) if method == "snapkv" else
             run.KVzipPress(compression_ratio=eviction, chunk_size=run.KVZIP_CHUNK_SIZE))
    rows = []
    with contextlib.nullcontext():
        started = time.monotonic()
        with press(model):
            model.model(input_ids=context_ids, past_key_values=cache, use_cache=True)
        prefill_seconds = time.monotonic() - started
        retained = [cache.get_seq_length(i) for i in range(len(cache))]
        if method == "snapkv":
            expected = math.ceil(context_length * (1 - eviction))
            if any(abs(x - expected) > 1 for x in retained):
                raise RuntimeError(f"retained KV mismatch: {retained[:4]} vs {expected}")
        for case, question in zip(cases, question_ids):
            started = time.monotonic()
            question = question.to(model.device)
            positions = torch.arange(context_length, context_length + question.shape[1], device=model.device)[None]
            output = model(input_ids=question, past_key_values=cache, position_ids=positions,
                           use_cache=True, logits_to_keep=1)
            token = output.logits[0, -1].argmax()
            generated = [token]
            eos = model.generation_config.eos_token_id
            eos_set = set(eos if isinstance(eos, list) else [eos])
            for i in range(case["max_new_tokens"] - 1):
                if token.item() in eos_set:
                    break
                pos = torch.tensor([[context_length + question.shape[1] + i]], device=model.device)
                output = model(input_ids=token.reshape(1, 1), past_key_values=cache,
                               position_ids=pos, use_cache=True, logits_to_keep=1)
                token = output.logits[0, -1].argmax()
                generated.append(token)
            prediction = tokenizer.decode(torch.stack(generated), skip_special_tokens=True)
            score, parsed = run.score(case, prediction)
            rows.append({"id": case["id"], "suite": case["suite"],
                         "category": case["category"], "source": case.get("source"),
                         "method": method, "eviction_fraction": eviction, "seed": 42,
                         "context_sha256": run.digest(case["context"]),
                         "question_sha256": run.digest(case["question"]),
                         "context_tokens": context_length, "question_tokens": question.shape[1],
                         "answer": case["answer"], "prediction": prediction,
                         "parsed_prediction": parsed, "score": score, "status": "complete",
                         "retained_kv_tokens": retained, "prefill_seconds_group": prefill_seconds,
                         "query_seconds": time.monotonic() - started})
            if on_row is not None:
                on_row(rows[-1])
            # Remove the query and generated tokens before the next independent QA.
            for layer, length in zip(cache.layers, retained):
                layer.keys = layer.keys[:, :, :length]
                layer.values = layer.values[:, :, :length]
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--suite", choices=("locomo", "mab"), required=True)
    p.add_argument("--max-per-category", type=int, default=0)
    p.add_argument("--methods", nargs="+", choices=run.METHODS, default=list(run.METHODS))
    p.add_argument("--all-native", action="store_true")
    p.add_argument("--categories", nargs="+", help="Optional category filter")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--max-context-tokens", type=int, default=262000)
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    torch.manual_seed(42)
    tokenizer = AutoTokenizer.from_pretrained(run.MODEL)
    cases = (run.locomo_cases(args.max_per_category) if args.suite == "locomo"
             else run.mab_cases(args.max_per_category, all_native=args.all_native))
    if args.categories:
        cases = [x for x in cases if x["category"] in args.categories]
    groups = defaultdict(list)
    for case in cases:
        groups[case["context"]].append(case)
    output = args.output or run.OUT / f"{args.suite}_snap_kvzip_grouped_seed42.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if output.exists():
        for line in output.read_text().splitlines():
            item = json.loads(line)
            done.add((item["id"], item["method"], item["eviction_fraction"]))
    model = AutoModelForCausalLM.from_pretrained(run.MODEL, dtype=torch.bfloat16,
        attn_implementation="sdpa", low_cpu_mem_usage=True).to(args.device).eval()
    for group in groups.values():
        for method in args.methods:
            for eviction in run.RATIOS:
                pending = [x for x in group if (x["id"], method, eviction) not in done]
                if not pending:
                    continue
                saved = []
                def save_row(row):
                    with output.open("a") as f:
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    saved.append(row)
                try:
                    rows = evaluate_group(model, tokenizer, pending, method, eviction,
                                          args.max_context_tokens, on_row=save_row)
                except Exception as exc:
                    saved_ids = {x["id"] for x in saved}
                    rows = [{"id": x["id"], "suite": x["suite"], "category": x["category"],
                             "source": x.get("source"), "method": method,
                             "eviction_fraction": eviction, "seed": 42,
                             "status": "failed", "error": f"{type(exc).__name__}: {exc}"}
                            for x in pending if x["id"] not in saved_ids]
                    for row in rows:
                        save_row(row)
                else:
                    rows = saved
                print(group[0]["category"], method, eviction,
                      f"{sum(x['status']=='complete' for x in rows)}/{len(rows)}", flush=True)
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
