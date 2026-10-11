"""Rerun only failed MAB rows with one question per cache and conservative memory use."""

import argparse
import gc
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from kv_cache_bench_20261010 import run
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--kvzip-chunk-size", type=int, default=1024)
    args = p.parse_args()

    failed = {}
    with args.input.open() as f:
        for line in f:
            row = json.loads(line)
            if row.get("status") != "failed":
                continue
            key = (row["id"], row["method"], row["eviction_fraction"])
            failed[key] = row

    done = set()
    if args.output.exists():
        with args.output.open() as f:
            for line in f:
                row = json.loads(line)
                done.add((row["id"], row["method"], row["eviction_fraction"]))

    cases = {x["id"]: x for x in run.mab_cases(0, all_native=True)}
    tokenizer = AutoTokenizer.from_pretrained(run.MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        run.MODEL, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    ).to(args.device).eval()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    for key, old in failed.items():
        if key in done:
            continue
        case = cases.get(old["id"])
        if case is None:
            row = {**old, "status": "failed", "error": "RetryLookupError: case id not found"}
        else:
            try:
                context_ids, question_ids = run.tokenize_case(tokenizer, case)
                response, retained, elapsed = run.infer(
                    model, tokenizer, context_ids, question_ids,
                    old["method"], old["eviction_fraction"], case["max_new_tokens"],
                    kvzip_chunk_size=args.kvzip_chunk_size,
                )
                value, clean = run.score(case, response)
                row = {**old, "status": "complete", "prediction": response,
                       "parsed_prediction": clean, "score": value,
                       "context_sha256": run.digest(case["context"]),
                       "question_sha256": run.digest(case["question"]),
                       "context_tokens": context_ids.shape[1],
                       "question_tokens": question_ids.shape[1],
                       "answer": case["answer"], "retained_kv_tokens": retained,
                       "elapsed_seconds": elapsed, "retry_chunk_size": args.kvzip_chunk_size}
            except Exception as exc:
                row = {**old, "status": "failed_retry",
                       "error": f"{type(exc).__name__}: {exc}",
                       "retry_chunk_size": args.kvzip_chunk_size}
        with args.output.open("a") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(row["id"], row["method"], row["eviction_fraction"], row["status"], flush=True)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
