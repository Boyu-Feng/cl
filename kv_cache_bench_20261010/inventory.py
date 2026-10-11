"""Record native Qwen3 context coverage for the official benchmark files."""

import json
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/kv_cache_bench_20261010/data"
MODEL = ROOT / "current_work/delta-Mem/model/Qwen3-4B-Instruct-2507"
OUT = ROOT / "results/kv_cache_bench_20261010/inventory.json"
MAX_TOKENS = 262144


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    result = {"model_max_position_embeddings": MAX_TOKENS, "categories": {}}
    mapping = {
        "Accurate_Retrieval": "AR", "Test_Time_Learning": "TTL",
        "Long_Range_Understanding": "LRU", "Conflict_Resolution": "SF",
    }
    for file in sorted(DATA.glob("*.parquet")):
        key = file.name.split("-00000")[0]
        rows = pq.read_table(file).to_pylist()
        sources = defaultdict(lambda: {"contexts": 0, "questions": 0,
                                       "native_contexts": 0, "native_questions": 0,
                                       "context_token_lengths": []})
        for row in rows:
            source = row["metadata"]["source"]
            bucket = sources[source]
            tokens = len(tokenizer.encode(row["context"], add_special_tokens=False))
            count = len(row["questions"])
            bucket["contexts"] += 1
            bucket["questions"] += count
            bucket["context_token_lengths"].append(tokens)
            if tokens + 200 < MAX_TOKENS:
                bucket["native_contexts"] += 1
                bucket["native_questions"] += count
        result["categories"][mapping[key]] = dict(sources)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2))
    for category, sources in result["categories"].items():
        total = sum(x["questions"] for x in sources.values())
        native = sum(x["native_questions"] for x in sources.values())
        print(category, f"native {native}/{total} questions")
        for source, x in sources.items():
            print(" ", source, f"{x['native_questions']}/{x['questions']}",
                  f"token range {min(x['context_token_lengths'])}-{max(x['context_token_lengths'])}")


if __name__ == "__main__":
    main()
