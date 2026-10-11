"""Small, auditable Qwen3 KV budget experiment; see README for scope."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import json
import math
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "data/kv_cache_bench_20261010/python_deps"))
sys.path.insert(0, str(ROOT / "data/kv_cache_bench_20261010/upstream/kvpress"))

import pyarrow.parquet as pq
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from kvpress import KVzipPress, SnapKVPress  # noqa: E402


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


LOCOMO = load_module(
    "kvpilot_locomo_protocol",
    ROOT / "current_work/delta-Mem/deltamem/eval/locomo_protocol.py",
)
MAB_TEMPLATES = load_module(
    "kvpilot_mab_templates",
    ROOT / "data/kv_cache_bench_20261010/upstream/MemoryAgentBench/templates.py",
)
MODEL = ROOT / "current_work/delta-Mem/model/Qwen3-4B-Instruct-2507"
DATA = ROOT / "data/kv_cache_bench_20261010"
OUT = ROOT / "results/kv_cache_bench_20261010"
RATIOS = (0.75, 0.50, 0.25)
METHODS = ("snapkv", "kvzip")
# Small reconstruction chunks reduce KVzip's temporary peak allocation on
# long MemoryAgentBench contexts.  This is intentionally conservative for the
# failure-only retry path as well as ordinary single-example inference.
KVZIP_CHUNK_SIZE = 1024


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def locomo_cases(limit: int):
    samples = json.loads((ROOT / "current_work/delta-Mem/data/locomo10.json").read_text())
    cases = []
    for sample in samples:
        conv = sample["conversation"]
        context = LOCOMO.OFFICIAL_CONV_START_PROMPT.format(conv["speaker_a"], conv["speaker_b"])
        sessions = sorted(int(k.split("_")[-1]) for k in conv if k.startswith("session_") and not k.endswith("date_time"))
        for n in sessions:
            context += LOCOMO.build_session_message(conv, n)["content"] + "\n"
        for i, question in enumerate(sample["qa"]):
            category = int(question["category"])
            if category not in (1, 2, 3, 4):
                continue
            spec = LOCOMO.prepare_locomo_question(question, sample_id=sample["sample_id"], question_index=i, seed=42)
            cases.append({
                "id": f"{sample['sample_id']}:{i}", "suite": "locomo",
                "category": LOCOMO.CATEGORY_NAMES[category],
                "context": context, "question": "\n\n" + LOCOMO.build_official_question_prompt(spec),
                "system": LOCOMO.OFFICIAL_SYSTEM_PROMPT,
                "answer": question["answer"], "question_record": question,
                "max_new_tokens": 50,
            })
    return select_cases(cases, limit)


def mab_cases(limit: int, all_native: bool = False):
    selected_sources = {
        "accurate_retrieval": ("eventqa_65536",),
        "test_time_learning": ("icl_trec_coarse_6600shot_balance",),
        "long_range_understanding": ("detective_qa",),
        "selective_forgetting": ("factconsolidation_sh_6k",),
    }
    if all_native:
        selected_sources = {
            "accurate_retrieval": ("ruler_qa1_197K", "eventqa_65536", "eventqa_131072"),
            "test_time_learning": ("icl_banking77_5900shot_balance", "icl_clinic150_7050shot_balance",
                                   "icl_nlu_8296shot_balance", "icl_trec_coarse_6600shot_balance",
                                   "icl_trec_fine_6400shot_balance"),
            "long_range_understanding": ("detective_qa",),
            "selective_forgetting": ("factconsolidation_mh_6k", "factconsolidation_mh_32k",
                                     "factconsolidation_mh_64k", "factconsolidation_sh_6k",
                                     "factconsolidation_sh_32k", "factconsolidation_sh_64k"),
        }
    sections = {"accurate_retrieval": "Accurate_Retrieval",
                "test_time_learning": "Test_Time_Learning",
                "long_range_understanding": "Long_Range_Understanding",
                "selective_forgetting": "Conflict_Resolution"}
    caps = {"accurate_retrieval": 40, "test_time_learning": 20,
            "long_range_understanding": 2000, "selective_forgetting": 10}
    cases = []
    for category, sources in selected_sources.items():
        section = sections[category]
        rows = pq.read_table(DATA / "data" / f"{section}-00000-of-00001.parquet").to_pylist()
        for row_index, row in enumerate(rows):
            source = row["metadata"]["source"]
            if source not in sources:
                continue
            memo = MAB_TEMPLATES.get_template(source, "memorize", "Long_context_agent")
            context = memo.format(time_stamp="", context=row["context"])
            for i, (question, answers) in enumerate(zip(row["questions"], row["answers"])):
                q = MAB_TEMPLATES.get_template(source, "query", "Long_context_agent").format(question=question)
                qid = row["metadata"].get("qa_pair_ids") or []
                cases.append({
                    "id": qid[i] if i < len(qid) else f"{source}:{row_index}:{i}",
                    "suite": "mab", "category": category, "source": source,
                    "context": context, "question": "\n\n" + q,
                    "system": MAB_TEMPLATES.SYSTEM_MESSAGE,
                    "answer": answers, "max_new_tokens": 50 if source.startswith("ruler_qa") else caps[category],
                })
    if limit:
        # A small pilot must fit the single A100's full-KV Quest arm as well.
        cases = [x for x in cases if x["category"] != "long_range_understanding"
                 or len(x["context"]) < 500_000]
    return select_cases(cases, limit)


def select_cases(cases, limit):
    if limit == 0:
        return cases
    groups = defaultdict(list)
    for case in cases:
        groups[case["category"]].append(case)
    chosen = []
    for category in sorted(groups):
        # Fixed seed selects inputs before seeing any model predictions.
        ranked = sorted(groups[category], key=lambda x: digest(f"42|{x['id']}"))
        chosen.extend(ranked[:limit])
    return chosen


def tokenize_case(tokenizer, case):
    marker = "<KV_BENCH_QUESTION_SPLIT_91f39b>"
    rendered = tokenizer.apply_chat_template(
        [{"role": "system", "content": case["system"]},
         {"role": "user", "content": case["context"] + marker}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    prefix, suffix = rendered.split(marker)
    context_ids = tokenizer.encode(prefix, add_special_tokens=False, return_tensors="pt")
    question_ids = tokenizer.encode(case["question"] + suffix, add_special_tokens=False, return_tensors="pt")
    return context_ids, question_ids


def normalize_answer(value):
    value = str(value).lower()
    value = re.sub(r"[^\w\s]", "", value)
    value = re.sub(r"\b(a|an|the)\b", " ", value)
    return " ".join(value.split())


def score(case, prediction):
    if case["suite"] == "locomo":
        spec = LOCOMO.prepare_locomo_question(case["question_record"],
            sample_id=case["id"].rsplit(":", 1)[0],
            question_index=int(case["id"].rsplit(":", 1)[1]), seed=42)
        clean = LOCOMO.canonicalize_locomo_prediction(prediction, spec)
        return LOCOMO.score_locomo_prediction(case["question_record"], clean), clean
    clean = prediction.splitlines()[0].removeprefix("Answer:").strip()
    references = case["answer"] if isinstance(case["answer"], list) else [case["answer"]]
    candidate = normalize_answer(clean)
    refs = [normalize_answer(x) for x in references]
    if case["category"] in ("accurate_retrieval", "selective_forgetting"):
        result = any(x and x in candidate for x in refs)
    else:
        result = candidate in refs
    return float(result), clean


@torch.inference_mode()
def infer(model, tokenizer, context_ids, question_ids, method, eviction, max_new_tokens,
          kvzip_chunk_size: int = KVZIP_CHUNK_SIZE):
    cache = DynamicCache()
    context_ids = context_ids.to(model.device)
    question_ids = question_ids.to(model.device)
    original_length = context_ids.shape[1]
    press = (SnapKVPress(compression_ratio=eviction) if method == "snapkv" else
             KVzipPress(compression_ratio=eviction, chunk_size=kvzip_chunk_size))
    started = time.monotonic()
    with contextlib.nullcontext():
        with press(model):
            model.model(input_ids=context_ids, past_key_values=cache, use_cache=True)
        retained = [cache.get_seq_length(i) for i in range(len(cache))]
        if method == "snapkv":
            expected = math.ceil(original_length * (1 - eviction))
            if any(abs(x - expected) > 1 for x in retained):
                raise RuntimeError(f"unexpected retained KV length: {retained[:4]}, expected {expected}")
        position = torch.arange(original_length, original_length + question_ids.shape[1], device=model.device)[None]
        out = model(input_ids=question_ids, past_key_values=cache, position_ids=position,
                    use_cache=True, logits_to_keep=1)
        token = out.logits[0, -1].argmax()
        generated = [token]
        eos = model.generation_config.eos_token_id
        eos_set = set(eos if isinstance(eos, list) else [eos])
        for i in range(max_new_tokens - 1):
            if token.item() in eos_set:
                break
            pos = torch.tensor([[original_length + question_ids.shape[1] + i]], device=model.device)
            out = model(input_ids=token.reshape(1, 1), past_key_values=cache,
                        position_ids=pos, use_cache=True, logits_to_keep=1)
            token = out.logits[0, -1].argmax()
            generated.append(token)
    return tokenizer.decode(torch.stack(generated), skip_special_tokens=True), retained, time.monotonic() - started


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--suite", choices=("locomo", "mab"), required=True)
    p.add_argument("--max-per-category", type=int, default=1,
                   help="0 means all questions; default is a small pilot")
    p.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    p.add_argument("--all-native", action="store_true")
    p.add_argument("--device", default="cuda:6")
    p.add_argument("--max-context-tokens", type=int, default=262000)
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    cases = (locomo_cases(args.max_per_category) if args.suite == "locomo"
             else mab_cases(args.max_per_category, all_native=args.all_native))
    output = args.output or OUT / f"{args.suite}_snap_kvzip_pilot_seed42.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    completed = set()
    if output.exists():
        for line in output.read_text().splitlines():
            row = json.loads(line)
            completed.add((row["id"], row["method"], row["eviction_fraction"]))
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    ).to(args.device).eval()
    for case in cases:
        context_ids, question_ids = tokenize_case(tokenizer, case)
        total_length = context_ids.shape[1] + question_ids.shape[1] + case["max_new_tokens"]
        for method in args.methods:
            for eviction in RATIOS:
                key = (case["id"], method, eviction)
                if key in completed:
                    continue
                row = {
                    "id": case["id"], "suite": case["suite"], "category": case["category"],
                    "source": case.get("source"), "method": method,
                    "eviction_fraction": eviction, "seed": 42,
                    "context_sha256": digest(case["context"]),
                    "question_sha256": digest(case["question"]),
                    "context_tokens": context_ids.shape[1],
                    "question_tokens": question_ids.shape[1],
                    "answer": case["answer"],
                }
                if total_length > args.max_context_tokens:
                    row.update(status="skipped_context_limit", model_max_tokens=args.max_context_tokens)
                else:
                    try:
                        response, retained, elapsed = infer(model, tokenizer, context_ids, question_ids,
                                                            method, eviction, case["max_new_tokens"])
                        value, clean = score(case, response)
                        row.update(status="complete", prediction=response, parsed_prediction=clean,
                                   score=value, retained_kv_tokens=retained,
                                   elapsed_seconds=elapsed)
                    except Exception as exc:
                        row.update(status="failed", error=f"{type(exc).__name__}: {exc}")
                with output.open("a") as f:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                print(f"{case['category']} {case['id']} {method} {eviction}: {row['status']} "
                      f"{row.get('score', '')}", flush=True)
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
