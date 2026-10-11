# Prefill KV cache benchmark

This directory records a Qwen3-4B-Instruct-2507 comparison on LoCoMo and
MemoryAgentBench at seed 42. The current comparison is **SnapKV versus
KVzip**, both using prefill-time KV compression. Raw data, model
weights, dependencies, and outputs live in the repository's ignored `data/`,
`models/`, and `results/` trees.

The requested compression values mean **KV eviction fractions**: 0.75, 0.50,
and 0.25. Both active methods physically retain approximately 25%, 50%, or
75% of the prefilling KV pairs. KVzip scores importance with context
reconstruction during the prefill/compression phase. The KVPress implementation
uses attention masks for the compressed cache, so its physical tensor length is
unchanged while subsequent attention excludes the evicted entries.
For the 100K native MemoryAgentBench contexts, the reconstruction chunk is set
to 1024 tokens for the H200 retry server. If memory is insufficient, lower it
with `--kvzip-chunk-size` in the failure-only retry script.

KVzip is the selected prominent recent prefill method (NeurIPS 2025 Oral), with
official code supporting Qwen3 and a public KVPress implementation.
Method sources:

- SnapKV: https://arxiv.org/abs/2404.14469
- KVzip: https://arxiv.org/abs/2505.23416
- KVzip code: https://github.com/snu-mllab/KVzip
- MemoryAgentBench: https://github.com/HUST-AI-HYZ/MemoryAgentBench
- LoCoMo: https://github.com/snap-research/locomo

Use `python run.py --help` for the pilot runner. Every JSONL output records
method, eviction fraction, seed, exact input digest, raw prediction, score,
token lengths, and any failure. A summary is produced only from completed rows;
it includes coverage counts, so partial runs are never labeled official scores.

## Inputs and execution

The local Qwen model is read from
`current_work/delta-Mem/model/Qwen3-4B-Instruct-2507`; LoCoMo is read from
`current_work/delta-Mem/data/locomo10.json`. `inventory.py` checks the official
MemoryAgentBench Parquet release at revision
`7ea066982b140a19337e17e60d45d4076e042faf`, downloaded into ignored
`data/kv_cache_bench_20261010/data/`. The four original category names are
AR, TTL, LRU, and Conflict Resolution; the requested table calls the last SF.

The local runner adapts the official MemoryAgentBench long-context prompts for
Qwen3 and uses the official exact or substring match criterion. Its LRU score
covers DetectiveQA only: InfBench summarization requires an external LLM judge
and must be reported separately. The benchmark's native range also excludes
some contexts. `results/kv_cache_bench_20261010/inventory.json` records the
source-by-source coverage; prompt overhead and GPU limits can exclude more.

| MemoryAgentBench category | Native-context questions | Directly scored here |
|---|---:|---:|
| AR | 1100 / 2000 | 1100 |
| TTL | 500 / 700 | 500 |
| LRU | 156 / 171 | 71 DetectiveQA; 85 summaries need a judge |
| SF / Conflict Resolution | 600 / 800 | 600 |

The local LoCoMo protocol reuses the repository's official-category mapping and
F1 scorer. It reports categories 1–4 only; category 5 was not requested.

The grouped runner prefills each common context once per method and ratio, then
restores that cache after every independent question. Its answers were checked
against one-question-at-a-time inference for all three methods.

```bash
# From the repository root, with the mainline Python environment available.
ttcl/.runtime/alf_delta_env/bin/python -m pip install --no-deps \
  --target data/kv_cache_bench_20261010/python_deps \
  transformers==4.57.3 tokenizers==0.22.2 fire==0.7.1
# Clone the EchoPress KVPress branch at commit 39748e9c8301944128eb2e681bfb074e9a2de860
# into data/kv_cache_bench_20261010/upstream/echopress_repo/;
# retain its LICENSE. Copy the MemoryAgentBench templates.py and LICENSE from
# commit 538026089d1a8a8eff05121d0db89b388f360eba into its upstream folder.
ttcl/.runtime/alf_delta_env/bin/python kv_cache_bench_20261010/inventory.py
CUDA_VISIBLE_DEVICES=6 ttcl/.runtime/alf_delta_env/bin/python \
  kv_cache_bench_20261010/run_grouped.py --suite locomo --device cuda:0 \
  --output results/kv_cache_bench_20261010/locomo_snap_echo_seed42.jsonl
CUDA_VISIBLE_DEVICES=2 ttcl/.runtime/alf_delta_env/bin/python \
  kv_cache_bench_20261010/run_grouped.py --suite mab --all-native --device cuda:0 \
  --output results/kv_cache_bench_20261010/mab_snap_echo_seed42.jsonl
```

`--max-per-category N` requests a fixed seed 42 pilot with N questions per
category. The script writes one row at a time and resumes an existing output
without repeating completed cells. See `freeze_snap_echo.json` for SHA-256
hashes of the actual model, code, and data used in this run. The earlier
SnapKV/Quest/DropKV JSONL files are archived separately and excluded from the
current summaries.
