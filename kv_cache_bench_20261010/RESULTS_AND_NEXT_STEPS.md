# Qwen3-4B-Instruct-2507 KV compression experiment

## Scope

- Model: `Qwen3-4B-Instruct-2507`
- Seed: `42`
- Suites: LoCoMo and native-context MemoryAgentBench
- Methods: SnapKV and KVzip
- User-facing budgets below mean **KV retained ratio**. The raw JSONL field is `eviction_fraction`, so retained 75/50/25% corresponds to eviction 25/50/75%.
- HotpotQA and Quest were excluded.

## Current results

### LoCoMo

LoCoMo is complete: 9,240/9,240 records, with no failures.

| Category | Method | Retain 75% | Retain 50% | Retain 25% |
|---|---:|---:|---:|---:|
| Single-hop | SnapKV | 0.3433 | 0.2227 | 0.1133 |
| Single-hop | KVzip | 0.4762 | 0.4738 | 0.4622 |
| Temporal | SnapKV | 0.2312 | 0.1465 | 0.0727 |
| Temporal | KVzip | 0.2889 | 0.2852 | 0.2539 |
| Multi-hop | SnapKV | 0.3186 | 0.2531 | 0.1566 |
| Multi-hop | KVzip | 0.3840 | 0.3830 | 0.3414 |
| Open-domain | SnapKV | 0.1004 | 0.0971 | 0.0805 |
| Open-domain | KVzip | 0.0983 | 0.0931 | 0.0902 |

### MemoryAgentBench

The native-context target is 13,626 records. The paused run contains 11,892 successful records, 1,600 CUDA OOM failures, and 134 records not yet started. The 134 pending records are in Long-range Understanding.

The following means are over successful records only; OOM rows are not silently removed from the final denominator.

| Category | Method | Retain 75% | Retain 50% | Retain 25% |
|---|---:|---:|---:|---:|
| Selective Forgetting | SnapKV | 0.1800 | 0.1350 | 0.0950 |
| Selective Forgetting | KVzip | 0.1917 | 0.1967 | 0.2033 |
| Test-time Learning | SnapKV | 0.3380 | 0.3000 | 0.1940 |
| Test-time Learning | KVzip | 0.4000 | 0.3800 | 0.1800 |
| Accurate Retrieval | SnapKV | 0.7830 | 0.6482 | 0.5300 |
| Accurate Retrieval | KVzip | 0.8270 | 0.8150 | 0.7860 |
| Long-range Understanding | SnapKV | pending | pending | pending |
| Long-range Understanding | KVzip | pending | pending | pending |

The raw outputs are intentionally kept under the repository's ignored `results/` tree. The checked-in tables above are the reproducible summary of those outputs.

## Work remaining on the H200 server

1. Re-run the 1,600 rows whose original status is `failed` (the recorded errors are CUDA OOM).
2. Run the 134 missing Long-range Understanding records.
3. Merge retry outputs with the original JSONL using `(id, method, eviction_fraction)` as the key.
4. Compute final scores with the full 13,626-record denominator, counting unresolved OOM as failures.

The current retry script is paused. It now defaults to `KVzipPress(chunk_size=1024)` and supports overriding the value with `--kvzip-chunk-size`. It runs one question at a time, clears CUDA memory after each record, and enables expandable CUDA allocator segments.

Example on the H200 server:

```bash
cd /path/to/cl
source ttcl/.runtime/alf_delta_env/bin/activate
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:256

python -u kv_cache_bench_20261010/retry_failed.py \
  --input results/kv_cache_bench_20261010/mab_snapkv_kvzip_seed42.jsonl \
  --output results/kv_cache_bench_20261010/mab_snapkv_kvzip_failed_retry_seed42.jsonl \
  --device cuda:0 \
  --kvzip-chunk-size 1024
```

After that retry completes, run the missing native Long-range Understanding cases with `run_grouped.py` and merge only keys absent from the existing output.
