"""Save content hashes and dependency versions beside ignored experiment output."""

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "current_work/delta-Mem/model/Qwen3-4B-Instruct-2507"
OUT = ROOT / "results/kv_cache_bench_20261010/freeze_snap_kvzip.json"


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    paths = [ROOT / "kv_cache_bench_20261010" / x
             for x in ("run.py", "run_grouped.py", "inventory.py")]
    paths += [ROOT / "current_work/delta-Mem/deltamem/eval/locomo_protocol.py",
              ROOT / "current_work/delta-Mem/data/locomo10.json",
              ROOT / "data/kv_cache_bench_20261010/upstream/MemoryAgentBench/templates.py"]
    paths += list((ROOT / "data/kv_cache_bench_20261010/upstream/kvpress/kvpress").rglob("*.py"))
    paths += list((ROOT / "data/kv_cache_bench_20261010/data").glob("*.parquet"))
    paths += list(MODEL.glob("*.safetensors"))
    paths += [MODEL / "config.json", MODEL / "tokenizer.json"]
    record = {"seed": 42, "eviction_fractions": [0.75, 0.50, 0.25],
              "model": "Qwen3-4B-Instruct-2507",
              "torch_version": "2.7.0+cu126", "transformers_version": "4.57.3",
              "python": subprocess.check_output(
                  [str(ROOT / "ttcl/.runtime/alf_delta_env/bin/python"), "--version"], text=True).strip(),
              "upstream_commits": {"MemoryAgentBench": "538026089d1a8a8eff05121d0db89b388f360eba",
                                   "kvpress_snapkv_kvzip": "7136c224105ba7b046a6e538742c3589698680b8",
                                   "KVzip_official": "main_snapshot_downloaded_2026-10-11"},
              "files": {str(x.relative_to(ROOT)): sha256(x) for x in paths}}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(record, indent=2))
    print(OUT, len(paths), "files")


if __name__ == "__main__":
    main()
