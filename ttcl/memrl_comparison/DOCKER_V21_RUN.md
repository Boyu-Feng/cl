# v21 on the two Docker CLBench domains

`evaluate_v21_docker.py` runs the canonical ordered sequence separately for native
MemRL and v21. Sales Prediction keeps one `/app` workspace across its 12 years;
Codebase Adaptation follows its 19 official PR instances. The runner updates
memory only after an official per-instance outcome. A partial `--limit` run is a
pilot and must not be reported as the official last-80% score.

The current host account `fengboyu` cannot open `/var/run/docker.sock` (owned by
`root:docker`, mode `0660`; `fengboyu` is not in `docker`). The installed rootless
Docker lacks `newuidmap` and `newgidmap`. A single-ID user namespace cannot
unpack the official Sales image, which contains GID 42. A working host Docker
endpoint or administrator-provided full rootless UID/GID mapping is required.
No Docker-domain result was obtained from this setup failure.

Once Docker access works, run from the repository root with the existing frozen
Qwen actor service. Do not interrupt or replace any running GPU service.

```bash
export TTCL_WORKSPACE="$PWD"
export TTCL_BENCH="$PWD/current_work/continual-learning-bench"
export PYTHONPATH="$PWD:$TTCL_BENCH:$PWD/ttcl/.runtime/structured_memory_deps:$PWD/ttcl/.runtime/deltamem_benchmark_deps"
docker info --format '{{.ServerVersion}}'
ttcl/.runtime/alf_delta_env/bin/python -m ttcl.memrl_comparison.evaluate_v21_docker \
  --origin results/memrl_comparison/20260928_budgeted \
  --output results/memrl_comparison/v21_docker_sales_pilot_303 \
  --task sales_prediction --repeat 303 --limit 2 --url http://127.0.0.1:18557
ttcl/.runtime/alf_delta_env/bin/python -m ttcl.memrl_comparison.evaluate_v21_docker \
  --origin results/memrl_comparison/20260928_budgeted \
  --output results/memrl_comparison/v21_docker_codebase_pilot_303 \
  --task codebase_adaptation --repeat 303 --limit 2 --url http://127.0.0.1:18557
```

Inspect each pilot's `analysis.json`, `vanilla/progress.json`,
`typed_grounded/progress.json`, and per-episode `row.json` before running full
sequences. For a full run, use a new output path and omit `--limit`. Repeat 303
and 404 are the original paired actor seeds. A failed or interrupted chain keeps
its records but cannot be resumed because the live Docker workspace is gone.
The script never substitutes missing outcomes with zero. It binds task/data
hashes, implementation hashes, image IDs, public calibration prompts, official
instance IDs, actor token counts, and memory snapshots in the run directory.
