# AGENTS.md

This package orchestrates benchmark runs above the single interaction loop.

- `single.py` owns one rollout, `baseline.py` owns stateless baselines, and `benchmark.py` owns repeated/parallel runs.
- Preserve score, gain, run-mode, baseline-alignment, and aggregate semantics unless a behavior change is approved.
- Respect task/system `parallel_safe` flags and keep worker payloads process-serializable.
- Keep trace, artifact, usage, and execution-summary ordering stable for downstream viewers.
