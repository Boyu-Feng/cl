"""Tests for CohortStudiesTask within-stage permutation across runs.

Invariants covered:
- Stage sequence is preserved across runs (only within-stage order changes).
- prepare_run(0) is a no-op on the canonical ordering.
- prepare_run(run_index>0) permutes instances only within each stage.
- Permutation is deterministic given the same seed + run_index.
"""

from __future__ import annotations

from collections import Counter

import pytest

from src.interface import Response

# The cohort_studies task pulls in optional analytics dependencies
# (lifelines, pandas).  Skip the whole module when they aren't installed so the
# rest of the suite can collect.
cohort_studies_task = pytest.importorskip(
    "src.tasks.cohort_studies.task",
    reason="cohort_studies optional dependencies (lifelines, pandas) are not installed.",
)
CohortStudiesTask = cohort_studies_task.CohortStudiesTask
_default_dataset_dir = cohort_studies_task._default_dataset_dir


SCHEDULE_ID = "default"


def _dataset_available() -> bool:
    return _default_dataset_dir(SCHEDULE_ID).exists()


pytestmark = pytest.mark.skipif(
    not _dataset_available(),
    reason=f"Frozen dataset for schedule '{SCHEDULE_ID}' not found on disk.",
)


def _make_task() -> CohortStudiesTask:
    return CohortStudiesTask(schedule=SCHEDULE_ID)


def _stages(task: CohortStudiesTask) -> list[int]:
    return [inst.stage_index for inst in task.instances]


def _variants(task: CohortStudiesTask) -> list[str]:
    return [inst.variant_id for inst in task.instances]


def test_canonical_and_prepare_run_zero_are_identical():
    a = _make_task()
    b = _make_task()
    b.prepare_run(0)

    assert _variants(a) == _variants(b), (
        "prepare_run(0) must be a no-op: baseline and run_index=0 must match."
    )


def test_stage_sequence_preserved_across_runs():
    canonical = _make_task()
    run = _make_task()
    run.prepare_run(2)

    assert _stages(canonical) == _stages(run), (
        "Stage sequence must be identical across runs; only within-stage "
        "order may change."
    )


def test_within_stage_permutation_only():
    canonical = _make_task()
    run = _make_task()
    run.prepare_run(2)

    canonical_stages = _stages(canonical)
    canonical_variants = _variants(canonical)
    run_variants = _variants(run)

    assert run_variants != canonical_variants, (
        "prepare_run(run_index>0) should permute at least some variants."
    )

    buckets_canonical: dict[int, list[str]] = {}
    buckets_run: dict[int, list[str]] = {}
    for stage, variant_c, variant_r in zip(
        canonical_stages,
        canonical_variants,
        run_variants,
    ):
        buckets_canonical.setdefault(stage, []).append(variant_c)
        buckets_run.setdefault(stage, []).append(variant_r)

    assert set(buckets_canonical) == set(buckets_run)
    for stage in buckets_canonical:
        assert Counter(buckets_canonical[stage]) == Counter(buckets_run[stage]), (
            f"Stage {stage}: variants leaked across stages during permutation."
        )


def test_permutation_is_deterministic():
    a = _make_task()
    b = _make_task()
    a.prepare_run(3)
    b.prepare_run(3)

    assert _variants(a) == _variants(b), (
        "Same (seed, run_index) must produce identical instance ordering."
    )


def test_distinct_run_indices_produce_distinct_orderings():
    a = _make_task()
    b = _make_task()
    a.prepare_run(1)
    b.prepare_run(2)

    assert _variants(a) != _variants(b), (
        "Different run indices should produce different within-stage orderings."
    )


def test_executor_runtime_error_surfaces_as_observation():
    """Runtime-time tool errors must not crash the run.

    Regression guard: errors raised inside the in-process ``ToolExecutor``
    (e.g. invalid SQL) must be surfaced to the agent as an observation
    rather than aborting the rollout. Bad SQL is the cleanest case because
    the action passes Pydantic validation but fails when the executor
    runs the query against SQLite.

    (Pydantic validation of malformed *arguments* is handled in the system
    layer's structured-output retry loop and never reaches ``task.step``.)
    """
    task = _make_task()
    task.build_current_query()
    assert task._executor is not None
    budget_before = task._executor.actions_used

    from src.tasks.cohort_studies.tool_schemas import SQLQuery, ToolCallResponse

    bad_response = Response(
        action=ToolCallResponse(
            thought="",
            tool_call=SQLQuery(sql="SELEC nonsense FROM nowhere"),
        )
    )
    result = task.step(bad_response)

    assert not result.done, "Executor errors should be recoverable, not fatal."
    assert result.next_query is not None
    assert "SQL ERROR" in result.observation.content
    assert task._executor.actions_used == budget_before + 1, (
        "Failed actions should still consume budget to prevent spam."
    )
