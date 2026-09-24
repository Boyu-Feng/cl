import json

from src.tasks.codebase_adaptation.curation.sequential_curation import (
    SequentialCurationConfig,
    curate_sequential_instances,
)
from src.tasks.codebase_adaptation.task_loader import load_tasks


def _make_patch(path: str, changed_lines: int = 8) -> str:
    patch_lines = [
        f"diff --git a/{path} b/{path}",
        f"--- a/{path}",
        f"+++ b/{path}",
        "@@ -1,4 +1,4 @@",
    ]
    for idx in range(changed_lines // 2):
        patch_lines.append(f"-old_{idx}")
        patch_lines.append(f"+new_{idx}")
    return "\n".join(patch_lines) + "\n"


def _make_row(repo: str, idx: int, created_at: str) -> dict:
    return {
        "repo": repo,
        "instance_id": f"{repo.replace('/', '__')}-{idx}",
        "pull_number": idx,
        "base_commit": f"deadbeef{idx}",
        "patch": _make_patch(f"src/core/module_{idx}.py", changed_lines=10),
        "test_patch": _make_patch(f"tests/test_module_{idx}.py", changed_lines=6),
        "problem_statement": "x" * 900,
        "created_at": created_at,
    }


def test_sequential_curation_picks_most_recent_valid_window():
    repo = "fresh/repo"
    rows = [
        _make_row(repo, 1, "2024-01-01T00:00:00Z"),
        _make_row(repo, 2, "2024-02-01T00:00:00Z"),
        _make_row(repo, 3, "2024-03-01T00:00:00Z"),
        _make_row(repo, 4, "2025-01-01T00:00:00Z"),
        _make_row(repo, 5, "2025-01-10T00:00:00Z"),
        _make_row(repo, 6, "2025-01-20T00:00:00Z"),
    ]

    result = curate_sequential_instances(
        rows,
        SequentialCurationConfig(
            instances_per_repo=3,
            min_repo_tasks=3,
            max_window_span_days=45,
            max_task_gap_days=20,
            min_unique_code_files=3,
        ),
    )

    assert [row["instance_id"] for row in result.selected_rows] == [
        "fresh__repo-4",
        "fresh__repo-5",
        "fresh__repo-6",
    ]
    assert [row["sequence_rank"] for row in result.selected_rows] == [1, 2, 3]
    assert all(
        row["selection_strategy"] == "most_recent_contiguous_window"
        for row in result.selected_rows
    )


def test_sequential_curation_respects_repo_blocklist():
    repo = "contaminated/repo"
    rows = [
        _make_row(repo, 1, "2025-01-01T00:00:00Z"),
        _make_row(repo, 2, "2025-01-10T00:00:00Z"),
        _make_row(repo, 3, "2025-01-20T00:00:00Z"),
    ]

    result = curate_sequential_instances(
        rows,
        SequentialCurationConfig(
            instances_per_repo=3,
            min_repo_tasks=3,
            min_unique_code_files=3,
            repo_blocklist=(repo,),
        ),
    )

    assert result.selected_rows == []
    assert result.report["counts"]["repo_eligible_total"] == 0
    assert result.report["repo_drop_reasons"]["repo_in_blocklist"] == 1


def test_loader_preserves_sequence_order_for_single_repo_dataset(tmp_path):
    dataset_path = tmp_path / "tasks.jsonl"
    rows = [
        {
            "instance_id": "example__repo-3",
            "repo": "example/repo",
            "patch": _make_patch("src/core/module_3.py"),
            "FAIL_TO_PASS": ["tests/test_example.py::test_a"],
            "PASS_TO_PASS": ["tests/test_example.py::test_b"],
            "image_name": "example/image",
            "problem_statement": "x" * 900,
            "sequence_rank": 3,
        },
        {
            "instance_id": "example__repo-1",
            "repo": "example/repo",
            "patch": _make_patch("src/core/module_1.py"),
            "FAIL_TO_PASS": ["tests/test_example.py::test_a"],
            "PASS_TO_PASS": ["tests/test_example.py::test_b"],
            "image_name": "example/image",
            "problem_statement": "x" * 900,
            "sequence_rank": 1,
        },
        {
            "instance_id": "example__repo-2",
            "repo": "example/repo",
            "patch": _make_patch("src/core/module_2.py"),
            "FAIL_TO_PASS": ["tests/test_example.py::test_a"],
            "PASS_TO_PASS": ["tests/test_example.py::test_b"],
            "image_name": "example/image",
            "problem_statement": "x" * 900,
            "sequence_rank": 2,
        },
    ]
    with open(dataset_path, "w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    tasks = load_tasks(str(dataset_path), num_instances=3, same_repo=True, seed=123)

    assert [task.instance_id for task in tasks] == [
        "example__repo-1",
        "example__repo-2",
        "example__repo-3",
    ]
