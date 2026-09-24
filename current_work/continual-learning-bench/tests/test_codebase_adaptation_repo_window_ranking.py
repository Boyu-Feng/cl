from src.tasks.codebase_adaptation.curation.repo_window_ranking import (
    RepoWindowRankingConfig,
    rank_repo_windows,
)
from src.tasks.codebase_adaptation.curation.sequential_curation import (
    SequentialCurationConfig,
)
from src.tasks.codebase_adaptation.curation.solvability_heuristics import (
    derive_cheap_task_solvability_metadata,
)


def _make_patch(paths: list[str]) -> str:
    chunks: list[str] = []
    for idx, path in enumerate(paths):
        chunks.extend(
            [
                f"diff --git a/{path} b/{path}",
                f"--- a/{path}",
                f"+++ b/{path}",
                "@@ -1,2 +1,2 @@",
                f"-old_{idx}",
                f"+new_{idx}",
                f"-oldb_{idx}",
                f"+newb_{idx}",
            ]
        )
    return "\n".join(chunks) + "\n"


def _make_row(
    repo: str,
    idx: int,
    created_at: str,
    paths: list[str],
    *,
    problem_statement: str | None = None,
    test_paths: list[str] | None = None,
    fail_to_pass: int = 2,
    pass_to_pass: int = 20,
) -> dict:
    return {
        "repo": repo,
        "instance_id": f"{repo.replace('/', '__')}-{idx}",
        "pull_number": idx,
        "base_commit": f"deadbeef{idx}",
        "patch": _make_patch(paths),
        "test_patch": _make_patch(test_paths or [f"tests/test_{idx}.py"]),
        "problem_statement": problem_statement or "x" * 900,
        "created_at": created_at,
        "FAIL_TO_PASS": [f"tests/test_{idx}.py::test_{n}" for n in range(fail_to_pass)],
        "PASS_TO_PASS": [
            f"tests/test_{idx}.py::test_pass_{n}" for n in range(pass_to_pass)
        ],
    }


def test_rank_repo_windows_prefers_related_window():
    rows = [
        _make_row("alpha/repo", 1, "2025-01-01T00:00:00Z", ["src/a.py"]),
        _make_row("alpha/repo", 2, "2025-01-02T00:00:00Z", ["src/a.py"]),
        _make_row("alpha/repo", 3, "2025-01-03T00:00:00Z", ["src/a.py"]),
        _make_row("beta/repo", 1, "2025-02-01T00:00:00Z", ["src/x.py"]),
        _make_row("beta/repo", 2, "2025-02-02T00:00:00Z", ["src/y.py"]),
        _make_row("beta/repo", 3, "2025-02-03T00:00:00Z", ["src/z.py"]),
    ]

    report = rank_repo_windows(
        rows,
        RepoWindowRankingConfig(
            window_sizes=(3,),
            curation=SequentialCurationConfig(
                instances_per_repo=3,
                min_repo_tasks=3,
                min_changed_lines=4,
                min_test_changed_lines=4,
                min_unique_code_files=1,
                max_top_code_file_share=1.0,
            ),
        ),
    )

    assert report["ranked_repos"][0]["repo"] == "alpha/repo"
    best = report["ranked_repos"][0]["best_overall_window"]
    assert best["selected_instance_ids"] == [
        "alpha__repo-1",
        "alpha__repo-2",
        "alpha__repo-3",
    ]
    assert best["repeated_code_file_touches"] == 3


def test_rank_repo_windows_dedupes_duplicate_instance_ids():
    rows = [
        _make_row("alpha/repo", 1, "2025-01-01T00:00:00Z", ["src/a.py"]),
        _make_row("alpha/repo", 1, "2025-01-01T00:00:00Z", ["src/a.py"]),
        _make_row("alpha/repo", 2, "2025-01-02T00:00:00Z", ["src/a.py"]),
        _make_row("alpha/repo", 3, "2025-01-03T00:00:00Z", ["src/a.py"]),
    ]

    report = rank_repo_windows(
        rows,
        RepoWindowRankingConfig(
            window_sizes=(3,),
            curation=SequentialCurationConfig(
                instances_per_repo=3,
                min_repo_tasks=3,
                min_changed_lines=4,
                min_test_changed_lines=4,
                min_unique_code_files=1,
                max_top_code_file_share=1.0,
            ),
        ),
    )

    assert report["counts"]["raw_total"] == 4
    assert report["counts"]["deduped_total"] == 3
    assert report["ranked_repos"][0]["best_overall_window"][
        "selected_instance_ids"
    ] == [
        "alpha__repo-1",
        "alpha__repo-2",
        "alpha__repo-3",
    ]


def test_cheap_solvability_metadata_prefers_localized_bugfix():
    focused = derive_cheap_task_solvability_metadata(
        _make_row(
            "alpha/repo",
            1,
            "2025-01-01T00:00:00Z",
            ["src/a.py"],
            problem_statement=(
                "Bug: parser fails on empty input.\n\n"
                "Steps to reproduce:\n1. call parse('')\n"
                "Expected: empty result\nActual: traceback"
            ),
            fail_to_pass=2,
            pass_to_pass=30,
        )
    )
    sprawling = derive_cheap_task_solvability_metadata(
        _make_row(
            "beta/repo",
            1,
            "2025-01-01T00:00:00Z",
            [
                "src/a.py",
                "src/b.py",
                "src/c.py",
                "src/d.py",
                "src/e.py",
            ],
            problem_statement=(
                "Feature request: add support for a new workflow.\n\n"
                "```py\nexample()\n```\n" * 4
            ),
            test_paths=[
                "tests/test_1.py",
                "tests/test_2.py",
                "tests/test_3.py",
            ],
            fail_to_pass=30,
            pass_to_pass=200,
        )
    )

    assert focused["solvability_label"] == "preferred"
    assert focused["solvability_score"] > sprawling["solvability_score"]
    assert sprawling["solvability_label"] == "avoid"


def test_rank_repo_windows_exposes_solvability_summary_fields():
    rows = [
        _make_row(
            "alpha/repo",
            1,
            "2025-01-01T00:00:00Z",
            ["src/a.py"],
            problem_statement=(
                "Bug: parser fails.\nExpected behavior differs from actual behavior.\n"
                + ("details " * 120)
            ),
            fail_to_pass=2,
        ),
        _make_row(
            "alpha/repo",
            2,
            "2025-01-02T00:00:00Z",
            ["src/a.py"],
            problem_statement="Bug: regression in parser.\n" + ("details " * 120),
            fail_to_pass=2,
        ),
        _make_row(
            "alpha/repo",
            3,
            "2025-01-03T00:00:00Z",
            ["src/a.py"],
            problem_statement="Bug: traceback while parsing.\n" + ("details " * 120),
            fail_to_pass=2,
        ),
    ]

    report = rank_repo_windows(
        rows,
        RepoWindowRankingConfig(
            window_sizes=(3,),
            curation=SequentialCurationConfig(
                instances_per_repo=3,
                min_repo_tasks=3,
                min_changed_lines=4,
                min_test_changed_lines=4,
                min_unique_code_files=1,
                max_top_code_file_share=1.0,
            ),
        ),
    )

    repo_report = report["ranked_repos"][0]
    assert repo_report["preferred_task_count"] == 3
    assert repo_report["screenable_task_count"] == 3
    assert repo_report["average_solvability_score"] >= 4.0
    best_window = repo_report["best_overall_window"]
    assert best_window["preferred_task_count"] == 3
    assert best_window["screenable_task_count"] == 3
    assert best_window["average_solvability_score"] >= 4.0
    assert repo_report["top_solvable_tasks"][0]["solvability_label"] == "preferred"
