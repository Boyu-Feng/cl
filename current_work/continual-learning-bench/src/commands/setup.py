"""
clbench setup

Runs one-time setup for a task (e.g. downloading datasets from Hugging Face).
Tasks that require setup declare ``has_setup = True`` on their class and
override :meth:`ContinualLearningTask.setup`.
"""

import argparse
import sys

from ..registry import get_task_class, list_tasks


def cmd_setup(args: list[str]) -> None:
    parser = argparse.ArgumentParser(
        prog="clbench setup",
        description=(
            "Run one-time setup for a benchmark task. "
            "With no arguments, lists tasks that have custom setup."
        ),
    )
    parser.add_argument(
        "task_name",
        nargs="?",
        help="Task to set up (omit to list tasks with custom setup).",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run setup for every task that has custom setup.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run setup even when output artifacts already exist.",
    )
    parsed = parser.parse_args(args)

    if parsed.all:
        _setup_all(force=parsed.force)
    elif parsed.task_name:
        _setup_one(parsed.task_name, force=parsed.force)
    else:
        _list_setup_tasks()


def _list_setup_tasks() -> None:
    """Print a table of tasks and whether they have custom setup."""
    tasks = list_tasks()
    rows = []
    for name in tasks:
        try:
            cls = get_task_class(name)
        except Exception:
            continue
        has = getattr(cls, "has_setup", False)
        rows.append((name, has))

    print("Tasks with custom setup:\n")
    has_any = False
    for name, has in rows:
        marker = "[setup]" if has else "      "
        print(f"  {marker}  {name}")
        if has:
            has_any = True

    if not has_any:
        print("\n  (no tasks currently require setup)")
    else:
        print(
            "\nRun 'clbench setup <task>' to set up a specific task, "
            "or 'clbench setup --all' to set up all of them."
        )


def _setup_one(task_name: str, *, force: bool) -> None:
    import difflib

    try:
        cls = get_task_class(task_name)
    except KeyError:
        available = list_tasks()
        matches = difflib.get_close_matches(task_name, available, n=3)
        print(f"Error: Unknown task '{task_name}'.", file=sys.stderr)
        if matches:
            print(f"  Closest matches: {', '.join(matches)}", file=sys.stderr)
        print("  Run 'clbench setup' to see available tasks.", file=sys.stderr)
        sys.exit(1)

    if not getattr(cls, "has_setup", False):
        print(f"Task '{task_name}' has no custom setup — nothing to do.")
        return

    print(f"Setting up '{task_name}' (force={force})...")
    try:
        cls.setup(force=force)
    except SystemExit:
        raise
    except Exception as e:
        print(f"Error: Setup for '{task_name}' failed: {e}", file=sys.stderr)
        sys.exit(1)


def _setup_all(*, force: bool) -> None:
    tasks = list_tasks()
    to_run = []
    for name in tasks:
        try:
            cls = get_task_class(name)
        except Exception:
            continue
        if getattr(cls, "has_setup", False):
            to_run.append((name, cls))

    if not to_run:
        print("No tasks require setup.")
        return

    print(
        f"Running setup for {len(to_run)} task(s): {', '.join(n for n, _ in to_run)}\n"
    )
    failed = []
    for name, cls in to_run:
        print(f"--- {name} ---")
        try:
            cls.setup(force=force)
        except SystemExit as e:
            print(f"Setup for '{name}' exited early: {e}", file=sys.stderr)
            failed.append(name)
        except Exception as e:
            print(f"Setup for '{name}' failed: {e}", file=sys.stderr)
            failed.append(name)
        print()

    if failed:
        print(f"Failed: {', '.join(failed)}", file=sys.stderr)
        sys.exit(1)
    else:
        print("All setups complete.")
