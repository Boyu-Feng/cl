"""
clbench doctor

Checks prerequisites and environment configuration, printing a clear
pass/fail/warn for each check with fix hints. Exits 0 if all checks pass.
"""

import importlib.util
import os
import shutil
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _check(label: str, ok: bool, fix: str = "", warn: bool = False) -> bool:
    """Print one check line. Returns True if ok."""
    if ok:
        print(f"  [ok]   {label}")
    elif warn:
        print(f"  [warn] {label}")
        if fix:
            print(f"         Fix: {fix}")
    else:
        print(f"  [err]  {label}")
        if fix:
            print(f"         Fix: {fix}")
    return ok


def cmd_doctor(_args: list[str]) -> None:
    print("clbench doctor — environment checks\n")
    failures = 0

    # ── Python version ──────────────────────────────────────────────────────
    print("Python")
    major, minor = sys.version_info[:2]
    ok = major == 3 and minor >= 13
    if not _check(
        f"Python {major}.{minor} (need >=3.13)",
        ok,
        fix="Install Python 3.13+ or use: uv python install 3.13",
    ):
        failures += 1

    # ── Core dependencies ────────────────────────────────────────────────────
    print("\nCore dependencies")
    core_deps = {
        "pydantic": "pydantic",
        "litellm": "litellm",
        "tqdm": "tqdm",
        "python-dotenv": "dotenv",
    }
    for label, module in core_deps.items():
        found = importlib.util.find_spec(module.replace("-", "_")) is not None
        if not _check(
            f"{label}",
            found,
            fix=f"uv sync  (or: uv add {label})",
        ):
            failures += 1

    # ── API keys ─────────────────────────────────────────────────────────────
    print("\nAPI keys")
    keys = {
        "ANTHROPIC_API_KEY": "required by claude-* models (ACE, icl with claude)",
        "OPENAI_API_KEY": "required by gpt-* models (icl default)",
    }
    for key, note in keys.items():
        present = bool(os.environ.get(key))
        _check(
            f"{key} set  ({note})",
            present,
            fix=f"export {key}=<your-key>  or add it to .env",
            warn=True,  # warn rather than error — depends on which models you use
        )

    # ── Optional: Docker ─────────────────────────────────────────────────────
    print("\nOptional tools")
    docker_found = shutil.which("docker") is not None
    _check(
        "docker available  (required by codebase_adaptation task)",
        docker_found,
        fix="Install Docker Desktop from https://www.docker.com/products/docker-desktop",
        warn=True,
    )

    # ── Task-specific optional extras ────────────────────────────────────────
    print("\nTask extras (install with: uv sync --extra <name>)")
    task_extras = {
        "exploitable_poker": ("texasholdem", "texasholdem"),
        "codebase_adaptation": ("datasets", "datasets"),
        "sales_prediction": ("pandas", "pandas"),
        "cohort_studies": ("lifelines", "lifelines"),
        "database_exploration": ("huggingface_hub", "huggingface-hub"),
    }
    for task_label, (module, pkg) in task_extras.items():
        found = importlib.util.find_spec(module) is not None
        _check(
            f"{task_label} extra ({pkg})",
            found,
            fix=f"uv sync --extra {task_label}",
            warn=True,
        )

    # ── Registry smoke check ─────────────────────────────────────────────────
    print("\nRegistry")
    try:
        from src.registry import list_tasks, list_systems

        tasks = list_tasks()
        systems = list_systems()
        _check(
            f"discovered {len(tasks)} task(s): {', '.join(tasks)}",
            bool(tasks),
            fix="Ensure src/tasks/<name>/task.py exists with @register_task('<name>')",
        )
        _check(
            f"discovered {len(systems)} system(s): {', '.join(systems)}",
            bool(systems),
            fix="Ensure src/systems/<name>/system.py exists with @register_system('<name>')",
        )
    except Exception as e:
        print(f"  [err]  registry import failed: {e}")
        failures += 1

    print()
    if failures:
        print(f"{failures} error(s) found. Fix them before running benchmarks.")
        sys.exit(2)
    else:
        print("All required checks passed.")
        sys.exit(0)
