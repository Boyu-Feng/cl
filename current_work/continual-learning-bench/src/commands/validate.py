"""
clbench validate task <name>
clbench validate system <name>

Validates that a task or system implementation is correctly structured before
running a long benchmark. Exits 0 on success, 1 on warnings, 2 on errors.
"""

import argparse
import importlib
import inspect
import json
import sys
from pathlib import Path
from typing import Any

_SRC_ROOT = Path(__file__).resolve().parent.parent
_TASKS_ROOT = _SRC_ROOT / "tasks"
_SYSTEMS_ROOT = _SRC_ROOT / "systems"


class _Result:
    def __init__(self) -> None:
        self._errors: list[str] = []
        self._warnings: list[str] = []
        self._ok: list[str] = []

    def ok(self, msg: str) -> None:
        self._ok.append(msg)

    def warn(self, msg: str) -> None:
        self._warnings.append(msg)

    def error(self, msg: str) -> None:
        self._errors.append(msg)

    def print_summary(self) -> int:
        for msg in self._ok:
            print(f"  [ok]   {msg}")
        for msg in self._warnings:
            print(f"  [warn] {msg}")
        for msg in self._errors:
            print(f"  [err]  {msg}")
        if self._errors:
            return 2
        if self._warnings:
            return 1
        return 0


def _check_json_file(path: Path, result: _Result) -> dict[str, Any] | None:
    try:
        with open(path) as f:
            data = json.load(f)
        result.ok(f"{path.name} is valid JSON")
        return data
    except json.JSONDecodeError as e:
        result.error(f"{path.name}: invalid JSON — {e}")
        return None


def validate_task(name: str) -> int:
    result = _Result()
    task_dir = _TASKS_ROOT / name

    print(f"Validating task: {name}")

    # 1. Directory structure
    if not task_dir.exists():
        print(f"  [err]  src/tasks/{name}/ not found")
        return 2
    result.ok(f"src/tasks/{name}/ exists")

    task_file = task_dir / "task.py"
    if not task_file.exists():
        result.error(f"src/tasks/{name}/task.py not found")
    else:
        result.ok("task.py found")

    init_file = task_dir / "__init__.py"
    if not init_file.exists():
        result.warn("__init__.py missing — add it for package consistency")
    else:
        result.ok("__init__.py found")

    # 2. Import
    cls = None
    try:
        importlib.import_module(f"src.tasks.{name}.task")
        result.ok("module imports cleanly")
    except Exception as e:
        result.error(f"import failed: {e}")
        return result.print_summary()

    # 3. Registry
    try:
        from src.registry import get_task_class, clear_registry_cache

        clear_registry_cache()
        cls = get_task_class(name)
        result.ok(f"registered as '{name}'")
    except KeyError as e:
        result.error(f"not registered: {e}")
    except Exception as e:
        result.error(f"registry error: {e}")

    if cls is None:
        return result.print_summary()

    from src.interface import require_query_instance_identity

    # 4. num_instances in __init__
    try:
        sig = inspect.signature(cls.__init__)
        if "num_instances" in sig.parameters:
            result.ok("num_instances parameter present")
        else:
            result.error(
                "num_instances missing from __init__ — the interface requires it. "
                "Add: num_instances: int = 10"
            )
    except Exception as e:
        result.warn(f"could not inspect __init__: {e}")

    # 5. Abstract methods implemented
    if inspect.isabstract(cls):
        abstract = {
            name
            for name, val in inspect.getmembers(cls)
            if getattr(val, "__isabstractmethod__", False)
        }
        result.error(f"abstract methods not implemented: {abstract}")
    else:
        result.ok(
            "build_canonical_run_state(), build_current_query(), step(), evaluate(), and name all implemented"
        )

    # 6. Instantiate and call reset() with minimal params
    try:
        task = cls(num_instances=1)
        result.ok("instantiates with num_instances=1")
        try:
            query = task.reset()
            result.ok("reset() returns a Query")
            if not hasattr(query, "prompt") or not hasattr(query, "response_schema"):
                result.error(
                    "reset() returned invalid Query (missing prompt or response_schema)"
                )
            else:
                try:
                    require_query_instance_identity(
                        query, context=f"{cls.__name__}.reset()"
                    )
                    result.ok("reset() query includes stable instance identity")
                except ValueError as e:
                    result.error(str(e))
        except NotImplementedError:
            result.error(
                "reset() raises NotImplementedError — implement the task lifecycle methods"
            )
        except Exception as e:
            result.warn(f"reset() raised an exception with num_instances=1: {e}")
    except Exception as e:
        result.warn(f"could not instantiate with num_instances=1: {e}")

    # 7. get_instance_outcomes()
    try:
        task_check = cls(num_instances=1)
        outcomes = task_check.get_instance_outcomes()
        if isinstance(outcomes, list):
            result.ok("get_instance_outcomes() returns a list")
        else:
            result.error("get_instance_outcomes() must return list[InstanceOutcome]")
    except Exception as e:
        result.warn(f"get_instance_outcomes() raised: {e}")

    # 8. Baseline-instance reset contract
    try:
        baseline_zero = cls(num_instances=2)
        query_zero = baseline_zero.reset_baseline_instance(0)
        if hasattr(query_zero, "prompt") and hasattr(query_zero, "response_schema"):
            result.ok("reset_baseline_instance(0) returns a Query")
            try:
                require_query_instance_identity(
                    query_zero, context=f"{cls.__name__}.reset_baseline_instance(0)"
                )
                result.ok("reset_baseline_instance(0) query includes instance identity")
            except ValueError as e:
                result.error(str(e))
        else:
            result.error(
                "reset_baseline_instance(0) returned invalid Query "
                "(missing prompt or response_schema)"
            )

        baseline_one = cls(num_instances=2)
        query_one = baseline_one.reset_baseline_instance(1)
        if hasattr(query_one, "prompt") and hasattr(query_one, "response_schema"):
            result.ok("reset_baseline_instance(1) returns a Query")
            try:
                require_query_instance_identity(
                    query_one, context=f"{cls.__name__}.reset_baseline_instance(1)"
                )
                result.ok("reset_baseline_instance(1) query includes instance identity")
            except ValueError as e:
                result.error(str(e))
        else:
            result.error(
                "reset_baseline_instance(1) returned invalid Query "
                "(missing prompt or response_schema)"
            )

        distinguishable = (
            getattr(query_zero, "instance_id", None)
            != getattr(query_one, "instance_id", None)
            or getattr(query_zero, "instance_index", None)
            != getattr(query_one, "instance_index", None)
            or getattr(query_zero, "prompt", None) != getattr(query_one, "prompt", None)
        )
        if distinguishable:
            result.ok("baseline instances produce distinguishable first queries")
        else:
            result.error(
                "reset_baseline_instance(0) and (1) produced indistinguishable "
                "first queries; baseline workers must target different instances."
            )
    except NotImplementedError as e:
        result.error(f"reset_baseline_instance() not implemented: {e}")
    except Exception as e:
        result.warn(f"reset_baseline_instance() raised with num_instances=2: {e}")

    # 9. Variant JSON files
    variant_dir = task_dir / "variants"
    if variant_dir.exists():
        variant_files = list(variant_dir.glob("*.json"))
        if variant_files:
            try:
                from src.tasks.variants import TaskVariantSpec, clear_task_variant_cache

                clear_task_variant_cache()
                ok_count = 0
                for vf in sorted(variant_files):
                    data = _check_json_file(vf, result)
                    if data is not None:
                        try:
                            data.pop("_canary", None)
                            spec = TaskVariantSpec.model_validate(data)
                            if spec.id != vf.stem:
                                result.error(
                                    f"{vf.name}: id '{spec.id}' must match filename '{vf.stem}'"
                                )
                            else:
                                ok_count += 1
                        except Exception as e:
                            result.error(f"{vf.name}: schema validation failed — {e}")
                if ok_count == len(variant_files):
                    result.ok(f"{ok_count} variant(s) valid")
            except Exception as e:
                result.warn(f"variant validation skipped: {e}")
        else:
            result.warn("no variant files found in variants/ (add at least one)")
    else:
        result.warn("no variants/ directory — add variants/*.json for named presets")

    # 10. Schedule JSON files
    schedule_dir = task_dir / "schedules"
    legacy_rollout_dir = task_dir / "rollouts"
    if not schedule_dir.exists() and legacy_rollout_dir.exists():
        schedule_dir = legacy_rollout_dir
        result.warn("using legacy rollouts/ directory; prefer schedules/")

    if schedule_dir.exists():
        schedule_files = list(schedule_dir.glob("*.json"))
        if schedule_files:
            try:
                from src.tasks.schedules import (
                    TaskScheduleSpec,
                    clear_task_schedule_cache,
                )

                clear_task_schedule_cache()
                ok_count = 0
                for sf in sorted(schedule_files):
                    data = _check_json_file(sf, result)
                    if data is not None:
                        try:
                            data.pop("_canary", None)
                            spec = TaskScheduleSpec.model_validate(data)
                            if spec.id != sf.stem:
                                result.error(
                                    f"{sf.name}: id '{spec.id}' must match filename '{sf.stem}'"
                                )
                            else:
                                ok_count += 1
                        except Exception as e:
                            result.error(f"{sf.name}: schema validation failed — {e}")
                if ok_count == len(schedule_files):
                    result.ok(f"{ok_count} schedule(s) valid")
                if any(sf.stem == "default" for sf in schedule_files):
                    result.ok("default.json schedule present (used by clbench run-all)")
                else:
                    result.error(
                        "default.json schedule missing — every task must provide "
                        "schedules/default.json so it can be included in clbench run-all"
                    )
            except Exception as e:
                result.warn(f"schedule validation skipped: {e}")
        else:
            result.error("no schedule files found — add schedules/*.json at minimum")
    else:
        result.error(
            "no schedules/ directory — add schedules/*.json for benchmark entry points"
        )

    return result.print_summary()


def validate_system(name: str) -> int:
    result = _Result()
    system_dir = _SYSTEMS_ROOT / name

    print(f"Validating system: {name}")

    # 1. Directory structure
    if not system_dir.exists():
        print(f"  [err]  src/systems/{name}/ not found")
        return 2
    result.ok(f"src/systems/{name}/ exists")

    system_file = system_dir / "system.py"
    if not system_file.exists():
        result.error(f"src/systems/{name}/system.py not found")
    else:
        result.ok("system.py found")

    init_file = system_dir / "__init__.py"
    if not init_file.exists():
        result.warn("__init__.py missing — add it for package consistency")
    else:
        result.ok("__init__.py found")

    # 2. Import
    try:
        importlib.import_module(f"src.systems.{name}.system")
        result.ok("module imports cleanly")
    except Exception as e:
        result.error(f"import failed: {e}")
        return result.print_summary()

    # 3. Registry
    cls = None
    try:
        from src.registry import get_system_class, clear_registry_cache

        clear_registry_cache()
        cls = get_system_class(name)
        result.ok(f"registered as '{name}'")
    except KeyError as e:
        result.error(f"not registered: {e}")
    except Exception as e:
        result.error(f"registry error: {e}")

    if cls is None:
        return result.print_summary()

    # 4. Abstract methods implemented
    if inspect.isabstract(cls):
        abstract = {
            attr
            for attr, val in inspect.getmembers(cls)
            if getattr(val, "__isabstractmethod__", False)
        }
        result.error(f"abstract methods not implemented: {abstract}")
    else:
        result.ok("respond(), reset(), name all implemented")

    # 5. Instantiate
    instance = None
    try:
        instance = cls()
        result.ok("instantiates with default params")
    except Exception as e:
        result.warn(f"could not instantiate with default params: {e}")

    if instance is None:
        return result.print_summary()

    # 6. name property
    try:
        n = instance.name
        if isinstance(n, str) and n:
            result.ok(f"name = '{n}'")
        else:
            result.error("name property must return a non-empty string")
    except Exception as e:
        result.error(f"name property raised: {e}")

    # 7. reset() callable
    try:
        instance.reset()
        result.ok("reset() runs without error")
    except NotImplementedError:
        result.error("reset() raises NotImplementedError — implement it")
    except Exception as e:
        result.warn(f"reset() raised: {e}")

    # 8. Runtime capability flags
    for attr in ("supports_baseline", "parallel_safe"):
        value = getattr(cls, attr, None)
        if isinstance(value, bool):
            result.ok(f"{attr} = {value}")
        else:
            result.warn(f"{attr} should be a class-level bool")

    observe = getattr(instance, "observe", None)
    if callable(observe):
        result.ok("observe() available for post-response feedback")
    else:
        result.warn(
            "observe() missing — implement it if the system learns after feedback"
        )

    return result.print_summary()


def cmd_validate(args: list[str]) -> None:
    parser = argparse.ArgumentParser(
        prog="clbench validate",
        description="Validate a task or system implementation.",
    )
    parser.add_argument("kind", choices=["task", "system"])
    parser.add_argument("name", help="Registered name (e.g. poker, icl).")
    parsed = parser.parse_args(args)

    if parsed.kind == "task":
        code = validate_task(parsed.name)
    else:
        code = validate_system(parsed.name)

    if code == 0:
        print("\nAll checks passed.")
    elif code == 1:
        print("\nPassed with warnings.")
    else:
        print("\nValidation failed. Fix errors above before running the benchmark.")

    sys.exit(code)
