"""
Human baseline system for continual learning benchmark.

Allows a human to interact with tasks through a CLI interface,
providing a baseline for comparison with automated systems.
"""

import ast
from types import UnionType
from typing import Any, Union, get_args, get_origin

from pydantic import ValidationError

from ...interface import ContinualLearningSystem, Observation, Query, Response
from ...registry import register_system


def _unwrap_optional(annotation: Any) -> tuple[Any, bool]:
    """Return the inner type for Optional[T] annotations."""
    origin = get_origin(annotation)
    if origin not in (Union, UnionType):
        return annotation, False

    args = get_args(annotation)
    non_none_args = [arg for arg in args if arg is not type(None)]
    if len(non_none_args) == 1 and len(non_none_args) != len(args):
        return non_none_args[0], True
    return annotation, False


def _format_annotation(annotation: Any) -> str:
    base, optional = _unwrap_optional(annotation)
    if hasattr(base, "__name__"):
        label = base.__name__
    else:
        label = str(base)
    return f"{label} | None" if optional else label


def _parse_scalar_value(raw: str, annotation: Any) -> Any:
    """Parse a guided-input value using the field annotation."""
    base, optional = _unwrap_optional(annotation)
    if raw == "":
        if optional:
            return None
        raise ValueError("This field is required.")

    if base is str:
        return raw
    if base is int:
        return int(raw)
    if base is float:
        return float(raw)
    if base is bool:
        normalized = raw.strip().lower()
        if normalized in {"true", "t", "yes", "y", "1"}:
            return True
        if normalized in {"false", "f", "no", "n", "0"}:
            return False
        raise ValueError("Enter true/false, yes/no, or 1/0.")

    try:
        return ast.literal_eval(raw)
    except (SyntaxError, ValueError) as exc:
        raise ValueError("Enter a valid value for this field.") from exc


@register_system("human")
class Human(ContinualLearningSystem):
    """
    Interactive CLI system where a human provides responses to tasks.

    This serves as a human baseline for benchmark evaluation.
    """

    supports_baseline: bool = False
    parallel_safe: bool = False  # reads from stdin; multiple instances would conflict

    def __init__(self, name: str = "human_baseline"):
        self._name = name
        self.interaction_count = 0
        self.pending_feedback: str | None = None

    def _read_input_line(self, prompt: str = "") -> str:
        try:
            return input(prompt)
        except (EOFError, KeyboardInterrupt):
            print("\n\n⚠️  Interrupted. Exiting...")
            raise KeyboardInterrupt("User interrupted")

    def _read_string_field(self, field_name: str, type_label: str) -> str:
        print(f"   {field_name} ({type_label}) [single line, or <<EOF for multiline]")
        raw = self._read_input_line("   > ")
        if raw.strip() != "<<EOF":
            return raw

        print("   Enter lines, then EOF on its own line.")
        lines = []
        while True:
            line = self._read_input_line()
            if line == "EOF":
                return "\n".join(lines)
            lines.append(line)

    def _prompt_for_guided_payload(self, schema) -> dict[str, Any]:
        print("\n✍️  Guided input mode")
        print("   Fill in each field. Leave optional fields blank.")

        payload: dict[str, Any] = {}
        for field_name, field in schema.model_fields.items():
            type_label = _format_annotation(field.annotation)
            if field.description:
                print(f"\n   {field_name}: {field.description}")

            while True:
                try:
                    base_type, _ = _unwrap_optional(field.annotation)
                    if base_type is str:
                        value = self._read_string_field(field_name, type_label)
                    else:
                        raw = self._read_input_line(
                            f"\n   {field_name} ({type_label}): "
                        )
                        value = _parse_scalar_value(raw, field.annotation)
                    payload[field_name] = value
                    break
                except ValueError as exc:
                    print(f"   Invalid value for {field_name}: {exc}")

        return payload

    def _collect_action(self, schema):
        while True:
            try:
                payload = self._prompt_for_guided_payload(schema)
                return schema.model_validate(payload)
            except (ValidationError, ValueError, TypeError) as exc:
                print(f"\n⚠️  Invalid response: {exc}")
                print("   Let's try that again.")

    def respond(self, query: Query) -> Response:
        """
        Display query to human and collect their response via CLI.
        """
        # Display header
        print("\n" + "=" * 80)
        self.interaction_count += 1
        print(f"INTERACTION {self.interaction_count}")
        print("=" * 80)

        if self.pending_feedback:
            print("\n📊 FEEDBACK:")
            print(f"   {self.pending_feedback}")
            self.pending_feedback = None

        # Display current prompt
        if query.prompt:
            print(f"\n{query.prompt}")

        schema = query.response_schema
        fields = {
            k: _format_annotation(v.annotation) for k, v in schema.model_fields.items()
        }
        print(f"\n📋 Expected fields: {fields}")

        action = self._collect_action(schema)

        return Response(
            action=action,
            metadata={
                "interaction_count": self.interaction_count,
                "system_type": "human",
            },
        )

    def reset(self) -> None:
        """Reset for a new task."""
        self.interaction_count = 0
        self.pending_feedback = None
        print("\n" + "🔄" * 40)
        print("TASK RESET - Starting new task instance")
        print("🔄" * 40)

    def get_run_artifacts(self) -> dict[str, Any]:
        """Export final human-run bookkeeping for analysis."""
        return {
            "artifact_type": "human",
            "interaction_count": self.interaction_count,
            "pending_feedback": self.pending_feedback,
        }

    def observe(
        self, observation: Observation, next_query: Query | None = None
    ) -> None:
        """Hold task feedback to display before the next prompt."""
        self.pending_feedback = observation.content

    @property
    def name(self) -> str:
        return self._name


def run_human_baseline(task, display_results: bool = True):
    """
    Run a task with human baseline system and optionally display results.

    Args:
        task: ContinualLearningTask instance to run
        display_results: Whether to display final results (default: True)

    Returns:
        TaskResult with evaluation metrics
    """
    from ...interface import run_task

    # Display task information
    print("\n" + "🎯" * 40)
    print(f"TASK: {task.name}")
    if task.description:
        print(f"DESCRIPTION: {task.description}")
    print("🎯" * 40)
    print("\nYou will be prompted to respond to queries.")
    print("The task will provide feedback after each response.")
    print("Responses use guided per-field input.")
    print("Try to learn and improve over time!\n")

    input("Press Enter to start...")

    # Create human system and run task
    human_system = Human()
    result = run_task(task, human_system)

    # Display results
    if display_results:
        print("\n" + "🏁" * 40)
        print("TASK COMPLETE - RESULTS")
        print("🏁" * 40)
        print(f"\n📊 FINAL SCORE: {result.score:.4f}")
        print("\n📝 SUMMARY:")
        print(f"   {result.summary}")
        print("\n📈 DETAILED METRICS:")
        for metric, value in result.metrics.items():
            if isinstance(value, float):
                print(f"   {metric}: {value:.4f}")
            else:
                print(f"   {metric}: {value}")
        print("\n" + "=" * 80 + "\n")

    return result
