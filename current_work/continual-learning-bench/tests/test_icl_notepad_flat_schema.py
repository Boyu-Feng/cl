"""Tests for ``_build_flat_schema`` wrapper handling in icl_notepad."""

from __future__ import annotations

import json

import pytest
from pydantic import BaseModel, Field

from src.systems.icl_notepad.system import _build_flat_schema


class _SubmissionLike(BaseModel):
    """Stand-in for a flat-keyed submission schema."""

    a: float = Field(ge=0, le=1)
    b: float = Field(ge=0, le=1)
    c: float = Field(ge=0, le=1)


def _full_task_payload() -> dict[str, float]:
    return {"a": 0.1, "b": 0.5, "c": 0.9}


class TestFlatSchemaCleanResponse:
    def test_full_payload_validates(self):
        schema = _build_flat_schema(_SubmissionLike)
        parsed = schema.model_validate({**_full_task_payload(), "notepad_update": "x"})
        assert parsed.a == 0.1
        assert parsed.b == 0.5
        assert parsed.c == 0.9
        assert parsed.notepad_update == "x"

    def test_notepad_update_is_optional(self):
        schema = _build_flat_schema(_SubmissionLike)
        parsed = schema.model_validate(_full_task_payload())
        assert parsed.notepad_update is None

    def test_extra_thought_is_ignored(self):
        schema = _build_flat_schema(_SubmissionLike)
        parsed = schema.model_validate(
            {"thought": "stray", **_full_task_payload(), "notepad_update": None}
        )
        assert "thought" not in parsed.model_dump()


class TestFlatSchemaWrapperUnwrap:
    def test_double_nested_wrapper_with_notepad_at_top(self):
        schema = _build_flat_schema(_SubmissionLike)
        parsed = schema.model_validate(
            {
                "thought": "Submitting now.",
                "tool_call": {
                    "tool": "submit",
                    "submission": _full_task_payload(),
                },
                "notepad_update": "after submission",
            }
        )
        assert parsed.a == 0.1
        assert parsed.b == 0.5
        assert parsed.c == 0.9
        assert parsed.notepad_update == "after submission"

    def test_single_nested_wrapper_preserves_notepad(self):
        schema = _build_flat_schema(_SubmissionLike)
        parsed = schema.model_validate(
            {
                "submission": _full_task_payload(),
                "notepad_update": "noted",
            }
        )
        assert parsed.a == 0.1
        assert parsed.notepad_update == "noted"

    def test_validate_json_path_unwraps_wrapper(self):
        schema = _build_flat_schema(_SubmissionLike)
        raw = json.dumps(
            {
                "thought": "...",
                "tool_call": {"submission": _full_task_payload()},
                "notepad_update": None,
            }
        )
        parsed = schema.model_validate_json(raw)
        assert parsed.b == 0.5
        assert parsed.notepad_update is None

    def test_wrapper_with_no_outer_notepad_still_parses(self):
        schema = _build_flat_schema(_SubmissionLike)
        parsed = schema.model_validate(
            {
                "thought": "...",
                "tool_call": {"submission": _full_task_payload()},
            }
        )
        assert parsed.c == 0.9
        assert parsed.notepad_update is None


class TestFlatSchemaFailureCases:
    def test_missing_task_fields_raises(self):
        schema = _build_flat_schema(_SubmissionLike)
        with pytest.raises(Exception):
            schema.model_validate({"thought": "...", "notepad_update": None})

    def test_partial_top_level_keys_does_not_descend(self):
        schema = _build_flat_schema(_SubmissionLike)
        with pytest.raises(Exception):
            schema.model_validate(
                {
                    "a": 0.1,
                    "deeply_nested": _full_task_payload(),
                    "notepad_update": None,
                }
            )


class TestFlatSchemaConflict:
    def test_task_schema_with_notepad_update_field_raises(self):
        class _Conflict(BaseModel):
            notepad_update: str | None = None

        with pytest.raises(ValueError, match="notepad_update"):
            _build_flat_schema(_Conflict)
