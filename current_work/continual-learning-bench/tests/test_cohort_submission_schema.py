"""Tests for the dynamic CohortSubmission schema's envelope-stripping validator.

Regression guards for Gemini's prompt-based structured-output fallback, which
sometimes carries the agent-loop ``{thought, tool_call: {...}}`` envelope into
the submission step.  Without normalisation, all 108 flat fields are reported
missing and the task aborts after the structured-output retry budget is
exhausted.
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip(
    "src.tasks.cohort_studies.tool_schemas",
    reason="cohort_studies optional dependencies (lifelines, pandas) are not installed.",
)

from src.tasks.cohort_studies.tool_schemas import build_submission_schema


def _full_payload() -> dict[str, float]:
    schema = build_submission_schema()
    return {name: 0.5 for name in schema.model_fields}


def test_flat_payload_validates():
    schema = build_submission_schema()
    payload = _full_payload()
    parsed = schema.model_validate(payload)
    assert parsed.model_dump()["voc_3_6__s12"] == 0.5


def test_top_level_thought_is_stripped():
    schema = build_submission_schema()
    payload = {"thought": "I will provide estimates...", **_full_payload()}
    parsed = schema.model_validate(payload)
    assert "thought" not in parsed.model_dump()


def test_double_nested_envelope_is_unwrapped():
    """Gemini's observed failure mode: ``{thought, tool_call: {tool, submission: {...}}}``."""
    schema = build_submission_schema()
    payload = {
        "thought": "I need to provide estimates for all 36 cohorts.",
        "tool_call": {
            "tool": "submit_cohort_report",
            "submission": _full_payload(),
        },
    }
    parsed = schema.model_validate(payload)
    assert parsed.model_dump()["voc_3_6__s12"] == 0.5


def test_single_wrapper_envelope_is_unwrapped():
    schema = build_submission_schema()
    payload = {"estimates": _full_payload()}
    parsed = schema.model_validate(payload)
    assert parsed.model_dump()["family_history_yes__s24"] == 0.5


def test_envelope_with_extra_sibling_fields_is_unwrapped():
    schema = build_submission_schema()
    payload = {
        "thought": "...",
        "tool_call": {
            "tool": "submit_cohort_report",
            "metadata": {"confidence": "low"},
            "submission": _full_payload(),
        },
    }
    parsed = schema.model_validate(payload)
    assert parsed.model_dump()["acron_lt2__s36"] == 0.5


def test_validate_json_path_unwraps_envelope():
    """``litellm`` callers go through ``model_validate_json``; cover that path too."""
    schema = build_submission_schema()
    payload = {
        "thought": "...",
        "tool_call": {
            "tool": "submit_cohort_report",
            "submission": _full_payload(),
        },
    }
    parsed = schema.model_validate_json(json.dumps(payload))
    dumped = parsed.model_dump()
    assert dumped["voc_3_6__s12"] == 0.5
    assert "thought" not in dumped


def test_partial_top_level_keys_keeps_top_level_view():
    """If any expected key is at the top level, the validator does not descend.

    This guards against a malformed payload where someone places a single
    expected key at the top while burying the rest of the data — we prefer
    failing loudly to silently selecting the wrong dict.
    """
    schema = build_submission_schema()
    payload = {"voc_3_6__s12": 0.5, "deeply_nested": _full_payload()}
    with pytest.raises(Exception):
        schema.model_validate(payload)
