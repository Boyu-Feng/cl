"""Tests for feedback visibility in icl_notepad system.

When clear_context_between_instances=True (the default), the notepad is
the sole cross-instance memory.  The model must see the previous instance's
feedback so it can revise the notepad.  These tests verify that feedback
survives context clearing and is presented to the model on the next turn.
"""

from unittest.mock import patch

from pydantic import BaseModel

from src.interface import Observation, Query
from src.systems.icl_notepad.system import ICLNotepadSystem
from src.usage import UsageEvent


class DummyAction(BaseModel):
    answer: str = "x"


def _make_query(prompt: str = "What next?") -> Query:
    return Query(prompt=prompt, response_schema=DummyAction)


def _make_observation(content: str, instance_complete: bool = True) -> Observation:
    return Observation(
        content=content, metadata={"instance_complete": instance_complete}
    )


def _capture_prompt(system: ICLNotepadSystem, query: Query) -> str:
    """Call respond() with a mocked LLM and return the user message sent."""
    dummy_parsed = type(
        "Obj",
        (),
        {
            "answer": "x",
            "notepad_update": None,
            "model_dump_json": lambda self: '{"answer":"x"}',
        },
    )()
    dummy_usage = UsageEvent(
        call_type="completion",
        model="test-model",
        cost_usd=0.0,
        pricing_source="test",
    )

    with patch(
        "src.systems.icl_notepad.system.completion_with_structured_output",
        return_value=(dummy_parsed, dummy_usage),
    ) as mock_llm:
        system.respond(query)
        messages = mock_llm.call_args[1].get("messages") or mock_llm.call_args[0][0]
        # The user message is the last one (or the one after system prompt)
        user_msgs = [m for m in messages if m["role"] == "user"]
        return user_msgs[-1]["content"] if user_msgs else ""


class TestFeedbackSurvivesClearContext:
    """Feedback from observe() must be visible in the next respond() call,
    even when clear_context_between_instances=True."""

    def test_feedback_lost_on_clear(self):
        """Core bug: with clear_context=True, feedback should still appear
        in the next turn's prompt."""
        system = ICLNotepadSystem(
            model="gpt-5",
            clear_context_between_instances=True,
            provider_mode="litellm_chat",
        )
        system.reset()

        # Turn 1: model responds (writes notepad before seeing feedback)
        _capture_prompt(system, _make_query("Scan 1 data"))

        # Feedback arrives, instance complete → context cleared
        system.observe(_make_observation("Ground truth: channel A at 5 MHz"))

        # Context should be empty after clear
        assert system.messages == []

        # Turn 2: model should see the feedback from turn 1
        prompt_t2 = _capture_prompt(system, _make_query("Scan 2 data"))

        assert "Ground truth: channel A at 5 MHz" in prompt_t2, (
            "Feedback from previous instance must be visible in the next "
            "turn's prompt when clear_context_between_instances=True. "
            "Currently the feedback is added to messages and then immediately "
            "cleared, so the model never sees it."
        )

    def test_feedback_visible_when_context_not_cleared(self):
        """Sanity check: with clear_context=False, feedback is in the
        conversation history and naturally visible."""
        system = ICLNotepadSystem(
            model="gpt-5",
            clear_context_between_instances=False,
            provider_mode="litellm_chat",
        )
        system.reset()

        _capture_prompt(system, _make_query("Scan 1 data"))
        system.observe(_make_observation("Ground truth: channel A at 5 MHz"))

        # Messages should still contain the feedback
        assert any("Ground truth" in m["content"] for m in system.messages)

        _capture_prompt(system, _make_query("Scan 2 data"))
        # Feedback is in the conversation history, so it's visible
        # (the model sees all messages, not just the latest user message)
        feedback_in_history = any(
            "Ground truth" in m["content"]
            for m in system.messages
            if m["role"] == "user"
        )
        assert feedback_in_history

    def test_stateless_suppresses_feedback(self):
        """In stateless mode, feedback should NOT carry forward."""
        system = ICLNotepadSystem(
            model="gpt-5",
            stateless=True,
            provider_mode="litellm_chat",
        )
        system.reset()

        _capture_prompt(system, _make_query("Scan 1 data"))
        system.observe(_make_observation("Ground truth: channel A at 5 MHz"))

        prompt_t2 = _capture_prompt(system, _make_query("Scan 2 data"))

        assert "Ground truth" not in prompt_t2, (
            "Stateless mode should suppress cross-instance feedback."
        )

    def test_within_instance_feedback_not_affected(self):
        """Mid-instance feedback (instance_complete=False) should work
        as before — it stays in the message history."""
        system = ICLNotepadSystem(
            model="gpt-5",
            clear_context_between_instances=True,
            provider_mode="litellm_chat",
        )
        system.reset()

        _capture_prompt(system, _make_query("Step 1"))
        system.observe(_make_observation("Partial result", instance_complete=False))

        # Messages should NOT be cleared (instance not complete)
        assert len(system.messages) > 0
        assert any("Partial result" in m["content"] for m in system.messages)
