import pytest
from pydantic import BaseModel

from src.interface import Query
from src.systems.icl import system as icl_module
from src.systems.icl.system import ICLSystem
from src.systems.icl_notepad import system as icl_notepad_module
from src.systems.icl_notepad.system import ICLNotepadSystem
from src.systems.utils.token_budget import (
    get_model_context_window,
    resolve_context_token_limit,
)
from src.usage import UsageEvent


class DummyAction(BaseModel):
    answer: str = "ok"


class DummyParsedResponse:
    answer: str = "ok"
    notepad_update = None

    def model_dump_json(self) -> str:
        return '{"answer":"ok"}'


def _usage_event(input_tokens: int) -> UsageEvent:
    return UsageEvent(
        call_type="completion",
        model="test-model",
        input_tokens=input_tokens,
        output_tokens=1,
        total_tokens=input_tokens + 1,
        cost_usd=0.0,
        pricing_source="test",
    )


@pytest.mark.parametrize(
    ("system_cls", "module"),
    [
        (ICLSystem, icl_module),
        (ICLNotepadSystem, icl_notepad_module),
    ],
)
def test_fifo_systems_truncate_from_api_reported_input_tokens(
    monkeypatch, system_cls, module
):
    monkeypatch.setattr(
        module.litellm,
        "token_counter",
        lambda *, model, messages: len(messages) * 100,
    )
    monkeypatch.setattr(
        module,
        "completion_with_structured_output",
        lambda **kwargs: (DummyParsedResponse(), _usage_event(200)),
    )

    system = system_cls(model="test-model", max_tokens=350, reserve_tokens=0)
    response = system.respond(Query(prompt="hello", response_schema=DummyAction))

    assert response.metadata["has_truncated"] is True
    assert system.truncation_count == 1
    assert [message["role"] for message in system.messages] == ["assistant"]
    assert response.metadata["context_tokens"] == 200


def test_icl_default_context_limit_is_model_based(monkeypatch):
    monkeypatch.setattr(
        "src.systems.utils.token_budget.litellm.get_model_info",
        lambda model: {"max_input_tokens": 128_000},
    )

    assert resolve_context_token_limit("some-provider-model", None) == 126_976
    assert ICLSystem(model="gpt-4o-mini").max_tokens == 126_976
    assert ICLSystem(model="gpt-4o-mini", max_tokens=32_768).max_tokens == 32_768


def test_model_context_window_requires_provider_metadata(monkeypatch):
    monkeypatch.setattr(
        "src.systems.utils.token_budget.litellm.get_model_info",
        lambda model: {},
    )

    with pytest.raises(ValueError, match="Unknown context window"):
        get_model_context_window("unknown-model")


def test_icl_truncates_against_system_prompt_and_schema_overhead(monkeypatch):
    monkeypatch.setattr(
        icl_module.litellm,
        "token_counter",
        lambda *, model, messages: len(messages) * 100,
    )
    monkeypatch.setattr(ICLSystem, "_response_schema_tokens", lambda *args: 75)

    captured = {}

    def fake_completion(**kwargs):
        captured["messages"] = kwargs["messages"]
        return DummyParsedResponse(), _usage_event(200)

    monkeypatch.setattr(
        icl_module, "completion_with_structured_output", fake_completion
    )

    system = ICLSystem(
        model="test-model",
        max_tokens=250,
        reserve_tokens=0,
        system_prompt="system",
    )
    system.respond(Query(prompt="hello", response_schema=DummyAction))

    response_roles = [message["role"] for message in captured["messages"]]
    assert response_roles
    assert response_roles == ["system"]
    assert system.has_truncated_flag is True
