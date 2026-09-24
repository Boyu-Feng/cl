import io
import json
import ssl
import urllib.error
from typing import Optional

import litellm
import pytest
from pydantic import BaseModel, Field

from src.errors import ProviderRefusalError
from src.systems.utils.provider_adapters import (
    ProviderTurnClient,
    _openai_strict_json_schema,
    _is_retryable_responses_exception,
    _resolve_model_max_output_tokens,
    detect_provider,
)
from src.usage import UsageEvent, summarize_usage_events


def _content_policy_error(message: str = "Invalid prompt: your prompt was flagged"):
    return litellm.ContentPolicyViolationError(
        message=message, model="gpt-5.4", llm_provider="openai"
    )


class AnswerSchema(BaseModel):
    answer: str


class OptionalActionSchema(BaseModel):
    thinking: str
    action: str
    amount: Optional[int] = Field(default=None)


class FakeResponse:
    def __init__(self, response_id: str, text: str):
        self.id = response_id
        self.output_text = text
        self.output = [
            {"type": "reasoning", "id": f"rs_{response_id}"},
            {
                "type": "message",
                "content": [{"type": "output_text", "text": text}],
            },
        ]
        self.usage = {
            "input_tokens": 10,
            "output_tokens": 5,
            "total_tokens": 15,
            "input_tokens_details": {"cached_tokens": 3},
            "output_tokens_details": {"reasoning_tokens": 2},
        }


def test_detect_provider_prefers_native_reasoning_families():
    assert detect_provider("gpt-5.4") == "openai"
    assert detect_provider("openai/o4-mini") == "openai"
    assert detect_provider("claude-opus-4-6") == "anthropic"
    assert detect_provider("gemini-2.5-pro") == "gemini"


def test_resolve_model_max_output_tokens_uses_model_info(monkeypatch):
    monkeypatch.setattr(
        "src.systems.utils.provider_adapters.litellm.get_model_info",
        lambda model: {"max_output_tokens": 12345},
    )

    assert _resolve_model_max_output_tokens("claude-test", None) == 12345
    assert _resolve_model_max_output_tokens("claude-test", 777) == 777


def test_openai_responses_chains_previous_response_id(monkeypatch):
    calls = []

    def fake_responses(**kwargs):
        calls.append(kwargs)
        return FakeResponse(f"resp_{len(calls)}", '{"answer": "ok"}')

    monkeypatch.setattr(
        "src.systems.utils.provider_adapters.litellm.responses", fake_responses
    )

    client = ProviderTurnClient(model="gpt-5.4", system_prompt="Be concise.")
    messages = [{"role": "user", "content": "first"}]
    first = client.respond_structured(messages=messages, response_schema=AnswerSchema)
    assert first.action.answer == "ok"

    messages = [
        *messages,
        {"role": "assistant", "content": first.assistant_record},
        {"role": "user", "content": "second"},
    ]
    second = client.respond_structured(messages=messages, response_schema=AnswerSchema)

    assert second.action.answer == "ok"
    assert calls[0]["store"] is True
    assert "previous_response_id" not in calls[0]
    assert calls[1]["previous_response_id"] == "resp_1"
    assert calls[1]["instructions"] == "Be concise."
    assert client.state.hidden_state_used is True


def test_openai_strict_schema_adds_required_additional_properties():
    schema = _openai_strict_json_schema(OptionalActionSchema.model_json_schema())

    assert schema["additionalProperties"] is False
    assert schema["required"] == ["thinking", "action", "amount"]
    assert "default" not in schema["properties"]["amount"]


def test_openai_responses_retries_bad_gateway_html(monkeypatch):
    calls = []

    def fake_responses(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise RuntimeError("BadGatewayError: OpenAIException - <!DOCTYPE html>")
        return FakeResponse("resp_ok", '{"answer": "ok"}')

    monkeypatch.setattr(
        "src.systems.utils.provider_adapters.litellm.responses", fake_responses
    )
    monkeypatch.setattr(
        "src.systems.utils.provider_adapters.time.sleep", lambda _: None
    )

    client = ProviderTurnClient(model="gpt-5.4")
    result = client.respond_structured(
        messages=[{"role": "user", "content": "hello"}],
        response_schema=AnswerSchema,
    )

    assert result.action.answer == "ok"
    assert len(calls) == 2
    assert _is_retryable_responses_exception(
        RuntimeError("BadGatewayError: OpenAIException - <!DOCTYPE html>")
    )


class TestResponsesRetryPolicy:
    """Coverage for the OpenAI Responses retry classifier."""

    def test_cloudflare_5xx_status_attribute_is_retryable(self):
        exc = RuntimeError("cf edge error")
        exc.status_code = 524
        assert _is_retryable_responses_exception(exc) is True

    def test_cloudflare_5xx_status_in_message_is_retryable(self):
        exc = RuntimeError("APIError: status_code: 520 origin returned unknown error")
        assert _is_retryable_responses_exception(exc) is True

    def test_request_timeout_status_is_retryable(self):
        exc = RuntimeError("HTTP Error 408: Request Timeout")
        assert _is_retryable_responses_exception(exc) is True

    def test_unrelated_4xx_is_not_retryable(self):
        exc = RuntimeError("HTTP Error 401: Unauthorized")
        assert _is_retryable_responses_exception(exc) is False

    def test_anthropic_overloaded_529_is_retryable(self):
        """Regression: 529 is Anthropic-specific (``overloaded_error``) and
        is not in any provider SDK's typed exception hierarchy. The policy
        of "all 5xx are transient" must catch it without an explicit code
        carve-out."""
        exc = RuntimeError("APIError: status_code: 529 overloaded_error")
        assert _is_retryable_responses_exception(exc) is True


class TestOpenAIContentPolicyRetry:
    """The reasoning-model safety classifier intermittently refuses long
    structured prompts whose individual moderation scores are well below any
    flagging threshold (mechanism: org-level cumulative-risk gate).  Without
    bounded retry these refusals abort whole rollouts even though a clean
    re-issue of the same prompt almost always succeeds.
    """

    def test_recovers_after_transient_refusal(self, monkeypatch):
        calls = []

        def fake_responses(**kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                raise _content_policy_error()
            return FakeResponse("resp_ok", '{"answer": "ok"}')

        monkeypatch.setattr(
            "src.systems.utils.provider_adapters.litellm.responses", fake_responses
        )
        monkeypatch.setattr(
            "src.systems.utils.provider_adapters.time.sleep", lambda _: None
        )

        client = ProviderTurnClient(model="gpt-5.4")
        result = client.respond_structured(
            messages=[{"role": "user", "content": "hello"}],
            response_schema=AnswerSchema,
        )

        assert result.action.answer == "ok"
        assert len(calls) == 2

    def test_exhausts_budget_and_raises_provider_refusal(self, monkeypatch):
        calls = []

        def fake_responses(**kwargs):
            calls.append(kwargs)
            raise _content_policy_error()

        monkeypatch.setattr(
            "src.systems.utils.provider_adapters.litellm.responses", fake_responses
        )
        monkeypatch.setattr(
            "src.systems.utils.provider_adapters.time.sleep", lambda _: None
        )

        client = ProviderTurnClient(
            model="gpt-5.4", content_policy_max_retries=2, content_policy_retry_delay=0
        )
        with pytest.raises(ProviderRefusalError) as excinfo:
            client.respond_structured(
                messages=[{"role": "user", "content": "hello"}],
                response_schema=AnswerSchema,
            )

        # 1 initial attempt + 2 retries = 3 total calls
        assert len(calls) == 3
        assert excinfo.value.kind == "content_policy"
        assert excinfo.value.provider == "openai"

    def test_zero_retries_disables_policy_loop(self, monkeypatch):
        calls = []

        def fake_responses(**kwargs):
            calls.append(kwargs)
            raise _content_policy_error()

        monkeypatch.setattr(
            "src.systems.utils.provider_adapters.litellm.responses", fake_responses
        )
        monkeypatch.setattr(
            "src.systems.utils.provider_adapters.time.sleep", lambda _: None
        )

        client = ProviderTurnClient(model="gpt-5.4", content_policy_max_retries=0)
        with pytest.raises(ProviderRefusalError):
            client.respond_structured(
                messages=[{"role": "user", "content": "hello"}],
                response_schema=AnswerSchema,
            )

        assert len(calls) == 1


def test_openai_stateless_passes_encrypted_reasoning(monkeypatch):
    calls = []

    def fake_responses(**kwargs):
        calls.append(kwargs)
        response = FakeResponse(f"resp_{len(calls)}", '{"answer": "ok"}')
        response.output[0]["encrypted_content"] = f"encrypted-{len(calls)}"
        return response

    monkeypatch.setattr(
        "src.systems.utils.provider_adapters.litellm.responses", fake_responses
    )

    client = ProviderTurnClient(model="gpt-5.4", openai_store=False)
    messages = [{"role": "user", "content": "first"}]
    client.respond_structured(messages=messages, response_schema=AnswerSchema)
    messages = [
        *messages,
        {"role": "assistant", "content": '{"answer": "ok"}'},
        {"role": "user", "content": "second"},
    ]
    client.respond_structured(messages=messages, response_schema=AnswerSchema)

    assert calls[0]["store"] is False
    assert calls[0]["include"] == ["reasoning.encrypted_content"]
    assert "previous_response_id" not in calls[1]
    assert calls[1]["input"][0]["encrypted_content"] == "encrypted-1"


def test_anthropic_preserves_thinking_blocks_between_turns():
    payloads = []
    responses = [
        {
            "id": "msg_1",
            "content": [
                {"type": "thinking", "thinking": "", "signature": "sig-1"},
                {"type": "text", "text": '{"answer": "one"}'},
            ],
            "usage": {"input_tokens": 4, "output_tokens": 6},
        },
        {
            "id": "msg_2",
            "content": [{"type": "text", "text": '{"answer": "two"}'}],
            "usage": {"input_tokens": 5, "output_tokens": 7},
        },
    ]

    client = ProviderTurnClient(model="claude-opus-4-6")

    def fake_anthropic_request(payload):
        payloads.append(payload)
        return responses[len(payloads) - 1]

    client._anthropic_request = fake_anthropic_request  # type: ignore[method-assign]

    messages = [{"role": "user", "content": "first"}]
    first = client.respond_structured(messages=messages, response_schema=AnswerSchema)
    messages = [
        *messages,
        {"role": "assistant", "content": first.assistant_record},
        {"role": "user", "content": "second"},
    ]
    second = client.respond_structured(messages=messages, response_schema=AnswerSchema)

    assert second.action.answer == "two"
    assistant_turns = [
        message for message in payloads[1]["messages"] if message["role"] == "assistant"
    ]
    assert assistant_turns[0]["content"][0]["type"] == "thinking"
    assert assistant_turns[0]["content"][0]["signature"] == "sig-1"
    assert client.state.hidden_state_used is True


def test_gemini_adapter_is_visible_context_fallback(monkeypatch):
    def fake_completion_with_structured_output(**kwargs):
        return AnswerSchema(answer="ok"), UsageEvent(
            call_type="completion",
            model=kwargs["model"],
            input_tokens=1,
            output_tokens=1,
        )

    monkeypatch.setattr(
        "src.systems.utils.provider_adapters.completion_with_structured_output",
        fake_completion_with_structured_output,
    )

    client = ProviderTurnClient(model="gemini-2.5-pro")
    result = client.respond_structured(
        messages=[{"role": "user", "content": "hello"}],
        response_schema=AnswerSchema,
    )

    assert result.action.answer == "ok"
    assert result.metadata["continuity_mode"] == "visible_context_only"


class _FakeUrlopenContext:
    """Mimic the ``with urllib.request.urlopen(...)`` context manager."""

    def __init__(self, payload: dict):
        self._body = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self):
        return self._body


def _http_error(status: int, body: str = "") -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        url="https://api.anthropic.com/v1/messages",
        code=status,
        msg=str(status),
        hdrs=None,  # type: ignore[arg-type]
        fp=io.BytesIO(body.encode("utf-8")),
    )


def _anthropic_response(text: str) -> dict:
    return {
        "id": "msg_x",
        "content": [{"type": "text", "text": text}],
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }


def _make_anthropic_client(monkeypatch) -> ProviderTurnClient:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setattr(
        "src.systems.utils.provider_adapters.time.sleep", lambda _: None
    )
    return ProviderTurnClient(model="claude-sonnet-4-6")


class TestAnthropicRequestRetry:
    """Regression guards for transient network/HTTP failures on the native
    Anthropic adapter.  Without retry, a single SSL ``BAD_RECORD_MAC`` flap
    or a 502/503 surfaced from the urllib stack kills an entire run-all
    benchmark — observed on flaky local networks where the OpenAI Responses
    path (which already retries) keeps working in the same session.
    """

    def test_retries_on_ssl_error(self, monkeypatch):
        client = _make_anthropic_client(monkeypatch)
        calls = {"count": 0}
        success = _FakeUrlopenContext(_anthropic_response('{"answer": "ok"}'))

        def fake_urlopen(_request, timeout=None):
            calls["count"] += 1
            if calls["count"] < 3:
                raise ssl.SSLError("[SSL: SSLV3_ALERT_BAD_RECORD_MAC] bad mac")
            return success

        monkeypatch.setattr(
            "src.systems.utils.provider_adapters.urllib.request.urlopen", fake_urlopen
        )

        result = client._anthropic_request({"messages": []})
        assert result["content"][0]["text"] == '{"answer": "ok"}'
        assert calls["count"] == 3

    def test_retries_on_broken_pipe(self, monkeypatch):
        client = _make_anthropic_client(monkeypatch)
        calls = {"count": 0}
        success = _FakeUrlopenContext(_anthropic_response('{"answer": "ok"}'))

        def fake_urlopen(_request, timeout=None):
            calls["count"] += 1
            if calls["count"] == 1:
                raise urllib.error.URLError("[Errno 32] Broken pipe")
            return success

        monkeypatch.setattr(
            "src.systems.utils.provider_adapters.urllib.request.urlopen", fake_urlopen
        )

        client._anthropic_request({"messages": []})
        assert calls["count"] == 2

    def test_retries_on_5xx(self, monkeypatch):
        client = _make_anthropic_client(monkeypatch)
        calls = {"count": 0}
        success = _FakeUrlopenContext(_anthropic_response('{"answer": "ok"}'))

        def fake_urlopen(_request, timeout=None):
            calls["count"] += 1
            if calls["count"] == 1:
                raise _http_error(503, "service unavailable")
            return success

        monkeypatch.setattr(
            "src.systems.utils.provider_adapters.urllib.request.urlopen", fake_urlopen
        )

        client._anthropic_request({"messages": []})
        assert calls["count"] == 2

    def test_retries_on_429(self, monkeypatch):
        client = _make_anthropic_client(monkeypatch)
        calls = {"count": 0}
        success = _FakeUrlopenContext(_anthropic_response('{"answer": "ok"}'))

        def fake_urlopen(_request, timeout=None):
            calls["count"] += 1
            if calls["count"] == 1:
                raise _http_error(429, "rate limit")
            return success

        monkeypatch.setattr(
            "src.systems.utils.provider_adapters.urllib.request.urlopen", fake_urlopen
        )

        client._anthropic_request({"messages": []})
        assert calls["count"] == 2

    def test_retries_on_529_overloaded(self, monkeypatch):
        """Regression for the 2026-05-02 cohort_studies failure: Anthropic
        returned HTTP 529 ``overloaded_error`` and the run aborted because
        529 was missing from the retry set. The new "all 5xx" policy must
        cover this without an explicit carve-out."""
        client = _make_anthropic_client(monkeypatch)
        calls = {"count": 0}
        success = _FakeUrlopenContext(_anthropic_response('{"answer": "ok"}'))

        def fake_urlopen(_request, timeout=None):
            calls["count"] += 1
            if calls["count"] == 1:
                raise _http_error(
                    529,
                    '{"type":"error","error":{"type":"overloaded_error",'
                    '"message":"Overloaded"}}',
                )
            return success

        monkeypatch.setattr(
            "src.systems.utils.provider_adapters.urllib.request.urlopen", fake_urlopen
        )

        client._anthropic_request({"messages": []})
        assert calls["count"] == 2

    def test_retries_on_arbitrary_future_5xx(self, monkeypatch):
        """Forward-compat guard: any 5xx code a provider invents in the
        future (e.g. 530, 540) must be retried automatically."""
        client = _make_anthropic_client(monkeypatch)
        calls = {"count": 0}
        success = _FakeUrlopenContext(_anthropic_response('{"answer": "ok"}'))

        def fake_urlopen(_request, timeout=None):
            calls["count"] += 1
            if calls["count"] == 1:
                raise _http_error(530, "site frozen")
            return success

        monkeypatch.setattr(
            "src.systems.utils.provider_adapters.urllib.request.urlopen", fake_urlopen
        )

        client._anthropic_request({"messages": []})
        assert calls["count"] == 2

    def test_does_not_retry_on_400(self, monkeypatch):
        client = _make_anthropic_client(monkeypatch)
        calls = {"count": 0}

        def fake_urlopen(_request, timeout=None):
            calls["count"] += 1
            raise _http_error(400, "bad request")

        monkeypatch.setattr(
            "src.systems.utils.provider_adapters.urllib.request.urlopen", fake_urlopen
        )

        with pytest.raises(RuntimeError, match="bad request"):
            client._anthropic_request({"messages": []})
        assert calls["count"] == 1

    def test_does_not_retry_on_401(self, monkeypatch):
        client = _make_anthropic_client(monkeypatch)
        calls = {"count": 0}

        def fake_urlopen(_request, timeout=None):
            calls["count"] += 1
            raise _http_error(401, "unauthorized")

        monkeypatch.setattr(
            "src.systems.utils.provider_adapters.urllib.request.urlopen", fake_urlopen
        )

        with pytest.raises(RuntimeError, match="unauthorized"):
            client._anthropic_request({"messages": []})
        assert calls["count"] == 1

    def test_exhausts_retry_budget_and_reraises(self, monkeypatch):
        client = _make_anthropic_client(monkeypatch)
        calls = {"count": 0}

        def fake_urlopen(_request, timeout=None):
            calls["count"] += 1
            raise ssl.SSLError("[SSL: SSLV3_ALERT_BAD_RECORD_MAC] bad mac")

        monkeypatch.setattr(
            "src.systems.utils.provider_adapters.urllib.request.urlopen", fake_urlopen
        )

        with pytest.raises(ssl.SSLError):
            client._anthropic_request({"messages": []})
        assert calls["count"] == 6

    def test_rate_limit_uses_longer_backoff_base(self):
        from src.systems.utils.provider_adapters import _anthropic_retry_delay

        rate_limit = _anthropic_retry_delay(_http_error(429), attempt=0)
        ssl_flap = _anthropic_retry_delay(ssl.SSLError("bad record mac"), attempt=0)
        assert rate_limit > ssl_flap


class TestAnthropicParseRetry:
    """Native Anthropic path re-samples on JSON/schema failures.

    Provider HTTP loop returns a 200 OK with a malformed body once (here a
    response with no extractable text content), and a clean response on the
    next attempt. The client must hide the bad sample, defer state capture
    until parse succeeds, and surface the second response unchanged.
    """

    def _bad_response(self):
        # Empty content list triggers ``_anthropic_text`` to raise — same
        # failure shape as a body whose JSON is unparseable.
        return {
            "id": "msg_bad",
            "content": [{"type": "text", "text": "{not valid json"}],
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }

    def _good_response(self, value: str):
        return {
            "id": f"msg_{value}",
            "content": [{"type": "text", "text": json.dumps({"answer": value})}],
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }

    def test_resamples_after_parse_failure(self, monkeypatch):
        monkeypatch.setattr(
            "src.systems.utils.provider_adapters.time.sleep", lambda _s: None
        )
        responses = [self._bad_response(), self._good_response("ok")]
        calls = {"count": 0}

        def fake_anthropic_request(_payload):
            response = responses[calls["count"]]
            calls["count"] += 1
            return response

        client = ProviderTurnClient(model="claude-opus-4-6")
        client._anthropic_request = fake_anthropic_request  # type: ignore[method-assign]

        result = client.respond_structured(
            messages=[{"role": "user", "content": "hi"}],
            response_schema=AnswerSchema,
        )
        assert result.action.answer == "ok"
        assert calls["count"] == 2
        # State should reflect only the GOOD assistant response.
        assistant_turns = [
            m for m in client.state.native_messages if m["role"] == "assistant"
        ]
        assert len(assistant_turns) == 1
        assert client.state.previous_response_id == "msg_ok"

    def test_exhausts_parse_retry_budget_and_raises(self, monkeypatch):
        monkeypatch.setattr(
            "src.systems.utils.provider_adapters.time.sleep", lambda _s: None
        )
        calls = {"count": 0}

        def fake_anthropic_request(_payload):
            calls["count"] += 1
            return self._bad_response()

        client = ProviderTurnClient(model="claude-opus-4-6")
        client._anthropic_request = fake_anthropic_request  # type: ignore[method-assign]

        with pytest.raises(Exception):
            client.respond_structured(
                messages=[{"role": "user", "content": "hi"}],
                response_schema=AnswerSchema,
            )
        # 4 = _PARSE_RETRY_MAX_ATTEMPTS
        assert calls["count"] == 4
        # No assistant message captured because all attempts failed.
        assistant_turns = [
            m for m in client.state.native_messages if m["role"] == "assistant"
        ]
        assert assistant_turns == []
        assert client.state.previous_response_id is None


class TestOpenAIParseRetry:
    """Native OpenAI Responses path re-samples on JSON/schema failures."""

    def test_resamples_after_parse_failure(self, monkeypatch):
        monkeypatch.setattr(
            "src.systems.utils.provider_adapters.time.sleep", lambda _s: None
        )
        responses = [
            FakeResponse("resp_bad", "{not valid json"),
            FakeResponse("resp_ok", '{"answer": "ok"}'),
        ]
        calls = {"count": 0}

        def fake_responses(**_kwargs):
            response = responses[calls["count"]]
            calls["count"] += 1
            return response

        monkeypatch.setattr(
            "src.systems.utils.provider_adapters.litellm.responses", fake_responses
        )

        client = ProviderTurnClient(model="gpt-5.4")
        result = client.respond_structured(
            messages=[{"role": "user", "content": "hi"}],
            response_schema=AnswerSchema,
        )
        assert result.action.answer == "ok"
        assert calls["count"] == 2
        # State should chain off the GOOD response, not the failed one.
        assert client.state.previous_response_id == "resp_ok"
        assert "resp_bad" not in client.state.response_ids

    def test_exhausts_parse_retry_budget_and_raises(self, monkeypatch):
        monkeypatch.setattr(
            "src.systems.utils.provider_adapters.time.sleep", lambda _s: None
        )
        calls = {"count": 0}

        def fake_responses(**_kwargs):
            calls["count"] += 1
            return FakeResponse(f"resp_bad_{calls['count']}", "{not valid json")

        monkeypatch.setattr(
            "src.systems.utils.provider_adapters.litellm.responses", fake_responses
        )

        client = ProviderTurnClient(model="gpt-5.4")
        with pytest.raises(Exception):
            client.respond_structured(
                messages=[{"role": "user", "content": "hi"}],
                response_schema=AnswerSchema,
            )
        assert calls["count"] == 4
        assert client.state.previous_response_id is None


def test_usage_summary_includes_reasoning_and_cache_tokens():
    summary = summarize_usage_events(
        [
            UsageEvent(
                call_type="completion",
                model="gpt-5.4",
                provider="openai",
                input_tokens=10,
                output_tokens=5,
                total_tokens=15,
                reasoning_tokens=2,
                cached_input_tokens=3,
                cache_creation_input_tokens=4,
            )
        ]
    )

    assert summary["reasoning_tokens"] == 2
    assert summary["cached_input_tokens"] == 3
    assert summary["cache_creation_input_tokens"] == 4
