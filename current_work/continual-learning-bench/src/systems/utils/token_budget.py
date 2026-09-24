"""Helpers for provider-aware context token budgeting."""

from __future__ import annotations

import json
from typing import Any, Callable

import litellm


MessageList = list[dict[str, Any]]
EstimateFn = Callable[[MessageList], int]

DEFAULT_CONTEXT_BUFFER_TOKENS = 1024


def count_tokens(model: str, text: str) -> int:
    """Count tokens in text using LiteLLM, with character-based fallback."""
    try:
        return len(litellm.encode(model=model, text=text))
    except Exception:
        return len(text) // 4


def get_model_context_window(model: str) -> int:
    """Return the provider prompt/context limit known for a model."""
    try:
        info = litellm.get_model_info(model)
        max_input_tokens = info.get("max_input_tokens")
        if max_input_tokens:
            return int(max_input_tokens)
    except Exception:
        raise ValueError(f"Unknown model: {model}")
    raise ValueError(f"Unknown context window for model: {model}")


def resolve_context_token_limit(
    model: str,
    configured_max_tokens: int | None,
    *,
    buffer_tokens: int = DEFAULT_CONTEXT_BUFFER_TOKENS,
) -> int:
    """Resolve an ICL context limit, applying a safety buffer by default."""
    if configured_max_tokens is not None:
        return configured_max_tokens
    try:
        return max(1, get_model_context_window(model) - max(0, buffer_tokens))
    except ValueError:
        # Custom/local model identifiers may not be present in LiteLLM's model
        # table. Keep them usable while still applying the same safety buffer.
        return max(1, 128_000 - max(0, buffer_tokens))


def _message_signature(messages: MessageList) -> str:
    """Create a stable signature for an exact prompt payload."""
    return json.dumps(
        messages,
        sort_keys=True,
        ensure_ascii=True,
        default=str,
        separators=(",", ":"),
    )


class TokenBudgetTracker:
    """Tracks prompt-token corrections using API-reported usage counts.

    LiteLLM's local token counting can drift by provider family. We therefore
    anchor the most recent prompt against the API's reported ``input_tokens``
    and reuse that calibration for later truncation decisions between calls.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._scale: float = 1.0
        self._exact_signature: str | None = None
        self._exact_tokens: int | None = None

    def count(self, messages: MessageList, estimate_fn: EstimateFn) -> int:
        if not messages:
            return 0

        signature = _message_signature(messages)
        if self._exact_signature == signature and self._exact_tokens is not None:
            return self._exact_tokens

        estimated_tokens = max(0, int(estimate_fn(messages)))
        if estimated_tokens == 0:
            return 0

        corrected_tokens = int(round(estimated_tokens * self._scale))
        return max(1, corrected_tokens)

    def note_usage(
        self,
        *,
        messages: MessageList,
        input_tokens: Any,
        estimate_fn: EstimateFn,
    ) -> None:
        if not messages:
            return
        try:
            resolved_input_tokens = int(input_tokens)
        except (TypeError, ValueError):
            return
        if resolved_input_tokens < 0:
            return

        self._exact_signature = _message_signature(messages)
        self._exact_tokens = resolved_input_tokens

        estimated_tokens = max(0, int(estimate_fn(messages)))
        if estimated_tokens > 0:
            self._scale = max(0.01, resolved_input_tokens / estimated_tokens)
