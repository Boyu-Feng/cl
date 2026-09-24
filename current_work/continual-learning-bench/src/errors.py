"""Shared benchmark error types."""

from __future__ import annotations

import hashlib
import logging
from typing import Any

import litellm

logger = logging.getLogger(__name__)

_CONTENT_POLICY_BAD_REQUEST_SUBSTRINGS = (
    "contentpolicyviolationerror",
    "content policy violation",
    "flagged as potentially violating our usage policy",
    "your prompt was flagged",
)


def _rebuild_provider_refusal_error(
    message: str,
    kind: str,
    provider: str | None,
    model: str | None,
    instance_id: str | None,
    instance_index: int | None,
    step_number: int | None,
    prompt_digest: str | None,
    prompt_chars: int | None,
) -> "ProviderRefusalError":
    """Reconstruct a pickled ``ProviderRefusalError``."""
    err = ProviderRefusalError(
        kind=kind,
        message=message,
        provider=provider,
        model=model,
        instance_id=instance_id,
        instance_index=instance_index,
    )
    err.step_number = step_number
    err.prompt_digest = prompt_digest
    err.prompt_chars = prompt_chars
    return err


class ProviderRefusalError(RuntimeError):
    """Raised when the upstream model provider refuses a prompt/request."""

    def __init__(
        self,
        *,
        kind: str,
        message: str,
        provider: str | None = None,
        model: str | None = None,
        instance_id: str | None = None,
        instance_index: int | None = None,
    ) -> None:
        super().__init__(message)
        self.kind = kind
        self.provider = provider
        self.model = model
        self.instance_id = instance_id
        self.instance_index = instance_index
        # Optional diagnostic context populated by the runner / call site.
        self.step_number: int | None = None
        self.prompt_digest: str | None = None
        self.prompt_chars: int | None = None

    def with_instance(
        self,
        *,
        instance_id: str | None,
        instance_index: int | None,
    ) -> "ProviderRefusalError":
        """Return a copy of this refusal annotated with benchmark instance identity."""
        copy = type(self)(
            kind=self.kind,
            message=str(self),
            provider=self.provider,
            model=self.model,
            instance_id=instance_id,
            instance_index=instance_index,
        )
        copy.step_number = self.step_number
        copy.prompt_digest = self.prompt_digest
        copy.prompt_chars = self.prompt_chars
        return copy

    def to_record(self) -> dict[str, Any]:
        """Serialize the refusal into benchmark metadata."""
        payload: dict[str, Any] = {
            "kind": self.kind,
            "message": str(self),
        }
        if self.provider:
            payload["provider"] = self.provider
        if self.model:
            payload["model"] = self.model
        if self.instance_id:
            payload["instance_id"] = self.instance_id
        if self.instance_index is not None:
            payload["instance_index"] = self.instance_index
        if self.step_number is not None:
            payload["step_number"] = self.step_number
        if self.prompt_digest is not None:
            payload["prompt_digest"] = self.prompt_digest
        if self.prompt_chars is not None:
            payload["prompt_chars"] = self.prompt_chars
        return payload

    def __reduce__(self) -> Any:
        """Make the exception safe to pass through multiprocessing boundaries."""
        return (
            _rebuild_provider_refusal_error,
            (
                str(self),
                self.kind,
                self.provider,
                self.model,
                self.instance_id,
                self.instance_index,
                self.step_number,
                self.prompt_digest,
                self.prompt_chars,
            ),
        )


def _summarize_messages(messages: Any) -> tuple[str | None, int | None]:
    """Return a stable digest and total char count of the prompt messages.

    The digest lets us correlate refusals across runs without persisting the
    user content itself; the char count helps spot pathologically long prompts
    (e.g. an unbounded notepad) that may be triggering refusals.
    """
    if not isinstance(messages, list):
        return None, None
    parts: list[str] = []
    total = 0
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if isinstance(content, str):
            parts.append(content)
            total += len(content)
        elif isinstance(content, list):
            for chunk in content:
                if isinstance(chunk, dict):
                    text = chunk.get("text")
                    if isinstance(text, str):
                        parts.append(text)
                        total += len(text)
    if not parts:
        return None, None
    digest = hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()[:16]
    return digest, total


def _is_content_policy_refusal(exc: Exception) -> bool:
    """Return True when *exc* represents a content-policy refusal."""
    content_policy_cls = getattr(litellm, "ContentPolicyViolationError", None)
    if content_policy_cls is not None and isinstance(exc, content_policy_cls):
        return True
    if isinstance(exc, litellm.BadRequestError):
        message = str(exc).lower()
        return any(s in message for s in _CONTENT_POLICY_BAD_REQUEST_SUBSTRINGS)
    return False


def as_provider_refusal(
    exc: Exception,
    *,
    provider: str | None = None,
    model: str | None = None,
    messages: Any = None,
) -> ProviderRefusalError | None:
    """Return a normalized provider-refusal error when *exc* matches one."""
    if isinstance(exc, ProviderRefusalError):
        return exc
    if not _is_content_policy_refusal(exc):
        return None

    refusal = ProviderRefusalError(
        kind="content_policy",
        message=str(exc),
        provider=provider,
        model=model,
    )
    refusal.prompt_digest, refusal.prompt_chars = _summarize_messages(messages)
    logger.warning(
        "Provider refusal: kind=%s model=%s prompt_chars=%s prompt_digest=%s",
        refusal.kind,
        refusal.model,
        refusal.prompt_chars,
        refusal.prompt_digest,
    )
    return refusal
