"""Helpers for describing system execution behavior in manifests."""

from __future__ import annotations

from typing import Any


def build_system_manifest_entry(
    system_name: str,
    system_params: dict[str, Any],
) -> dict[str, Any]:
    """Return the system block used in trace/live/summary manifests."""
    params = dict(system_params)
    if system_name in {"codex", "claude"}:
        single_conversation = params.get("single_conversation", True)
        if not isinstance(single_conversation, bool):
            raise ValueError(
                f"single_conversation must be a bool, got {single_conversation!r}"
            )
        system_label = "Codex" if system_name == "codex" else "Claude"
        return {
            "name": system_name,
            "params": params,
            "continuity": {
                "mode": "single_conversation"
                if single_conversation
                else "per_instance_conversation",
                "single_conversation": single_conversation,
                "reset_behavior": "reset_clears_conversation",
                "description": (
                    f"{system_label} resumes the same CLI conversation across benchmark "
                    f"instances until reset clears all persistent {system_label} state."
                    if single_conversation
                    else f"{system_label} resumes within one benchmark instance, then starts "
                    "fresh when the instance changes or the system resets."
                ),
                "baseline_scope": (
                    "Baseline workers instantiate a fresh system per baseline "
                    "instance; conversation continuity does not cross baseline "
                    "instances."
                ),
            },
        }

    always_continue = bool(params.get("always_continue", False))
    allow_resume = bool(params.get("allow_resume_during_instance", False))
    provider_mode = str(params.get("provider_mode", "") or "")
    openai_store = bool(params.get("openai_store", True))

    if provider_mode in {"auto", "native"}:
        if openai_store:
            mode = "provider_native_stateful"
            description = (
                "The system routes supported provider models through native APIs "
                "that preserve provider reasoning/session state when available."
            )
            reset_behavior = "provider_state_follows_system_reset"
        else:
            mode = "encrypted_reasoning_stateless"
            description = (
                "The system requests encrypted reasoning artifacts for stateless "
                "OpenAI Responses continuity when supported."
            )
            reset_behavior = "provider_state_follows_system_reset"
    elif provider_mode == "litellm_chat":
        mode = "chat_completion_fallback"
        description = (
            "The system uses LiteLLM chat completions and preserves only visible "
            "message history."
        )
        reset_behavior = "reset_clears_conversation"
    elif always_continue:
        mode = "always_continue"
        description = (
            "The system preserves and resumes the same CLI conversation across "
            "benchmark resets/instances when a prior conversation exists."
        )
        reset_behavior = "reset_preserves_conversation"
    elif allow_resume:
        mode = "resume_during_instance"
        description = (
            "The system may resume the CLI conversation within an instance, but "
            "reset clears the conversation between instances."
        )
        reset_behavior = "reset_clears_conversation"
    else:
        mode = "fresh_per_turn"
        description = (
            "The system starts fresh CLI turns unless its own non-conversation "
            "memory mechanism is configured."
        )
        reset_behavior = "reset_clears_conversation"

    return {
        "name": system_name,
        "params": params,
        "continuity": {
            "mode": mode,
            "always_continue": always_continue,
            "allow_resume_during_instance": allow_resume,
            "reset_behavior": reset_behavior,
            "description": description,
            "baseline_scope": (
                "Baseline workers instantiate a fresh system per baseline instance; "
                "conversation continuity does not cross baseline instances."
            ),
        },
    }
