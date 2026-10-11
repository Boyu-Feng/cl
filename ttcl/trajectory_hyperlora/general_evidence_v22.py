"""Format-neutral, provenance-bound evidence for additive parameter/text use.

This module does not choose between a LoRA adapter and evidence. A caller may
mount its adapter and supply the selected public evidence in the same inference
call. JSON actions can retain the typed operators in typed_evidence_v20;
text commands use the observation/action/feedback route below.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from typing import Any, Iterable


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":"), default=str).encode()).hexdigest()


def words(value: str) -> set[str]:
    return set(re.findall(r"\w+", value.casefold()))


@dataclass(frozen=True)
class Evidence:
    source_index: int
    step_index: int
    source_sha256: str
    action_sha256: str
    feedback_sha256: str
    observation: str
    action: Any
    feedback: str
    reward: float


def extract_records(records: Iterable[dict], *, source_index: int,
                    source_sha256: str, reward: float) -> list[Evidence]:
    """Accept public text commands or structured actions without parsing JSON."""
    if not source_sha256 or not 0 <= reward <= 1:
        raise ValueError("Source digest and bounded source reward required")
    output = []
    for step_index, row in enumerate(records):
        if not all(key in row for key in ("observation", "action", "feedback")):
            raise ValueError("Incomplete public source step")
        output.append(Evidence(source_index, step_index, source_sha256,
            digest(row["action"]), digest(row["feedback"]),
            str(row["observation"]), row["action"], str(row["feedback"]),
            float(reward)))
    return output


def verify(evidence: Evidence, records: list[dict], source_sha256: str) -> bool:
    if (evidence.source_sha256 != source_sha256 or
            digest(records) != source_sha256 or
            not 0 <= evidence.step_index < len(records)):
        return False
    row = records[evidence.step_index]
    return (evidence.action_sha256 == digest(row["action"])
            and evidence.feedback_sha256 == digest(row["feedback"])
            and evidence.observation == str(row["observation"])
            and evidence.action == row["action"])


def select_text(evidence: Iterable[Evidence], *, target_index: int,
                task: str, observation: str, available: Iterable[str],
                limit: int = 3) -> list[Evidence]:
    """Retrieve past command evidence using only current public context."""
    if limit < 0:
        raise ValueError("Negative evidence limit")
    query = words(task + " " + observation)
    verbs = {command.split(" ", 1)[0].casefold() for command in available}
    ranked = []
    for item in evidence:
        if item.source_index >= target_index or not isinstance(item.action, str):
            continue
        verb = item.action.split(" ", 1)[0].casefold()
        if verb not in verbs:
            continue
        source = words(item.observation)
        overlap = len(query & source) / max(1, len(query | source))
        ranked.append((overlap, item.reward, -item.source_index,
                       -item.step_index, item))
    ranked.sort(key=lambda row: row[:4], reverse=True)
    return [row[4] for row in ranked[:limit]]


def render_text(items: Iterable[Evidence], *, max_chars: int = 1400) -> str:
    """Render evidence as prior examples, never as mandatory current commands."""
    parts = ["Past examples may involve different objects. Choose only from the current available commands."]
    for item in items:
        part = (f"Source {item.source_index}, step {item.step_index}, outcome {item.reward:g}:\n"
                f"Observed: {item.observation}\n"
                f"Action: {item.action}\n"
                f"Environment reply: {item.feedback}")
        if len("\n".join(parts + [part])) > max_chars:
            break
        parts.append(part)
    return "\n".join(parts) if len(parts) > 1 else ""


def text_provider(items: Iterable[Evidence], *, target_index: int,
                  max_chars: int = 1400):
    """Return the callback accepted by the ALFWorld actor's per-step prompt."""
    frozen = tuple(items)

    def provide(task: str, observation: str, available: list[str]) -> str:
        chosen = select_text(frozen, target_index=target_index, task=task,
                             observation=observation, available=available)
        return render_text(chosen, max_chars=max_chars)

    return provide


def compose_structured(current_action: dict, evidence: Iterable[Evidence],
                       *, schema: dict, gate_model: dict, target_index: int,
                       target_query: str) -> tuple[dict, dict]:
    """Apply approved typed evidence to an existing LoRA-generated JSON action.

    The base action is supplied by the caller; no frozen-policy replacement is
    generated here. This is the structured counterpart of text_provider.
    """
    from jsonschema import Draft202012Validator
    from ttcl.trajectory_hyperlora.typed_evidence_v20 import (
        extract as typed_extract, proposals, route,
    )

    validator = Draft202012Validator(schema)
    if not isinstance(current_action, dict) or not validator.is_valid(current_action):
        return current_action, {"decision": "unchanged", "reason": "invalid_base_action"}
    candidates = []
    for item in evidence:
        if item.source_index >= target_index or not isinstance(item.action, dict):
            continue
        for typed in typed_extract(item.action, source_index=item.source_index,
                trajectory_sha256=item.source_sha256, feedback=item.feedback,
                reward=item.reward, query=item.observation):
            candidates.extend(proposals(current_action, typed, schema))
    operator, selected, lower = route(candidates, current_action, gate_model,
        target_index=target_index, target_query=target_query)
    if selected is None or selected.action is None:
        return current_action, {"decision": "unchanged", "considered": len(candidates)}
    if not validator.is_valid(selected.action):
        raise ValueError("Selected evidence action violates current schema")
    return selected.action, {"decision": operator, "lower_utility": lower,
        "source_index": selected.evidence.source_index,
        "source_trajectory_sha256": selected.evidence.trajectory_sha256,
        "source_action_sha256": selected.evidence.action_sha256,
        "considered": len(candidates)}
