"""Source-bound, typed history records for training-free evidence prompting."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import re
from typing import Any, Iterable


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), default=str).encode()).hexdigest()


def kind(value: Any) -> str:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, str):
        return "text"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return "null"


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


def extract_records(records: list[dict], *, source_index: int,
                    reward: float) -> list[Evidence]:
    if source_index < 0 or not math.isfinite(reward):
        raise ValueError("Source index and finite official reward required")
    source_hash = digest(records)
    output = []
    for step_index, row in enumerate(records):
        if not all(key in row for key in ("observation", "action", "feedback")):
            raise ValueError("Incomplete public source step")
        output.append(Evidence(source_index, step_index, source_hash,
            digest(row["action"]), digest(row["feedback"]),
            str(row["observation"]), row["action"], str(row["feedback"]),
            float(reward)))
    return output


def verify(item: Evidence, records: list[dict]) -> bool:
    if (item.source_sha256 != digest(records) or
            not 0 <= item.step_index < len(records)):
        return False
    row = records[item.step_index]
    return (item.action_sha256 == digest(row["action"]) and
            item.feedback_sha256 == digest(row["feedback"]) and
            item.observation == str(row["observation"]) and
            item.action == row["action"] and
            item.feedback == str(row["feedback"]))


def words(value: str) -> set[str]:
    return set(re.findall(r"\w+", value.casefold()))


def select(items: Iterable[Evidence], *, target_index: int,
           target_query: str, limit: int = 3) -> list[Evidence]:
    """Lexical, outcome-aware retrieval with at most one step per source."""
    query = words(target_query)
    ranked = []
    for item in items:
        if item.source_index >= target_index:
            continue
        source = words(item.observation)
        overlap = len(query & source) / max(1, len(query | source))
        ranked.append((overlap, item.reward, item.source_index,
                       item.step_index, item))
    ranked.sort(key=lambda x: x[:4], reverse=True)
    chosen = []
    seen = set()
    for _, _, _, _, item in ranked:
        if item.source_index not in seen:
            chosen.append(item)
            seen.add(item.source_index)
        if len(chosen) >= limit:
            break
    return chosen


def presentation(item: Evidence) -> dict:
    action = item.action
    return {"source_index": item.source_index,
        "step_index": item.step_index,
        "source_sha256": item.source_sha256,
        "action_sha256": item.action_sha256,
        "feedback_sha256": item.feedback_sha256,
        "observation": item.observation[:260],
        "action_type": kind(action),
        "field_types": ({key: kind(value) for key, value in action.items()}
                        if isinstance(action, dict) else {}),
        "action": action, "feedback": item.feedback[:260],
        "official_reward": item.reward}


def render(items: Iterable[Evidence], *, max_chars: int = 4500) -> str:
    """Retain typed fields; ask the actor to adapt them to current schema."""
    intro = ("Past typed evidence follows as JSON. Each record binds a public "
        "observation, action, feedback, official outcome, and provenance hashes. "
        "Treat it as examples, check relevance and current feedback, and produce "
        "only an action allowed by the current schema. Do not copy stale names, "
        "SQL, or bet sizes blindly.\n")
    output = intro
    for item in items:
        value = presentation(item)
        row = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":")) + "\n"
        if len(output) + len(row) > max_chars:
            continue
        output += row
    return output if output != intro else ""
