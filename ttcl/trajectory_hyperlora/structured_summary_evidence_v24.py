"""Model-distilled evidence with fixed fields and verifiable source bindings."""

from __future__ import annotations

import json
import math
from typing import Any

from ttcl.trajectory_hyperlora.typed_prompt_evidence_v23 import (
    digest, kind, words,
)

FIELDS = ("observation_condition", "action_pattern", "feedback_effect",
          "applicability", "caution")


def sample_steps(records: list[dict], limit: int = 6) -> list[int]:
    """Cover a trajectory's start, middle and end without reward peeking."""
    if not records:
        return []
    if len(records) <= limit:
        return list(range(len(records)))
    return sorted({round(i * (len(records) - 1) / (limit - 1))
                   for i in range(limit)})


def field_types(records: list[dict], indices: list[int]) -> dict[str, str]:
    output = {}
    for index in indices:
        action = records[index]["action"]
        if isinstance(action, dict):
            for key, value in action.items():
                output[key] = kind(value)
        else:
            output["$action"] = kind(action)
    return output


def writer_input(records: list[dict], *, source_index: int, reward: float,
                 initial_query: str) -> dict:
    if source_index < 0 or not math.isfinite(reward) or not records:
        raise ValueError("Completed finite-reward source trajectory required")
    indices = sample_steps(records)
    shown = []
    for index in indices:
        row = records[index]
        action = json.dumps(row["action"], ensure_ascii=False,
                            sort_keys=True, default=str)
        shown.append({"step_index": index,
            "observation": str(row["observation"])[:500],
            "action": action[:1000],
            "action_sha256": digest(row["action"]),
            "feedback": str(row["feedback"])[:500],
            "feedback_sha256": digest(row["feedback"])})
    return {"source_index": source_index,
        "source_trajectory_sha256": digest(records),
        "initial_query": initial_query[:1000],
        "official_reward": reward,
        "field_types": field_types(records, indices),
        "selected_steps": shown}


def writer_messages(value: dict) -> list[dict]:
    schema = {key: "nonempty string, at most 180 characters" for key in FIELDS}
    schema["supporting_step_indices"] = "array of selected step_index integers"
    return [{"role": "system", "content":
        "Turn this completed public trajectory into ONE transferable, structured "
        "evidence record. Return only a JSON object with exactly the required "
        "fields. Ground each claim in selected observations, actions and feedback. "
        "Describe conditions and actions using roles or variable names, not "
        "instance-specific table names, objects, frequencies or bet sizes. "
        "Distinguish observed feedback from a causal claim. A bad outcome may "
        "support a caution, not a positive rule. Do not invent unseen steps."},
        {"role": "user", "content": "Required JSON fields:\n" +
            json.dumps(schema, ensure_ascii=False, sort_keys=True) +
            "\nSource:\n" + json.dumps(value, ensure_ascii=False,
                                       sort_keys=True)}]


def build_summary(raw: str, value: dict, *, writer_prompt_sha256: str,
                  writer_response_sha256: str) -> dict:
    generated = json.loads(raw)
    if not isinstance(generated, dict) or set(generated) != set(FIELDS) | {
            "supporting_step_indices"}:
        raise ValueError("Writer returned wrong evidence schema")
    for key in FIELDS:
        field = generated[key]
        if not isinstance(field, str) or not field.strip() or len(field) > 180:
            raise ValueError(f"Invalid structured evidence field {key}")
    support = generated["supporting_step_indices"]
    selected = {step["step_index"] for step in value["selected_steps"]}
    if (not isinstance(support, list) or not support or len(support) > 6 or
            any(type(index) is not int or index not in selected
                for index in support) or len(support) != len(set(support))):
        raise ValueError("Evidence cites an unobserved or duplicate step")
    by_index = {step["step_index"]: step for step in value["selected_steps"]}
    excerpts = [{"step_index": index,
        "observation": by_index[index]["observation"][:180],
        "action": by_index[index]["action"][:280],
        "action_sha256": by_index[index]["action_sha256"],
        "feedback": by_index[index]["feedback"][:180],
        "feedback_sha256": by_index[index]["feedback_sha256"]}
        for index in support]
    return {"source_index": value["source_index"],
        "source_trajectory_sha256": value["source_trajectory_sha256"],
        "writer_input_sha256": digest(value),
        "writer_prompt_sha256": writer_prompt_sha256,
        "writer_response_sha256": writer_response_sha256,
        "official_reward": value["official_reward"],
        "field_types": value["field_types"],
        "source_excerpts": excerpts,
        **generated}


def select(summaries: list[dict], *, target_index: int,
           target_query: str, limit: int = 3) -> list[dict]:
    query = words(target_query)
    ranked = []
    for item in summaries:
        if item["source_index"] >= target_index:
            continue
        source = words(item["observation_condition"] + " " +
                       item["applicability"])
        overlap = len(query & source) / max(1, len(query | source))
        ranked.append((overlap, item["official_reward"],
                       item["source_index"], item))
    ranked.sort(key=lambda row: row[:3], reverse=True)
    return [row[3] for row in ranked[:limit]]


def render(items: list[dict], *, max_chars: int = 4200) -> str:
    intro = ("Past structured evidence records follow as JSON. Their fields "
        "are grounded in completed trajectories with hashes and step citations. "
        "Check applicability against the CURRENT observation and schema; do "
        "not copy source-specific identifiers or treat a bad outcome as advice. "
        "Return only the current action JSON.\n")
    output = intro
    for item in items:
        row = json.dumps(item, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":")) + "\n"
        if len(output) + len(row) <= max_chars:
            output += row
    return output if output != intro else ""
