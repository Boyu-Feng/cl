"""Feedback-trained SQL query operator for provenance-bound typed evidence.

This is a database action adapter for the general evidence interface. Its
training target is immediate public query feedback, not final CLBench reward.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re

from ttcl.trajectory_hyperlora.general_evidence_v22 import Evidence, digest


def feedback_grade(feedback: str) -> int:
    """Distinguish errors, empty results, and informative SQL results."""
    if re.search(r"\bERROR\s*:", feedback, re.I):
        return -1
    if "(no results)" in feedback.lower():
        return 0
    return 1 if "Query result (" in feedback else 0


def missing_identifiers(feedback: str) -> set[str]:
    return {match.group(1).lower() for match in re.finditer(
        r"no such (?:table|column)\s*:\s*([\w.]+)", feedback, re.I)}


def mentions_identifier(sql: str, identifier: str) -> bool:
    return bool(re.search(r"(?<![\w])" + re.escape(identifier) + r"(?![\w])",
                          sql, re.I))


def is_schema_query(sql: str) -> bool:
    normalized = " ".join(sql.lower().split())
    return ("sqlite_master" in normalized and "select" in normalized and
            "name" in normalized and "table" in normalized)


@dataclass(frozen=True)
class Candidate:
    source: Evidence
    action: dict
    failed_identifier: str


def candidate(current: dict, history: list[Evidence], target_index: int,
              known_bad: set[str]) -> Candidate | None:
    if current.get("action", "").upper() != "QUERY":
        return None
    sql = current.get("content")
    if not isinstance(sql, str):
        return None
    bad = next((name for name in sorted(known_bad)
                if mentions_identifier(sql, name)), None)
    if bad is None:
        return None
    sources = [item for item in history if item.source_index < target_index
        and isinstance(item.action, dict)
        and item.action.get("action", "").upper() == "QUERY"
        and isinstance(item.action.get("content"), str)
        and is_schema_query(item.action["content"])
        and feedback_grade(item.feedback) > 0]
    if not sources:
        return None
    source = min(sources, key=lambda item: (item.source_index, item.step_index))
    if source.action["content"] == sql:
        return None
    return Candidate(source, {**current, "content": source.action["content"]}, bad)


def fit(labels: list[dict], *, min_examples: int = 3) -> dict:
    """Fit a conservative empirical gate on train-only immediate feedback."""
    train = [row for row in labels if row["split"] == "train"]
    dev = [row for row in labels if row["split"] == "dev"]
    by_target = {}
    for row in train:
        by_target.setdefault(row["target_index"], []).append(row["feedback_delta"])
    dev_targets = {}
    for row in dev:
        dev_targets.setdefault(row["target_index"], []).append(row["feedback_delta"])
    if len(by_target) < min_examples:
        return {"operator": "sql_feedback_schema_query_copy",
                "enabled": False, "reason": "insufficient_train_labels",
                "train_count": len(train), "train_targets": len(by_target),
                "dev_count": len(dev)}
    # Repeated actions within one task share a source and are not independent.
    target_means = [sum(values)/len(values) for values in by_target.values()]
    n = len(target_means)
    mean = sum(target_means)/n
    variance = sum((value-mean)**2 for value in target_means)/(n-1)
    lower = mean - 1.96 * math.sqrt(variance/n)
    dev_means = [sum(values)/len(values) for values in dev_targets.values()]
    enabled = lower > 0 and len(dev_means) >= min_examples and \
        all(value > 0 for value in dev_means)
    return {"operator": "sql_feedback_schema_query_copy", "enabled": enabled,
            "train_count": len(train), "train_targets": n,
            "train_target_mean_delta": mean,
            "train_target_mean_lower_normal_approx": lower,
            "dev_count": len(dev), "dev_targets": len(dev_means),
            "dev_positive_mean_targets": sum(value > 0 for value in dev_means),
            "target": "immediate_public_query_feedback_delta",
            "note": "Official final reward is evaluated separately"}


def apply(current: dict, history: list[Evidence], target_index: int,
          known_bad: set[str], model: dict) -> tuple[dict, dict]:
    selected = candidate(current, history, target_index, known_bad)
    if selected is None:
        return current, {"decision": "unchanged", "reason": "no_supported_candidate"}
    if not model.get("enabled"):
        return current, {"decision": "unchanged", "reason": "feedback_gate_abstained"}
    item = selected.source
    return selected.action, {"decision": model["operator"],
        "source_index": item.source_index, "source_step_index": item.step_index,
        "source_sha256": item.source_sha256,
        "source_action_sha256": item.action_sha256,
        "source_feedback_sha256": item.feedback_sha256,
        "failed_identifier": selected.failed_identifier,
        "candidate_action_sha256": digest(selected.action)}
