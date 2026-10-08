"""Task-neutral, provenance-bound evidence and candidate-use operators.

Operators describe possible uses; a fitted utility gate must approve one.
Untrained operators abstain and leave the existing parameter-memory path in
control. No CLBench domain, action field, or environment command is encoded.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from copy import deepcopy
import json
import math
import re
from typing import Any

import numpy as np
from jsonschema import Draft202012Validator

from ttcl.trajectory_hyperlora.alfworld_online_from_empty import digest


OPERATORS = ("array_union", "numeric_mean", "scalar_copy", "text_hint")
FEATURES = ("source_reward", "log_age", "query_overlap", "same_value",
            "log_source_size", "log_current_size", "numeric_distance")


@dataclass(frozen=True)
class Evidence:
    source_index: int
    trajectory_sha256: str
    action_sha256: str
    feedback_sha256: str
    path: str
    kind: str
    value: Any
    source_reward: float
    source_query: str

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class Proposal:
    operator: str
    evidence: Evidence
    action: Any | None
    hint: str | None = None


def _escape(key: str) -> str:
    return key.replace("~", "~0").replace("/", "~1")


def _unescape(key: str) -> str:
    return key.replace("~1", "/").replace("~0", "~")


def _locate(value: Any, path: str):
    if path == "":
        return value
    current = value
    for segment in path.split("/")[1:]:
        if not isinstance(current, dict) or _unescape(segment) not in current:
            raise KeyError(path)
        current = current[_unescape(segment)]
    return current


def _replace(action: dict, path: str, value: Any) -> dict:
    output = deepcopy(action)
    current = output
    segments = [_unescape(x) for x in path.split("/")[1:]]
    for segment in segments[:-1]:
        current = current[segment]
    current[segments[-1]] = deepcopy(value)
    return output


def extract(action: Any, *, source_index: int, trajectory_sha256: str,
            feedback: Any, reward: float, query: str) -> list[Evidence]:
    """Keep typed public action values, including text commands and scalars."""
    action_hash, feedback_hash = digest(action), digest(feedback)
    output = []

    def visit(value: Any, path: str):
        if isinstance(value, dict):
            for key, child in value.items():
                visit(child, path + "/" + _escape(str(key)))
            return
        if isinstance(value, list):
            kind = "array"
        elif isinstance(value, bool):
            kind = "boolean"
        elif isinstance(value, (int, float)) and math.isfinite(float(value)):
            kind = "number"
        elif isinstance(value, str):
            kind = "text" if path == "" else "string"
        else:
            return
        output.append(Evidence(source_index, trajectory_sha256, action_hash,
            feedback_hash, path, kind, deepcopy(value), float(reward), query))

    visit(action, "")
    return output


def verified(evidence: Evidence, *, source_action: Any,
             source_feedback: Any, trajectory_sha256: str) -> bool:
    try:
        value = _locate(source_action, evidence.path)
    except KeyError:
        return False
    return (evidence.trajectory_sha256 == trajectory_sha256 and
            evidence.action_sha256 == digest(source_action) and
            evidence.feedback_sha256 == digest(source_feedback) and
            evidence.value == value)


def proposals(current: Any, evidence: Evidence, schema: dict | None = None,
              *, max_hint_chars: int = 256) -> list[Proposal]:
    """Produce schema-safe candidates; never execute one merely because it exists."""
    if isinstance(current, str) and evidence.kind == "text":
        hint = f"Earlier public action ({evidence.source_index}): {evidence.value}"
        return [Proposal("text_hint", evidence, None, hint[:max_hint_chars])]
    if not isinstance(current, dict) or not evidence.path:
        return []
    try:
        now = _locate(current, evidence.path)
    except KeyError:
        return []
    choices = []
    if evidence.kind == "array" and isinstance(now, list):
        old = {json.dumps(x, sort_keys=True, ensure_ascii=False) for x in now}
        merged = deepcopy(now)
        for item in evidence.value:
            key = json.dumps(item, sort_keys=True, ensure_ascii=False)
            if key not in old:
                merged.append(deepcopy(item))
                old.add(key)
        if merged != now:
            choices.append(("array_union", merged))
    elif (evidence.kind == "number" and isinstance(now, (int, float)) and
          not isinstance(now, bool) and math.isfinite(float(now))):
        choices.append(("numeric_mean", (float(now) + float(evidence.value)) / 2))
        choices.append(("scalar_copy", evidence.value))
    elif evidence.kind in ("boolean", "string") and type(now) is type(evidence.value):
        choices.append(("scalar_copy", evidence.value))
    output = []
    validator = Draft202012Validator(schema) if schema is not None else None
    for operator, value in choices:
        if value == now:
            continue
        action = _replace(current, evidence.path, value)
        if validator is None or validator.is_valid(action):
            output.append(Proposal(operator, evidence, action))
    return output


def _words(value: str) -> set[str]:
    return set(re.findall(r"[\w]+", value.lower()))


def features(proposal: Proposal, current: Any, *, target_index: int,
             target_query: str) -> np.ndarray:
    evidence = proposal.evidence
    now = _locate(current, evidence.path)
    old = evidence.value
    query, source = _words(target_query), _words(evidence.source_query)
    overlap = len(query & source) / max(1, len(query | source))
    numeric_distance = (abs(float(now) - float(old))
        if evidence.kind == "number" else 0.0)
    return np.array([evidence.source_reward,
        math.log1p(max(1, target_index - evidence.source_index)), overlap,
        float(now == old), math.log1p(len(json.dumps(old, ensure_ascii=False))),
        math.log1p(len(json.dumps(now, ensure_ascii=False))),
        math.log1p(numeric_distance)], dtype=np.float64)


def fit(rows: list[tuple[str, np.ndarray, float]], *, penalty: float = 1.0,
        min_examples: int = 8) -> dict:
    """Fit one reward-difference regressor per operator; sparse types abstain."""
    if penalty <= 0 or min_examples < 2:
        raise ValueError("Positive penalty and at least two examples required")
    model = {"features": FEATURES, "operators": {}, "penalty": penalty,
             "min_examples": min_examples}
    for operator in OPERATORS:
        selected = [(x, y) for op, x, y in rows if op == operator]
        if len(selected) < min_examples:
            continue
        x = np.stack([value for value, _ in selected])
        y = np.array([value for _, value in selected])
        mean, std = x.mean(0), np.maximum(x.std(0), 1e-6)
        design = np.column_stack([np.ones(len(x)), (x - mean) / std])
        regularizer = np.diag([0.] + [penalty] * x.shape[1])
        weight = np.linalg.solve(design.T @ design + regularizer, design.T @ y)
        error = y - design @ weight
        model["operators"][operator] = {"count": len(y),
            "positive": int((y > 0).sum()), "negative": int((y < 0).sum()),
            "mean": mean.tolist(), "std": std.tolist(),
            "weight": weight.tolist(),
            "residual_margin": float(np.quantile(abs(error), .9))}
    return model


def route(candidates: list[Proposal], current: Any, model: dict, *,
          target_index: int, target_query: str) -> tuple[str, Proposal | None, float]:
    """Choose a positive supported operator; otherwise retain LoRA routing."""
    if tuple(model["features"]) != FEATURES:
        raise ValueError("Feature schema changed")
    best = None
    for proposal in candidates:
        fitted = model["operators"].get(proposal.operator)
        if fitted is None:
            continue
        x = features(proposal, current, target_index=target_index,
                     target_query=target_query)
        scaled = (x - np.array(fitted["mean"])) / np.array(fitted["std"])
        estimate = float(np.dot(np.r_[1., scaled], fitted["weight"]))
        lower = estimate - fitted["residual_margin"]
        if lower > 0 and (best is None or lower > best[2]):
            best = (proposal.operator, proposal, lower)
    return best if best is not None else ("lora", None, 0.0)
