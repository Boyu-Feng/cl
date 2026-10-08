"""Provenance-bound JSON evidence and a small reward-difference read gate.

The representation and features refer to public schema paths and values, not
CLBench domain names or hand-written transmitter, SQL, or Cohort fields.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, asdict

import numpy as np

from ttcl.trajectory_hyperlora.alfworld_online_from_empty import digest
from ttcl.trajectory_hyperlora.train_clbench_cumulative_hyperlora_v4 import merge_object_arrays


FEATURES = (
    "added_count", "prior_count", "current_count", "duplicate_fraction",
    "object_key_overlap", "query_word_overlap", "source_reward", "log_age",
)


@dataclass(frozen=True)
class Evidence:
    source_index: int
    source_trajectory_sha256: str
    source_action_sha256: str
    public_feedback_sha256: str
    schema_path: str
    value: list[dict]
    source_reward: float
    source_query: str

    def as_dict(self):
        return asdict(self)


def extract(action: dict, *, source_index: int, trajectory_sha256: str,
            feedback: object, reward: float, query: str) -> list[Evidence]:
    if not isinstance(action, dict):
        return []
    result = []
    for key, value in action.items():
        if not (isinstance(value, list) and value and
                all(isinstance(item, dict) for item in value)):
            continue
        escaped = key.replace("~", "~0").replace("/", "~1")
        result.append(Evidence(source_index, trajectory_sha256, digest(action),
            digest(feedback), "/" + escaped,
            json.loads(json.dumps(value)), float(reward), query))
    return result


def _key(evidence: Evidence) -> str:
    return evidence.schema_path[1:].replace("~1", "/").replace("~0", "~")


def _words(value: str) -> set[str]:
    return set(re.findall(r"[A-Za-z0-9_]+", value.lower()))


def features(current: dict, evidence: Evidence, *, target_index: int,
             target_query: str) -> np.ndarray | None:
    key = _key(evidence)
    value = current.get(key)
    if not (isinstance(value, list) and all(isinstance(item, dict) for item in value)):
        return None
    old = evidence.value
    seen = {json.dumps(item, ensure_ascii=False, sort_keys=True) for item in value}
    added = sum(json.dumps(item, ensure_ascii=False, sort_keys=True) not in seen
                for item in old)
    keys_current = {k for item in value for k in item}
    keys_old = {k for item in old for k in item}
    words_current, words_old = _words(target_query), _words(evidence.source_query)
    return np.array([
        math.log1p(added), math.log1p(len(old)), math.log1p(len(value)),
        1 - added / max(1, len(old)),
        len(keys_current & keys_old) / max(1, len(keys_current | keys_old)),
        len(words_current & words_old) / max(1, len(words_current | words_old)),
        evidence.source_reward,
        math.log1p(max(1, target_index - evidence.source_index)),
    ], dtype=np.float64)


def candidate(current: dict, evidence: Evidence) -> dict:
    return merge_object_arrays(current, [{_key(evidence): evidence.value}])


def fit_ridge(rows: list[tuple[np.ndarray, float]], penalty: float) -> dict:
    if not rows or penalty <= 0:
        raise ValueError("Need reviewed utility labels and positive regularization")
    x = np.stack([row[0] for row in rows])
    y = np.array([row[1] for row in rows], dtype=np.float64)
    mean, std = x.mean(0), np.maximum(x.std(0), 1e-6)
    scaled = (x - mean) / std
    design = np.column_stack([np.ones(len(x)), scaled])
    regularizer = np.diag([0.0] + [penalty] * x.shape[1])
    weight = np.linalg.solve(design.T @ design + regularizer,
                             design.T @ y)
    return {"features": FEATURES, "penalty": penalty,
            "mean": mean.tolist(), "std": std.tolist(),
            "weight": weight.tolist(),
            "train_rows": len(rows), "train_positive": int((y > 0).sum()),
            "train_negative": int((y < 0).sum()),
            "train_zero": int((y == 0).sum())}


def predict(model: dict, value: np.ndarray) -> float:
    if tuple(model["features"]) != FEATURES:
        raise ValueError("Evidence feature schema changed")
    x = (value - np.array(model["mean"])) / np.array(model["std"])
    return float(np.dot(np.r_[1.0, x], np.array(model["weight"])))
