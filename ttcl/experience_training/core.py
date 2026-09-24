"""Pure selection and paired evaluation; missing outcomes never become zero."""

import copy
import statistics


KEEP = {
    "trajectory_summary": "Preserve the existing experience bank.",
    "reward_interpretation": "The completed episode's reward does not by itself establish a transferable improvement.",
    "decision": "KEEP",
    "operations": [],
}


def select(candidates, rewards, seeds, epsilon=1e-6):
    """Require nonnegative gain in BOTH repeats and positive average gain.

    This is a noisy training filter, not proof of positive expected utility.
    KEEP labels are allowed only when every generated candidate was valid and
    scored, none helped in either repeat, and at least one harmed performance.
    """
    baseline = rewards.get("keep", {})
    if any(baseline.get(str(s)) is None for s in seeds):
        return {"selected": None, "reason": "missing_control", "deltas": {}}
    deltas = {}
    for name, candidate in candidates.items():
        scores = rewards.get(name, {})
        if not candidate["accepted"] or any(scores.get(str(s)) is None for s in seeds):
            continue
        deltas[name] = [scores[str(s)] - baseline[str(s)] for s in seeds]
    positive = [
        name
        for name, values in deltas.items()
        if min(values) >= -epsilon and statistics.mean(values) > epsilon
    ]
    if positive:
        best = max(positive, key=lambda n: (statistics.mean(deltas[n]), n))
        return {
            "selected": best,
            "reason": "positive_in_mean_nonnegative_each_repeat",
            "deltas": deltas,
        }
    if (
        deltas
        and len(deltas) == len(candidates)
        and all(max(v) <= epsilon for v in deltas.values())
        and any(min(v) < -epsilon for v in deltas.values())
    ):
        return {
            "selected": "keep",
            "reason": "keep_beats_nonbeneficial_candidates",
            "deltas": deltas,
        }
    return {
        "selected": None,
        "reason": "tie_mixed_sign_or_missing_candidate",
        "deltas": deltas,
    }


def restore_bank(state, limits):
    from ttcl.llm_memory.trajectory_bank import TrajectoryBank

    bank = TrajectoryBank(**limits)
    bank.entries = copy.deepcopy(state["entries"])
    bank.version = state["version"]
    bank.last_observed = state["last_observed"]
    return bank


def comparisons(rows, arms, expected):
    """Use the same complete task/seed/index triples for every arm."""
    grouped = {
        arm: {
            (r["task"], r["repeat"], r["canonical_index"]): r
            for r in rows
            if r["arm"] == arm and r["status"] == "complete"
        }
        for arm in arms
    }
    common = set.intersection(*(set(g) for g in grouped.values()))
    pairs = []
    for key in sorted(common):
        values = [grouped[a][key] for a in arms]
        if len({v["instance_id"] for v in values}) != 1:
            raise ValueError("Compared different task instances")
        pairs.append(
            {
                "task": key[0],
                "repeat": key[1],
                "index": key[2],
                "rewards": {a: grouped[a][key]["reward"] for a in arms},
            }
        )
    result = {
        "paired_count": len(pairs),
        "expected": expected,
        "complete": len(pairs) == expected,
        "pairs": pairs,
        "arms": {},
    }
    for arm in arms:
        scores = [p["rewards"][arm] for p in pairs]
        ds = [p["rewards"][arm] - p["rewards"][arms[0]] for p in pairs]
        result["arms"][arm] = {
            "mean_reward": statistics.mean(scores) if scores else None,
            "delta_vs_none": statistics.mean(ds) if ds else None,
            "wins": sum(x > 1e-12 for x in ds),
            "losses": sum(x < -1e-12 for x in ds),
            "ties": sum(abs(x) <= 1e-12 for x in ds),
        }
    return result
