"""Pure helpers for enriching and aggregating benchmark trace metrics."""

from __future__ import annotations

from typing import Any, Optional


def _coerce_int(value: Any) -> Optional[int]:
    """Return ``value`` as an int when possible, otherwise ``None``."""
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_float(value: Any) -> Optional[float]:
    """Return ``value`` as a float when possible, otherwise ``None``."""
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _interaction_section(interaction: Any, key: str) -> dict[str, Any]:
    """Extract a dict section from either a dataclass or serialized interaction."""
    if hasattr(interaction, key):
        section = getattr(interaction, key, {})
    elif isinstance(interaction, dict):
        section = interaction.get(key, {})
    else:
        section = {}
    return section if isinstance(section, dict) else {}


def _new_metric_bucket() -> dict[str, Any]:
    return {
        "cost_usd": 0.0,
        "has_cost_usd": False,
        "latency_seconds": 0.0,
        "has_latency_seconds": False,
    }


def _accumulate_interaction_metrics(bucket: dict[str, Any], interaction: Any) -> bool:
    """Add cost/latency from one interaction into ``bucket``."""
    usage = _interaction_section(interaction, "usage")
    interaction_usage = usage.get("interaction", {})
    if not isinstance(interaction_usage, dict):
        interaction_usage = {}
    cost = _coerce_float(interaction_usage.get("cost_usd"))

    timing = _interaction_section(interaction, "timing")
    latency = _coerce_float(timing.get("response_latency_seconds"))

    added = False
    if cost is not None:
        bucket["cost_usd"] += cost
        bucket["has_cost_usd"] = True
        added = True
    if latency is not None:
        bucket["latency_seconds"] += latency
        bucket["has_latency_seconds"] = True
        added = True
    return added


def _write_metric_bucket(outcome: dict[str, Any], bucket: dict[str, Any]) -> None:
    """Write aggregated metrics from ``bucket`` onto one serialized outcome."""
    if bucket.get("has_cost_usd"):
        outcome["cost_usd"] = round(float(bucket["cost_usd"]), 10)
    if bucket.get("has_latency_seconds"):
        outcome["latency_seconds"] = round(float(bucket["latency_seconds"]), 6)


def attach_interaction_metrics_to_outcomes(
    serialized_outcomes: list[dict[str, Any]],
    interactions: list[Any],
) -> list[dict[str, Any]]:
    """Attach per-instance interaction aggregates onto serialized outcomes."""
    if not serialized_outcomes or not interactions:
        return serialized_outcomes

    metrics_by_id: dict[str, dict[str, Any]] = {}
    metrics_by_index: dict[int, dict[str, Any]] = {}
    keyed_metrics_found = False

    for interaction in interactions:
        query = _interaction_section(interaction, "query")
        instance_id = query.get("instance_id")
        instance_index = _coerce_int(query.get("instance_index"))

        bucket_ids: list[dict[str, Any]] = []
        if instance_id is not None:
            bucket_ids.append(
                metrics_by_id.setdefault(str(instance_id), _new_metric_bucket())
            )
        if instance_index is not None:
            bucket_ids.append(
                metrics_by_index.setdefault(instance_index, _new_metric_bucket())
            )

        if not bucket_ids:
            continue

        added = False
        for bucket in bucket_ids:
            added = _accumulate_interaction_metrics(bucket, interaction) or added
        keyed_metrics_found = keyed_metrics_found or added

    if keyed_metrics_found:
        for outcome in serialized_outcomes:
            bucket = None
            outcome_id = outcome.get("instance_id")
            if outcome_id is not None:
                bucket = metrics_by_id.get(str(outcome_id))
            if bucket is None:
                outcome_index = _coerce_int(outcome.get("instance_index"))
                if outcome_index is not None:
                    bucket = metrics_by_index.get(outcome_index)
            if bucket is not None:
                _write_metric_bucket(outcome, bucket)
        return serialized_outcomes

    return serialized_outcomes


def summarize_instance_outcomes(
    outcomes: list[dict[str, Any]],
) -> dict[str, Any]:
    """Summarize a single run's normalized outcomes."""
    rewards = [float(outcome["reward"]) for outcome in outcomes]
    total_reward = sum(rewards)
    mean_reward = total_reward / len(rewards) if rewards else 0.0

    def _summarize_optional_series(
        key: str,
        *,
        total_key: str,
        mean_key: str,
        series_key: str,
    ) -> dict[str, Any]:
        series: list[float | None] = []
        values: list[float] = []
        for outcome in outcomes:
            raw_value = outcome.get(key)
            if raw_value is None:
                series.append(None)
                continue
            value = float(raw_value)
            values.append(value)
            series.append(round(value, 6))
        if not values:
            return {}
        return {
            total_key: round(sum(values), 6),
            mean_key: round(sum(values) / len(values), 6),
            series_key: series,
        }

    raw_metric_name: Optional[str] = None
    raw_values: list[float] = []
    for outcome in outcomes:
        name = outcome.get("raw_metric_name")
        value = outcome.get("raw_metric_value")
        if name is None or value is None:
            continue
        if raw_metric_name is None:
            raw_metric_name = str(name)
        if raw_metric_name != name:
            raw_metric_name = None
            raw_values = []
            break
        raw_values.append(float(value))

    summary: dict[str, Any] = {
        "instances_completed": len(outcomes),
        "total_reward": round(total_reward, 6),
        "mean_reward": round(mean_reward, 6),
        "reward_by_index": [round(value, 6) for value in rewards],
    }
    summary.update(
        _summarize_optional_series(
            "cost_usd",
            total_key="total_cost",
            mean_key="mean_cost",
            series_key="cost_by_index",
        )
    )
    gain_summary = _summarize_optional_series(
        "gain",
        total_key="total_gain",
        mean_key="mean_gain",
        series_key="gain_by_index",
    )
    if gain_summary:
        gain_count = sum(1 for outcome in outcomes if outcome.get("gain") is not None)
        summary.update(gain_summary)
        summary["gain_instances_matched"] = gain_count
        summary["gain_coverage_ratio"] = (
            round(gain_count / len(outcomes), 6) if outcomes else 0.0
        )
    summary.update(
        _summarize_optional_series(
            "cost_increase",
            total_key="total_cost_increase",
            mean_key="mean_cost_increase",
            series_key="cost_increase_by_index",
        )
    )
    if raw_metric_name is not None and len(raw_values) == len(outcomes):
        summary["raw_metric_name"] = raw_metric_name
        summary["raw_metric_by_index"] = [round(value, 6) for value in raw_values]
    return summary


def build_sequence_signature(
    outcomes: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return the canonical sequence signature for a set of normalized outcomes."""
    return [
        {
            "instance_id": str(outcome["instance_id"]),
            "instance_index": int(outcome["instance_index"]),
        }
        for outcome in outcomes
    ]


def _validate_rollout_instance_set(
    baseline_ids: set[str],
    rollout_ids: set[str],
    *,
    allow_partial_baseline: bool,
    allow_partial_rollout: bool = False,
    aggregate_context: bool,
) -> None:
    """Validate rollout instance coverage against the successful baseline subset.

    ``allow_partial_baseline`` permits the baseline to be a strict subset of
    the rollout (e.g. some baseline instances were content-policy blocked).
    ``allow_partial_rollout`` permits the rollout to be a strict subset of the
    baseline (e.g. a rollout was refused mid-trajectory).  When both are true,
    we tolerate any overlap (including empty), which downstream aggregation
    surfaces as ``no gain data``.
    """
    if allow_partial_rollout and allow_partial_baseline:
        return

    extra_in_rollout = rollout_ids - baseline_ids
    missing_from_rollout = baseline_ids - rollout_ids
    # Whichever side we said was allowed to be a subset, suppress its diff.
    if allow_partial_rollout:
        missing_from_rollout = set()
    if allow_partial_baseline:
        extra_in_rollout = set()

    if not extra_in_rollout and not missing_from_rollout:
        return

    parts: list[str] = []
    if missing_from_rollout:
        parts.append(f"Missing from rollout: {sorted(missing_from_rollout)}.")
    if extra_in_rollout:
        parts.append(f"Extra in rollout: {sorted(extra_in_rollout)}.")
    detail = " ".join(parts)
    header = (
        "Cannot aggregate rollouts with different benchmark instance sets. "
        "Ensure all rollouts play the same instances as the baseline."
        if aggregate_context
        else "Rollout instance set does not match baseline."
    )
    raise ValueError(f"{header} {detail}")


def build_benchmark_aggregate(
    baseline_outcomes: list[dict[str, Any]],
    run_outcomes: list[list[dict[str, Any]]],
    *,
    allow_partial_baseline: bool = False,
    allow_partial_rollout: bool = False,
) -> dict[str, Any]:
    """Compute reward-first benchmark aggregates from aligned baseline and rollouts."""

    def _mean_by_index(
        matrix: list[list[float | None]],
        *,
        length: int,
    ) -> list[float | None]:
        result: list[float | None] = []
        for index in range(length):
            values = [
                float(run[index])
                for run in matrix
                if index < len(run) and run[index] is not None
            ]
            result.append(round(sum(values) / len(values), 6) if values else None)
        return result

    def _optional_mean_by_index(
        matrix: list[list[float | None]],
        *,
        length: int,
    ) -> list[float | None]:
        has_any = any(any(v is not None for v in run) for run in matrix)
        if not has_any:
            return []
        return _mean_by_index(matrix, length=length)

    if not baseline_outcomes:
        mean_reward_by_index: list[float | None] = []
        if run_outcomes and run_outcomes[0]:
            seq_len = max(len(run) for run in run_outcomes)
            for index in range(seq_len):
                values = [
                    float(run[index]["reward"])
                    for run in run_outcomes
                    if index < len(run) and run[index].get("reward") is not None
                ]
                mean_reward_by_index.append(
                    round(sum(values) / len(values), 6) if values else None
                )
        return {
            "baseline_reward_by_index": [],
            "baseline_cost_by_index": [],
            "mean_reward_by_index": mean_reward_by_index,
            "mean_gain_by_index": [],
            "mean_cost_increase_by_index": [],
            "cumulative_mean_gain_by_index": [],
            "final_cumulative_mean_gain": 0.0,
        }

    baseline_reward_by_id = {
        str(outcome["instance_id"]): float(outcome["reward"])
        for outcome in baseline_outcomes
    }
    baseline_cost_by_id = {
        str(outcome["instance_id"]): (
            None if outcome.get("cost_usd") is None else float(outcome["cost_usd"])
        )
        for outcome in baseline_outcomes
    }
    baseline_ids = set(baseline_reward_by_id)
    baseline_reward_by_index = [
        float(outcome["reward"]) for outcome in baseline_outcomes
    ]
    baseline_cost_by_index = [
        (
            None
            if outcome.get("cost_usd") is None
            else round(float(outcome["cost_usd"]), 6)
        )
        for outcome in baseline_outcomes
    ]
    reward_matrix: list[list[float]] = []
    gain_matrix: list[list[float | None]] = []
    cost_matrix: list[list[float | None]] = []
    cost_increase_matrix: list[list[float | None]] = []
    latency_matrix: list[list[float | None]] = []
    matched_baseline_reward_matrix: list[list[float | None]] = []
    matched_baseline_cost_matrix: list[list[float | None]] = []
    raw_metric_name: Optional[str] = None
    raw_metric_matrix: list[list[float]] = []
    sequence_length = len(baseline_reward_by_index)

    if not run_outcomes:
        return {
            "baseline_reward_by_index": [
                round(value, 6) for value in baseline_reward_by_index
            ],
            "baseline_cost_by_index": (
                baseline_cost_by_index
                if any(value is not None for value in baseline_cost_by_index)
                else []
            ),
            "mean_reward_by_index": [],
            "mean_gain_by_index": [],
            "mean_cost_by_index": [],
            "mean_cost_increase_by_index": [],
            "mean_latency_by_index": [],
            "cumulative_mean_gain_by_index": [],
            "final_cumulative_mean_gain": 0.0,
        }

    for outcomes in run_outcomes:
        rollout_ids = {str(outcome["instance_id"]) for outcome in outcomes}
        _validate_rollout_instance_set(
            baseline_ids,
            rollout_ids,
            allow_partial_baseline=allow_partial_baseline,
            allow_partial_rollout=allow_partial_rollout,
            aggregate_context=True,
        )

        rewards = [float(outcome["reward"]) for outcome in outcomes]
        costs = [
            None if outcome.get("cost_usd") is None else float(outcome["cost_usd"])
            for outcome in outcomes
        ]
        gains: list[float | None] = []
        matched_baseline_rewards: list[float | None] = []
        matched_baseline_costs: list[float | None] = []
        cost_increases: list[float | None] = []
        for outcome in outcomes:
            baseline_reward = baseline_reward_by_id.get(str(outcome["instance_id"]))
            if baseline_reward is None:
                gains.append(None)
                matched_baseline_rewards.append(None)
            else:
                gains.append(float(outcome["reward"]) - baseline_reward)
                matched_baseline_rewards.append(baseline_reward)
            rollout_cost = outcome.get("cost_usd")
            baseline_cost = baseline_cost_by_id.get(str(outcome["instance_id"]))
            matched_baseline_costs.append(baseline_cost)
            if rollout_cost is None or baseline_cost is None:
                cost_increases.append(None)
            else:
                cost_increases.append(float(rollout_cost) - baseline_cost)
        latencies = [outcome.get("latency_seconds") for outcome in outcomes]
        reward_matrix.append(rewards)
        gain_matrix.append(gains)
        cost_matrix.append(costs)
        cost_increase_matrix.append(cost_increases)
        latency_matrix.append(latencies)
        matched_baseline_reward_matrix.append(matched_baseline_rewards)
        matched_baseline_cost_matrix.append(matched_baseline_costs)

        if allow_partial_baseline:
            sequence_length = max(sequence_length, len(outcomes))

        run_raw_name: Optional[str] = None
        run_raw_values: list[float] = []
        for outcome in outcomes:
            name = outcome.get("raw_metric_name")
            value = outcome.get("raw_metric_value")
            if name is None or value is None:
                run_raw_name = None
                run_raw_values = []
                break
            if run_raw_name is None:
                run_raw_name = str(name)
            if run_raw_name != name:
                run_raw_name = None
                run_raw_values = []
                break
            run_raw_values.append(float(value))

        if run_raw_name is not None and len(run_raw_values) == len(outcomes):
            if raw_metric_name is None:
                raw_metric_name = run_raw_name
            if raw_metric_name == run_raw_name:
                raw_metric_matrix.append(run_raw_values)

    mean_reward_by_index = _mean_by_index(
        [[float(value) for value in run] for run in reward_matrix],
        length=sequence_length,
    )
    if allow_partial_baseline:
        baseline_reward_series = _optional_mean_by_index(
            matched_baseline_reward_matrix,
            length=sequence_length,
        )
        baseline_cost_series = _optional_mean_by_index(
            matched_baseline_cost_matrix,
            length=sequence_length,
        )
        mean_gain_by_index = _optional_mean_by_index(
            gain_matrix,
            length=sequence_length,
        )
    else:
        baseline_reward_series = [round(value, 6) for value in baseline_reward_by_index]
        baseline_cost_series = (
            baseline_cost_by_index
            if any(value is not None for value in baseline_cost_by_index)
            else []
        )
        mean_gain_by_index = _mean_by_index(gain_matrix, length=sequence_length)

    cumulative_mean_gain_by_index: list[float] = []
    running_gain = 0.0
    running_gain_count = 0
    for gain in mean_gain_by_index:
        if gain is not None:
            running_gain += gain
            running_gain_count += 1
            cumulative_mean_gain_by_index.append(
                round(running_gain / running_gain_count, 6)
            )
        else:
            cumulative_mean_gain_by_index.append(
                None
                if running_gain_count == 0
                else round(running_gain / running_gain_count, 6)
            )

    mean_cost_by_index = _optional_mean_by_index(cost_matrix, length=sequence_length)
    mean_cost_increase_by_index = _optional_mean_by_index(
        cost_increase_matrix,
        length=sequence_length,
    )
    mean_latency_by_index = _optional_mean_by_index(
        latency_matrix,
        length=sequence_length,
    )

    aggregate: dict[str, Any] = {
        "baseline_reward_by_index": baseline_reward_series,
        "baseline_cost_by_index": baseline_cost_series,
        "mean_reward_by_index": mean_reward_by_index,
        "mean_gain_by_index": mean_gain_by_index,
        "mean_cost_by_index": mean_cost_by_index,
        "mean_cost_increase_by_index": mean_cost_increase_by_index,
        "mean_latency_by_index": mean_latency_by_index,
        "cumulative_mean_gain_by_index": cumulative_mean_gain_by_index,
        "final_cumulative_mean_gain": next(
            (
                value
                for value in reversed(cumulative_mean_gain_by_index)
                if value is not None
            ),
            0.0,
        ),
    }

    if raw_metric_name is not None and len(raw_metric_matrix) == len(run_outcomes):
        aggregate["raw_metric_name"] = raw_metric_name
        aggregate["mean_raw_metric_by_index"] = _mean_by_index(
            [[float(value) for value in run] for run in raw_metric_matrix],
            length=sequence_length,
        )

    return aggregate


def attach_baseline_to_outcomes(
    baseline_outcomes: list[dict[str, Any]],
    outcomes: list[dict[str, Any]],
    *,
    allow_partial_baseline: bool = False,
    allow_partial_rollout: bool = False,
) -> list[dict[str, Any]]:
    """Attach instance-matched baseline reward/cost deltas to rollout outcomes."""
    baseline_by_id = {
        str(outcome["instance_id"]): outcome for outcome in baseline_outcomes
    }
    baseline_ids = set(baseline_by_id)
    rollout_ids = {str(outcome["instance_id"]) for outcome in outcomes}
    _validate_rollout_instance_set(
        baseline_ids,
        rollout_ids,
        allow_partial_baseline=allow_partial_baseline,
        allow_partial_rollout=allow_partial_rollout,
        aggregate_context=False,
    )

    enriched: list[dict[str, Any]] = []
    for outcome in outcomes:
        baseline_outcome = baseline_by_id.get(str(outcome["instance_id"]))
        if baseline_outcome is None:
            enriched.append(dict(outcome))
            continue
        baseline_reward = float(baseline_outcome["reward"])
        reward = float(outcome["reward"])
        enriched_outcome = {
            **outcome,
            "baseline_reward": round(baseline_reward, 6),
            "gain": round(reward - baseline_reward, 6),
        }
        baseline_cost = baseline_outcome.get("cost_usd")
        rollout_cost = outcome.get("cost_usd")
        if baseline_cost is not None:
            baseline_cost_value = float(baseline_cost)
            enriched_outcome["baseline_cost"] = round(baseline_cost_value, 6)
            if rollout_cost is not None:
                enriched_outcome["cost_increase"] = round(
                    float(rollout_cost) - baseline_cost_value, 6
                )
        enriched.append(enriched_outcome)
    return enriched
