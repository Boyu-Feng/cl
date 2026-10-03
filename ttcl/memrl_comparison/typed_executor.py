"""Deterministic, schema-shaped actions from unverified historical submissions."""
from __future__ import annotations

import copy
import math


def apply_candidate(actor_action, candidate, episode_count):
    """Project a typed candidate only after a task-conditioned selector chooses it.

    The caller validates the result with the current response schema. Sources
    remain attached to the candidate for audit; they do not certify truth.
    """
    result = copy.deepcopy(actor_action)
    if candidate['kind'] == 'numeric_consensus':
        if set(result) != set(candidate['values']):
            raise ValueError('Numerical field set changed')
        n = candidate['sample_count']
        if n < 2:
            raise ValueError('Insufficient historical predictions')
        for key, mean in candidate['values'].items():
            current = result[key]
            if (not isinstance(current, (int, float)) or isinstance(current, bool)
                    or not math.isfinite(current) or not math.isfinite(mean)):
                raise ValueError('Non-finite numerical prediction')
            result[key] = (n * mean + current) / (n + 1)
        return result
    if candidate['kind'] == 'recurring_records':
        path = candidate['path']
        records = result.get(path)
        if not isinstance(records, list):
            raise ValueError('Record path changed')
        identity = candidate['identity_field']
        tolerance = candidate['tolerance']
        minimum = max(2, math.ceil(.05 * episode_count))
        for item in candidate['records']:
            if item['seen_in_episodes'] < minimum:
                continue
            value = item['identity_value']
            if any(isinstance(record, dict) and
                   isinstance(record.get(identity), (int, float)) and
                   abs(record[identity] - value) <= tolerance for record in records):
                continue
            historical = copy.deepcopy(item['sample_record'])
            historical[identity] = value
            # A historical record absent from the present submission cannot
            # be asserted to be currently active. All Boolean properties in
            # that record are conservatively false until observed again.
            for key, field in historical.items():
                if isinstance(field, bool):
                    historical[key] = False
            records.append(historical)
        return result
    raise ValueError('Unknown typed candidate')
