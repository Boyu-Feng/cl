"""Task-name-free candidates from repeated JSON action shapes.

These are summaries of *submitted actions*, not verified environment facts.
They give a schema-aware reader exact numerical state without pretending that
free-form notebook claims or source hashes establish correctness.
"""
from __future__ import annotations

import math
import itertools
import statistics


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _numeric_map(value):
    return (isinstance(value, dict) and len(value) >= 8 and
            all(_number(x) for x in value.values()))


def numeric_consensus(events, current):
    """Mean of prior matching numerical submissions, without using reward."""
    if not _numeric_map(current):
        return None
    same = {}
    for event in events:
        action = event['submitted_action']
        if _numeric_map(action) and action.keys() == current.keys():
            same[event['episode']] = event
    if len(same) < 2:
        return None
    prior = list(same.values())
    values = {key: statistics.fmean(e['submitted_action'][key] for e in prior)
              for key in current}
    return dict(kind='numeric_consensus', status='unverified_submission_aggregate',
                sample_count=len(prior), values=values,
                source_sha256=[e['source_sha256'] for e in prior[-8:]])


def _record_lists(events, current):
    if not isinstance(current, dict):
        return []
    result = []
    for path, records in current.items():
        if not isinstance(records, list) or not records or not all(
                isinstance(record, dict) for record in records):
            continue
        historical = []
        for event in events:
            action = event['submitted_action']
            values = action.get(path) if isinstance(action, dict) else None
            if isinstance(values, list):
                historical.extend((event, item) for item in values
                                  if isinstance(item, dict))
        if historical:
            result.append((path, records, historical))
    return result


def _identity_field(historical):
    shared = set.intersection(*(set(item) for _, item in historical))
    fields = [field for field in sorted(shared) if all(
              _number(item[field]) for _, item in historical)]
    episodes = sorted({e['episode'] for e, _ in historical})
    if len(episodes) < 2:
        return None
    candidates = []
    for field in fields:
        by_episode = {episode: sorted(item[field] for e, item in historical
                                      if e['episode'] == episode)
                      for episode in episodes}
        gaps = [b-a for values in by_episode.values()
                for a, b in zip(values, values[1:]) if b > a]
        if not gaps:
            continue
        tolerance = max(0.01, .15 * statistics.median(gaps))
        hits, comparisons = 0, 0
        for previous, current in itertools.combinations(episodes, 2):
            if current - previous > 10:
                continue
            old, new = by_episode[previous], by_episode[current]
            for value in new:
                hits += any(abs(value - prior) <= tolerance for prior in old)
                comparisons += 1
        within_hits = sum(abs(a-b) <= tolerance for values in by_episode.values()
                          for a, b in itertools.combinations(values, 2))
        within_pairs = sum(len(values) * (len(values)-1) // 2
                           for values in by_episode.values())
        if comparisons:
            recurrence = hits / comparisons
            collision = within_hits / within_pairs if within_pairs else 0.
            candidates.append((recurrence-collision, hits, field, tolerance))
    if not candidates:
        return None
    score, hits, field, tolerance = max(candidates)
    return (field, tolerance) if hits >= 2 and score >= .1 else None


def recurring_records(events, current):
    """Find recurrence in list-valued structured submissions by a learned key."""
    output = []
    for path, _, historical in _record_lists(events, current):
        identity = _identity_field(historical)
        if identity is None:
            continue
        field, tolerance = identity
        clusters = []
        for event, item in historical:
            value = float(item[field])
            matches = [cluster for cluster in clusters
                       if event['episode'] not in cluster['episodes']
                       and abs(value - statistics.median(cluster['values'])) <= tolerance]
            if matches:
                cluster = min(matches, key=lambda x: abs(
                    value - statistics.median(x['values'])))
            else:
                cluster = dict(values=[], episodes=set(), sources=[], records=[])
                clusters.append(cluster)
            cluster['values'].append(value)
            cluster['episodes'].add(event['episode'])
            cluster['sources'].append(event['source_sha256'])
            cluster['records'].append(item)
        durable = sorted((cluster for cluster in clusters if len(cluster['episodes']) >= 2),
                         key=lambda x: (-len(x['episodes']), statistics.median(x['values'])))
        if not durable:
            continue
        output.append(dict(kind='recurring_records', path=path,
                           status='unverified_submission_recurrence',
                           identity_field=field, tolerance=tolerance,
                           records=[dict(identity_value=statistics.median(c['values']),
                                         seen_in_episodes=len(c['episodes']),
                                         sample_record=c['records'][-1],
                                         source_sha256=list(dict.fromkeys(c['sources']))[-8:])
                                    for c in durable[:16]]))
    return output


def typed_candidates(events, current):
    output = []
    consensus = numeric_consensus(events, current)
    if consensus:
        output.append(consensus)
    output.extend(recurring_records(events, current))
    return output
