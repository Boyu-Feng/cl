"""Find environment responses that recur after the same public action.

This source-bound readout is interface-independent: it inspects public action
and feedback fields, with no task names, schema keys, or benchmark labels.
Repetition is evidence of stability across episodes, not proof that the
action or the response applies to the current task.
"""
from __future__ import annotations

from collections import defaultdict
import json
import re


MIN_SOURCE_EPISODES = 2


def _payload(feedback: object) -> str | None:
    if not isinstance(feedback, str):
        return None
    parts = re.split(r'\n\s*\n', feedback, maxsplit=1)
    text = parts[-1].strip()
    if len(text) < 24 or len(set(re.findall(r'\w+', text.lower()))) < 3:
        return None
    if re.match(r'(?i)^(?:error|exception|failed)\s*[:\s]', text):
        return None
    return text


def stable_feedback(events: list[dict]) -> list[dict]:
    """Return exact action/observed-response recurrences across episodes."""
    groups = defaultdict(dict)
    for event in events:
        if event.get('instance_complete'):
            continue
        payload = _payload(event.get('observed_feedback'))
        if payload is None:
            continue
        action = json.dumps(event.get('submitted_action'), ensure_ascii=False,
                            sort_keys=True, separators=(',', ':'), allow_nan=False)
        groups[(action, payload)][event['episode']] = event
    result = []
    for (action, payload), by_episode in groups.items():
        if len(by_episode) < MIN_SOURCE_EPISODES:
            continue
        sources = sorted(by_episode.values(), key=lambda event: event['episode'])
        result.append(dict(action=json.loads(action), feedback=payload,
                           episode_count=len(sources),
                           source_sha256=[event['source_sha256'] for event in sources],
                           public_trace_sha256=[event['public_trace_sha256']
                                                for event in sources],
                           prior_inputs=[event['public_input'] for event in sources]))
    return sorted(result, key=lambda row: (-row['episode_count'],
                                           json.dumps(row['action'], sort_keys=True),
                                           row['feedback']))
