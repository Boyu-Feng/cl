"""Reserve public-command guidance for the final failed-task attempt.

Two native attempts run first. Only if both fail and at least one has a
high, public invalid-action rate does the third attempt use constrained
decoding. This preserves native second-attempt successes by construction.
"""
from __future__ import annotations

from .retry_trigger_actor import should_guide_retry


def should_guide_final_retry(episodes: list[dict]) -> bool:
    if len(episodes) < 2:
        return False
    if len(episodes) != 2:
        raise ValueError('Final retry rule expects exactly two prior attempts')
    if any(episode.get('reward') for episode in episodes):
        return False
    return any(should_guide_retry(episode) for episode in episodes)
