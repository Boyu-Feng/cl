"""Domain-independent action repetition control from public interaction events.

The caller supplies an observation, available actions, and the observed result.
No command syntax, environment family, or reward scale is built into the guard.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import json
from typing import Any, Iterable


def stable_key(value: Any) -> str:
    """Give JSON-like observations and actions a stable comparable key."""
    try:
        return json.dumps(value, sort_keys=True, ensure_ascii=False,
                          separators=(",", ":"), default=str)
    except (TypeError, ValueError):
        return repr(value)


@dataclass
class ActionFeedbackGuard:
    """Suppress unproductive repeats while always leaving an action available.

    `progressed` is optional public feedback from the environment. If it is
    absent, a changed observation or positive reward counts as progress. Repeated
    state-action visits are counted even if their next observations differ:
    this detects loops that return to a previously seen state.
    """

    repeat_limit: int = 2
    stagnation_limit: int = 2
    visits: Counter[tuple[str, str]] = field(default_factory=Counter)
    stagnant_actions: Counter[str] = field(default_factory=Counter)
    last_action: str | None = None
    last_no_progress: bool = False
    proposals: int = 0
    suppressed: int = 0
    fallbacks: int = 0

    def __post_init__(self) -> None:
        if self.repeat_limit < 1 or self.stagnation_limit < 1:
            raise ValueError("Action repetition limits must be positive")

    def allowed(self, observation: Any,
                candidates: Iterable[Any]) -> list[Any]:
        choices = list(candidates)
        if not choices:
            return []
        state = stable_key(observation)
        accepted = []
        for action in choices:
            key = stable_key(action)
            repeated_state = self.visits[(state, key)] >= self.repeat_limit
            stagnant_streak = (key == self.last_action and
                self.last_no_progress and
                self.stagnant_actions[key] >= self.stagnation_limit)
            if not (repeated_state or stagnant_streak):
                accepted.append(action)
        self.proposals += len(choices)
        if accepted:
            self.suppressed += len(choices) - len(accepted)
            return accepted
        # Do not invent an action or deadlock when every supplied choice loops.
        self.fallbacks += 1
        return choices

    def observe(self, observation: Any, action: Any,
                next_observation: Any, *, reward: float = 0.0,
                progressed: bool | None = None) -> None:
        state, choice = stable_key(observation), stable_key(action)
        self.visits[(state, choice)] += 1
        made_progress = (reward > 0 or stable_key(next_observation) != state
                         if progressed is None else bool(progressed))
        if made_progress:
            self.stagnant_actions[choice] = 0
        elif choice == self.last_action and self.last_no_progress:
            self.stagnant_actions[choice] += 1
        else:
            self.stagnant_actions[choice] = 1
        self.last_action = choice
        self.last_no_progress = not made_progress

    def summary(self) -> dict[str, int]:
        return {"proposals": self.proposals, "suppressed": self.suppressed,
                "fallbacks": self.fallbacks}
