"""Deterministic poker experience extracted exclusively from agent-visible text.

No benchmark imports, opponent identifiers, evaluator history, private cards, or
policy labels are used. Action shares are explicitly conditional on visibility;
the task omits some terminal actions and street labels for batched actions.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from collections import Counter
from typing import Any


_ACTIONS = {"FOLD", "CALL", "CHECK", "RAISE", "ALL_IN"}
_AUTO_MARKER = "Additional auto-resolved hand while advancing to the next prompt:"
_GUIDANCE = (
    "Historical public poker observations. These are observed action counts, "
    "not a hidden opponent policy. Action shares use observed actions as the "
    "denominator; missing terminal actions and unobserved opportunities are not "
    "counted. UNKNOWN_STREET means a transition prevented reliable attribution. "
    "After-raise counts cover only visible responses on an identifiable street, "
    "not every raise opportunity. Net chips depend on cards and our actions; "
    "these averages are not causal estimates of action value. Showdown examples "
    "are selectively revealed. Use the current hand to choose a legal action."
)


def _line(text: str, field: str) -> str:
    match = re.search(r"^\s*" + re.escape(field) + r":\s*([^\n]+)", text, re.M)
    return match.group(1).strip() if match else ""


def _new_profile() -> dict[str, Any]:
    return {
        "completed_hands": 0,
        "net_chips": 0,
        "wins": 0,
        "losses": 0,
        "ties": 0,
        "showdowns": 0,
        "opponent_actions": {},
        "our_actions": {},
        "our_raise_decisions": 0,
        "responses_after_our_raise": {
            "known_street_observed_responses": 0,
            "counts": {},
        },
        "public_raise_totals": {"n": 0, "sum": 0, "min": None, "max": None},
        "outcome_by_our_last_action": {},
        "showdown_examples": [],
    }


def _count(counts: dict[str, int], action: str) -> None:
    counts[action] = counts.get(action, 0) + 1


class PokerMemory:
    """Aggregate only facts made available via query/observation text.

    ``instance_id`` is used solely to avoid duplicate/incompatible updates; it
    never appears in returned memory because benchmark IDs can encode variants.
    Call observe once after each task step, then context for a later hand.
    """

    def __init__(self, max_context_chars: int = 9000, *, max_chars: int | None = None):
        if max_chars is not None:
            max_context_chars = max_chars
        if max_context_chars < 1000:
            raise ValueError("max_context_chars must be at least 1000")
        self.max_context_chars = max_context_chars
        self._opponents: dict[str, dict[str, Any]] = {}
        self._current_instance: str | None = None
        self._current_name = ""
        self._last_query_hash = ""
        self._previous_street = ""
        self._previous_action = ""
        self._completed_ids: set[str] = set()
        self._unattributed_auto_hands = 0

    def observe(
        self,
        query: str,
        action: dict,
        observation: str,
        *,
        instance_id: str,
        instance_complete: bool,
    ) -> None:
        if instance_id in self._completed_ids:
            return
        name = _line(query, "Opponent")[:80]
        hand = re.search(r"^Hand #(\d+) - ([A-Z_]+)\s*$", query, re.M)
        if not name or not hand:
            return
        street = hand.group(2)
        if instance_id != self._current_instance:
            self._current_instance = instance_id
            self._current_name = name
            self._last_query_hash = ""
            self._previous_street = ""
            self._previous_action = ""
        if name != self._current_name:
            return  # A hand cannot silently switch named opponent.
        profile = self._opponents.setdefault(name, _new_profile())
        # The task drops its one-time brief/change notice on an invalid retry.
        # Fingerprint the hand state rather than those presentation prefixes.
        query_hash = hashlib.sha256(query[hand.start() :].encode()).hexdigest()
        if query_hash != self._last_query_hash:
            raw_actions = _line(query, "Opponent's actions") or _line(
                query, "Opponent's action"
            )
            actions = [
                part.strip()
                for part in raw_actions.split("->")
                if part.strip() in _ACTIONS
            ]
            known_street = street == self._previous_street or (
                not self._previous_street and street == "PREFLOP"
            )
            attributed_street = street if known_street else "UNKNOWN_STREET"
            if actions:
                bucket = profile["opponent_actions"].setdefault(
                    attributed_street, {"observed_actions": 0, "counts": {}}
                )
                for opponent_action in actions:
                    bucket["observed_actions"] += 1
                    _count(bucket["counts"], opponent_action)
                if self._previous_action == "RAISE" and known_street:
                    after_raise = profile["responses_after_our_raise"]
                    after_raise["known_street_observed_responses"] += 1
                    _count(after_raise["counts"], actions[0])
                # Amounts are public only in a current "raised to" situation;
                # action lists alone contain no amounts. Attribute only to an
                # explicitly observed last RAISE on the same known street.
                raised = re.search(
                    r"^Situation: Opponent raised to (\d+)\b", query, re.M
                )
                if raised and known_street and actions[-1] == "RAISE":
                    amount = int(raised.group(1))
                    amounts = profile["public_raise_totals"]
                    amounts["n"] += 1
                    amounts["sum"] += amount
                    amounts["min"] = (
                        amount
                        if amounts["min"] is None
                        else min(amounts["min"], amount)
                    )
                    amounts["max"] = (
                        amount
                        if amounts["max"] is None
                        else max(amounts["max"], amount)
                    )
            self._last_query_hash = query_hash

        if observation.startswith("Invalid poker action:"):
            return
        chosen = str(action.get("action", "")).upper()
        if chosen in _ACTIONS:
            _count(profile["our_actions"].setdefault(street, {}), chosen)
            profile["our_raise_decisions"] += int(chosen == "RAISE")
            self._previous_action = chosen
        self._previous_street = street
        if not instance_complete:
            return

        primary, *auto_parts = observation.split(_AUTO_MARKER)
        profit_match = re.search(
            r"^Net chip change this hand:\s*([+-]?\d+) chips", primary, re.M
        )
        if not profit_match:
            return  # An aborted/capped hand has no fabricated zero outcome.
        profit = int(profit_match.group(1))
        self._completed_ids.add(instance_id)
        profile["completed_hands"] += 1
        profile["net_chips"] += profit
        profile["wins" if profit > 0 else "losses" if profit < 0 else "ties"] += 1
        terminal_action = chosen if chosen in _ACTIONS else "UNKNOWN"
        outcome = profile["outcome_by_our_last_action"].setdefault(
            terminal_action, {"hands": 0, "net_chips": 0}
        )
        outcome["hands"] += 1
        outcome["net_chips"] += profit
        if "SHOWDOWN:" in primary and _line(primary, "Opponent's hand"):
            profile["showdowns"] += 1
            profile["showdown_examples"].append(
                {
                    "public_hand_number": int(hand.group(1)),
                    "board": _line(primary, "Board")[:140],
                    "our_hand": _line(primary, "Your hand")[:80],
                    "opponent_hand": _line(primary, "Opponent's hand")[:80],
                    "net_chips": profit,
                }
            )
            profile["showdown_examples"] = profile["showdown_examples"][-3:]
        # The combined terminal text can include another auto-resolved hand,
        # possibly after an opponent switch. Its opponent is not identified.
        self._unattributed_auto_hands += sum(
            len(re.findall(r"^Net chip change this hand:", part, re.M))
            for part in auto_parts
        )

    def state_dict(self) -> dict:
        state = copy.deepcopy(
            {
                "format": "public_poker_statistics_v1",
                "opponents": self._opponents,
                "unattributed_auto_completed_hands": self._unattributed_auto_hands,
            }
        )
        for profile in state["opponents"].values():
            count = profile["completed_hands"]
            profile["mean_net_chips"] = (
                round(profile["net_chips"] / count, 4) if count else None
            )
            for bucket in profile["opponent_actions"].values():
                denominator = bucket["observed_actions"]
                bucket["shares_among_observed_actions"] = {
                    action: round(n / denominator, 4)
                    for action, n in sorted(bucket["counts"].items())
                }
        return state

    def context(self, query: str = "") -> str:
        state = self.state_dict()
        queried_opponent = _line(query, "Opponent")[:80]
        if queried_opponent:
            selected = [queried_opponent] if queried_opponent in self._opponents else []
        else:
            selected = sorted(self._opponents)
        if not selected:
            return ""
        lines = [_GUIDANCE]
        for name in selected:
            profile = state["opponents"][name]
            text = json.dumps(
                {"opponent": name, **profile},
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            if sum(map(len, lines)) + len(lines) + len(text) > self.max_context_chars:
                profile = copy.deepcopy(profile)
                profile.pop("showdown_examples")
                profile.pop("outcome_by_our_last_action")
                text = json.dumps(
                    {"opponent": name, **profile},
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            if sum(map(len, lines)) + len(lines) + len(text) > self.max_context_chars:
                # Keep complete JSON and the most important observed counts;
                # never truncate a card, field name, or probability denominator.
                counts = Counter()
                for bucket in profile["opponent_actions"].values():
                    counts.update(bucket["counts"])
                text = json.dumps(
                    {
                        "opponent": name,
                        "completed_hands": profile["completed_hands"],
                        "mean_net_chips": profile["mean_net_chips"],
                        "visible_actions_all_streets": dict(counts),
                        "observed_actions": sum(counts.values()),
                        "details_omitted_for_budget": True,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            if sum(map(len, lines)) + len(lines) + len(text) > self.max_context_chars:
                break
            lines.append(text)
        return "\n".join(lines) if len(lines) > 1 else ""
