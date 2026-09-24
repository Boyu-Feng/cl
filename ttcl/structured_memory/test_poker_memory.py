"""Public-trace-only tests; no benchmark opponents or hidden state are imported."""

import unittest

from ttcl.structured_memory.poker_memory import PokerMemory


def prompt(
    hand=1, opponent="Rin", street="PREFLOP", actions="", situation="Action to you"
):
    action_line = f"\nOpponent's actions: {actions}" if actions else ""
    return (
        f"Hand #{hand} - {street}\nOpponent: {opponent}\nYour position: big blind\n"
        f"Your hand: [A♠] [K♠]\nBoard: No cards yet\nPot: 15 chips\n"
        f"Your chips: 990{action_line}\n\nSituation: {situation}\n"
    )


def terminal(profit=10, showdown=False):
    shown = (
        "\nSHOWDOWN:\n  Board: [2♠] [3♠] [4♠]\n  Your hand: [A♠] [K♠]\n  Opponent's hand: [Q♥] [J♥]"
        if showdown
        else ""
    )
    return f"Hand 1 complete: You WON!{shown}\nFinal pot: 30 chips\nEnding stack: 1010 chips\nNet chip change this hand: {profit:+d} chips\nTotal profit: {profit:+d} chips"


class PokerMemoryTest(unittest.TestCase):
    def test_uniform_factory_keyword(self):
        memory = PokerMemory(max_chars=1600)
        self.assertEqual(memory.max_context_chars, 1600)
        memory.observe(
            prompt(actions="CALL"),
            {"action": "CHECK"},
            terminal(showdown=True),
            instance_id="x",
            instance_complete=True,
        )
        self.assertLessEqual(len(memory.context()), 1600)

    def test_visible_actions_separated_by_name_and_street(self):
        memory = PokerMemory()
        memory.observe(
            prompt(actions="CALL"),
            {"action": "CHECK"},
            "Action taken",
            instance_id="hidden-policy-a",
            instance_complete=False,
        )
        memory.observe(
            prompt(street="FLOP", actions="CHECK"),
            {"action": "CHECK"},
            terminal(showdown=True),
            instance_id="hidden-policy-a",
            instance_complete=True,
        )
        memory.observe(
            prompt(
                hand=2,
                opponent="Kai",
                actions="RAISE",
                situation="Opponent raised to 40 (you need 30 to call)",
            ),
            {"action": "FOLD"},
            terminal(-10),
            instance_id="hidden-policy-b",
            instance_complete=True,
        )
        state = memory.state_dict()
        rin = state["opponents"]["Rin"]
        self.assertEqual(rin["opponent_actions"]["PREFLOP"]["counts"]["CALL"], 1)
        self.assertEqual(
            rin["opponent_actions"]["UNKNOWN_STREET"]["counts"]["CHECK"], 1
        )
        self.assertEqual(rin["completed_hands"], 1)
        self.assertEqual(rin["showdowns"], 1)
        kai = state["opponents"]["Kai"]
        self.assertEqual(kai["net_chips"], -10)
        self.assertEqual(kai["public_raise_totals"]["sum"], 40)
        self.assertNotIn("hidden-policy", memory.context())
        self.assertNotIn("Q♥", memory.context(prompt(opponent="Kai")))

    def test_invalid_retry_does_not_duplicate_actions_or_own_decisions(self):
        memory = PokerMemory()
        query = prompt(actions="RAISE")
        memory.observe(
            "=== Brief ===\nTemporary one-time task instructions\n\n" + query,
            {"action": "CHECK"},
            "Invalid poker action: cannot check",
            instance_id="x",
            instance_complete=False,
        )
        memory.observe(
            query,
            {"action": "CALL"},
            "Action taken",
            instance_id="x",
            instance_complete=False,
        )
        profile = memory.state_dict()["opponents"]["Rin"]
        self.assertEqual(profile["opponent_actions"]["PREFLOP"]["observed_actions"], 1)
        self.assertEqual(profile["our_actions"]["PREFLOP"], {"CALL": 1})

    def test_opportunity_denominators_are_not_win_based_fold_rates(self):
        memory = PokerMemory()
        memory.observe(
            prompt(),
            {"action": "RAISE", "amount": 50},
            "Action taken",
            instance_id="x",
            instance_complete=False,
        )
        memory.observe(
            prompt(
                actions="RAISE",
                situation="Opponent raised to 100 (you need 50 to call)",
            ),
            {"action": "RAISE", "amount": 200},
            terminal(),
            instance_id="x",
            instance_complete=True,
        )
        p = memory.state_dict()["opponents"]["Rin"]
        self.assertEqual(p["our_raise_decisions"], 2)
        self.assertEqual(
            p["responses_after_our_raise"]["known_street_observed_responses"], 1
        )
        self.assertEqual(p["responses_after_our_raise"]["counts"], {"RAISE": 1})
        self.assertNotIn("FOLD", p["opponent_actions"]["PREFLOP"]["counts"])

    def test_terminal_only_reveals_cards_when_present_and_auto_hands_unattributed(self):
        memory = PokerMemory()
        obs = (
            terminal()
            + "\nAdditional auto-resolved hand while advancing to the next prompt:\n"
            + terminal(-5)
        )
        memory.observe(
            prompt(),
            {"action": "RAISE", "amount": 50},
            obs,
            instance_id="x",
            instance_complete=True,
        )
        memory.observe(
            prompt(),
            {"action": "RAISE", "amount": 50},
            obs,
            instance_id="x",
            instance_complete=True,
        )
        state = memory.state_dict()
        self.assertEqual(state["opponents"]["Rin"]["completed_hands"], 1)
        self.assertEqual(state["opponents"]["Rin"]["showdown_examples"], [])
        self.assertEqual(state["unattributed_auto_completed_hands"], 1)

    def test_context_is_bounded_deterministic_and_unknown_opponent_is_empty(self):
        memory = PokerMemory(max_context_chars=2400)
        for i in range(50):
            memory.observe(
                prompt(hand=i + 1, opponent=f"player-{i}"),
                {"action": "FOLD"},
                terminal(-5, showdown=True),
                instance_id=str(i),
                instance_complete=True,
            )
        self.assertLessEqual(len(memory.context()), 2400)
        self.assertEqual(memory.context(), memory.context())
        self.assertEqual(memory.context(prompt(opponent="new-player")), "")


if __name__ == "__main__":
    unittest.main()
