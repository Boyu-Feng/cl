"""Checks for the synthetic action-transfer protocol."""

import random

from ttcl.trajectory_hyperlora.experience_pilot import (
    TEST_NUMBERS, TEST_SOURCE_TEMPLATES, action, trajectory,
)
from ttcl.trajectory_hyperlora.support_transfer_pilot import source_text
from ttcl.trajectory_hyperlora.verify_support_transfer import FRESH_NUMBERS


def test_query_readings_are_absent_from_history() -> None:
    for policy in (0, 1):
        source, _, source_numbers = trajectory(policy, random.Random(2001),
                                               TEST_SOURCE_TEMPLATES)
        assert source
        assert len(source_numbers) == len(set(source_numbers)) == 4
        assert not set(TEST_NUMBERS).intersection(source_numbers)
        assert sum(number % 2 for number in source_numbers) == 2


def test_opposite_policy_changes_all_correct_actions() -> None:
    for number in (*TEST_NUMBERS, 5, 20):
        assert action(0, number) != action(1, number)


def test_ordered_pair_requires_action_transfer() -> None:
    for policy in (0, 1):
        source, _, numbers = trajectory(policy, random.Random(3001),
                                        TEST_SOURCE_TEMPLATES, ordered_pair=True)
        assert source
        assert len(numbers) == 2 and numbers[0] % 2 == 1 and numbers[1] % 2 == 0
        assert action(policy, TEST_NUMBERS[0]) == action(policy, numbers[0])
        assert action(policy, TEST_NUMBERS[1]) == action(policy, numbers[1])
        assert TEST_NUMBERS[0] not in numbers and TEST_NUMBERS[1] not in numbers


def test_source_actions_and_new_readings_are_content_bound() -> None:
    numbers = [3, 8]
    history = source_text(0, numbers)
    assert "reading 3 (ODD); agent pressed RIGHT; environment feedback: success" in history
    assert "reading 8 (EVEN); agent pressed LEFT; environment feedback: success" in history
    assert not set(numbers).intersection(TEST_NUMBERS)
    assert not set(numbers).intersection(FRESH_NUMBERS)
