import io
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest.mock import patch

from src.interface import Query, Response
from src.cli import main
from src.tasks.codebase_adaptation.task import CodebaseAdaptationTask
from src.tasks.codebase_adaptation.task_loader import load_tasks
from src.tasks.codebase_adaptation.task_loader import TaskInstance
from src.tasks.database_exploration.task import _build_schema_drift_question_sequence
from src.tasks.exploitable_poker.task import Poker, PokerAction
from src.tasks.variants import (
    clear_task_variant_cache,
    get_task_variant,
    list_task_variants,
)
from src.tasks.schedules import clear_task_schedule_cache, get_task_schedule


class TaskVariantsTests(unittest.TestCase):
    def tearDown(self):
        clear_task_variant_cache()
        clear_task_schedule_cache()

    def test_list_task_variants_returns_expected_poker_variants(self):
        variants = list_task_variants("exploitable_poker")
        self.assertEqual(
            [variant.id for variant in variants],
            ["calling_station", "fit_or_fold", "loose_aggressive"],
        )

    def test_cli_inspect_shows_task_variants(self):
        stdout = io.StringIO()
        with patch("sys.argv", ["clbench", "inspect", "task", "exploitable_poker"]):
            with redirect_stdout(stdout):
                main()

        self.assertIn("calling_station", stdout.getvalue())
        self.assertIn("loose_aggressive", stdout.getvalue())

    def test_poker_variant_rejects_conflicting_fixed_overrides(self):
        with self.assertRaises(ValueError):
            Poker(variant="calling_station", starting_chips=500)

    def test_poker_variant_keeps_policy_without_leaking_query_metadata(self):
        task = Poker(variant="fit_or_fold")
        query = task.reset()

        self.assertEqual(task.opponent_policy, "fit_or_fold")
        self.assertEqual(task.opponent_name, "Adam")
        self.assertIn("Opponent: Adam", query.prompt)
        self.assertNotIn("opponent_policy", query.metadata)
        self.assertNotIn("variant_id", query.metadata)
        self.assertNotIn("variant_display_name", query.metadata)

    def test_poker_query_includes_table_size_and_position(self):
        task = Poker(
            num_instances=1, opponent_policy="calling_station", opponent_name="Tom"
        )
        query = task.reset()

        self.assertIn("Table: Heads-up Texas Hold'em (2 players)", query.prompt)
        self.assertIn("Your position:", query.prompt)
        self.assertEqual(query.metadata["num_players"], 2)
        self.assertIn(query.metadata["position"], {"Button / small blind", "Big blind"})

    def test_database_schema_drift_shuffles_each_stage_without_crossing_boundary(self):
        pre = [{"question_id": f"pre-{i}"} for i in range(20)]
        post = [{"question_id": f"post-{i}"} for i in range(20)]

        seq = _build_schema_drift_question_sequence(
            pre_questions=pre,
            post_questions=post,
            pre_drift_count=20,
            post_drift_count=20,
            seed=42,
            run_index=0,
        )
        seq_2 = _build_schema_drift_question_sequence(
            pre_questions=pre,
            post_questions=post,
            pre_drift_count=20,
            post_drift_count=20,
            seed=42,
            run_index=1,
        )

        first_stage = [q["question_id"] for q in seq[:20]]
        second_stage = [q["question_id"] for q in seq[20:]]
        self.assertEqual(set(first_stage), {f"pre-{i}" for i in range(20)})
        self.assertEqual(set(second_stage), {f"post-{i}" for i in range(20)})
        self.assertNotEqual(first_stage, [f"pre-{i}" for i in range(20)])
        self.assertNotEqual(second_stage, [f"post-{i}" for i in range(20)])
        self.assertNotEqual(first_stage, [q["question_id"] for q in seq_2[:20]])

    def test_poker_rejects_check_when_facing_raise(self):
        task = Poker(
            num_instances=1, opponent_policy="loose_aggressive", opponent_name="Tom"
        )
        query = task.reset()

        self.assertGreater(query.metadata["chips_to_call"], 0)
        self.assertIn("CALL", query.metadata["legal_actions"])
        self.assertNotIn("CHECK", query.metadata["legal_actions"])

        result = task.step(
            Response(action=PokerAction(thinking="", action="CHECK", amount=None))
        )

        self.assertFalse(result.done)
        self.assertFalse(result.observation.instance_complete)
        self.assertIn("Invalid poker action", result.observation.content)
        self.assertEqual(task.hands_played, 0)
        self.assertEqual(task.system_profit, 0)
        self.assertTrue(task.game.is_hand_running())
        self.assertEqual(
            result.next_query.metadata["hand_num"], query.metadata["hand_num"]
        )

    def test_poker_preserves_feedback_when_next_hand_auto_resolves(self):
        task = Poker(
            num_instances=3, opponent_policy="calling_station", opponent_name="Tom"
        )
        task.reset()

        call_count = [0]

        def fake_start_fresh_hand():
            call_count[0] += 1
            if call_count[0] == 1:
                task.game = SimpleNamespace(
                    is_hand_running=lambda: False,
                    players=[
                        SimpleNamespace(chips=task.starting_chips + 5),
                        SimpleNamespace(chips=task.starting_chips - 5),
                    ],
                )
                task.system_folded = False
                task.bot_folded = True
            else:
                task.game = SimpleNamespace(
                    is_hand_running=lambda: True,
                    players=[
                        SimpleNamespace(chips=task.starting_chips),
                        SimpleNamespace(chips=task.starting_chips),
                    ],
                )
                task.system_folded = False
                task.bot_folded = False

        task._start_fresh_hand = fake_start_fresh_hand
        task._get_current_query = lambda: Query(
            prompt="next hand",
            response_schema=PokerAction,
            metadata={"hand_num": task.hands_played + 1},
        )

        task.hands_played = 0
        task.system_profit = 5
        task.hand_history = [{"profit": 5}]
        task.current_hand_actions = []
        task.system_folded = False
        task.bot_folded = True
        task.game = SimpleNamespace(
            is_hand_running=lambda: False,
            players=[
                SimpleNamespace(chips=task.starting_chips + 5),
                SimpleNamespace(chips=task.starting_chips - 5),
            ],
        )

        result = task._handle_hand_end(None)

        self.assertFalse(result.done)
        self.assertIn("Hand 1 complete:", result.observation.content)
        self.assertEqual(call_count[0], 2)

    def test_poker_rollout_continues_after_zero_chip_hand(self):
        task = Poker(
            num_instances=2, opponent_policy="calling_station", opponent_name="Tom"
        )
        task.game = SimpleNamespace(
            players=[
                SimpleNamespace(chips=0),
                SimpleNamespace(chips=task.starting_chips),
            ]
        )
        task.system_folded = True
        task.bot_folded = False
        task.current_hand_actions = []
        task.hands_played = 0
        task.system_profit = 0

        def fake_start_fresh_hand():
            task.game = SimpleNamespace(
                is_hand_running=lambda: True,
                players=[
                    SimpleNamespace(chips=task.starting_chips),
                    SimpleNamespace(chips=task.starting_chips),
                ],
            )

        task._start_fresh_hand = fake_start_fresh_hand
        task._get_current_query = lambda: Query(
            prompt="next hand",
            response_schema=PokerAction,
            metadata={},
        )

        result = task._handle_hand_end(None)

        self.assertFalse(result.done)
        self.assertEqual(task.hands_played, 1)
        self.assertEqual(task.system_profit, -task.starting_chips)
        self.assertEqual(result.next_query.prompt, "next hand")

    def test_poker_baseline_reset_keeps_identity_for_auto_resolved_hand(self):
        task = Poker(
            num_instances=1, opponent_policy="calling_station", opponent_name="Tom"
        )

        def fake_start_fresh_hand():
            task.game = SimpleNamespace(
                is_hand_running=lambda: False,
                players=[
                    SimpleNamespace(chips=0),
                    SimpleNamespace(chips=task.starting_chips),
                ],
            )
            task.system_folded = True
            task.bot_folded = False

        task._start_fresh_hand = fake_start_fresh_hand

        query = task.reset_baseline_instance(0)
        outcome = task.get_instance_outcomes()[0]

        self.assertTrue(query.metadata["done"])
        self.assertEqual(query.instance_id, outcome.instance_id)
        self.assertEqual(query.instance_index, outcome.instance_index)
        self.assertEqual(query.metadata["instance_id"], outcome.instance_id)

    def test_codebase_issue_query_hides_instance_and_variant_identity(self):
        task = CodebaseAdaptationTask()
        task.instances = [
            TaskInstance(
                instance_id="example__repo.123abc",
                repo="example/repo",
                patch="",
                fail_to_pass=[],
                pass_to_pass=[],
                image_name="example:latest",
                problem_statement="Fix the failing behavior.",
            )
        ]
        task.current_issue_idx = 0
        task.variant = "tablib"
        task.variant_display_name = "Tablib Final"
        task.variant_repo = "example/repo"

        with patch(
            "src.tasks.codebase_adaptation.task._load_mswea_templates",
            return_value={
                "system_template": "SYSTEM",
                "instance_template": "{{ task }}",
                "observation_template": "{{ output }}",
            },
        ):
            issue_query = task._make_issue_query()

        self.assertIn("Repository: example/repo", issue_query.prompt)
        self.assertNotIn("Instance:", issue_query.prompt)
        self.assertNotIn("example__repo.123abc", issue_query.prompt)
        self.assertNotIn("issue_id", issue_query.metadata)
        self.assertNotIn("variant_id", issue_query.metadata)
        self.assertNotIn("variant_display_name", issue_query.metadata)
        self.assertNotIn("variant_repo", issue_query.metadata)

        task._new_issue_pending = False
        task.current_steps = 3
        followup_query = task._next_query()

        self.assertEqual(
            followup_query.prompt,
            "Repository: example/repo\nWhat's your next command?",
        )
        self.assertNotIn("issue_id", followup_query.metadata)
        self.assertNotIn("variant_id", followup_query.metadata)

    def test_codebase_variant_rejects_conflicting_fixed_overrides(self):
        with self.assertRaises(ValueError):
            CodebaseAdaptationTask(variant="tablib", max_steps_per_issue=5)

    def test_codebase_loader_preserves_explicit_instance_order(self):
        variant = get_task_variant("codebase_adaptation", "tablib")
        reversed_ids = list(reversed(variant.config["instance_ids"][:3]))

        tasks = load_tasks(
            variant.defaults["dataset_path"],
            instance_ids=reversed_ids,
        )

        self.assertEqual([task.instance_id for task in tasks], reversed_ids)

    def test_codebase_final_variant_uses_final_dataset(self):
        task = CodebaseAdaptationTask(variant="tablib")

        self.assertEqual(
            task.dataset_path,
            "data/codebase_adaptation/final-dataset.jsonl",
        )
        self.assertEqual(task.max_steps_per_issue, 40)
        self.assertEqual(task.variant_repo, "jazzband/tablib")
        self.assertEqual(
            task.variant_instance_ids,
            [
                "jazzband__tablib-534",
                "jazzband__tablib-540",
                "jazzband__tablib-547",
                "jazzband__tablib-579",
                "jazzband__tablib-584",
                "jazzband__tablib-594",
                "jazzband__tablib-595",
                "jazzband__tablib-596",
                "jazzband__tablib-613",
            ],
        )

    def test_codebase_final_schedule_initializes(self):
        schedule = get_task_schedule("codebase_adaptation", "default")
        self.assertEqual(len(schedule.stages), 2)

        task = CodebaseAdaptationTask(schedule="default")
        self.assertEqual(
            task.dataset_path,
            "data/codebase_adaptation/final-dataset.jsonl",
        )
        self.assertEqual(task.num_instances, 19)
        self.assertEqual(task.max_steps_per_issue, 40)
        self.assertEqual(task.variant, "tablib")


if __name__ == "__main__":
    unittest.main()
