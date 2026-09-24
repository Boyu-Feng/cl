"""Protocol and extraction checks with invented public evidence."""

import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from pydantic import BaseModel

from ttcl.structured_memory.run_benchmark import normalize_action, StructuredSystem
from ttcl.structured_memory.focused_memory import (
    FocusedDatabaseMemory,
    FocusedCohortMemory,
    ConditionalPokerMemory,
    ProceduralDatabaseMemory,
)
from ttcl.structured_memory.ablation import AblationSystem, AblationRecorder
from ttcl.structured_memory.test_poker_memory import prompt, terminal
from ttcl.structured_memory.test_runner import FrozenFakeModel
from src.interface import Observation, InstanceOutcome, Query


class DatabaseAction(BaseModel):
    action: str
    content: str


class Params(BaseModel):
    tool: str
    sql: str


class ToolAction(BaseModel):
    thought: str
    tool_call: Params


class AblationTests(unittest.TestCase):
    def test_diagnostic_procedures_use_only_observed_identifiers_and_no_results(self):
        m = ProceduralDatabaseMemory()
        self.assertEqual(m.context("Reviews in 2020"), "")
        m.observe(
            "",
            {"action": "QUERY", "content": "PRAGMA table_info(observed_table)"},
            "Query result\n\ncid | name | type | pk\n----+------+-----+---\n0 | ts | INTEGER | 0\n1 | main_cat | TEXT | 0",
            instance_id="a",
            instance_complete=False,
        )
        text = m.context("How many reviews in 2020?")
        parsed = json.loads(text)
        probes = parsed["records"][0]["candidate_diagnostic_queries"]
        self.assertEqual(len(probes), 2)
        self.assertTrue(all('"observed_table"' in p["sql"] for p in probes))
        self.assertNotIn("unixepoch", m.context("What is the average rating?"))
        self.assertNotIn("expected_answer", text)
        self.assertNotIn("observed_table", m.context("NOTICE: database schema changed"))

    def test_format_retry_is_logged_and_counted_without_inventing_an_action(self):
        class Model(FrozenFakeModel):
            def generate(self, messages, seed):
                result = super().generate(messages, seed)
                result["raw_response"] = (
                    '{"action":"ANSWER"}'
                    if len(self.requests) == 1
                    else '{"action":"QUERY","content":"SELECT 7"}'
                )
                return result

        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(
                task="database_exploration",
                memory_chars=16000,
                seed=42,
                num_instances=1,
                max_turns_per_instance=10,
                action_retries=1,
                normalize_action=True,
            )
            model = Model()
            system = StructuredSystem(args, "independent", model, Path(directory), "")
            result = system.respond(
                Query(
                    prompt="Inspect",
                    response_schema=DatabaseAction,
                    instance_id="a",
                    instance_index=0,
                )
            )
            self.assertEqual(result.action.content, "SELECT 7")
            self.assertEqual(
                (system.calls, system.input_tokens, system.output_tokens), (2, 20, 6)
            )
            events = [
                json.loads(s)
                for s in (Path(directory) / "responses.jsonl").read_text().splitlines()
            ]
            self.assertIsNotNone(events[0]["parse_error"])
            self.assertNotIn("action", events[0])
            self.assertEqual(events[1]["format_retry"], 1)
            self.assertNotEqual(
                events[0]["generation_seed"], events[1]["generation_seed"]
            )
            self.assertEqual(system.turn, 1)

    def test_packaging_repairs_preserve_query_and_reject_missing_values(self):
        action, kind = normalize_action(
            '{"action":"PRAGMA table_info(things)"}', DatabaseAction
        )
        self.assertEqual(
            action.model_dump(),
            {"action": "QUERY", "content": "PRAGMA table_info(things)"},
        )
        self.assertEqual(kind, "move_sql_to_content")
        raw = {
            "thought": "test",
            "tool_call": {"tool": "query_sql", "tool_call_params": {"sql": "SELECT 7"}},
        }
        action, kind = normalize_action(json.dumps(raw), ToolAction)
        self.assertEqual(action.tool_call.sql, "SELECT 7")
        self.assertEqual(kind, "flatten_tool_call_params")
        for invalid in ('{"action":"ANSWER"}', '{"action":"DROP TABLE things"}'):
            with self.assertRaises(ValueError):
                normalize_action(invalid, DatabaseAction)

    def test_cross_street_response_is_attributed_to_preceding_raise_once(self):
        m = ConditionalPokerMemory()
        m.observe(
            prompt(actions="CALL"),
            {"action": "RAISE", "amount": 30},
            "Action taken",
            instance_id="x",
            instance_complete=False,
        )
        q = prompt(street="FLOP", actions="CALL -> CHECK")
        m.observe(
            q,
            {"action": "CALL"},
            "Invalid poker action: cannot call",
            instance_id="x",
            instance_complete=False,
        )
        m.observe(
            q,
            {"action": "CHECK"},
            terminal(-20, showdown=True),
            instance_id="x",
            instance_complete=True,
        )
        profile = m.state_dict()["opponents"]["Rin"]
        bucket = next(iter(profile["responses"].values()))
        self.assertEqual(bucket["counts"], {"CALL": 1})
        self.assertEqual(bucket["observed_responses"], 1)
        self.assertEqual(bucket["our_raise_opportunities"], 1)
        context = m.context(prompt())
        self.assertNotIn("net_chips", context)
        self.assertNotIn("net_chip", context)
        self.assertEqual(m.context(prompt(opponent="Other")), "")

    def test_unobserved_terminal_response_is_not_inferred_from_reward(self):
        m = ConditionalPokerMemory()
        m.observe(
            prompt(),
            {"action": "RAISE", "amount": 30},
            terminal(100),
            instance_id="x",
            instance_complete=True,
        )
        context = json.loads(m.context(prompt()))
        self.assertEqual(context["all_streets_after_our_raise"]["missing"], 1)
        self.assertEqual(context["all_streets_after_our_raise"]["counts"], {})

    def test_database_cases_keep_episode_verdict_scope_and_hide_stale_schema(self):
        m = FocusedDatabaseMemory()
        q = "Question 1/12\n\nHow many things?\n\nYou have access"
        m.observe(
            q,
            {"action": "QUERY", "content": "SELECT COUNT(*) FROM things"},
            "Query result\n\nCOUNT(*)\n--------\n7",
            instance_id="a",
            instance_complete=False,
        )
        m.observe(
            q,
            {"action": "ANSWER", "content": "7"},
            "Question 1: CORRECT!",
            instance_id="a",
            instance_complete=True,
        )
        r = json.loads(m.context(q))["records"][0]
        self.assertEqual(r["episode_verdict"], "correct")
        self.assertIn("not each query", r["verification_scope"])
        changed = m.context("NOTICE: database schema changed")
        self.assertNotIn("SELECT COUNT", changed)

    def test_cohort_recipe_retains_required_instrument_scope(self):
        m = FocusedCohortMemory()
        m.observe(
            "## Study 1/2: A\n**Enrollment:** Clinic",
            {"tool": "get_database_metadata"},
            "=== Columns ===\n  score: Instrument Z; not interchangeable with Instrument Y",
            instance_id="a",
            instance_complete=False,
        )
        m.observe(
            "",
            {
                "tool": "estimate_survival_by_group",
                "group_expression": "CASE WHEN score < 7 THEN 'a' ELSE 'b' END",
            },
            "=== Per-Group Survival ===\n  a: n=10 (100%)  S(12m)=0.9000  S(24m)=0.8000  S(36m)=0.7000",
            instance_id="a",
            instance_complete=True,
        )
        ctx = json.loads(m.context("Study A"))
        self.assertIn(
            "Instrument Z", ctx["records"][0]["required_column_definitions"]["score"]
        )
        self.assertEqual(ctx["records"][0]["study_scope"]["enrollment"], "Clinic")

    def test_only_completed_scalar_reaches_reward_variant_never_metadata(self):
        for variant in ["focused", "focused_reward"]:
            with (
                self.subTest(variant=variant),
                tempfile.TemporaryDirectory() as directory,
            ):
                args = SimpleNamespace(
                    task="database_exploration",
                    memory_chars=16000,
                    seed=42,
                    num_instances=2,
                    max_turns_per_instance=10,
                    variant=variant,
                )
                model = FrozenFakeModel()

                class Action(BaseModel):
                    command: str

                system = AblationSystem(args, "structured", model, Path(directory), "")
                system.respond(
                    Query(
                        prompt="Public question",
                        response_schema=Action,
                        instance_id="SECRET_POLICY_ID",
                        instance_index=0,
                    )
                )
                recorder = AblationRecorder(Path(directory), system, 2)
                outcome = InstanceOutcome(
                    "SECRET_POLICY_ID", 0, 0.731, metadata={"gold": "SECRET_GOLD"}
                )
                recorder.sync_instance_outcomes([outcome])
                self.assertFalse(system.finished_cases)
                system.observe(
                    Observation(content="Public done", instance_complete=True)
                )
                recorder.sync_instance_outcomes([outcome])
                recorder.sync_instance_outcomes([outcome])
                context = system.experience_context("Next question")
                self.assertEqual("0.731" in context, variant == "focused_reward")
                self.assertNotIn("SECRET_POLICY_ID", context)
                self.assertNotIn("SECRET_GOLD", context)
                self.assertEqual(
                    len(
                        (Path(directory) / "experience_cases.jsonl")
                        .read_text()
                        .splitlines()
                    ),
                    1,
                )


if __name__ == "__main__":
    unittest.main()
