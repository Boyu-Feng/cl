"""Focused tests for Claude Code CLI system continuation controls."""

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pydantic import BaseModel

from src.artifacts import save_artifacts
from src.interface import Observation, Query

_FAKE_CONTAINER_ID = "abc123def456"
_SESSION_ID = "sess_abc123"


class _SimpleAction(BaseModel):
    value: str


def _make_jsonl(*events: dict) -> str:
    return "\n".join(json.dumps(e) for e in events) + "\n"


def _system_event(session_id: str = _SESSION_ID) -> dict:
    return {"type": "system", "subtype": "init", "session_id": session_id}


def _assistant_event(text: str) -> dict:
    return {
        "type": "assistant",
        "message": {
            "content": [
                {"type": "text", "text": text},
            ],
        },
    }


def _result_event(
    *,
    input_tokens: int = 100,
    output_tokens: int = 50,
    cache_read_input_tokens: int = 0,
    cache_creation_input_tokens: int = 0,
    total_cost_usd: float | None = None,
) -> dict:
    event = {
        "type": "result",
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cache_read_input_tokens": cache_read_input_tokens,
            "cache_creation_input_tokens": cache_creation_input_tokens,
        },
    }
    if total_cost_usd is not None:
        event["total_cost_usd"] = total_cost_usd
    return event


def _standard_jsonl(action_json: str = '{"value": "hello"}') -> str:
    return _make_jsonl(
        _system_event(),
        _assistant_event(action_json),
        _result_event(),
    )


def _mock_run_side_effect(*_args, **_kwargs):
    cmd = _args[0] if _args else _kwargs.get("args", [])
    if cmd[:2] == ["docker", "run"]:
        return subprocess.CompletedProcess(cmd, 0, stdout=_FAKE_CONTAINER_ID + "\n")
    if cmd[:2] == ["docker", "exec"]:
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
    if cmd[:2] == ["docker", "rm"]:
        return subprocess.CompletedProcess(cmd, 0)
    return subprocess.CompletedProcess(cmd, 0)


def _make_system(**overrides):
    from src.systems.claude.system import ClaudeCodeSystem

    defaults = {"model": "test-model"}
    defaults.update(overrides)
    with patch("subprocess.run", side_effect=_mock_run_side_effect):
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test"}, clear=False):
            return ClaudeCodeSystem(**defaults)


def _respond_with_mock(sys, query, jsonl_stdout=None):
    if jsonl_stdout is None:
        jsonl_stdout = _standard_jsonl()

    def side_effect(cmd, **kwargs):
        if cmd[:2] == ["docker", "exec"]:
            return subprocess.CompletedProcess(cmd, 0, stdout=jsonl_stdout, stderr="")
        return _mock_run_side_effect(cmd, **kwargs)

    with patch("subprocess.run", side_effect=side_effect):
        return sys.respond(query)


class SingleConversationTests(unittest.TestCase):
    def test_build_cli_args_default_reasoning_effort_is_low(self):
        sys = _make_system()
        args = sys._build_cli_args("fresh")
        self.assertIn("--effort", args)
        self.assertEqual(args[args.index("--effort") + 1], "low")

    def test_build_cli_args_includes_configured_reasoning_effort(self):
        sys = _make_system(reasoning_effort="high")
        args = sys._build_cli_args("fresh")
        self.assertIn("--effort", args)
        self.assertEqual(args[args.index("--effort") + 1], "high")

    def test_build_cli_args_omits_reasoning_effort_when_disabled(self):
        sys = _make_system(reasoning_effort=None)
        args = sys._build_cli_args("fresh")
        self.assertNotIn("--effort", args)

    def test_invalid_reasoning_effort_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            _make_system(reasoning_effort="ultra")
        self.assertIn("reasoning_effort", str(ctx.exception))

    def test_rejects_resume_during_instance_knob(self):
        with self.assertRaises(TypeError):
            _make_system(allow_resume_during_instance=True)

    def test_removed_always_continue_knob_is_rejected(self):
        with self.assertRaises(TypeError):
            _make_system(always_continue=True)

    def test_single_conversation_uses_resume_session_id_after_prior_session(self):
        sys = _make_system(single_conversation=True)
        sys._conversation_id = "sess_xyz"
        args = sys._build_cli_args("resume-sid")
        self.assertNotIn("--continue", args)
        self.assertIn("--resume", args)
        self.assertEqual(args[args.index("--resume") + 1], "sess_xyz")

    def test_single_conversation_disabled_clears_only_on_instance_complete(self):
        sys = _make_system(single_conversation=False)
        sys._conversation_id = "sess_xyz"

        sys.observe(Observation(content="mid", instance_complete=False))
        self.assertEqual(sys._conversation_id, "sess_xyz")

        sys.observe(Observation(content="done", instance_complete=True))
        self.assertIsNone(sys._conversation_id)

    def test_reset_is_hard_boundary_when_single_conversation_enabled(self):
        sys = _make_system(single_conversation=True)
        sys._conversation_id = "sess_xyz"
        with patch("subprocess.run", side_effect=_mock_run_side_effect):
            sys.reset()
        self.assertIsNone(sys._conversation_id)

    def test_reset_deletes_claude_projects_memory_and_watched_files(self):
        sys = _make_system(single_conversation=False, other_memory_files=["notes/*.md"])
        sys._conversation_id = "sess_xyz"
        sys._claude_conversation_session_ids = {"sess_xyz"}
        memory_file = sys.memory_dir / "note.md"
        memory_file.write_text("keep memory")
        transcript_file = (
            Path(sys._tmp_dir)
            / ".claude"
            / "projects"
            / "-workspace"
            / "sess_xyz.jsonl"
        )
        transcript_file.write_text("{}\n")
        notes_dir = Path(sys._tmp_dir) / "notes"
        notes_dir.mkdir()
        watched_file = notes_dir / "topic.md"
        watched_file.write_text("delete me")
        skill_file = Path(sys._tmp_dir) / ".claude" / "skills" / "manual" / "SKILL.md"
        skill_file.parent.mkdir(parents=True)
        skill_file.write_text("keep skill")

        with patch("subprocess.run", side_effect=_mock_run_side_effect):
            sys.reset()

        self.assertIsNone(sys._conversation_id)
        self.assertEqual(sys._claude_conversation_session_ids, set())
        self.assertFalse(memory_file.exists())
        self.assertFalse(transcript_file.exists())
        self.assertFalse(watched_file.exists())
        self.assertEqual(skill_file.read_text(), "keep skill")

    def test_respond_resumes_within_instance_but_not_across_instances_when_disabled(
        self,
    ):
        sys = _make_system(single_conversation=False)
        same_instance = Query(
            prompt="x",
            response_schema=_SimpleAction,
            instance_id="inst-0",
            instance_index=0,
        )
        next_instance = Query(
            prompt="x",
            response_schema=_SimpleAction,
            instance_id="inst-1",
            instance_index=1,
        )
        outputs = [
            _make_jsonl(
                _system_event("session-one"),
                _assistant_event('{"value": "first"}'),
                _result_event(),
            ),
            _make_jsonl(
                _system_event("session-one"),
                _assistant_event('{"value": "second"}'),
                _result_event(),
            ),
            _make_jsonl(
                _system_event("session-two"),
                _assistant_event('{"value": "third"}'),
                _result_event(),
            ),
        ]
        claude_commands: list[str] = []

        def side_effect(cmd, **kwargs):
            if cmd[:2] == ["docker", "exec"]:
                command = cmd[-1]
                if command.startswith("claude "):
                    claude_commands.append(command)
                    return subprocess.CompletedProcess(
                        cmd,
                        0,
                        stdout=outputs.pop(0),
                        stderr="",
                    )
                return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
            return _mock_run_side_effect(cmd, **kwargs)

        with patch("subprocess.run", side_effect=side_effect):
            first = sys.respond(same_instance)
            second = sys.respond(same_instance)
            sys.observe(Observation(content="done", instance_complete=True))
            third = sys.respond(next_instance)

        self.assertEqual(
            first.metadata["continuity"]["mode"], "per_instance_conversation"
        )
        self.assertFalse(first.metadata["continuity"]["single_conversation"])
        self.assertEqual(first.metadata["continuity"]["invocation_mode"], "fresh")
        self.assertEqual(second.metadata["continuity"]["invocation_mode"], "resume-sid")
        self.assertEqual(third.metadata["continuity"]["invocation_mode"], "fresh")
        self.assertEqual(len(claude_commands), 3)
        self.assertNotIn("--continue", claude_commands[0])
        self.assertNotIn("--continue", claude_commands[1])
        self.assertIn("--resume session-one", claude_commands[1])
        self.assertNotIn("--continue", claude_commands[2])

    def test_respond_uses_single_conversation_across_instances(self):
        sys = _make_system()
        same_instance = Query(
            prompt="x",
            response_schema=_SimpleAction,
            instance_id="inst-0",
            instance_index=0,
        )
        next_instance = Query(
            prompt="x",
            response_schema=_SimpleAction,
            instance_id="inst-1",
            instance_index=1,
        )
        outputs = [
            _make_jsonl(
                _system_event("session-one"),
                _assistant_event('{"value": "first"}'),
                _result_event(input_tokens=100, output_tokens=50),
            ),
            _make_jsonl(
                _system_event("session-one"),
                _assistant_event('{"value": "second"}'),
                _result_event(input_tokens=140, output_tokens=75),
            ),
            _make_jsonl(
                _system_event("session-one"),
                _assistant_event('{"value": "third"}'),
                _result_event(input_tokens=180, output_tokens=90),
            ),
        ]
        claude_commands: list[str] = []

        def side_effect(cmd, **kwargs):
            if cmd[:2] == ["docker", "exec"]:
                command = cmd[-1]
                if command.startswith("claude "):
                    claude_commands.append(command)
                    return subprocess.CompletedProcess(
                        cmd,
                        0,
                        stdout=outputs.pop(0),
                        stderr="",
                    )
                return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
            return _mock_run_side_effect(cmd, **kwargs)

        with patch("subprocess.run", side_effect=side_effect):
            first = sys.respond(same_instance)
            second = sys.respond(same_instance)
            third = sys.respond(next_instance)

        self.assertEqual(first.metadata["continuity"]["mode"], "single_conversation")
        self.assertTrue(first.metadata["continuity"]["single_conversation"])
        self.assertFalse(first.metadata["continuity"]["resumed_this_turn"])
        self.assertTrue(second.metadata["continuity"]["resumed_this_turn"])
        self.assertTrue(third.metadata["continuity"]["resumed_this_turn"])
        self.assertEqual(len(claude_commands), 3)
        self.assertNotIn("--resume", claude_commands[0])
        self.assertIn("--resume session-one", claude_commands[1])
        self.assertIn("--resume session-one", claude_commands[2])

    def test_skill_file_installs_and_invokes_claude_command_tags(self):
        with tempfile.TemporaryDirectory() as tmp:
            skill_file = Path(tmp) / "learner.md"
            skill_file.write_text("Use memory well.")
            sys = _make_system(skill=str(skill_file))
        prompt = sys._build_prompt(
            Query(prompt="prompt", response_schema=_SimpleAction)
        )
        installed = Path(sys._tmp_dir) / ".claude" / "skills" / "learner" / "SKILL.md"
        self.assertTrue(
            prompt.startswith(
                "<command-message>learner</command-message>\n"
                "<command-name>/learner</command-name>\n"
                "<command-args>"
            )
        )
        self.assertIn("prompt", prompt)
        self.assertTrue(prompt.endswith("</command-args>"))
        self.assertEqual(installed.read_text(), "Use memory well.")

    def test_build_prompt_can_omit_skill_invocation(self):
        with tempfile.TemporaryDirectory() as tmp:
            skill_file = Path(tmp) / "learner.md"
            skill_file.write_text("Use memory well.")
            sys = _make_system(skill=str(skill_file))

        prompt = sys._build_prompt(
            Query(prompt="prompt", response_schema=_SimpleAction),
            include_skill=False,
        )

        self.assertIn("prompt", prompt)
        self.assertFalse(
            prompt.startswith("<command-message>learner</command-message>")
        )
        self.assertNotIn("<command-args>", prompt)

    def test_repair_prompt_does_not_invoke_skill(self):
        with tempfile.TemporaryDirectory() as tmp:
            skill_file = Path(tmp) / "learner.md"
            skill_file.write_text("Use memory well.")
            sys = _make_system(skill=str(skill_file))

        repair_prompt = sys._build_repair_prompt("original", "bad", "not json")

        self.assertFalse(repair_prompt.startswith("<command-message>learner"))
        self.assertNotIn("<command-args>", repair_prompt)
        self.assertIn("Original prompt:\noriginal", repair_prompt)

    def test_respond_invokes_skill_only_on_first_conversation_turn(self):
        with tempfile.TemporaryDirectory() as tmp:
            skill_file = Path(tmp) / "learner.md"
            skill_file.write_text("Use memory well.")
            sys = _make_system(skill=str(skill_file))

        commands: list[str] = []
        outputs = [
            _make_jsonl(
                _system_event("session-one"),
                _assistant_event('{"value": "first"}'),
                _result_event(input_tokens=100, output_tokens=50),
            ),
            _make_jsonl(
                _system_event("session-one"),
                _assistant_event('{"value": "second"}'),
                _result_event(input_tokens=140, output_tokens=75),
            ),
        ]

        def side_effect(cmd, **kwargs):
            if cmd[:2] == ["docker", "exec"]:
                command = cmd[-1]
                if command.startswith("claude "):
                    commands.append(command)
                    return subprocess.CompletedProcess(
                        cmd,
                        0,
                        stdout=outputs.pop(0),
                        stderr="",
                    )
                return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
            return _mock_run_side_effect(cmd, **kwargs)

        with patch("subprocess.run", side_effect=side_effect):
            sys.respond(Query(prompt="x", response_schema=_SimpleAction))
            sys.respond(Query(prompt="x", response_schema=_SimpleAction))

        self.assertIn("<command-message>learner</command-message>", commands[0])
        self.assertNotIn("<command-message>learner</command-message>", commands[1])

    def test_repair_turn_does_not_repeat_skill_invocation(self):
        with tempfile.TemporaryDirectory() as tmp:
            skill_file = Path(tmp) / "learner.md"
            skill_file.write_text("Use memory well.")
            sys = _make_system(skill=str(skill_file))

        commands: list[str] = []
        outputs = [
            _make_jsonl(
                _system_event("session-one"),
                _assistant_event('{"description": "not schema"}'),
                _result_event(input_tokens=100, output_tokens=50),
            ),
            _make_jsonl(
                _system_event("session-one"),
                _assistant_event('{"value": "fixed"}'),
                _result_event(input_tokens=140, output_tokens=75),
            ),
        ]

        def side_effect(cmd, **kwargs):
            if cmd[:2] == ["docker", "exec"]:
                command = cmd[-1]
                if command.startswith("claude "):
                    commands.append(command)
                    return subprocess.CompletedProcess(
                        cmd,
                        0,
                        stdout=outputs.pop(0),
                        stderr="",
                    )
                return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
            return _mock_run_side_effect(cmd, **kwargs)

        with patch("subprocess.run", side_effect=side_effect):
            response = sys.respond(Query(prompt="x", response_schema=_SimpleAction))

        self.assertEqual(response.action.value, "fixed")
        self.assertIn("<command-message>learner</command-message>", commands[0])
        self.assertNotIn("<command-message>learner</command-message>", commands[1])


class UsageTests(unittest.TestCase):
    def test_extract_token_usage_sums_result_tokens_and_cache_buckets(self):
        from src.systems.claude.system import _extract_token_usage

        events = [
            _result_event(
                input_tokens=100,
                output_tokens=50,
                cache_read_input_tokens=0,
                cache_creation_input_tokens=5,
            ),
            _result_event(
                input_tokens=140,
                output_tokens=75,
                cache_read_input_tokens=90,
                cache_creation_input_tokens=7,
            ),
        ]

        usage = _extract_token_usage(events)

        self.assertEqual(usage["input_tokens"], 342)
        self.assertEqual(usage["output_tokens"], 125)
        self.assertEqual(usage["cached_input_tokens"], 90)
        self.assertEqual(usage["cache_creation_input_tokens"], 12)

    def test_extract_cli_cost_usd_sums_result_costs(self):
        from src.systems.claude.system import _extract_cli_cost_usd

        events = [
            _result_event(total_cost_usd=0.12),
            {"type": "assistant"},
            _result_event(total_cost_usd=0.03),
        ]

        self.assertAlmostEqual(_extract_cli_cost_usd(events), 0.15)

    def test_extract_cli_cost_usd_requires_all_results_to_report_cost(self):
        from src.systems.claude.system import _extract_cli_cost_usd

        events = [
            _result_event(total_cost_usd=0.12),
            _result_event(),
        ]

        self.assertIsNone(_extract_cli_cost_usd(events))

    def test_respond_records_per_call_output_usage_when_continuing(self):
        sys = _make_system(single_conversation=True)
        query = Query(prompt="x", response_schema=_SimpleAction)
        first_jsonl = _make_jsonl(
            _system_event(),
            _assistant_event('{"value": "first"}'),
            _result_event(input_tokens=100, output_tokens=50),
        )
        second_jsonl = _make_jsonl(
            _system_event(),
            _assistant_event('{"value": "second"}'),
            _result_event(input_tokens=140, output_tokens=75),
        )

        _respond_with_mock(sys, query, jsonl_stdout=first_jsonl)
        first_event = sys.consume_usage_events()[0]
        _respond_with_mock(sys, query, jsonl_stdout=second_jsonl)
        second_event = sys.consume_usage_events()[0]

        self.assertEqual(first_event.output_tokens, 50)
        self.assertEqual(second_event.output_tokens, 75)
        self.assertEqual(sys._cumulative_tokens["output_tokens"], 125)

    def test_respond_records_cached_usage_event_fields(self):
        sys = _make_system(single_conversation=True)
        jsonl = _make_jsonl(
            _system_event(),
            _assistant_event('{"value": "cached"}'),
            _result_event(
                input_tokens=100,
                output_tokens=50,
                cache_read_input_tokens=80,
                cache_creation_input_tokens=20,
            ),
        )

        _respond_with_mock(
            sys,
            Query(prompt="x", response_schema=_SimpleAction),
            jsonl_stdout=jsonl,
        )

        event = sys.consume_usage_events()[0]
        self.assertEqual(event.input_tokens, 200)
        self.assertEqual(event.output_tokens, 50)
        self.assertEqual(event.cached_input_tokens, 80)
        self.assertEqual(event.cache_creation_input_tokens, 20)

    def test_respond_prefers_claude_cli_cost_when_reported(self):
        sys = _make_system(single_conversation=True)
        jsonl = _make_jsonl(
            _system_event(),
            _assistant_event('{"value": "priced"}'),
            _result_event(total_cost_usd=0.123456),
        )

        _respond_with_mock(
            sys,
            Query(prompt="x", response_schema=_SimpleAction),
            jsonl_stdout=jsonl,
        )

        event = sys.consume_usage_events()[0]
        self.assertEqual(event.cost_usd, 0.123456)
        self.assertEqual(event.pricing_source, "claude_cli.total_cost_usd")

    def test_respond_sums_fresh_repair_output_tokens(self):
        sys = _make_system(single_conversation=False)
        first = _make_jsonl(
            _system_event("session-one"),
            _assistant_event('{"description": "not schema"}'),
            _result_event(input_tokens=100, output_tokens=50),
        )
        repaired = _make_jsonl(
            _system_event("session-two"),
            _assistant_event('{"value": "fixed"}'),
            _result_event(input_tokens=140, output_tokens=75),
        )
        outputs = [first, repaired]

        def side_effect(cmd, **kwargs):
            if cmd[:2] == ["docker", "exec"]:
                return subprocess.CompletedProcess(
                    cmd,
                    0,
                    stdout=outputs.pop(0),
                    stderr="",
                )
            return _mock_run_side_effect(cmd, **kwargs)

        with patch("subprocess.run", side_effect=side_effect):
            sys.respond(Query(prompt="x", response_schema=_SimpleAction))

        event = sys.consume_usage_events()[0]
        self.assertEqual(event.output_tokens, 125)
        self.assertEqual(sys._cumulative_tokens["output_tokens"], 125)

    def test_respond_retries_empty_claude_output(self):
        sys = _make_system(single_conversation=True)
        first = _make_jsonl(
            _system_event("session-one"),
            _result_event(input_tokens=100, output_tokens=0),
        )
        retried = _make_jsonl(
            _system_event("session-two"),
            _assistant_event('{"value": "retried"}'),
            _result_event(input_tokens=140, output_tokens=75),
        )
        outputs = [first, retried]

        def side_effect(cmd, **kwargs):
            if cmd[:2] == ["docker", "exec"]:
                return subprocess.CompletedProcess(
                    cmd,
                    0,
                    stdout=outputs.pop(0),
                    stderr="",
                )
            return _mock_run_side_effect(cmd, **kwargs)

        with patch("subprocess.run", side_effect=side_effect):
            response = sys.respond(Query(prompt="x", response_schema=_SimpleAction))

        self.assertEqual(response.action.value, "retried")
        event = sys.consume_usage_events()[0]
        self.assertEqual(event.input_tokens, 240)
        self.assertEqual(event.output_tokens, 75)

    def test_respond_wraps_repeated_empty_claude_output_as_llm_failure(self):
        sys = _make_system(single_conversation=True)
        empty = _make_jsonl(
            _system_event(),
            _result_event(input_tokens=100, output_tokens=0),
        )
        outputs = [empty, empty]

        def side_effect(cmd, **kwargs):
            if cmd[:2] == ["docker", "exec"]:
                return subprocess.CompletedProcess(
                    cmd,
                    0,
                    stdout=outputs.pop(0),
                    stderr="",
                )
            return _mock_run_side_effect(cmd, **kwargs)

        with patch("subprocess.run", side_effect=side_effect):
            with self.assertRaises(RuntimeError) as ctx:
                sys.respond(Query(prompt="x", response_schema=_SimpleAction))

        self.assertIn("LLM call failed:", str(ctx.exception))
        self.assertIn("no assistant message", str(ctx.exception))

    def test_respond_wraps_claude_timeout_as_llm_failure(self):
        sys = _make_system(single_conversation=True)

        def side_effect(cmd, **kwargs):
            if cmd[:2] == ["docker", "exec"]:
                raise subprocess.TimeoutExpired(cmd=cmd, timeout=300)
            return _mock_run_side_effect(cmd, **kwargs)

        with patch("subprocess.run", side_effect=side_effect):
            with self.assertRaises(RuntimeError) as ctx:
                sys.respond(Query(prompt="x", response_schema=_SimpleAction))

        self.assertIn("LLM call failed:", str(ctx.exception))
        self.assertIn("timed out after 300 seconds", str(ctx.exception))


class ArtifactTests(unittest.TestCase):
    def test_respond_metadata_reports_unified_memory_snapshot(self):
        sys = _make_system()
        (sys.memory_dir / "topic.md").write_text("learned note")

        resp = _respond_with_mock(
            sys,
            Query(prompt="x", response_schema=_SimpleAction),
        )

        self.assertEqual(resp.metadata["memory_files"], {"topic.md": "learned note"})
        self.assertNotIn("memory_backend", resp.metadata)
        self.assertNotIn("memory_dir", resp.metadata)
        self.assertNotIn("native_memory_files", resp.metadata)
        self.assertNotIn("other_memory_files", resp.metadata)
        self.assertNotIn("memory_file_sources", resp.metadata)
        self.assertNotIn("other_memory_file_mapping", resp.metadata)
        self.assertNotIn("other_memory_file_patterns", resp.metadata)
        self.assertNotIn("memory_units", resp.metadata)
        self.assertEqual(resp.metadata["continuity"]["mode"], "single_conversation")
        self.assertTrue(resp.metadata["continuity"]["single_conversation"])
        self.assertNotIn("always_continue", resp.metadata["continuity"])
        self.assertNotIn(
            "allow_resume_during_instance",
            resp.metadata["continuity"],
        )

    def test_respond_metadata_collects_other_memory_files(self):
        sys = _make_system(other_memory_files=["scratchpad/*.md"])
        (sys.memory_dir / "MEMORY.md").write_text("claude notes")
        scratchpad = Path(sys._tmp_dir) / "scratchpad"
        scratchpad.mkdir()
        (scratchpad / "continual-learn.md").write_text("notes")

        resp = _respond_with_mock(
            sys,
            Query(prompt="x", response_schema=_SimpleAction),
        )

        self.assertEqual(
            resp.metadata["memory_files"],
            {
                "MEMORY.md": "claude notes",
                "scratchpad/continual-learn.md": "notes",
            },
        )
        self.assertNotIn("native_memory_files", resp.metadata)
        self.assertNotIn("other_memory_files", resp.metadata)
        self.assertNotIn("memory_file_sources", resp.metadata)
        self.assertNotIn("other_memory_file_patterns", resp.metadata)
        self.assertNotIn("memory_units", resp.metadata)

    def test_respond_metadata_does_not_hide_claude_memory_on_other_file_collision(self):
        sys = _make_system(other_memory_files=["MEMORY.md"])
        (sys.memory_dir / "MEMORY.md").write_text("claude notes")
        (Path(sys._tmp_dir) / "MEMORY.md").write_text("workspace notes")

        resp = _respond_with_mock(
            sys,
            Query(prompt="x", response_schema=_SimpleAction),
        )

        self.assertEqual(resp.metadata["memory_files"]["MEMORY.md"], "claude notes")
        self.assertEqual(
            resp.metadata["memory_files"]["other_memory/MEMORY.md"],
            "workspace notes",
        )
        self.assertNotIn("other_memory_file_mapping", resp.metadata)
        self.assertNotIn("memory_file_sources", resp.metadata)

    def test_respond_metadata_collects_derived_project_memory_directory(self):
        sys = _make_system()
        derived_memory_dir = (
            Path(sys._tmp_dir)
            / ".claude"
            / "projects"
            / "-workspace-derived"
            / "memory"
        )

        def side_effect(cmd, **kwargs):
            if cmd[:2] == ["docker", "exec"]:
                derived_memory_dir.mkdir(parents=True, exist_ok=True)
                (derived_memory_dir / "topic.md").write_text("derived note")
                return subprocess.CompletedProcess(
                    cmd,
                    0,
                    stdout=_standard_jsonl(),
                    stderr="",
                )
            return _mock_run_side_effect(cmd, **kwargs)

        with patch("subprocess.run", side_effect=side_effect):
            resp = sys.respond(Query(prompt="x", response_schema=_SimpleAction))

        self.assertEqual(resp.metadata["memory_files"], {"topic.md": "derived note"})
        self.assertNotIn("memory_units", resp.metadata)

    def test_get_run_artifacts_reports_unified_memory_files(self):
        sys = _make_system()
        (sys.memory_dir / "topic.md").write_text("note body")
        _respond_with_mock(sys, Query(prompt="x", response_schema=_SimpleAction))

        artifacts = sys.get_run_artifacts()

        self.assertEqual(artifacts["artifact_type"], "claude")
        self.assertEqual(artifacts["conversation_id"], _SESSION_ID)
        self.assertEqual(artifacts["interaction_count"], 1)
        self.assertEqual(artifacts["memory_files"]["topic.md"], "note body")
        self.assertNotIn("native_memory_files", artifacts)
        self.assertNotIn("other_memory_files", artifacts)
        self.assertNotIn("memory_file_sources", artifacts)
        self.assertNotIn("other_memory_file_mapping", artifacts)
        self.assertNotIn("other_memory_file_patterns", artifacts)
        self.assertNotIn("memory_dir", artifacts)
        self.assertNotIn("memory_backend", artifacts)

    def test_get_run_artifacts_collects_only_conversation_jsonl_files(self):
        sys = _make_system()
        project_dir = Path(sys._tmp_dir) / ".claude" / "projects" / "-workspace"
        project_dir.mkdir(parents=True, exist_ok=True)
        conversation_path = project_dir / f"{_SESSION_ID}.jsonl"
        conversation_path.write_text('{"type":"conversation"}\n')
        (project_dir / "subagent-session.jsonl").write_text("subagent\n")

        resp = _respond_with_mock(
            sys,
            Query(
                prompt="x",
                response_schema=_SimpleAction,
                instance_id="i0",
                instance_index=0,
            ),
        )
        artifacts = sys.get_run_artifacts()

        rel = f".claude/projects/-workspace/{_SESSION_ID}.jsonl"
        self.assertEqual(
            artifacts["claude_conversation_jsonl_files"],
            {rel: '{"type":"conversation"}\n'},
        )
        self.assertEqual(resp.metadata["claude_conversation_jsonl_files"], [rel])
        self.assertEqual(
            artifacts["claude_instance_conversations"][0]["conversation_jsonl_files"],
            {rel: '{"type":"conversation"}\n'},
        )
        self.assertNotIn(
            ".claude/projects/-workspace/subagent-session.jsonl",
            artifacts["claude_conversation_jsonl_files"],
        )

    def test_get_run_artifacts_includes_other_memory_files_as_memory(self):
        sys = _make_system(other_memory_files=["scratchpad/*.md"])
        (sys.memory_dir / "MEMORY.md").write_text("claude notes")
        scratchpad = Path(sys._tmp_dir) / "scratchpad"
        scratchpad.mkdir()
        (scratchpad / "continual-learn.md").write_text("notes")
        _respond_with_mock(sys, Query(prompt="x", response_schema=_SimpleAction))

        artifacts = sys.get_run_artifacts()

        self.assertEqual(
            artifacts["memory_files"],
            {
                "MEMORY.md": "claude notes",
                "scratchpad/continual-learn.md": "notes",
            },
        )
        self.assertNotIn("native_memory_files", artifacts)
        self.assertNotIn("other_memory_files", artifacts)
        self.assertNotIn("memory_file_sources", artifacts)
        self.assertNotIn("other_memory_file_patterns", artifacts)

    def test_save_artifacts_uses_registered_claude_exporter(self):
        import src.systems.claude  # noqa: F401

        artifacts = {
            "artifact_type": "claude",
            "conversation_id": "sess_123",
            "version": "2.1.119",
            "interaction_count": 1,
            "cumulative_tokens": {"input_tokens": 10},
            "jsonl_events": [{"type": "result"}],
            "memory_history": [{"step": 1, "files": {"topic.md": "note"}}],
            "memory_files": {"topic.md": "topic body"},
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            trace_path = Path(tmpdir) / "trace.json"
            trace_path.write_text("{}", encoding="utf-8")

            artifact_dir = save_artifacts(artifacts, trace_path)

            self.assertIsNotNone(artifact_dir)
            self.assertTrue((artifact_dir / "events.jsonl").exists())
            self.assertEqual(
                (artifact_dir / "memory" / "topic.md").read_text(),
                "topic body",
            )
            manifest = json.loads((artifact_dir / "manifest.json").read_text())
            self.assertEqual(manifest["artifact_type"], "claude")
            self.assertEqual(manifest["conversation_id"], "sess_123")
            self.assertEqual(manifest["memory_artifact_path"], "memory")
            self.assertEqual(manifest["memory_files"], ["topic.md"])
            self.assertNotIn("memory_dir", manifest)
            self.assertNotIn("memory_backend", manifest)
            self.assertNotIn("memory_sources_path", manifest)
            self.assertNotIn("native_memory_files", manifest)
            self.assertNotIn("other_memory_files", manifest)
            self.assertNotIn("memory_file_sources", manifest)
            self.assertFalse((artifact_dir / "memory_sources.json").exists())

    def test_save_artifacts_writes_claude_conversation_jsonl_per_run_and_instance(self):
        import src.systems.claude  # noqa: F401

        rel = ".claude/projects/-workspace/sess_123.jsonl"
        artifacts = {
            "artifact_type": "claude",
            "conversation_id": "sess_123",
            "claude_conversation_session_ids": ["sess_123"],
            "claude_conversation_jsonl_files": {rel: "main transcript\n"},
            "claude_conversation_history": [
                {
                    "step": 1,
                    "instance_id": "inst-0",
                    "instance_index": 0,
                    "conversation_jsonl_files": [rel],
                }
            ],
            "claude_instance_conversations": [
                {
                    "step": 1,
                    "instance_id": "inst-0",
                    "instance_index": 0,
                    "conversation_id": "sess_123",
                    "session_ids": ["sess_123"],
                    "conversation_jsonl_files": {rel: "instance transcript\n"},
                }
            ],
            "jsonl_events": [],
            "memory_history": [],
            "memory_files": {},
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            trace_path = Path(tmpdir) / "trace.json"
            trace_path.write_text("{}", encoding="utf-8")

            artifact_dir = save_artifacts(artifacts, trace_path)

            self.assertIsNotNone(artifact_dir)
            self.assertEqual(
                (
                    artifact_dir
                    / "claude_conversations"
                    / "final"
                    / ".claude"
                    / "projects"
                    / "-workspace"
                    / "sess_123.jsonl"
                ).read_text(),
                "main transcript\n",
            )
            self.assertEqual(
                (
                    artifact_dir
                    / "instances"
                    / "instance_0000"
                    / "claude_conversations"
                    / ".claude"
                    / "projects"
                    / "-workspace"
                    / "sess_123.jsonl"
                ).read_text(),
                "instance transcript\n",
            )
            manifest = json.loads((artifact_dir / "manifest.json").read_text())
            self.assertEqual(
                manifest["claude_conversation_jsonl_files"],
                [rel],
            )
            self.assertEqual(
                manifest["claude_instance_conversations"][0]["instance_index"],
                0,
            )

    def test_save_artifacts_writes_claude_other_memory_files_in_merged_memory_dir(self):
        import src.systems.claude  # noqa: F401

        artifacts = {
            "artifact_type": "claude",
            "conversation_id": "sess_123",
            "version": "2.1.119",
            "interaction_count": 1,
            "cumulative_tokens": {},
            "jsonl_events": [],
            "memory_history": [],
            "memory_files": {
                "MEMORY.md": "claude notes",
                "MENTAL_MODEL.md": "workspace notes",
            },
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            trace_path = Path(tmpdir) / "trace.json"
            trace_path.write_text("{}", encoding="utf-8")

            artifact_dir = save_artifacts(artifacts, trace_path)

            self.assertIsNotNone(artifact_dir)
            self.assertEqual(
                (artifact_dir / "memory" / "MEMORY.md").read_text(), "claude notes"
            )
            self.assertEqual(
                (artifact_dir / "memory" / "MENTAL_MODEL.md").read_text(),
                "workspace notes",
            )
            manifest = json.loads((artifact_dir / "manifest.json").read_text())
            self.assertEqual(manifest["memory_files"], ["MEMORY.md", "MENTAL_MODEL.md"])
            self.assertNotIn("native_memory_files", manifest)
            self.assertNotIn("other_memory_files", manifest)
            self.assertNotIn("memory_file_sources", manifest)

    def test_save_artifacts_writes_claude_baseline_instances(self):
        import src.systems.claude  # noqa: F401

        rel = ".claude/projects/-workspace/sess_baseline.jsonl"
        artifacts = {
            "artifact_type": "claude",
            "phase": "baseline",
            "baseline_instances": [
                {
                    "instance_index": 2,
                    "artifact_manifest": {"jsonl_event_count": 1},
                    "system_artifacts": {
                        "artifact_type": "claude",
                        "conversation_id": "sess_baseline",
                        "claude_conversation_session_ids": ["sess_baseline"],
                        "claude_conversation_jsonl_files": {
                            rel: "baseline transcript\n"
                        },
                        "interaction_count": 1,
                        "jsonl_events": [{"type": "result"}],
                        "memory_history": [],
                        "memory_files": {"note.md": "baseline note"},
                    },
                }
            ],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            trace_path = Path(tmpdir) / "baseline.json"
            trace_path.write_text("{}", encoding="utf-8")

            artifact_dir = save_artifacts(artifacts, trace_path)

            self.assertIsNotNone(artifact_dir)
            instance_dir = artifact_dir / "instances" / "instance_0002"
            self.assertTrue((instance_dir / "events.jsonl").exists())
            self.assertEqual(
                (instance_dir / "memory" / "note.md").read_text(),
                "baseline note",
            )
            self.assertEqual(
                (
                    instance_dir
                    / "claude_conversations"
                    / "final"
                    / ".claude"
                    / "projects"
                    / "-workspace"
                    / "sess_baseline.jsonl"
                ).read_text(),
                "baseline transcript\n",
            )
            manifest = json.loads((artifact_dir / "manifest.json").read_text())
            self.assertEqual(manifest["phase"], "baseline")
            self.assertEqual(manifest["baseline_instance_count"], 1)
            self.assertEqual(manifest["claude_conversation_jsonl_file_count"], 1)
            self.assertEqual(manifest["instances"][0]["instance_index"], 2)


if __name__ == "__main__":
    unittest.main()
