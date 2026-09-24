"""Tests for the Codex CLI system (Docker-based)."""

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pydantic import BaseModel

from src.artifacts import save_artifacts
from src.interface import Observation, Query, Response


# ---------------------------------------------------------------------------
# Test schema
# ---------------------------------------------------------------------------


class _SimpleAction(BaseModel):
    value: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CONV_ID = "conv_abc123"
_FAKE_CONTAINER_ID = "abc123def456"


def _make_jsonl(*events: dict) -> str:
    return "\n".join(json.dumps(e) for e in events) + "\n"


def _session_meta_event(conv_id: str = _CONV_ID) -> dict:
    return {"type": "session_meta", "payload": {"id": conv_id}}


def _item_completed_event(text: str) -> dict:
    return {
        "type": "item.completed",
        "item": {"type": "agent_message", "text": text},
    }


def _turn_completed_event(
    input_tokens: int = 100,
    output_tokens: int = 50,
    reasoning_tokens: int = 10,
    cached_tokens: int = 0,
) -> dict:
    return {
        "type": "turn.completed",
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "reasoning_tokens": reasoning_tokens,
            "cached_input_tokens": cached_tokens,
        },
    }


def _standard_jsonl(action_json: str = '{"value": "hello"}') -> str:
    return _make_jsonl(
        _session_meta_event(),
        _item_completed_event(action_json),
        _turn_completed_event(),
    )


def _write_session_jsonl(
    workspace: str | Path,
    *,
    conv_id: str = _CONV_ID,
    rel: str | None = None,
    contents: str | None = None,
) -> str:
    if rel is None:
        rel = f".codex/sessions/2025/01/03/rollout-2025-01-03T12-00-00-{conv_id}.jsonl"
    if contents is None:
        contents = _make_jsonl(
            {
                "timestamp": "2025-01-03T12:00:00Z",
                "type": "session_meta",
                "payload": {"id": conv_id},
            },
            {"type": "event_msg", "payload": {"type": "user_message"}},
        )
    path = Path(workspace) / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents, encoding="utf-8")
    return rel


def _mock_run_side_effect(*_args, **_kwargs):
    """Default side effect for subprocess.run that handles docker run/exec/rm."""
    cmd = _args[0] if _args else _kwargs.get("args", [])
    if cmd[:2] == ["docker", "run"]:
        return subprocess.CompletedProcess(cmd, 0, stdout=_FAKE_CONTAINER_ID + "\n")
    if cmd[:2] == ["docker", "exec"]:
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
    if cmd[:2] == ["docker", "rm"]:
        return subprocess.CompletedProcess(cmd, 0)
    return subprocess.CompletedProcess(cmd, 0)


def _make_system(**overrides):
    """Build a CodexSystem with mocked Docker subprocess calls."""
    from src.systems.codex.system import CodexSystem

    defaults = {"model": "test-model"}
    defaults.update(overrides)
    with patch("subprocess.run", side_effect=_mock_run_side_effect):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=False):
            return CodexSystem(**defaults)


def _respond_with_mock(sys, query, jsonl_stdout=None):
    """Call sys.respond() with a mocked docker exec returning canned JSONL."""
    if jsonl_stdout is None:
        jsonl_stdout = _standard_jsonl()

    def side_effect(cmd, **kwargs):
        if cmd[:2] == ["docker", "exec"]:
            return subprocess.CompletedProcess(cmd, 0, stdout=jsonl_stdout, stderr="")
        return _mock_run_side_effect(cmd, **kwargs)

    with patch("subprocess.run", side_effect=side_effect):
        return sys.respond(query)


# ---------------------------------------------------------------------------
# Init / environment tests
# ---------------------------------------------------------------------------


class InitTests(unittest.TestCase):
    def test_creates_host_workspace(self):
        sys = _make_system()
        self.assertTrue(Path(sys._tmp_dir).is_dir())

    def test_does_not_create_native_memory_dir(self):
        sys = _make_system()
        self.assertFalse((Path(sys._tmp_dir) / "memory").exists())
        self.assertFalse((Path(sys._tmp_dir) / ".codex" / "memories").exists())

    def test_removed_memory_options_are_rejected(self):
        removed_options = [
            {"memory_dir_name": None},
            {"seed_memory_dir": "/tmp/seed"},
            {"memory_instruction": "Use memory."},
        ]
        for option in removed_options:
            with self.subTest(option=option):
                with self.assertRaises(TypeError):
                    _make_system(**option)

    def test_container_started(self):
        sys = _make_system()
        self.assertEqual(sys._container_id, _FAKE_CONTAINER_ID)

    def test_copies_oauth_auth_json(self):
        with tempfile.TemporaryDirectory() as fake_home:
            codex_dir = Path(fake_home) / ".codex"
            codex_dir.mkdir()
            auth_file = codex_dir / "auth.json"
            auth_file.write_text('{"token": "test"}')

            with patch(
                "src.systems.codex.system.Path.home", return_value=Path(fake_home)
            ):
                sys = _make_system(use_oauth=True)

            copied = Path(sys._tmp_dir) / "auth.json"
            self.assertTrue(copied.exists())
            self.assertEqual(json.loads(copied.read_text()), {"token": "test"})

    def test_installs_skill_file_in_home(self):
        with tempfile.TemporaryDirectory() as tmp:
            skill_file = Path(tmp) / "edge-cases.md"
            skill_file.write_text("skill instructions")
            sys = _make_system(skill=str(skill_file))

        installed = Path(sys._tmp_dir) / ".codex" / "skills" / "edge-cases" / "SKILL.md"
        self.assertEqual(installed.read_text(), "skill instructions")

    def test_missing_skill_file_raises(self):
        with self.assertRaises(FileNotFoundError) as ctx:
            _make_system(skill="/missing/nope.md")
        self.assertIn("skill does not exist", str(ctx.exception))

    def test_requires_api_key_without_oauth(self):
        from src.systems.codex.system import CodexSystem

        with patch("subprocess.run", side_effect=_mock_run_side_effect):
            with patch.dict(os.environ, {}, clear=True):
                with self.assertRaises(ValueError):
                    CodexSystem(model="test-model")


# ---------------------------------------------------------------------------
# Command building tests
# ---------------------------------------------------------------------------


class CommandTests(unittest.TestCase):
    def test_build_command_first_call(self):
        sys = _make_system()
        cmd = sys._build_command("do stuff")
        self.assertNotIn("--resume", cmd)
        self.assertNotIn("--ephemeral", cmd)
        self.assertIn("codex --enable memories exec", cmd)
        self.assertIn("memories.use_memories=true", cmd)
        self.assertIn("memories.generate_memories=true", cmd)
        self.assertIn("--json", cmd)
        self.assertIn("--dangerously-bypass-approvals-and-sandbox", cmd)
        self.assertIn("--skip-git-repo-check", cmd)

    def test_build_command_can_disable_native_memories_flag_only(self):
        sys = _make_system(disable_native_memories=True)
        cmd = sys._build_command("do stuff")
        self.assertIn("codex exec", cmd)
        self.assertNotIn("--enable memories", cmd)
        self.assertNotIn("memories.use_memories", cmd)
        self.assertNotIn("memories.generate_memories", cmd)

    def test_build_command_resumes_when_conversation_exists(self):
        sys = _make_system()
        sys._conversation_id = "conv_xyz"
        cmd = sys._build_command("do stuff")
        self.assertIn("codex --enable memories exec resume", cmd)
        self.assertNotIn("--resume", cmd)
        self.assertIn("conv_xyz", cmd)

    def test_removed_resume_knobs_are_rejected(self):
        removed_options = [
            {"always_continue": True},
            {"allow_resume_during_instance": False},
        ]
        for option in removed_options:
            with self.subTest(option=option):
                with self.assertRaises(TypeError):
                    _make_system(**option)

    def test_invalid_single_conversation_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            _make_system(single_conversation="yes")
        self.assertIn("single_conversation", str(ctx.exception))

    def test_invalid_disable_native_memories_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            _make_system(disable_native_memories="yes")
        self.assertIn("disable_native_memories", str(ctx.exception))

    def test_build_command_passes_none_reasoning_effort_by_default(self):
        sys = _make_system()
        cmd = sys._build_command("do stuff")
        self.assertIn('-c model_reasoning_effort="none"', cmd)

    def test_build_command_omits_reasoning_effort_when_null(self):
        sys = _make_system(reasoning_effort=None)
        cmd = sys._build_command("do stuff")
        self.assertNotIn("model_reasoning_effort", cmd)

    def test_build_command_includes_reasoning_effort_when_set(self):
        sys = _make_system(reasoning_effort="high")
        cmd = sys._build_command("do stuff")
        self.assertIn('-c model_reasoning_effort="high"', cmd)

    def test_build_command_includes_auto_compact_token_limit_when_set(self):
        sys = _make_system(model_auto_compact_token_limit=64000)
        cmd = sys._build_command("do stuff")
        self.assertIn("-c model_auto_compact_token_limit=64000", cmd)

    def test_invalid_reasoning_effort_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            _make_system(reasoning_effort="ultra")
        self.assertIn("reasoning_effort", str(ctx.exception))

    def test_invalid_auto_compact_token_limit_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            _make_system(model_auto_compact_token_limit=0)
        self.assertIn("model_auto_compact_token_limit", str(ctx.exception))

    def test_reset_hard_clears_conversation_sessions_and_memory(self):
        sys = _make_system()
        sys._conversation_id = "conv_xyz"
        sys._last_cumulative_token_usage_by_conversation = {"conv_xyz": {}}
        session_file = Path(sys._tmp_dir) / ".codex" / "sessions" / "conv_xyz.jsonl"
        session_file.parent.mkdir(parents=True)
        session_file.write_text("{}")
        memory_file = Path(sys._tmp_dir) / ".codex" / "memories" / "topic.md"
        memory_file.parent.mkdir(parents=True)
        memory_file.write_text("learned")

        with patch("subprocess.run", side_effect=_mock_run_side_effect):
            sys.reset()

        self.assertIsNone(sys._conversation_id)
        self.assertEqual(sys._last_cumulative_token_usage_by_conversation, {})
        self.assertFalse(session_file.exists())
        self.assertFalse(memory_file.exists())


# ---------------------------------------------------------------------------
# Prompt building tests
# ---------------------------------------------------------------------------


class PromptTests(unittest.TestCase):
    def _query(self):
        return Query(
            prompt="What is 2+2?",
            response_schema=_SimpleAction,
        )

    def test_build_prompt_without_memory(self):
        sys = _make_system()
        prompt = sys._build_prompt(self._query())
        self.assertIn("What is 2+2?", prompt)
        self.assertNotIn("MEMORY.md", prompt)
        self.assertNotIn("/workspace/memory", prompt)
        self.assertNotIn("/workspace/.codex/memories", prompt)

    def test_build_prompt_includes_preamble_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            preamble_file = Path(tmp) / "preamble.md"
            preamble_file.write_text("Follow these rules")
            sys = _make_system(preamble_file=str(preamble_file))
        prompt = sys._build_prompt(self._query())
        self.assertIn("PRE-AMBLE:\nFollow these rules", prompt)

    def test_build_prompt_prepends_skill_command(self):
        with tempfile.TemporaryDirectory() as tmp:
            skill_file = Path(tmp) / "edge-cases.md"
            skill_file.write_text("skill instructions")
            sys = _make_system(skill=str(skill_file))
        prompt = sys._build_prompt(self._query())
        self.assertTrue(prompt.startswith("$edge-cases\n"))

    def test_build_prompt_can_prepend_skill_contents(self):
        with tempfile.TemporaryDirectory() as tmp:
            skill_file = Path(tmp) / "edge-cases.md"
            skill_file.write_text(
                "---\nname: edge-cases\ndescription: Test skill.\n---\n\n# Edge Cases\nUse them.\n"
            )
            sys = _make_system(skill=str(skill_file), invocation_type="prepend")
        prompt = sys._build_prompt(self._query())
        self.assertTrue(prompt.startswith("# Edge Cases\nUse them.\n\n"))
        self.assertFalse((Path(sys._tmp_dir) / ".codex" / "skills").exists())

    def test_build_prompt_does_not_inline_watched_memory_contents(self):
        sys = _make_system(other_memory_files=["scratchpad/*.md"])
        scratchpad = Path(sys._tmp_dir) / "scratchpad"
        scratchpad.mkdir()
        (scratchpad / "topic.md").write_text("should not appear")
        prompt = sys._build_prompt(self._query())
        self.assertNotIn("should not appear", prompt)
        self.assertNotIn("/workspace/.codex/memories", prompt)

    def test_build_prompt_omits_default_native_memory_nudge(self):
        sys = _make_system()
        sys._interaction_count = 1
        prompt = sys._build_prompt(self._query())
        self.assertNotIn("Codex memories", prompt)
        self.assertNotIn("durable task knowledge", prompt)
        self.assertNotIn("memory", prompt.lower())
        self.assertNotIn("MEMORY.md", prompt)
        self.assertNotIn("/workspace/memory", prompt)

    def test_build_prompt_with_pending_feedback_from_observe(self):
        sys = _make_system()
        sys.observe(Observation(content="Ground truth: 4"))
        prompt = sys._build_prompt(self._query())
        self.assertIn("Ground truth: 4", prompt)


# ---------------------------------------------------------------------------
# JSONL parsing tests
# ---------------------------------------------------------------------------


class ParseTests(unittest.TestCase):
    def test_parse_events_skips_blank_lines(self):
        from src.systems.codex.system import _parse_events

        stdout = "\n" + json.dumps({"type": "x"}) + "\n\n"
        events = _parse_events(stdout)
        self.assertEqual(len(events), 1)

    def test_parse_events_skips_malformed(self):
        from src.systems.codex.system import _parse_events

        stdout = "not json\n" + json.dumps({"type": "x"}) + "\n"
        events = _parse_events(stdout)
        self.assertEqual(len(events), 1)

    def test_extract_assistant_text(self):
        from src.systems.codex.system import _extract_assistant_text

        events = [
            _session_meta_event(),
            _item_completed_event("first"),
            _turn_completed_event(),
            _item_completed_event("second"),
            _turn_completed_event(),
        ]
        self.assertEqual(_extract_assistant_text(events), "second")

    def test_extract_assistant_text_no_messages(self):
        from src.systems.codex.system import _extract_assistant_text

        with self.assertRaises(ValueError):
            _extract_assistant_text([_session_meta_event(), {"type": "turn.started"}])

    def test_extract_token_usage_uses_latest_turn_values(self):
        from src.systems.codex.system import _extract_token_usage

        events = [
            _turn_completed_event(
                input_tokens=100,
                output_tokens=50,
                reasoning_tokens=10,
                cached_tokens=0,
            ),
            _turn_completed_event(
                input_tokens=140,
                output_tokens=75,
                reasoning_tokens=15,
                cached_tokens=90,
            ),
        ]

        usage = _extract_token_usage(events)

        self.assertEqual(usage["input_tokens"], 140)
        self.assertEqual(usage["output_tokens"], 75)
        self.assertEqual(usage["reasoning_tokens"], 15)
        self.assertEqual(usage["cached_input_tokens"], 90)

    def test_parse_action_text_prefers_latest_valid_json(self):
        from src.systems.codex.system import _parse_action_text

        text = """
```json
{"description": "schema"}
```
```json
{"value": "hello"}
```
"""
        action = _parse_action_text(text, _SimpleAction)
        self.assertEqual(action.value, "hello")


# ---------------------------------------------------------------------------
# respond() integration tests
# ---------------------------------------------------------------------------


class RespondTests(unittest.TestCase):
    def _query(self):
        return Query(prompt="do it", response_schema=_SimpleAction)

    def test_respond_returns_valid_response(self):
        sys = _make_system()
        resp = _respond_with_mock(sys, self._query())
        self.assertIsInstance(resp, Response)
        self.assertEqual(resp.action.value, "hello")

    def test_respond_sets_conversation_id(self):
        sys = _make_system()
        self.assertIsNone(sys._conversation_id)
        _respond_with_mock(sys, self._query())
        self.assertEqual(sys._conversation_id, _CONV_ID)

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
                _session_meta_event("conv-one"),
                _item_completed_event('{"value": "first"}'),
                _turn_completed_event(input_tokens=100, output_tokens=50),
            ),
            _make_jsonl(
                _session_meta_event("conv-one"),
                _item_completed_event('{"value": "second"}'),
                _turn_completed_event(input_tokens=160, output_tokens=75),
            ),
            _make_jsonl(
                _session_meta_event("conv-one"),
                _item_completed_event('{"value": "third"}'),
                _turn_completed_event(input_tokens=200, output_tokens=100),
            ),
        ]
        codex_commands: list[str] = []

        def side_effect(cmd, **kwargs):
            if cmd[:2] == ["docker", "exec"]:
                command = cmd[-1]
                if command.startswith("codex "):
                    codex_commands.append(command)
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
        self.assertEqual(len(codex_commands), 3)
        self.assertNotIn(" exec resume ", codex_commands[0])
        self.assertIn(" exec resume ", codex_commands[1])
        self.assertIn(" exec resume ", codex_commands[2])

    def test_respond_starts_new_instance_when_not_single_conversation(self):
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
                _session_meta_event("conv-one"),
                _item_completed_event('{"value": "first"}'),
                _turn_completed_event(input_tokens=100, output_tokens=50),
            ),
            _make_jsonl(
                _session_meta_event("conv-one"),
                _item_completed_event('{"value": "second"}'),
                _turn_completed_event(input_tokens=160, output_tokens=75),
            ),
            _make_jsonl(
                _session_meta_event("conv-two"),
                _item_completed_event('{"value": "third"}'),
                _turn_completed_event(input_tokens=40, output_tokens=25),
            ),
        ]
        codex_commands: list[str] = []

        def side_effect(cmd, **kwargs):
            if cmd[:2] == ["docker", "exec"]:
                command = cmd[-1]
                if command.startswith("codex "):
                    codex_commands.append(command)
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
        self.assertFalse(first.metadata["continuity"]["resumed_this_turn"])
        self.assertTrue(second.metadata["continuity"]["resumed_this_turn"])
        self.assertFalse(third.metadata["continuity"]["resumed_this_turn"])
        self.assertEqual(len(codex_commands), 3)
        self.assertNotIn(" exec resume ", codex_commands[0])
        self.assertIn(" exec resume ", codex_commands[1])
        self.assertNotIn(" exec resume ", codex_commands[2])

    def test_respond_invokes_skill_only_on_first_conversation_turn(self):
        with tempfile.TemporaryDirectory() as tmp:
            skill_file = Path(tmp) / "edge-cases.md"
            skill_file.write_text("skill instructions")
            sys = _make_system(skill=str(skill_file))

        codex_commands: list[str] = []
        outputs = [
            _make_jsonl(
                _session_meta_event("conv-one"),
                _item_completed_event('{"value": "first"}'),
                _turn_completed_event(input_tokens=100, output_tokens=50),
            ),
            _make_jsonl(
                _session_meta_event("conv-one"),
                _item_completed_event('{"value": "second"}'),
                _turn_completed_event(input_tokens=160, output_tokens=75),
            ),
        ]

        def side_effect(cmd, **kwargs):
            if cmd[:2] == ["docker", "exec"]:
                command = cmd[-1]
                if command.startswith("codex "):
                    codex_commands.append(command)
                    return subprocess.CompletedProcess(
                        cmd,
                        0,
                        stdout=outputs.pop(0),
                        stderr="",
                    )
                return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
            return _mock_run_side_effect(cmd, **kwargs)

        with patch("subprocess.run", side_effect=side_effect):
            sys.respond(self._query())
            sys.respond(self._query())

        self.assertIn("$edge-cases", codex_commands[0])
        self.assertNotIn("$edge-cases", codex_commands[1])

    def test_respond_prepends_skill_only_on_first_conversation_turn(self):
        with tempfile.TemporaryDirectory() as tmp:
            skill_file = Path(tmp) / "edge-cases.md"
            skill_file.write_text(
                "---\nname: edge-cases\ndescription: Test skill.\n---\n\n# Edge Cases\nUse them.\n"
            )
            sys = _make_system(skill=str(skill_file), invocation_type="prepend")

        codex_commands: list[str] = []
        outputs = [
            _make_jsonl(
                _session_meta_event("conv-one"),
                _item_completed_event('{"value": "first"}'),
                _turn_completed_event(input_tokens=100, output_tokens=50),
            ),
            _make_jsonl(
                _session_meta_event("conv-one"),
                _item_completed_event('{"value": "second"}'),
                _turn_completed_event(input_tokens=160, output_tokens=75),
            ),
        ]

        def side_effect(cmd, **kwargs):
            if cmd[:2] == ["docker", "exec"]:
                command = cmd[-1]
                if command.startswith("codex "):
                    codex_commands.append(command)
                    return subprocess.CompletedProcess(
                        cmd,
                        0,
                        stdout=outputs.pop(0),
                        stderr="",
                    )
                return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
            return _mock_run_side_effect(cmd, **kwargs)

        with patch("subprocess.run", side_effect=side_effect):
            sys.respond(self._query())
            sys.respond(self._query())

        self.assertIn("# Edge Cases", codex_commands[0])
        self.assertIn("Use them.", codex_commands[0])
        self.assertNotIn("# Edge Cases", codex_commands[1])
        self.assertNotIn("Use them.", codex_commands[1])

    def test_repair_turn_does_not_reinvoke_skill_or_inline_body(self):
        with tempfile.TemporaryDirectory() as tmp:
            skill_file = Path(tmp) / "edge-cases.md"
            skill_file.write_text("skill body should not be in prompt")
            sys = _make_system(skill=str(skill_file))

        first = _make_jsonl(
            _session_meta_event("conv-one"),
            _item_completed_event('{"description": "schema"}'),
            _turn_completed_event(),
        )
        repaired = _make_jsonl(
            _session_meta_event("conv-one"),
            _item_completed_event('{"value": "fixed"}'),
            _turn_completed_event(input_tokens=200, output_tokens=100),
        )
        codex_commands: list[str] = []

        def side_effect(cmd, **kwargs):
            if cmd[:2] == ["docker", "exec"]:
                command = cmd[-1]
                if command.startswith("codex "):
                    codex_commands.append(command)
                    stdout = first if len(codex_commands) == 1 else repaired
                    return subprocess.CompletedProcess(
                        cmd,
                        0,
                        stdout=stdout,
                        stderr="",
                    )
                return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
            return _mock_run_side_effect(cmd, **kwargs)

        with patch("subprocess.run", side_effect=side_effect):
            resp = sys.respond(self._query())

        self.assertEqual(resp.action.value, "fixed")
        self.assertIn("$edge-cases", codex_commands[0])
        self.assertNotIn("skill body should not be in prompt", codex_commands[0])
        self.assertNotIn("$edge-cases", codex_commands[1])
        self.assertNotIn("skill body should not be in prompt", codex_commands[1])

    def test_repair_turn_does_not_repeat_prepended_skill_body(self):
        with tempfile.TemporaryDirectory() as tmp:
            skill_file = Path(tmp) / "edge-cases.md"
            skill_file.write_text(
                "---\nname: edge-cases\ndescription: Test skill.\n---\n\n# Edge Cases\nUse them.\n"
            )
            sys = _make_system(skill=str(skill_file), invocation_type="prepend")

        first = _make_jsonl(
            _session_meta_event("conv-one"),
            _item_completed_event('{"description": "schema"}'),
            _turn_completed_event(),
        )
        repaired = _make_jsonl(
            _session_meta_event("conv-one"),
            _item_completed_event('{"value": "fixed"}'),
            _turn_completed_event(input_tokens=200, output_tokens=100),
        )
        codex_commands: list[str] = []

        def side_effect(cmd, **kwargs):
            if cmd[:2] == ["docker", "exec"]:
                command = cmd[-1]
                if command.startswith("codex "):
                    codex_commands.append(command)
                    stdout = first if len(codex_commands) == 1 else repaired
                    return subprocess.CompletedProcess(
                        cmd,
                        0,
                        stdout=stdout,
                        stderr="",
                    )
                return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
            return _mock_run_side_effect(cmd, **kwargs)

        with patch("subprocess.run", side_effect=side_effect):
            resp = sys.respond(self._query())

        self.assertEqual(resp.action.value, "fixed")
        self.assertIn("# Edge Cases", codex_commands[0])
        self.assertIn("Use them.", codex_commands[0])
        self.assertNotIn("# Edge Cases", codex_commands[1])
        self.assertNotIn("Use them.", codex_commands[1])

    def test_respond_increments_interaction_count(self):
        sys = _make_system()
        _respond_with_mock(sys, self._query())
        self.assertEqual(sys._interaction_count, 1)
        _respond_with_mock(sys, self._query())
        self.assertEqual(sys._interaction_count, 2)

    def test_respond_appends_jsonl_log(self):
        sys = _make_system()
        _respond_with_mock(sys, self._query())
        self.assertEqual(len(sys._jsonl_log), 3)

    def test_respond_includes_token_usage(self):
        sys = _make_system()
        resp = _respond_with_mock(sys, self._query())
        usage = resp.metadata["token_usage"]
        self.assertEqual(usage["input_tokens"], 100)
        self.assertEqual(usage["output_tokens"], 50)
        self.assertEqual(usage["reasoning_tokens"], 10)
        self.assertEqual(usage["cached_input_tokens"], 0)

    def test_respond_metadata_collects_watched_memory_files(self):
        sys = _make_system(other_memory_files=["scratchpad/*.md"])
        scratchpad = Path(sys._tmp_dir) / "scratchpad"
        scratchpad.mkdir()
        (scratchpad / "continual-learn.md").write_text("notes")

        resp = _respond_with_mock(sys, self._query())

        self.assertEqual(
            resp.metadata["memory_files"],
            {"scratchpad/continual-learn.md": "notes"},
        )
        self.assertNotIn("memory_backend", resp.metadata)
        self.assertNotIn("memory_dir", resp.metadata)
        self.assertNotIn("other_memory_files", resp.metadata)
        self.assertNotIn("other_memory_file_patterns", resp.metadata)
        self.assertNotIn("memory_units", resp.metadata)
        self.assertEqual(
            sys._memory_history[0]["files"],
            {"scratchpad/continual-learn.md": "notes"},
        )

    def test_respond_reads_existing_native_memory_files(self):
        sys = _make_system(
            disable_native_memories=True,
            other_memory_files=["scratchpad/*.md"],
        )
        codex_memory = Path(sys._tmp_dir) / ".codex" / "memories"
        codex_memory.mkdir(parents=True)
        (codex_memory / "native.md").write_text("codex note")
        scratchpad = Path(sys._tmp_dir) / "scratchpad"
        scratchpad.mkdir()
        (scratchpad / "continual-learn.md").write_text("watched note")

        resp = _respond_with_mock(sys, self._query())

        self.assertEqual(
            resp.metadata["memory_files"],
            {
                "native.md": "codex note",
                "scratchpad/continual-learn.md": "watched note",
            },
        )
        self.assertNotIn("memory_backend", resp.metadata)
        self.assertNotIn("memory_dir", resp.metadata)
        self.assertNotIn("other_memory_files", resp.metadata)
        self.assertNotIn("other_memory_file_patterns", resp.metadata)

    def test_respond_accumulates_tokens(self):
        sys = _make_system()
        first_jsonl = _standard_jsonl()
        second_jsonl = _make_jsonl(
            _session_meta_event(),
            _item_completed_event('{"value": "hello"}'),
            _turn_completed_event(input_tokens=200, output_tokens=150),
        )

        _respond_with_mock(sys, self._query(), jsonl_stdout=first_jsonl)
        _respond_with_mock(sys, self._query(), jsonl_stdout=second_jsonl)

        self.assertEqual(sys._cumulative_tokens["input_tokens"], 200)
        self.assertEqual(sys._cumulative_tokens["output_tokens"], 150)

    def test_respond_includes_feedback_from_observe(self):
        sys = _make_system()
        sys.observe(Observation(content="Ground truth: channel A"))

        captured: list[str] = []

        def side_effect(cmd, **kwargs):
            if cmd[:2] == ["docker", "exec"]:
                captured.append(cmd[-1])
                return subprocess.CompletedProcess(
                    cmd, 0, stdout=_standard_jsonl(), stderr=""
                )
            return _mock_run_side_effect(cmd, **kwargs)

        with patch("subprocess.run", side_effect=side_effect):
            sys.respond(self._query())

        self.assertTrue(any("Ground truth: channel A" in c for c in captured))
        self.assertIsNone(sys._pending_feedback)

    def test_respond_repairs_invalid_json_payload(self):
        sys = _make_system()
        first = _make_jsonl(
            _session_meta_event(),
            _item_completed_event('{"description": "schema"}'),
            _turn_completed_event(),
        )
        repaired = _make_jsonl(
            _session_meta_event(),
            _item_completed_event('{"value": "fixed"}'),
            _turn_completed_event(input_tokens=200, output_tokens=100),
        )
        exec_calls = 0

        def side_effect(cmd, **kwargs):
            nonlocal exec_calls
            if cmd[:2] == ["docker", "exec"]:
                exec_calls += 1
                stdout = first if exec_calls == 1 else repaired
                return subprocess.CompletedProcess(cmd, 0, stdout=stdout, stderr="")
            return _mock_run_side_effect(cmd, **kwargs)

        with patch("subprocess.run", side_effect=side_effect):
            resp = sys.respond(self._query())

        self.assertEqual(resp.action.value, "fixed")
        self.assertTrue(resp.metadata["repair_attempted"])
        self.assertEqual(sys._cumulative_tokens["input_tokens"], 200)
        self.assertEqual(sys._cumulative_tokens["output_tokens"], 100)
        events = sys.consume_usage_events()
        self.assertEqual(len(events), 2)
        self.assertEqual([event.output_tokens for event in events], [50, 50])

    def test_respond_no_usage_event_returns_zeros(self):
        sys = _make_system()
        jsonl = _make_jsonl(
            _session_meta_event(),
            _item_completed_event('{"value": "hi"}'),
            {"type": "turn.completed", "usage": {}},
        )
        resp = _respond_with_mock(sys, self._query(), jsonl_stdout=jsonl)
        usage = resp.metadata["token_usage"]
        self.assertEqual(usage["input_tokens"], 0)

    def test_subprocess_nonzero_exit_raises(self):
        sys = _make_system()

        def side_effect(cmd, **kwargs):
            if cmd[:2] == ["docker", "exec"]:
                return subprocess.CompletedProcess(
                    cmd, 1, stdout="", stderr="bad things"
                )
            return _mock_run_side_effect(cmd, **kwargs)

        with patch("subprocess.run", side_effect=side_effect):
            with self.assertRaises(RuntimeError) as ctx:
                sys.respond(self._query())
            self.assertIn("bad things", str(ctx.exception))

    def test_failed_respond_preserves_pending_feedback(self):
        sys = _make_system()
        sys.observe(Observation(content="Ground truth: 4"))

        def side_effect(cmd, **kwargs):
            if cmd[:2] == ["docker", "exec"]:
                return subprocess.CompletedProcess(
                    cmd, 1, stdout="", stderr="bad things"
                )
            return _mock_run_side_effect(cmd, **kwargs)

        with patch("subprocess.run", side_effect=side_effect):
            with self.assertRaises(RuntimeError):
                sys.respond(self._query())
        self.assertEqual(sys._pending_feedback, "Ground truth: 4")

    def test_subprocess_timeout_raises(self):
        sys = _make_system()

        def side_effect(cmd, **kwargs):
            if cmd[:2] == ["docker", "exec"]:
                raise subprocess.TimeoutExpired(cmd=cmd, timeout=5)
            return _mock_run_side_effect(cmd, **kwargs)

        with patch("subprocess.run", side_effect=side_effect):
            with self.assertRaises(subprocess.TimeoutExpired):
                sys.respond(self._query())


# ---------------------------------------------------------------------------
# reset() tests
# ---------------------------------------------------------------------------


class ResetTests(unittest.TestCase):
    def test_reset_clears_state(self):
        sys = _make_system(other_memory_files=["scratchpad/*.md"])
        _respond_with_mock(sys, Query(prompt="x", response_schema=_SimpleAction))
        self.assertIsNotNone(sys._conversation_id)
        sys.observe(Observation(content="learned stuff"))
        sys._last_cumulative_token_usage_by_conversation = {_CONV_ID: {}}

        scratchpad = Path(sys._tmp_dir) / "scratchpad"
        scratchpad.mkdir()
        (scratchpad / "topic.md").write_text("learned stuff")
        codex_memory = Path(sys._tmp_dir) / ".codex" / "memories"
        codex_memory.mkdir(parents=True)
        (codex_memory / "topic.md").write_text("codex learned stuff")

        with patch("subprocess.run", side_effect=_mock_run_side_effect):
            sys.reset()

        self.assertIsNone(sys._conversation_id)
        self.assertEqual(sys._last_cumulative_token_usage_by_conversation, {})
        self.assertEqual(sys._interaction_count, 0)
        self.assertEqual(sys._jsonl_log, [])
        self.assertEqual(sys._memory_history, [])
        self.assertEqual(sys._conversation_session_ids, set())
        self.assertEqual(sys._conversation_history, [])
        self.assertEqual(sys._instance_conversations, {})
        self.assertFalse((scratchpad / "topic.md").exists())
        self.assertFalse(codex_memory.exists())
        self.assertEqual(sys._cumulative_tokens["input_tokens"], 0)
        self.assertIsNone(sys._pending_feedback)

    def test_reset_clears_conversation_even_when_single_conversation(self):
        sys = _make_system(single_conversation=True)
        _respond_with_mock(sys, Query(prompt="x", response_schema=_SimpleAction))
        self.assertIsNotNone(sys._conversation_id)

        with patch("subprocess.run", side_effect=_mock_run_side_effect):
            sys.reset()

        self.assertIsNone(sys._conversation_id)
        self.assertEqual(sys._conversation_session_ids, set())
        self.assertEqual(sys._last_cumulative_token_usage_by_conversation, {})

    def test_reset_does_not_create_native_memory_dir(self):
        sys = _make_system()
        with patch("subprocess.run", side_effect=_mock_run_side_effect):
            sys.reset()
        self.assertFalse((Path(sys._tmp_dir) / "memory").exists())
        self.assertFalse((Path(sys._tmp_dir) / ".codex" / "memories").exists())


# ---------------------------------------------------------------------------
# Usage event tracking tests
# ---------------------------------------------------------------------------


class UsageEventTests(unittest.TestCase):
    def test_respond_records_usage_event(self):
        sys = _make_system()
        _respond_with_mock(sys, Query(prompt="x", response_schema=_SimpleAction))
        events = sys.consume_usage_events()
        self.assertEqual(len(events), 1)
        event = events[0]
        self.assertEqual(event.model, "test-model")
        self.assertEqual(event.provider, "openai")
        self.assertEqual(event.input_tokens, 100)
        self.assertEqual(event.output_tokens, 50)
        self.assertEqual(event.call_type, "completion")

    def test_respond_records_cached_usage_event_fields(self):
        sys = _make_system()
        jsonl = _make_jsonl(
            _session_meta_event(),
            _item_completed_event('{"value": "hi"}'),
            _turn_completed_event(
                input_tokens=1000,
                output_tokens=50,
                reasoning_tokens=25,
                cached_tokens=800,
            ),
        )
        _respond_with_mock(
            sys,
            Query(prompt="x", response_schema=_SimpleAction),
            jsonl_stdout=jsonl,
        )
        event = sys.consume_usage_events()[0]
        self.assertEqual(event.input_tokens, 1000)
        self.assertEqual(event.output_tokens, 50)
        self.assertEqual(event.reasoning_tokens, 25)
        self.assertEqual(event.cached_input_tokens, 800)

    def test_respond_records_latest_resumed_turn_usage(self):
        sys = _make_system()
        query = Query(prompt="x", response_schema=_SimpleAction)
        first_jsonl = _make_jsonl(
            _session_meta_event(),
            _item_completed_event('{"value": "first"}'),
            _turn_completed_event(input_tokens=100, output_tokens=50),
        )
        second_jsonl = _make_jsonl(
            _session_meta_event(),
            _item_completed_event('{"value": "first"}'),
            _turn_completed_event(input_tokens=100, output_tokens=50),
            _item_completed_event('{"value": "second"}'),
            _turn_completed_event(
                input_tokens=240,
                output_tokens=75,
                reasoning_tokens=15,
                cached_tokens=90,
            ),
        )

        _respond_with_mock(sys, query, jsonl_stdout=first_jsonl)
        first_event = sys.consume_usage_events()[0]
        _respond_with_mock(sys, query, jsonl_stdout=second_jsonl)
        second_event = sys.consume_usage_events()[0]

        self.assertEqual(first_event.output_tokens, 50)
        self.assertEqual(second_event.output_tokens, 25)
        self.assertEqual(second_event.input_tokens, 140)
        self.assertEqual(second_event.reasoning_tokens, 5)
        self.assertEqual(second_event.cached_input_tokens, 90)
        self.assertEqual(sys._cumulative_tokens["output_tokens"], 75)

    def test_respond_prefers_last_usage_when_present(self):
        sys = _make_system()
        query = Query(prompt="x", response_schema=_SimpleAction)
        first_jsonl = _make_jsonl(
            _session_meta_event(),
            _item_completed_event('{"value": "first"}'),
            _turn_completed_event(input_tokens=100, output_tokens=50),
        )
        second_jsonl = _make_jsonl(
            _session_meta_event(),
            _item_completed_event('{"value": "second"}'),
            {
                "type": "turn.completed",
                "usage": {
                    "input_tokens": 240,
                    "output_tokens": 75,
                    "reasoning_output_tokens": 15,
                    "cached_input_tokens": 90,
                },
                "last_usage": {
                    "input_tokens": 140,
                    "output_tokens": 25,
                    "reasoning_output_tokens": 5,
                    "cached_input_tokens": 90,
                },
            },
        )

        _respond_with_mock(sys, query, jsonl_stdout=first_jsonl)
        sys.consume_usage_events()
        _respond_with_mock(sys, query, jsonl_stdout=second_jsonl)
        second_event = sys.consume_usage_events()[0]

        self.assertEqual(second_event.input_tokens, 140)
        self.assertEqual(second_event.output_tokens, 25)
        self.assertEqual(second_event.reasoning_tokens, 5)
        self.assertEqual(second_event.cached_input_tokens, 90)

    def test_respond_records_full_usage_for_new_conversation(self):
        sys = _make_system()
        query = Query(prompt="x", response_schema=_SimpleAction)
        first_jsonl = _make_jsonl(
            _session_meta_event("conv-first"),
            _item_completed_event('{"value": "first"}'),
            _turn_completed_event(input_tokens=100, output_tokens=50),
        )
        second_jsonl = _make_jsonl(
            _session_meta_event("conv-second"),
            _item_completed_event('{"value": "second"}'),
            _turn_completed_event(input_tokens=140, output_tokens=75),
        )

        _respond_with_mock(sys, query, jsonl_stdout=first_jsonl)
        first_event = sys.consume_usage_events()[0]
        _respond_with_mock(sys, query, jsonl_stdout=second_jsonl)
        second_event = sys.consume_usage_events()[0]

        self.assertEqual(first_event.output_tokens, 50)
        self.assertEqual(second_event.output_tokens, 75)
        self.assertEqual(sys._cumulative_tokens["output_tokens"], 125)

    def test_multiple_responds_accumulate_usage_events(self):
        sys = _make_system()
        query = Query(prompt="x", response_schema=_SimpleAction)
        _respond_with_mock(sys, query)
        _respond_with_mock(sys, query)
        events = sys.consume_usage_events()
        self.assertEqual(len(events), 2)

    def test_consume_clears_buffer(self):
        sys = _make_system()
        _respond_with_mock(sys, Query(prompt="x", response_schema=_SimpleAction))
        sys.consume_usage_events()
        self.assertEqual(sys.consume_usage_events(), [])


# ---------------------------------------------------------------------------
# Artifacts tests
# ---------------------------------------------------------------------------


class ArtifactTests(unittest.TestCase):
    def test_get_run_artifacts(self):
        sys = _make_system(other_memory_files=["scratchpad/*.md"])
        codex_memory = Path(sys._tmp_dir) / ".codex" / "memories"
        codex_memory.mkdir(parents=True)
        (codex_memory / "native.md").write_text("codex note")
        scratchpad = Path(sys._tmp_dir) / "scratchpad"
        scratchpad.mkdir()
        (scratchpad / "topic.md").write_text("note body")
        _respond_with_mock(sys, Query(prompt="x", response_schema=_SimpleAction))

        artifacts = sys.get_run_artifacts()
        self.assertEqual(artifacts["artifact_type"], "codex")
        self.assertEqual(artifacts["conversation_id"], _CONV_ID)
        self.assertEqual(artifacts["interaction_count"], 1)
        self.assertIsInstance(artifacts["jsonl_events"], list)
        self.assertIn("memory_files", artifacts)
        self.assertEqual(
            artifacts["memory_files"],
            {"native.md": "codex note", "scratchpad/topic.md": "note body"},
        )
        self.assertNotIn("memory_dir", artifacts)
        self.assertNotIn("memory_backend", artifacts)
        self.assertNotIn("other_memory_files", artifacts)
        self.assertNotIn("other_memory_file_patterns", artifacts)
        self.assertIn("cumulative_tokens", artifacts)
        self.assertEqual(artifacts["cumulative_tokens"]["input_tokens"], 100)

    def test_get_run_artifacts_collects_only_current_conversation_session_jsonl(self):
        sys = _make_system()
        rel = _write_session_jsonl(
            sys._tmp_dir,
            contents="main transcript\n",
        )
        other_rel = _write_session_jsonl(
            sys._tmp_dir,
            conv_id="conv_other",
            contents="other transcript\n",
        )

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

        self.assertEqual(artifacts["codex_conversation_session_ids"], [_CONV_ID])
        self.assertEqual(
            artifacts["codex_conversation_jsonl_files"],
            {rel: "main transcript\n"},
        )
        self.assertEqual(resp.metadata["codex_conversation_jsonl_files"], [rel])
        self.assertEqual(
            artifacts["codex_instance_conversations"][0]["conversation_jsonl_files"],
            {rel: "main transcript\n"},
        )
        self.assertNotIn(other_rel, artifacts["codex_conversation_jsonl_files"])

    def test_get_run_artifacts_matches_session_meta_when_filename_differs(self):
        sys = _make_system()
        rel = _write_session_jsonl(
            sys._tmp_dir,
            rel=".codex/sessions/2025/01/03/rollout-without-id.jsonl",
        )

        _respond_with_mock(sys, Query(prompt="x", response_schema=_SimpleAction))
        artifacts = sys.get_run_artifacts()

        self.assertIn(rel, artifacts["codex_conversation_jsonl_files"])

    def test_save_artifacts_writes_conversation_jsonl_per_run_and_instance(self):
        import src.systems.codex  # noqa: F401

        rel = ".codex/sessions/2025/01/03/rollout-2025-01-03T12-00-00-conv_123.jsonl"
        artifacts = {
            "artifact_type": "codex",
            "conversation_id": "conv_123",
            "codex_conversation_session_ids": ["conv_123"],
            "codex_conversation_jsonl_files": {rel: "main transcript\n"},
            "codex_conversation_history": [
                {
                    "step": 1,
                    "instance_id": "inst-0",
                    "instance_index": 0,
                    "conversation_jsonl_files": [rel],
                }
            ],
            "codex_instance_conversations": [
                {
                    "step": 1,
                    "instance_id": "inst-0",
                    "instance_index": 0,
                    "conversation_id": "conv_123",
                    "session_ids": ["conv_123"],
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
                    / "codex_conversations"
                    / "final"
                    / ".codex"
                    / "sessions"
                    / "2025"
                    / "01"
                    / "03"
                    / "rollout-2025-01-03T12-00-00-conv_123.jsonl"
                ).read_text(),
                "main transcript\n",
            )
            self.assertEqual(
                (
                    artifact_dir
                    / "instances"
                    / "instance_0000"
                    / "codex_conversations"
                    / ".codex"
                    / "sessions"
                    / "2025"
                    / "01"
                    / "03"
                    / "rollout-2025-01-03T12-00-00-conv_123.jsonl"
                ).read_text(),
                "instance transcript\n",
            )
            manifest = json.loads((artifact_dir / "manifest.json").read_text())
            self.assertEqual(manifest["codex_conversation_jsonl_files"], [rel])
            self.assertEqual(
                manifest["codex_instance_conversations"][0]["instance_index"],
                0,
            )

    def test_save_artifacts_writes_baseline_instances(self):
        import src.systems.codex  # noqa: F401

        artifacts = {
            "artifact_type": "codex",
            "phase": "baseline",
            "baseline_instances": [
                {
                    "instance_index": 2,
                    "artifact_manifest": {"jsonl_event_count": 1},
                    "system_artifacts": {
                        "artifact_type": "codex",
                        "conversation_id": "conv_baseline",
                        "interaction_count": 1,
                        "jsonl_events": [{"type": "turn.completed"}],
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
            manifest = json.loads((artifact_dir / "manifest.json").read_text())
            self.assertEqual(manifest["artifact_type"], "codex")
            self.assertEqual(manifest["phase"], "baseline")
            self.assertEqual(manifest["baseline_instance_count"], 1)
            self.assertEqual(manifest["instances"][0]["instance_index"], 2)


# ---------------------------------------------------------------------------
# Name property
# ---------------------------------------------------------------------------


class NameTests(unittest.TestCase):
    def test_name_default(self):
        sys = _make_system()
        self.assertEqual(sys.name, "codex")

    def test_name_custom(self):
        sys = _make_system(name="my_codex")
        self.assertEqual(sys.name, "my_codex")


# ---------------------------------------------------------------------------
# Container lifecycle
# ---------------------------------------------------------------------------


class ContainerTests(unittest.TestCase):
    def test_stop_container_on_del(self):
        sys = _make_system()
        container_id = sys._container_id
        self.assertIsNotNone(container_id)

        with patch("subprocess.run", side_effect=_mock_run_side_effect) as mock_run:
            sys._stop_container()
            docker_rm_calls = [
                c for c in mock_run.call_args_list if c.args[0][:2] == ["docker", "rm"]
            ]
            self.assertEqual(len(docker_rm_calls), 1)
            self.assertIn(container_id, docker_rm_calls[0].args[0])

        self.assertIsNone(sys._container_id)
