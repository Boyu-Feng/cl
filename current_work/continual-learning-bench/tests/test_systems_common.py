"""Tests for shared system helpers."""

import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


class MemoryHelperTests(unittest.TestCase):
    def test_resolve_seed_memory_dir_requires_existing_directory(self):
        from src.systems.common import resolve_seed_memory_dir

        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(resolve_seed_memory_dir(tmp), Path(tmp).resolve())

        with self.assertRaises(FileNotFoundError) as ctx:
            resolve_seed_memory_dir("/missing/seed-dir")

        self.assertIn("seed_memory_dir", str(ctx.exception))

    def test_copy_memory_seed_copies_files_and_directories(self):
        from src.systems.common import copy_memory_seed

        with tempfile.TemporaryDirectory() as tmp:
            seed_dir = Path(tmp) / "seed"
            target_dir = Path(tmp) / "target"
            (seed_dir / "nested").mkdir(parents=True)
            (seed_dir / "topic.md").write_text("topic")
            (seed_dir / "nested" / "note.md").write_text("nested")

            copy_memory_seed(seed_dir, target_dir)

            self.assertEqual((target_dir / "topic.md").read_text(), "topic")
            self.assertEqual(
                (target_dir / "nested" / "note.md").read_text(),
                "nested",
            )

    def test_read_memory_snapshot_uses_relative_paths_and_later_dirs_win(self):
        from src.systems.common import read_memory_snapshot

        with tempfile.TemporaryDirectory() as tmp:
            first_dir = Path(tmp) / "first"
            second_dir = Path(tmp) / "second"
            first_dir.mkdir()
            second_dir.mkdir()
            (first_dir / "topic.md").write_text("old")
            (second_dir / "topic.md").write_text("new")
            (second_dir / "nested").mkdir()
            (second_dir / "nested" / "note.md").write_text("note")

            snapshot = read_memory_snapshot(
                [first_dir, second_dir],
                logger=Mock(),
            )

            self.assertEqual(
                snapshot,
                {
                    "nested/note.md": "note",
                    "topic.md": "new",
                },
            )

    def test_read_other_memory_files_expands_files_dirs_and_globs(self):
        from src.systems.common import read_other_memory_files

        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "scratchpad" / "nested").mkdir(parents=True)
            (workspace / "scratchpad" / "continual-learn.md").write_text("notes")
            (workspace / "scratchpad" / "nested" / "detail.txt").write_text("detail")
            (workspace / "logs").mkdir()
            (workspace / "logs" / "keep.md").write_text("log")
            (workspace / "logs" / "skip.txt").write_text("skip")

            snapshot = read_other_memory_files(
                workspace,
                ["scratchpad/continual-learn.md", "scratchpad/nested", "logs/*.md"],
                logger=Mock(),
            )

            self.assertEqual(
                snapshot,
                {
                    "logs/keep.md": "log",
                    "scratchpad/continual-learn.md": "notes",
                    "scratchpad/nested/detail.txt": "detail",
                },
            )

    def test_read_other_memory_files_rejects_paths_outside_workspace(self):
        from src.systems.common import read_other_memory_files

        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            with self.assertRaises(ValueError):
                read_other_memory_files(workspace, ["../secret.md"], logger=Mock())
            with self.assertRaises(ValueError):
                read_other_memory_files(
                    workspace, [str(workspace / "x.md")], logger=Mock()
                )

    def test_merge_memory_snapshots_preserves_native_on_collision(self):
        from src.systems.common import merge_memory_snapshots

        merged, sources, other_key_map = merge_memory_snapshots(
            {"MENTAL_MODEL.md": "native", "topic.md": "topic"},
            {"MENTAL_MODEL.md": "other", "scratchpad/note.md": "note"},
            native_source="claude_auto_memory",
        )

        self.assertEqual(
            merged,
            {
                "MENTAL_MODEL.md": "native",
                "other_memory/MENTAL_MODEL.md": "other",
                "scratchpad/note.md": "note",
                "topic.md": "topic",
            },
        )
        self.assertEqual(sources["MENTAL_MODEL.md"], "claude_auto_memory")
        self.assertEqual(sources["other_memory/MENTAL_MODEL.md"], "other_memory_file")
        self.assertEqual(
            other_key_map["MENTAL_MODEL.md"], "other_memory/MENTAL_MODEL.md"
        )
        self.assertEqual(other_key_map["scratchpad/note.md"], "scratchpad/note.md")


class SkillHelperTests(unittest.TestCase):
    def test_load_skill_prompt_prefix_installs_single_file_skill(self):
        from src.systems.common import load_skill_prompt_prefix

        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            workspace.mkdir()
            skill_file = Path(tmp) / "edge-cases.md"
            skill_file.write_text("skill instructions")

            prefix = load_skill_prompt_prefix(
                str(skill_file),
                agent_name="codex",
                workspace_dir=workspace,
                invocation_type="skill",
            )

            installed = workspace / ".codex" / "skills" / "edge-cases" / "SKILL.md"
            self.assertEqual(prefix, "$edge-cases\n")
            self.assertEqual(installed.read_text(), "skill instructions")

    def test_load_skill_prompt_prefix_copies_standard_skill_directory(self):
        from src.systems.common import load_skill_prompt_prefix

        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            workspace.mkdir()
            skill_dir = Path(tmp) / "continual-learn"
            refs_dir = skill_dir / "references"
            refs_dir.mkdir(parents=True)
            (skill_dir / "SKILL.md").write_text(
                "---\nname: continual-learn\ndescription: Learn as you go.\n---\n\n# Skill\n"
            )
            (refs_dir / "guide.md").write_text("details")

            prefix = load_skill_prompt_prefix(
                str(skill_dir),
                agent_name="claude",
                workspace_dir=workspace,
                invocation_type="skill",
            )

            installed = workspace / ".claude" / "skills" / "continual-learn"
            self.assertEqual(
                prefix,
                "<command-message>continual-learn</command-message>\n"
                "<command-name>/continual-learn</command-name>\n",
            )
            self.assertEqual(
                (installed / "SKILL.md").read_text(),
                (skill_dir / "SKILL.md").read_text(),
            )
            self.assertEqual(
                (installed / "references" / "guide.md").read_text(), "details"
            )

    def test_load_skill_prompt_prefix_accepts_skill_md_file_location(self):
        from src.systems.common import load_skill_prompt_prefix

        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            workspace.mkdir()
            skill_dir = Path(tmp) / "helper"
            skill_dir.mkdir()
            skill_file = skill_dir / "SKILL.md"
            skill_file.write_text("# Helper\n")

            prefix = load_skill_prompt_prefix(
                str(skill_file),
                agent_name="codex",
                workspace_dir=workspace,
                invocation_type="skill",
            )

            installed = workspace / ".codex" / "skills" / "helper" / "SKILL.md"
            self.assertEqual(prefix, "$helper\n")
            self.assertEqual(installed.read_text(), "# Helper\n")

    def test_load_skill_prompt_prefix_prepend_strips_frontmatter(self):
        from src.systems.common import load_skill_prompt_prefix

        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            workspace.mkdir()
            skill_file = Path(tmp) / "learner.md"
            skill_file.write_text(
                "---\nname: learner\ndescription: Test skill.\n---\n\n# Learner\nKeep notes.\n"
            )

            prefix = load_skill_prompt_prefix(
                str(skill_file),
                agent_name="claude",
                workspace_dir=workspace,
                invocation_type="prepend",
            )

            self.assertEqual(prefix, "# Learner\nKeep notes.\n\n")
            self.assertFalse((workspace / ".claude" / "skills").exists())


class DockerHelperTests(unittest.TestCase):
    def test_start_container_mounts_workspace_and_sets_env(self):
        from src.systems.common import CONTAINER_WORKSPACE, start_docker_container

        result = subprocess.CompletedProcess(
            ["docker", "run"],
            0,
            stdout="abc123\n",
            stderr="",
        )
        with patch("subprocess.run", return_value=result) as mock_run:
            container_id = start_docker_container(
                name_prefix="agent_bench",
                host_workspace="/tmp/run_123",
                docker_image="node:22-slim",
                env={"TOKEN": "secret", "CONFIG_DIR": "/workspace/.config"},
                logger=Mock(),
            )

        self.assertEqual(container_id, "abc123")
        cmd = mock_run.call_args.args[0]
        self.assertEqual(
            cmd[:5], ["docker", "run", "-d", "--name", "agent_bench_run_123"]
        )
        self.assertIn("-v", cmd)
        self.assertIn(f"/tmp/run_123:{CONTAINER_WORKSPACE}", cmd)
        self.assertIn("TOKEN=secret", cmd)
        self.assertIn("CONFIG_DIR=/workspace/.config", cmd)
        self.assertEqual(cmd[-3:], ["node:22-slim", "sleep", "infinity"])

    def test_docker_exec_requires_running_container(self):
        from src.systems.common import docker_exec

        with self.assertRaises(RuntimeError) as ctx:
            docker_exec(None, "echo hi", timeout=1, default_timeout=30)

        self.assertIn("Container not running", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
