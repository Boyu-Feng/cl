import gzip
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pydantic import BaseModel

from src.artifacts import build_artifact_manifest, save_artifacts
from src.interface import (
    EvalMetrics,
    InstanceOutcome,
    Observation,
    Query,
    Response,
    TaskResult,
)
import src.trace_storage as trace_storage
from src.trace_storage import (
    BaselineSummaryEntry,
    InteractionStep,
    RunSummary,
    RunSummaryRunEntry,
    attach_baseline_to_outcomes,
    build_benchmark_aggregate,
    load_run_summary,
    load_run_summaries,
    save_run_summary,
    save_viewer_artifact,
    summarize_instance_outcomes,
    save_trace,
    TraceRecorder,
)


class _TraceAction(BaseModel):
    value: str


class TraceStorageTests(unittest.TestCase):
    def test_save_trace_uses_custom_output_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "nested" / "trace.json"
            trace_data = {
                "task": {"params": {"variant": "calling_station"}},
                "result": {"score": 1.0},
            }

            saved_path = save_trace(trace_data, "exploitable_poker", output_path)

            self.assertEqual(saved_path, output_path)
            self.assertTrue(output_path.exists())
            with open(output_path) as f:
                self.assertEqual(json.load(f), trace_data)

    def test_save_artifacts_uses_registered_ace_exporter(self):
        import src.systems.ace  # noqa: F401

        artifacts = {
            "artifact_type": "ace",
            "initial_playbook": "## STRATEGIES & INSIGHTS\n",
            "final_playbook": "## STRATEGIES & INSIGHTS\n[sai-00001] helpful=1 harmful=0 :: Example\n",
            "playbook_snapshots": [
                {
                    "update_count": 1,
                    "interaction_count": 4,
                    "playbook_bullet_count": 1,
                    "reflection_ran": True,
                    "curation_ran": True,
                    "summary": "Reflected on 4 turn(s); ran curation.",
                    "playbook": "## STRATEGIES & INSIGHTS\n[sai-00001] helpful=0 harmful=0 :: Example\n",
                }
            ],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            trace_path = Path(tmpdir) / "trace.json"
            trace_path.write_text("{}", encoding="utf-8")

            artifact_dir = save_artifacts(artifacts, trace_path)

            self.assertIsNotNone(artifact_dir)
            self.assertTrue((artifact_dir / "initial_playbook.md").exists())
            self.assertTrue((artifact_dir / "final_playbook.md").exists())
            self.assertTrue(
                (artifact_dir / "playbook_snapshots" / "update_0001.md").exists()
            )
            manifest = json.loads((artifact_dir / "manifest.json").read_text())
            self.assertEqual(manifest["artifact_type"], "ace")
            self.assertEqual(len(manifest["playbook_snapshots"]), 1)

    def test_save_artifacts_uses_registered_codex_exporter(self):
        import src.systems.codex  # noqa: F401

        artifacts = {
            "artifact_type": "codex",
            "conversation_id": "conv_123",
            "version": "0.125.0",
            "interaction_count": 1,
            "cumulative_tokens": {"input_tokens": 10},
            "jsonl_events": [{"type": "turn.completed"}],
            "memory_history": [{"step": 1, "files": {"MEMORY.md": "note"}}],
            "memory_files": {"MEMORY.md": "note", "topic.md": "topic body"},
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            trace_path = Path(tmpdir) / "trace.json"
            trace_path.write_text("{}", encoding="utf-8")

            artifact_dir = save_artifacts(artifacts, trace_path)

            self.assertIsNotNone(artifact_dir)
            self.assertTrue((artifact_dir / "events.jsonl").exists())
            self.assertEqual(
                (artifact_dir / "memory" / "MEMORY.md").read_text(), "note"
            )
            self.assertEqual(
                (artifact_dir / "memory" / "topic.md").read_text(), "topic body"
            )
            manifest = json.loads((artifact_dir / "manifest.json").read_text())
            self.assertEqual(manifest["artifact_type"], "codex")
            self.assertEqual(manifest["conversation_id"], "conv_123")
            self.assertNotIn("memory_dir", manifest)
            self.assertNotIn("memory_backend", manifest)
            self.assertNotIn("other_memory_files", manifest)
            self.assertNotIn("other_memory_file_patterns", manifest)
            self.assertCountEqual(manifest["memory_files"], ["MEMORY.md", "topic.md"])

    def test_save_artifacts_writes_codex_watched_memory_files(self):
        import src.systems.codex  # noqa: F401

        artifacts = {
            "artifact_type": "codex",
            "conversation_id": "conv_123",
            "version": "0.125.0",
            "interaction_count": 1,
            "cumulative_tokens": {"input_tokens": 10},
            "jsonl_events": [],
            "memory_history": [
                {"step": 1, "files": {"scratchpad/continual-learn.md": "notes"}}
            ],
            "memory_files": {"scratchpad/continual-learn.md": "notes"},
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            trace_path = Path(tmpdir) / "trace.json"
            trace_path.write_text("{}", encoding="utf-8")

            artifact_dir = save_artifacts(artifacts, trace_path)

            self.assertIsNotNone(artifact_dir)
            self.assertEqual(
                (
                    artifact_dir / "memory" / "scratchpad" / "continual-learn.md"
                ).read_text(),
                "notes",
            )
            manifest = json.loads((artifact_dir / "manifest.json").read_text())
            self.assertNotIn("memory_dir", manifest)
            self.assertNotIn("memory_backend", manifest)
            self.assertNotIn("other_memory_files", manifest)
            self.assertNotIn("other_memory_file_patterns", manifest)
            self.assertEqual(
                manifest["memory_files"], ["scratchpad/continual-learn.md"]
            )

    def test_build_artifact_manifest_returns_generic_type_for_unknown_artifacts(self):
        manifest = build_artifact_manifest({"artifact_type": "custom"})
        self.assertEqual(manifest, {"type": "custom"})

    def test_trace_payload_collects_response_memory_files_in_top_level_metadata(self):
        recorder = TraceRecorder(
            system_name="codex",
            task_name="task",
            system_params={},
            task_params={},
        )
        query = Query(prompt="q", response_schema=_TraceAction)
        response = Response(
            action=_TraceAction(value="ok"),
            metadata={
                "memory_files": {"summary.md": "remember this"},
            },
        )
        recorder.record_interaction(
            1,
            query,
            response,
            observation=Observation(content="obs"),
            done=False,
        )

        payload = recorder._build_trace_payload(result=None)

        self.assertNotIn("backend", payload["system_memory"])
        self.assertNotIn("memory_dir", payload["system_memory"])
        self.assertEqual(
            payload["system_memory"]["files"], {"summary.md": "remember this"}
        )
        self.assertEqual(
            payload["system_memory"]["history"],
            [{"step": 1, "files": {"summary.md": "remember this"}}],
        )
        self.assertNotIn("units", payload["system_memory"])

    def test_trace_payload_collects_native_and_other_memory_metadata(self):
        recorder = TraceRecorder(
            system_name="claude",
            task_name="task",
            system_params={},
            task_params={},
        )
        query = Query(prompt="q", response_schema=_TraceAction)
        response = Response(
            action=_TraceAction(value="ok"),
            metadata={
                "memory_backend": "claude_auto_memory",
                "memory_dir": ".claude/projects/-workspace/memory",
                "memory_files": {
                    "MEMORY.md": "claude note",
                    "MENTAL_MODEL.md": "workspace note",
                },
                "native_memory_files": {"MEMORY.md": "claude note"},
                "other_memory_files": {"MENTAL_MODEL.md": "workspace note"},
                "memory_file_sources": {
                    "MEMORY.md": "claude_auto_memory",
                    "MENTAL_MODEL.md": "other_memory_file",
                },
                "other_memory_file_mapping": {"MENTAL_MODEL.md": "MENTAL_MODEL.md"},
                "other_memory_file_patterns": ["MENTAL_MODEL.md"],
            },
        )
        recorder.record_interaction(
            1,
            query,
            response,
            observation=Observation(content="obs"),
            done=False,
        )

        payload = recorder._build_trace_payload(result=None)

        self.assertEqual(
            payload["system_memory"]["files"],
            {"MEMORY.md": "claude note", "MENTAL_MODEL.md": "workspace note"},
        )
        self.assertEqual(
            payload["system_memory"]["native_files"], {"MEMORY.md": "claude note"}
        )
        self.assertEqual(
            payload["system_memory"]["other_files"],
            {"MENTAL_MODEL.md": "workspace note"},
        )
        self.assertEqual(
            payload["system_memory"]["file_sources"]["MENTAL_MODEL.md"],
            "other_memory_file",
        )
        self.assertEqual(
            payload["system_memory"]["other_file_patterns"], ["MENTAL_MODEL.md"]
        )

    def test_trace_payload_records_empty_memory_snapshots_when_backend_is_present(self):
        recorder = TraceRecorder(
            system_name="claude",
            task_name="task",
            system_params={},
            task_params={},
        )
        recorder.record_interaction(
            1,
            Query(prompt="q", response_schema=_TraceAction),
            Response(
                action=_TraceAction(value="ok"),
                metadata={
                    "memory_backend": "claude_auto_memory",
                    "memory_dir": ".claude/projects/-workspace/memory",
                    "memory_files": {},
                },
            ),
            observation=Observation(content="obs"),
            done=False,
        )

        payload = recorder._build_trace_payload(result=None)

        self.assertEqual(payload["system_memory"]["backend"], "claude_auto_memory")
        self.assertEqual(payload["system_memory"]["files"], {})
        self.assertEqual(
            payload["system_memory"]["history"], [{"step": 1, "files": {}}]
        )

    def test_trace_payload_uses_artifact_memory_history_when_available(self):
        recorder = TraceRecorder(
            system_name="codex",
            task_name="task",
            system_params={},
            task_params={},
        )
        recorder.record_system_artifacts(
            {
                "artifact_type": "codex",
                "memory_files": {"final.md": "final note"},
                "memory_history": [
                    {"step": 1, "files": {}},
                    {"step": 2, "files": {"final.md": "final note"}},
                ],
            }
        )

        payload = recorder._build_trace_payload(result=None)

        self.assertEqual(payload["system_memory"]["files"], {"final.md": "final note"})
        self.assertEqual(
            payload["system_memory"]["history"],
            [
                {"step": 1, "files": {}},
                {"step": 2, "files": {"final.md": "final note"}},
            ],
        )


def _make_run_summary(run_mode=None, **overrides):
    defaults = dict(
        run_group_id="abc123",
        system={"name": "sys", "params": {}},
        task={"name": "tsk", "params": {"variant": "v1"}},
        variant="v1",
        schedule=None,
        run_count=3,
        max_workers=2,
        aggregate={"score": {"mean": 0.5}},
        runs=[
            RunSummaryRunEntry(
                run_index=i,
                score=0.5,
                trace_path=f"t{i}.json",
                execution={"usage": {"interaction": {"cost_usd": 0.1}}},
            )
            for i in range(3)
        ],
        run_mode=run_mode,
    )
    defaults.update(overrides)
    return RunSummary(**defaults)


class RunSummaryTests(unittest.TestCase):
    def test_round_trip(self):
        s = _make_run_summary(run_mode="resample")
        d = s.to_dict()
        restored = RunSummary.from_dict(d)
        self.assertEqual(restored.run_group_id, s.run_group_id)
        self.assertEqual(restored.variant, "v1")
        self.assertEqual(restored.run_mode, "resample")
        self.assertEqual(len(restored.runs), 3)
        self.assertEqual(
            restored.runs[0].execution["usage"]["interaction"]["cost_usd"], 0.1
        )

    def test_from_dict_defaults_run_mode_to_none(self):
        s = _make_run_summary(run_mode="replicate")
        d = s.to_dict()
        del d["run_mode"]
        del d["rollout_mode"]
        restored = RunSummary.from_dict(d)
        self.assertIsNone(restored.run_mode)

    def test_run_mode_round_trips(self):
        for mode in [None, "replicate", "resample", "permute"]:
            s = _make_run_summary(run_mode=mode)
            d = s.to_dict()
            restored = RunSummary.from_dict(d)
            self.assertEqual(restored.run_mode, mode, f"Failed for mode={mode}")

    def test_save_and_load(self):
        s = _make_run_summary(run_mode="resample")
        with tempfile.TemporaryDirectory() as tmpdir:
            import os

            orig_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                path = save_run_summary(s, "tsk")
                loaded = load_run_summary(path)
                self.assertEqual(loaded.run_group_id, s.run_group_id)
                self.assertEqual(loaded.run_mode, "resample")
            finally:
                os.chdir(orig_cwd)

    def test_saved_summary_names_use_full_timestamp_run_id(self):
        run_id = "2026-04-29T15-30-12.123456Z"
        s = _make_run_summary(run_group_id=run_id)
        with tempfile.TemporaryDirectory() as tmpdir:
            import os

            orig_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                summary_path = save_run_summary(s, "tsk")
                viewer_path = save_viewer_artifact(s, "tsk")
            finally:
                os.chdir(orig_cwd)

        self.assertTrue(summary_path.name.startswith(f"run_summary_{run_id}_"))
        self.assertTrue(viewer_path.name.startswith(f"viewer_artifact_{run_id}_"))

    def test_save_viewer_artifact_embeds_summary_and_traces(self):
        s = _make_run_summary(baseline=BaselineSummaryEntry(trace_path=""))
        baseline_trace = {"status": "completed", "interactions": [], "execution": {}}
        run_traces = [
            {"status": "completed", "interactions": [], "execution": {"run_index": i}}
            for i in range(3)
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            import os

            orig_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                path = save_viewer_artifact(s, "tsk", baseline_trace, run_traces)
                payload = json.loads(gzip.decompress(path.read_bytes()))

                self.assertEqual(path.suffixes[-2:], [".json", ".gz"])
                self.assertEqual(payload["kind"], "viewer_artifact")
                self.assertEqual(payload["summary"]["run_group_id"], "abc123")
                self.assertEqual(payload["baseline_trace"]["status"], "completed")
                self.assertEqual(len(payload["run_traces"]), 3)
                self.assertEqual(
                    payload["run_traces"][0]["trace"]["execution"]["run_index"], 0
                )
            finally:
                os.chdir(orig_cwd)

    def test_load_run_summary_reads_embedded_viewer_artifact_summary(self):
        s = _make_run_summary(run_mode="permute")
        with tempfile.TemporaryDirectory() as tmpdir:
            import os

            orig_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                path = save_viewer_artifact(s, "tsk")
                self.assertEqual(path.suffixes[-2:], [".json", ".gz"])
                loaded = load_run_summary(path)
                self.assertEqual(loaded.run_group_id, s.run_group_id)
                self.assertEqual(loaded.run_mode, "permute")
            finally:
                os.chdir(orig_cwd)

    def test_load_run_summary_reads_legacy_uncompressed_viewer_artifact(self):
        s = _make_run_summary(run_mode="replicate")
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "viewer_artifact_abc123_20260408_120000.json"
            path.write_text(
                json.dumps(
                    {
                        "kind": "viewer_artifact",
                        "summary": s.to_dict(),
                        "baseline_trace": None,
                        "run_traces": [],
                    }
                ),
                encoding="utf-8",
            )

            loaded = load_run_summary(path)

        self.assertEqual(loaded.run_group_id, s.run_group_id)
        self.assertEqual(loaded.run_mode, "replicate")

    def test_load_run_summaries_reads_viewer_artifacts(self):
        s = _make_run_summary(run_mode="resample")
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            traces_dir = repo_root / "results" / "tsk"
            traces_dir.mkdir(parents=True, exist_ok=True)
            payload = {
                "kind": "viewer_artifact",
                "summary": s.to_dict(),
                "baseline_trace": None,
                "run_traces": [],
            }
            with gzip.open(
                traces_dir / "viewer_artifact_abc123_20260408_120000.json.gz",
                "wt",
                encoding="utf-8",
            ) as fh:
                json.dump(payload, fh)

            fake_module_path = repo_root / "src" / "trace_storage.py"
            fake_module_path.parent.mkdir(parents=True, exist_ok=True)
            fake_module_path.write_text("", encoding="utf-8")

            with patch.object(trace_storage, "__file__", str(fake_module_path)):
                loaded = load_run_summaries("tsk")

            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded[0].run_group_id, s.run_group_id)
            self.assertEqual(loaded[0].run_mode, "resample")

    def test_variant_falls_back_to_task_params(self):
        s = _make_run_summary()
        d = s.to_dict()
        del d["variant"]
        restored = RunSummary.from_dict(d)
        self.assertEqual(restored.variant, "v1")

    def test_attach_baseline_to_outcomes_adds_gain(self):
        baseline = [
            {"instance_id": "a", "instance_index": 0, "reward": 1.0},
            {"instance_id": "b", "instance_index": 1, "reward": 0.5},
        ]
        rollout = [
            {"instance_id": "a", "instance_index": 0, "reward": 1.5},
            {"instance_id": "b", "instance_index": 1, "reward": -0.5},
        ]

        enriched = attach_baseline_to_outcomes(baseline, rollout)

        self.assertEqual(enriched[0]["baseline_reward"], 1.0)
        self.assertEqual(enriched[0]["gain"], 0.5)
        self.assertEqual(enriched[1]["gain"], -1.0)

    def test_attach_baseline_to_outcomes_allows_partial_baseline(self):
        baseline = [
            {"instance_id": "a", "instance_index": 0, "reward": 1.0},
        ]
        rollout = [
            {"instance_id": "a", "instance_index": 0, "reward": 1.5},
            {"instance_id": "b", "instance_index": 1, "reward": -0.5},
        ]

        enriched = attach_baseline_to_outcomes(
            baseline,
            rollout,
            allow_partial_baseline=True,
        )

        self.assertEqual(enriched[0]["baseline_reward"], 1.0)
        self.assertEqual(enriched[0]["gain"], 0.5)
        self.assertNotIn("baseline_reward", enriched[1])
        self.assertNotIn("gain", enriched[1])

    def test_attach_baseline_matches_by_instance_id_not_position(self):
        # Rollout plays the same instances in reversed order
        baseline = [
            {"instance_id": "a", "instance_index": 0, "reward": 1.0},
            {"instance_id": "b", "instance_index": 1, "reward": 0.5},
        ]
        rollout = [
            {"instance_id": "b", "instance_index": 0, "reward": 0.8},
            {"instance_id": "a", "instance_index": 1, "reward": 1.2},
        ]

        enriched = attach_baseline_to_outcomes(baseline, rollout)

        # "b" at position 0 should compare to baseline "b" (reward 0.5), not "a" (1.0)
        self.assertEqual(enriched[0]["baseline_reward"], 0.5)
        self.assertAlmostEqual(enriched[0]["gain"], 0.3)
        # "a" at position 1 should compare to baseline "a" (reward 1.0)
        self.assertEqual(enriched[1]["baseline_reward"], 1.0)
        self.assertAlmostEqual(enriched[1]["gain"], 0.2)

    def test_attach_baseline_to_outcomes_adds_cost_increase(self):
        baseline = [
            {"instance_id": "a", "instance_index": 0, "reward": 1.0, "cost_usd": 0.3},
            {"instance_id": "b", "instance_index": 1, "reward": 0.5, "cost_usd": 0.2},
        ]
        rollout = [
            {"instance_id": "a", "instance_index": 0, "reward": 1.5, "cost_usd": 0.7},
            {"instance_id": "b", "instance_index": 1, "reward": -0.5, "cost_usd": 0.1},
        ]

        enriched = attach_baseline_to_outcomes(baseline, rollout)

        self.assertEqual(enriched[0]["baseline_cost"], 0.3)
        self.assertEqual(enriched[0]["cost_increase"], 0.4)
        self.assertEqual(enriched[1]["baseline_cost"], 0.2)
        self.assertEqual(enriched[1]["cost_increase"], -0.1)

    def test_attach_baseline_to_outcomes_skips_cost_increase_without_baseline_cost(
        self,
    ):
        baseline = [
            {"instance_id": "a", "instance_index": 0, "reward": 1.0},
            {"instance_id": "b", "instance_index": 1, "reward": 0.5, "cost_usd": 0.2},
        ]
        rollout = [
            {"instance_id": "a", "instance_index": 0, "reward": 1.5, "cost_usd": 0.7},
            {"instance_id": "b", "instance_index": 1, "reward": -0.5, "cost_usd": 0.1},
        ]

        enriched = attach_baseline_to_outcomes(baseline, rollout)

        self.assertNotIn("baseline_cost", enriched[0])
        self.assertNotIn("cost_increase", enriched[0])
        self.assertEqual(enriched[1]["baseline_cost"], 0.2)
        self.assertEqual(enriched[1]["cost_increase"], -0.1)

    def test_summarize_instance_outcomes_includes_cost_rollups(self):
        outcomes = [
            {
                "instance_id": "a",
                "instance_index": 0,
                "reward": 1.0,
                "cost_usd": 0.3,
                "gain": 0.5,
                "cost_increase": 0.1,
            },
            {
                "instance_id": "b",
                "instance_index": 1,
                "reward": 0.5,
                "cost_usd": 0.1,
                "gain": -0.2,
            },
        ]

        summary = summarize_instance_outcomes(outcomes)

        self.assertEqual(summary["total_cost"], 0.4)
        self.assertEqual(summary["mean_cost"], 0.2)
        self.assertEqual(summary["cost_by_index"], [0.3, 0.1])
        self.assertEqual(summary["total_cost_increase"], 0.1)
        self.assertEqual(summary["mean_cost_increase"], 0.1)
        self.assertEqual(summary["cost_increase_by_index"], [0.1, None])

    def test_summarize_instance_outcomes_reports_partial_gain_coverage(self):
        outcomes = [
            {
                "instance_id": "a",
                "instance_index": 0,
                "reward": 1.0,
                "gain": 0.5,
            },
            {
                "instance_id": "b",
                "instance_index": 1,
                "reward": 0.5,
            },
        ]

        summary = summarize_instance_outcomes(outcomes)

        self.assertEqual(summary["total_gain"], 0.5)
        self.assertEqual(summary["mean_gain"], 0.5)
        self.assertEqual(summary["gain_by_index"], [0.5, None])
        self.assertEqual(summary["gain_instances_matched"], 1)
        self.assertEqual(summary["gain_coverage_ratio"], 0.5)

    def test_build_benchmark_aggregate_gain_is_instance_matched_across_orderings(self):
        # Baseline plays [a, b]; rollout plays [b, a] — gain must still be per-instance
        baseline = [
            {"instance_id": "a", "instance_index": 0, "reward": 1.0},
            {"instance_id": "b", "instance_index": 1, "reward": 0.0},
        ]
        runs = [
            [
                {"instance_id": "b", "instance_index": 0, "reward": 1.0},
                {"instance_id": "a", "instance_index": 1, "reward": 2.0},
            ],
        ]

        aggregate = build_benchmark_aggregate(baseline, runs)

        # gain[0] = b_reward(1.0) - baseline_b(0.0) = 1.0
        # gain[1] = a_reward(2.0) - baseline_a(1.0) = 1.0
        self.assertEqual(aggregate["mean_gain_by_index"], [1.0, 1.0])

    def test_build_benchmark_aggregate_computes_reward_and_gain_curves(self):
        baseline = [
            {"instance_id": "a", "instance_index": 0, "reward": 1.0},
            {"instance_id": "b", "instance_index": 1, "reward": 0.0},
        ]
        runs = [
            [
                {"instance_id": "a", "instance_index": 0, "reward": 2.0},
                {"instance_id": "b", "instance_index": 1, "reward": 1.0},
            ],
            [
                {"instance_id": "a", "instance_index": 0, "reward": 0.0},
                {"instance_id": "b", "instance_index": 1, "reward": 3.0},
            ],
        ]

        aggregate = build_benchmark_aggregate(baseline, runs)

        self.assertEqual(aggregate["baseline_reward_by_index"], [1.0, 0.0])
        self.assertEqual(aggregate["mean_reward_by_index"], [1.0, 2.0])
        self.assertEqual(aggregate["mean_gain_by_index"], [0.0, 2.0])
        self.assertEqual(aggregate["cumulative_mean_gain_by_index"], [0.0, 1.0])
        self.assertEqual(aggregate["final_cumulative_mean_gain"], 1.0)

    def test_build_benchmark_aggregate_allows_partial_baseline(self):
        baseline = [
            {"instance_id": "a", "instance_index": 0, "reward": 1.0, "cost_usd": 0.1},
        ]
        runs = [
            [
                {
                    "instance_id": "a",
                    "instance_index": 0,
                    "reward": 3.0,
                    "cost_usd": 0.4,
                },
                {
                    "instance_id": "b",
                    "instance_index": 1,
                    "reward": 5.0,
                    "cost_usd": 0.8,
                },
            ],
        ]

        aggregate = build_benchmark_aggregate(
            baseline,
            runs,
            allow_partial_baseline=True,
        )

        self.assertEqual(aggregate["baseline_reward_by_index"], [1.0, None])
        self.assertEqual(aggregate["baseline_cost_by_index"], [0.1, None])
        self.assertEqual(aggregate["mean_reward_by_index"], [3.0, 5.0])
        self.assertEqual(aggregate["mean_gain_by_index"], [2.0, None])
        self.assertEqual(aggregate["mean_cost_increase_by_index"], [0.3, None])
        self.assertEqual(aggregate["cumulative_mean_gain_by_index"], [2.0, 2.0])
        self.assertEqual(aggregate["final_cumulative_mean_gain"], 2.0)

    def test_build_benchmark_aggregate_cost_present_on_all_runs(self):
        baseline = [
            {"instance_id": "a", "instance_index": 0, "reward": 0.5, "cost_usd": 0.05},
            {"instance_id": "b", "instance_index": 1, "reward": 0.3, "cost_usd": 0.15},
            {"instance_id": "c", "instance_index": 2, "reward": 0.1, "cost_usd": 0.25},
        ]
        runs = [
            [
                {
                    "instance_id": "a",
                    "instance_index": 0,
                    "reward": 0.8,
                    "cost_usd": 0.10,
                },
                {
                    "instance_id": "b",
                    "instance_index": 1,
                    "reward": 0.6,
                    "cost_usd": 0.20,
                },
                {
                    "instance_id": "c",
                    "instance_index": 2,
                    "reward": 0.4,
                    "cost_usd": 0.30,
                },
            ],
            [
                {
                    "instance_id": "a",
                    "instance_index": 0,
                    "reward": 0.6,
                    "cost_usd": 0.20,
                },
                {
                    "instance_id": "b",
                    "instance_index": 1,
                    "reward": 0.4,
                    "cost_usd": 0.40,
                },
                {
                    "instance_id": "c",
                    "instance_index": 2,
                    "reward": 0.2,
                    "cost_usd": 0.10,
                },
            ],
        ]

        aggregate = build_benchmark_aggregate(baseline, runs)

        self.assertEqual(aggregate["baseline_cost_by_index"], [0.05, 0.15, 0.25])
        # mean_cost[0] = (0.10 + 0.20) / 2 = 0.15
        # mean_cost[1] = (0.20 + 0.40) / 2 = 0.30
        # mean_cost[2] = (0.30 + 0.10) / 2 = 0.20
        self.assertEqual(aggregate["mean_cost_by_index"], [0.15, 0.3, 0.2])
        self.assertEqual(aggregate["mean_cost_increase_by_index"], [0.1, 0.15, -0.05])

    def test_build_benchmark_aggregate_cost_missing_entirely(self):
        baseline = [
            {"instance_id": "a", "instance_index": 0, "reward": 0.5},
            {"instance_id": "b", "instance_index": 1, "reward": 0.3},
        ]
        runs = [
            [
                {"instance_id": "a", "instance_index": 0, "reward": 0.8},
                {"instance_id": "b", "instance_index": 1, "reward": 0.6},
            ],
        ]

        aggregate = build_benchmark_aggregate(baseline, runs)

        self.assertEqual(aggregate["baseline_cost_by_index"], [])
        self.assertEqual(aggregate["mean_cost_by_index"], [])
        self.assertEqual(aggregate["mean_cost_increase_by_index"], [])

    def test_build_benchmark_aggregate_cost_partial_across_runs(self):
        baseline = [
            {"instance_id": "a", "instance_index": 0, "reward": 0.5, "cost_usd": 0.05},
            {"instance_id": "b", "instance_index": 1, "reward": 0.3},
        ]
        runs = [
            [
                {
                    "instance_id": "a",
                    "instance_index": 0,
                    "reward": 0.8,
                    "cost_usd": 0.10,
                },
                {
                    "instance_id": "b",
                    "instance_index": 1,
                    "reward": 0.6,
                    "cost_usd": 0.20,
                },
            ],
            [
                {"instance_id": "a", "instance_index": 0, "reward": 0.6},
                {"instance_id": "b", "instance_index": 1, "reward": 0.4},
            ],
        ]

        aggregate = build_benchmark_aggregate(baseline, runs)

        # Run 1 has costs, run 2 doesn't — averages only over non-None
        # mean_cost[0] = 0.10 (only run 1)
        # mean_cost[1] = 0.20 (only run 1)
        self.assertEqual(aggregate["baseline_cost_by_index"], [0.05, None])
        self.assertEqual(aggregate["mean_cost_by_index"], [0.1, 0.2])
        self.assertEqual(aggregate["mean_cost_increase_by_index"], [0.05, None])

    def test_build_benchmark_aggregate_cost_none_at_specific_indices(self):
        baseline = [
            {"instance_id": "a", "instance_index": 0, "reward": 0.5},
            {"instance_id": "b", "instance_index": 1, "reward": 0.3},
        ]
        runs = [
            [
                {
                    "instance_id": "a",
                    "instance_index": 0,
                    "reward": 0.8,
                    "cost_usd": 0.10,
                },
                {"instance_id": "b", "instance_index": 1, "reward": 0.6},
            ],
        ]

        aggregate = build_benchmark_aggregate(baseline, runs)

        # Index 0 has cost, index 1 does not
        self.assertEqual(aggregate["mean_cost_by_index"][0], 0.1)
        self.assertIsNone(aggregate["mean_cost_by_index"][1])

    def test_build_benchmark_aggregate_latency_present_on_all_runs(self):
        baseline = [
            {"instance_id": "a", "instance_index": 0, "reward": 0.5},
            {"instance_id": "b", "instance_index": 1, "reward": 0.3},
            {"instance_id": "c", "instance_index": 2, "reward": 0.1},
        ]
        runs = [
            [
                {
                    "instance_id": "a",
                    "instance_index": 0,
                    "reward": 0.8,
                    "latency_seconds": 1.0,
                },
                {
                    "instance_id": "b",
                    "instance_index": 1,
                    "reward": 0.6,
                    "latency_seconds": 2.0,
                },
                {
                    "instance_id": "c",
                    "instance_index": 2,
                    "reward": 0.4,
                    "latency_seconds": 3.0,
                },
            ],
            [
                {
                    "instance_id": "a",
                    "instance_index": 0,
                    "reward": 0.6,
                    "latency_seconds": 3.0,
                },
                {
                    "instance_id": "b",
                    "instance_index": 1,
                    "reward": 0.4,
                    "latency_seconds": 4.0,
                },
                {
                    "instance_id": "c",
                    "instance_index": 2,
                    "reward": 0.2,
                    "latency_seconds": 1.0,
                },
            ],
        ]

        aggregate = build_benchmark_aggregate(baseline, runs)

        # mean_latency[0] = (1.0 + 3.0) / 2 = 2.0
        # mean_latency[1] = (2.0 + 4.0) / 2 = 3.0
        # mean_latency[2] = (3.0 + 1.0) / 2 = 2.0
        self.assertEqual(aggregate["mean_latency_by_index"], [2.0, 3.0, 2.0])

    def test_build_benchmark_aggregate_latency_missing_entirely(self):
        baseline = [
            {"instance_id": "a", "instance_index": 0, "reward": 0.5},
        ]
        runs = [
            [{"instance_id": "a", "instance_index": 0, "reward": 0.8}],
        ]

        aggregate = build_benchmark_aggregate(baseline, runs)

        self.assertEqual(aggregate["mean_latency_by_index"], [])

    def test_build_benchmark_aggregate_latency_partial_across_runs(self):
        baseline = [
            {"instance_id": "a", "instance_index": 0, "reward": 0.5},
            {"instance_id": "b", "instance_index": 1, "reward": 0.3},
        ]
        runs = [
            [
                {
                    "instance_id": "a",
                    "instance_index": 0,
                    "reward": 0.8,
                    "latency_seconds": 1.2,
                },
                {
                    "instance_id": "b",
                    "instance_index": 1,
                    "reward": 0.6,
                    "latency_seconds": 2.4,
                },
            ],
            [
                {"instance_id": "a", "instance_index": 0, "reward": 0.6},
                {"instance_id": "b", "instance_index": 1, "reward": 0.4},
            ],
        ]

        aggregate = build_benchmark_aggregate(baseline, runs)

        # Only run 1 has latencies — averages only over non-None
        self.assertEqual(aggregate["mean_latency_by_index"], [1.2, 2.4])

    def test_build_benchmark_aggregate_latency_none_at_specific_indices(self):
        baseline = [
            {"instance_id": "a", "instance_index": 0, "reward": 0.5},
            {"instance_id": "b", "instance_index": 1, "reward": 0.3},
        ]
        runs = [
            [
                {"instance_id": "a", "instance_index": 0, "reward": 0.8},
                {
                    "instance_id": "b",
                    "instance_index": 1,
                    "reward": 0.6,
                    "latency_seconds": 3.0,
                },
            ],
        ]

        aggregate = build_benchmark_aggregate(baseline, runs)

        self.assertIsNone(aggregate["mean_latency_by_index"][0])
        self.assertEqual(aggregate["mean_latency_by_index"][1], 3.0)

    def test_build_benchmark_aggregate_mixed_cost_and_latency(self):
        """Both cost and latency present with different missing patterns."""
        baseline = [
            {"instance_id": "a", "instance_index": 0, "reward": 0.5},
            {"instance_id": "b", "instance_index": 1, "reward": 0.3},
        ]
        runs = [
            [
                {
                    "instance_id": "a",
                    "instance_index": 0,
                    "reward": 0.8,
                    "cost_usd": 0.10,
                    "latency_seconds": 1.0,
                },
                {
                    "instance_id": "b",
                    "instance_index": 1,
                    "reward": 0.6,
                    "latency_seconds": 2.0,
                },
            ],
            [
                {
                    "instance_id": "a",
                    "instance_index": 0,
                    "reward": 0.6,
                    "cost_usd": 0.30,
                },
                {
                    "instance_id": "b",
                    "instance_index": 1,
                    "reward": 0.4,
                    "cost_usd": 0.20,
                    "latency_seconds": 4.0,
                },
            ],
        ]

        aggregate = build_benchmark_aggregate(baseline, runs)

        # cost[0]: (0.10 + 0.30) / 2 = 0.20, cost[1]: 0.20 (only run 2)
        self.assertEqual(aggregate["mean_cost_by_index"][0], 0.2)
        self.assertEqual(aggregate["mean_cost_by_index"][1], 0.2)
        # latency[0]: 1.0 (only run 1), latency[1]: (2.0 + 4.0) / 2 = 3.0
        self.assertEqual(aggregate["mean_latency_by_index"][0], 1.0)
        self.assertEqual(aggregate["mean_latency_by_index"][1], 3.0)

    def test_build_benchmark_aggregate_no_cost_no_latency_returns_empty_lists(self):
        baseline = [
            {"instance_id": "a", "instance_index": 0, "reward": 0.5},
        ]
        runs = [
            [{"instance_id": "a", "instance_index": 0, "reward": 0.8}],
        ]

        aggregate = build_benchmark_aggregate(baseline, runs)

        self.assertEqual(aggregate["mean_cost_by_index"], [])
        self.assertEqual(aggregate["mean_latency_by_index"], [])

    def test_trace_recorder_attaches_cost_and_latency_to_outcomes(self):
        recorder = TraceRecorder(
            system_name="sys",
            task_name="task",
            system_params={},
            task_params={},
        )
        for i in range(3):
            recorder.interactions.append(
                InteractionStep(
                    step_number=i,
                    timestamp="2026-01-01T00:00:00",
                    query={"instance_id": f"inst-{i}", "instance_index": i},
                    response={},
                    observation={},
                    timing={"response_latency_seconds": 0.5 + i * 0.1},
                    usage={"interaction": {"cost_usd": 0.01 * (i + 1)}},
                    done=(i == 2),
                )
            )
            recorder.record_instance_outcome(
                InstanceOutcome(
                    instance_id=f"inst-{i}",
                    instance_index=i,
                    reward=float(i) / 2,
                )
            )

        result = recorder._build_trace_payload(result=None)
        outcomes = result["instance_outcomes"]

        self.assertEqual(len(outcomes), 3)
        self.assertAlmostEqual(outcomes[0]["cost_usd"], 0.01)
        self.assertAlmostEqual(outcomes[1]["cost_usd"], 0.02)
        self.assertAlmostEqual(outcomes[2]["cost_usd"], 0.03)
        self.assertAlmostEqual(outcomes[0]["latency_seconds"], 0.5)
        self.assertAlmostEqual(outcomes[1]["latency_seconds"], 0.6)
        self.assertAlmostEqual(outcomes[2]["latency_seconds"], 0.7)

    def test_trace_recorder_aggregates_metrics_across_interactions_per_instance(self):
        recorder = TraceRecorder(
            system_name="sys",
            task_name="task",
            system_params={},
            task_params={},
        )
        recorder.interactions.extend(
            [
                InteractionStep(
                    step_number=0,
                    timestamp="2026-01-01T00:00:00",
                    query={"instance_id": "inst-0", "instance_index": 0},
                    response={},
                    observation={},
                    timing={"response_latency_seconds": 0.5},
                    usage={"interaction": {"cost_usd": 0.01}},
                    done=False,
                ),
                InteractionStep(
                    step_number=1,
                    timestamp="2026-01-01T00:00:01",
                    query={"instance_id": "inst-0", "instance_index": 0},
                    response={},
                    observation={},
                    timing={"response_latency_seconds": 0.7},
                    usage={"interaction": {"cost_usd": 0.02}},
                    done=True,
                ),
                InteractionStep(
                    step_number=2,
                    timestamp="2026-01-01T00:00:02",
                    query={"instance_id": "inst-1", "instance_index": 1},
                    response={},
                    observation={},
                    timing={"response_latency_seconds": 1.1},
                    usage={"interaction": {"cost_usd": 0.03}},
                    done=True,
                ),
            ]
        )
        recorder.record_instance_outcome(
            InstanceOutcome(instance_id="inst-0", instance_index=0, reward=0.5)
        )
        recorder.record_instance_outcome(
            InstanceOutcome(instance_id="inst-1", instance_index=1, reward=1.0)
        )

        result = recorder._build_trace_payload(result=None)
        outcomes = result["instance_outcomes"]

        self.assertAlmostEqual(outcomes[0]["cost_usd"], 0.03)
        self.assertAlmostEqual(outcomes[0]["latency_seconds"], 1.2)
        self.assertAlmostEqual(outcomes[1]["cost_usd"], 0.03)
        self.assertAlmostEqual(outcomes[1]["latency_seconds"], 1.1)

    def test_trace_recorder_matches_metrics_by_instance_identity_not_order(self):
        recorder = TraceRecorder(
            system_name="sys",
            task_name="task",
            system_params={},
            task_params={},
        )
        recorder.interactions.extend(
            [
                InteractionStep(
                    step_number=0,
                    timestamp="2026-01-01T00:00:00",
                    query={"instance_id": "inst-1", "instance_index": 1},
                    response={},
                    observation={},
                    timing={"response_latency_seconds": 1.0},
                    usage={"interaction": {"cost_usd": 0.04}},
                    done=True,
                ),
                InteractionStep(
                    step_number=1,
                    timestamp="2026-01-01T00:00:01",
                    query={"instance_id": "inst-0", "instance_index": 0},
                    response={},
                    observation={},
                    timing={"response_latency_seconds": 0.25},
                    usage={"interaction": {"cost_usd": 0.01}},
                    done=False,
                ),
                InteractionStep(
                    step_number=2,
                    timestamp="2026-01-01T00:00:02",
                    query={"instance_id": "inst-0", "instance_index": 0},
                    response={},
                    observation={},
                    timing={"response_latency_seconds": 0.75},
                    usage={"interaction": {"cost_usd": 0.02}},
                    done=True,
                ),
            ]
        )
        recorder.record_instance_outcome(
            InstanceOutcome(instance_id="inst-0", instance_index=0, reward=0.5)
        )
        recorder.record_instance_outcome(
            InstanceOutcome(instance_id="inst-1", instance_index=1, reward=1.0)
        )

        result = recorder._build_trace_payload(result=None)
        outcomes = result["instance_outcomes"]

        self.assertAlmostEqual(outcomes[0]["cost_usd"], 0.03)
        self.assertAlmostEqual(outcomes[0]["latency_seconds"], 1.0)
        self.assertAlmostEqual(outcomes[1]["cost_usd"], 0.04)
        self.assertAlmostEqual(outcomes[1]["latency_seconds"], 1.0)

    def test_trace_recorder_omits_metrics_when_not_available(self):
        recorder = TraceRecorder(
            system_name="sys",
            task_name="task",
            system_params={},
            task_params={},
        )
        recorder.interactions.append(
            InteractionStep(
                step_number=0,
                timestamp="2026-01-01T00:00:00",
                query={"instance_id": "inst-0", "instance_index": 0},
                response={},
                observation={},
                timing={},
                usage={},
                done=True,
            )
        )
        recorder.record_instance_outcome(
            InstanceOutcome(
                instance_id="inst-0",
                instance_index=0,
                reward=0.5,
            )
        )

        result = recorder._build_trace_payload(result=None)
        outcomes = result["instance_outcomes"]

        self.assertNotIn("cost_usd", outcomes[0])
        self.assertNotIn("latency_seconds", outcomes[0])

    def test_trace_recorder_partial_metrics(self):
        """Cost present but latency missing, and vice versa."""
        recorder = TraceRecorder(
            system_name="sys",
            task_name="task",
            system_params={},
            task_params={},
        )
        recorder.interactions.append(
            InteractionStep(
                step_number=0,
                timestamp="2026-01-01T00:00:00",
                query={"instance_id": "inst-0", "instance_index": 0},
                response={},
                observation={},
                timing={"response_latency_seconds": 1.5},
                usage={},
                done=False,
            )
        )
        recorder.interactions.append(
            InteractionStep(
                step_number=1,
                timestamp="2026-01-01T00:00:01",
                query={"instance_id": "inst-1", "instance_index": 1},
                response={},
                observation={},
                timing={},
                usage={"interaction": {"cost_usd": 0.05}},
                done=True,
            )
        )
        for i in range(2):
            recorder.record_instance_outcome(
                InstanceOutcome(
                    instance_id=f"inst-{i}",
                    instance_index=i,
                    reward=0.5,
                )
            )

        result = recorder._build_trace_payload(result=None)
        outcomes = result["instance_outcomes"]

        self.assertAlmostEqual(outcomes[0]["latency_seconds"], 1.5)
        self.assertNotIn("cost_usd", outcomes[0])
        self.assertAlmostEqual(outcomes[1]["cost_usd"], 0.05)
        self.assertNotIn("latency_seconds", outcomes[1])

    def test_trace_recorder_writes_live_snapshot(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            live_path = Path(tmpdir) / "live.json"
            recorder = TraceRecorder(
                system_name="sys",
                task_name="task",
                system_params={},
                task_params={},
                live_trace_path=live_path,
            )

            recorder.record_instance_outcome(
                InstanceOutcome(
                    instance_id="inst-1",
                    instance_index=0,
                    reward=1.0,
                    success=True,
                )
            )

            self.assertTrue(live_path.exists())
            payload = json.loads(live_path.read_text())
            self.assertEqual(payload["status"], "running")
            self.assertEqual(len(payload["instance_outcomes"]), 1)

    def test_trace_recorder_finalize_updates_live_snapshot_with_result_outcomes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            live_path = Path(tmpdir) / "live.json"
            recorder = TraceRecorder(
                system_name="sys",
                task_name="task",
                system_params={},
                task_params={},
                live_trace_path=live_path,
            )

            recorder.record_instance_outcome(
                InstanceOutcome(
                    instance_id="stale-inst",
                    instance_index=0,
                    reward=0.5,
                    success=True,
                )
            )

            result = TaskResult(
                metrics={"hands_played": 2},
                summary="done",
                eval_metrics=EvalMetrics(
                    loss_curve=[],
                    optimal_performance=2.0,
                    actual_performance=2.0,
                ),
                instance_outcomes=[
                    InstanceOutcome(
                        instance_id="inst-1",
                        instance_index=0,
                        reward=1.0,
                        success=True,
                    ),
                    InstanceOutcome(
                        instance_id="inst-2",
                        instance_index=1,
                        reward=2.0,
                        success=True,
                    ),
                ],
            )

            recorder.finalize(result)

            payload = json.loads(live_path.read_text())
            self.assertEqual(payload["status"], "completed")
            self.assertEqual(
                [outcome["instance_id"] for outcome in payload["instance_outcomes"]],
                ["inst-1", "inst-2"],
            )
            self.assertEqual(payload["result"]["metrics"]["hands_played"], 2)

    def test_trace_recorder_finalize_blocked_writes_extra_to_live_snapshot(self):
        """Blocked-run extras (e.g. ``blocked_reason``) must reach the live JSON
        so the dashboard can render the refusal context, not just the status."""
        with tempfile.TemporaryDirectory() as tmpdir:
            live_path = Path(tmpdir) / "live.json"
            recorder = TraceRecorder(
                system_name="sys",
                task_name="task",
                system_params={},
                task_params={},
                live_trace_path=live_path,
            )

            result = TaskResult(
                metrics={"blocked": True},
                summary="blocked",
                eval_metrics=EvalMetrics(
                    loss_curve=[], optimal_performance=0.0, actual_performance=0.0
                ),
                instance_outcomes=[],
            )
            blocked_reason = {"kind": "content_policy", "message": "refused"}

            returned = recorder.finalize(
                result,
                status="blocked",
                extra={"blocked_reason": blocked_reason},
            )

            self.assertEqual(returned["status"], "blocked")
            self.assertEqual(returned["blocked_reason"], blocked_reason)

            live_payload = json.loads(live_path.read_text())
            self.assertEqual(live_payload["status"], "blocked")
            self.assertEqual(live_payload["blocked_reason"], blocked_reason)


if __name__ == "__main__":
    unittest.main()
