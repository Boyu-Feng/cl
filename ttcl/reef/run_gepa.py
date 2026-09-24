"""Run Reef's original GEPA proposer/archive/selector on CLBench episodes.

The adapter replaces only the episode runner and model transport. This is a
method-level integration, not a test of the Reef HTTP service or deployment.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import traceback
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [
    str(ROOT),
    str(ROOT / "current_work/reef"),
    str(ROOT / "ttcl/.runtime/structured_memory_deps"),
]

from recipes.gepa.archive import Archive  # noqa: E402
from recipes.gepa.method import (  # noqa: E402
    EpisodeRunner,
    GEPAProposer,
    GEPASelectorMixin,
    ScoreFeedback,
)
from reef.core.evaluation import EvaluationResult, UpdateCandidate  # noqa: E402
from reef.harness.adapters import get_adapter  # noqa: E402
from reef.harness.episodes.model_binding import ModelBinding, ModelBindings  # noqa: E402
from reef.harness.episodes.run import EpisodeResult  # noqa: E402
from reef.train.cordis_backend.strategies import EpisodeScorer  # noqa: E402
from reef.train.types import TrajectoryItem  # noqa: E402
from ttcl.common.local_qwen import LocalQwen  # noqa: E402
from ttcl.structured_memory.online_bank import run_episode  # noqa: E402
from ttcl.structured_memory.run_benchmark import append, make_task, write_json  # noqa: E402


def upstream_metadata(repo, workspace):
    """Read the original source revision after vendoring without nested Git."""
    manifest = workspace / "config/upstreams.json"
    if manifest.is_file():
        repositories = json.loads(manifest.read_text())["repositories"]
        entry = repositories.get(repo.name)
        if entry and entry.get("url") and entry.get("commit"):
            return {"url": entry["url"], "commit": entry["commit"]}
    # Git otherwise searches parent directories and could report this project's
    # revision instead of the upstream. Require metadata at the upstream root.
    if not (repo / ".git").exists():
        raise FileNotFoundError(
            f"Missing original upstream revision for {repo.name}; "
            f"restore its entry in {manifest}"
        )
    git_root = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "--show-toplevel"], text=True
    ).strip()
    if Path(git_root).resolve() != repo.resolve():
        raise ValueError(f"Git metadata does not belong to upstream {repo}")
    return {
        "url": subprocess.check_output(
            ["git", "-C", str(repo), "remote", "get-url", "origin"], text=True
        ).strip(),
        "commit": subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
        ).strip(),
    }


def public_output(episode):
    # The official episode helper contains public tool observations and the
    # terminal scalar, but no private grader metadata or future questions.
    return json.dumps(episode, ensure_ascii=False)


def trajectory_item(prompt, record, episode):
    output = public_output(episode)
    return TrajectoryItem(
        {
            "schema_version": "ATIF-v1.7",
            "agent": {"name": "clbench-qwen"},
            "steps": [
                {"step_id": 1, "source": "user", "message": prompt},
                {"step_id": 2, "source": "agent", "message": output},
            ],
            "extra": {
                "reef": {
                    "reward": record["reward"],
                    "records": [
                        {
                            "payload": {
                                "messages": [{"role": "user", "content": prompt}],
                                "response": {
                                    "choices": [
                                        {
                                            "message": {
                                                "role": "assistant",
                                                "content": output,
                                            }
                                        }
                                    ]
                                },
                            }
                        }
                    ],
                }
            },
        }
    )


@dataclass(frozen=True)
class LocalBinding(ModelBinding):
    backend: object = None
    log_path: Path = Path("reflection.jsonl")
    seed: int = 812

    def chat(self, messages, *, timeout_s=None, **params):
        result = self.backend.generate(messages, self.seed, temperature=0.0)
        append(self.log_path, {"messages": messages, **result})
        return result["raw_response"]


class OfficialScorer(EpisodeScorer):
    def __call__(self, task, result):
        return float(json.loads(result.stdout)["reward"])


class OfficialSelector(GEPASelectorMixin):
    def __init__(self, archive, runner, validation_indices):
        super().__init__(archive)
        self.archive, self.runner, self.validation_indices = (
            archive,
            runner,
            validation_indices,
        )

    def evaluate(self, candidate):
        context = self.archive.candidates[int(candidate.candidate_id)].texts["rules"]
        scores = [
            self.runner.execute(i, context, "candidate_validation")[0]["reward"]
            for i in self.validation_indices
        ]
        return EvaluationResult(
            "clbench",
            "1",
            {
                "candidate_scores": scores,
                "current_scores": self.archive.candidates[
                    self.archive.served
                ].val_scores,
            },
        )


class CLBenchRunner(EpisodeRunner):
    def __init__(self, args, model, output):
        self.args, self.model, self.output = args, model, output
        self.prompts = {}
        self.count = 0
        self.phase = "development"
        self.records = []

    def prompt(self, index):
        task = make_task(self.args.task, self.args.seed, independent=True)
        try:
            query = task.reset_baseline_instance(index)
            replay_id = hashlib.sha256(
                f"{self.args.task}:{index}".encode()
            ).hexdigest()[:12]
            prompt = f"Replay reference: {replay_id}\n\n{query.prompt}"
        finally:
            connection = getattr(task, "_conn", None)
            if connection is not None:
                connection.close()
        if prompt in self.prompts and self.prompts[prompt] != index:
            raise ValueError(
                "Duplicate initial task prompts require distinct replay identifiers"
            )
        self.prompts[prompt] = index
        return prompt

    def execute(self, index, context, label):
        self.count += 1
        directory = self.output / "episodes" / f"{self.count:04d}_{label}_{index:03d}"
        record, episode = run_episode(self.args, self.model, index, directory, context)
        record.update(phase=self.phase, label=label, directory=str(directory))
        self.records.append(record)
        append(self.output / "episodes.jsonl", record)
        print(
            json.dumps(
                {
                    k: record[k]
                    for k in ["phase", "label", "canonical_index", "reward", "status"]
                }
            ),
            flush=True,
        )
        if record["status"] != "complete":
            raise RuntimeError(
                f"Incomplete official episode: {directory}; missing reward is not zero"
            )
        return record, episode

    def run(
        self, descriptor, files, prompt, *, binary=None, timeout=600.0, executor=None
    ):
        context = files.get("native/RULES.md", "").strip()
        record, episode = self.execute(self.prompts[prompt], context, "candidate")
        return EpisodeResult(
            0,
            json.dumps(record),
            "",
            ({"role": "assistant", "content": public_output(episode)},),
            (),
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--task",
        choices=["database_exploration", "cohort_studies"],
        default="database_exploration",
    )
    parser.add_argument(
        "--model",
        default=str(ROOT / "current_work/delta-Mem/model/Qwen3-4B-Instruct-2507"),
    )
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--test-count", type=int, default=8)
    parser.add_argument("--preflight", action="store_true")
    cli = parser.parse_args()
    if cli.rounds < 1 or not 1 <= cli.test_count <= 8:
        parser.error("rounds must be positive; test-count must be 1..8")
    args = SimpleNamespace(
        model=cli.model,
        task=cli.task,
        seed=42,
        device="cuda:0",
        dtype="bfloat16",
        temperature=0.7,
        top_p=0.9,
        top_k=0,
        max_new_tokens=4096,
        context_limit=65536,
        memory_chars=16000,
        max_turns_per_instance=64,
        action_retries=2,
        normalize_action=True,
        allow_initial_experience=True,
        num_instances=1,
    )
    train_indices, validation_indices = [0, 1, 2], [3, 4, 5]
    test_indices = list(range(12, 12 + cli.test_count))
    cli.output = cli.output.resolve()
    cli.output.mkdir(parents=True, exist_ok=False)
    os.chdir(ROOT / "current_work/continual-learning-bench")
    upstream = upstream_metadata(ROOT / "current_work/reef", ROOT)
    plan = {
        "method": "REEF-GEPA",
        "integration": "original GEPA method classes; CLBench episode runner; direct local model transport",
        "task": cli.task,
        "model": vars(args),
        "rounds": cli.rounds,
        "train_indices": train_indices,
        "validation_indices": validation_indices,
        "test_indices": test_indices,
        "test_updates": False,
        "scope": "development-prefix prompt optimization followed by frozen held-out evaluation; not online weight training",
        "reef_commit": upstream["commit"],
        "reef_url": upstream["url"],
    }
    write_json(cli.output / "plan.json", plan)
    try:
        runner = CLBenchRunner(args, None, cli.output)
        train_prompts = [runner.prompt(i) for i in train_indices]
        for index in validation_indices:
            runner.prompt(index)
        if cli.preflight:
            write_json(
                cli.output / "status.json",
                {"status": "preflight_passed", "instances_checked": 6},
            )
            print(
                "Preflight passed: original Reef imports and six official task resets"
            )
            return
        model = LocalQwen(args)
        runner.model = model
        binding = LocalBinding(
            base_url="http://unused-local-binding",
            model=cli.model,
            backend=model,
            log_path=cli.output / "reflection.jsonl",
        )
        models = ModelBindings(served=binding, named={"reflection": binding})
        archive = Archive(cli.output / "archive.json")
        initial = archive.seed({"rules": ""})
        seed_scores = [
            runner.execute(i, "", "seed_validation")[0]["reward"]
            for i in validation_indices
        ]
        archive.record_validation(initial, seed_scores)
        archive.charge(len(validation_indices))
        proposer = GEPAProposer(
            archive=archive,
            descriptor=get_adapter("native"),
            binary=None,
            score_episode=OfficialScorer(),
            feedback=ScoreFeedback(),
            minibatch_size=3,
            rng_seed=812,
            skip_perfect_score=True,
            perfect_score=1.0,
            max_metric_calls=None,
            kinds=("rules",),
            valset_size=len(validation_indices),
            episode_runner=runner,
        )
        selector = OfficialSelector(archive, runner, validation_indices)
        for round_index in range(cli.rounds):
            context = archive.candidates[archive.served].texts["rules"]
            samples = []
            for index, prompt in zip(train_indices, train_prompts):
                record, episode = runner.execute(
                    index, context, f"round_{round_index}_train"
                )
                samples.append(trajectory_item(prompt, record, episode))
            mutations = proposer(
                (("rules", {"text": context}),), tuple(samples), models
            )
            if mutations is not None:
                pending = archive.pending
                candidate = UpdateCandidate(str(pending))
                decision = selector.decide(candidate, selector.evaluate(candidate))
                append(
                    cli.output / "decisions.jsonl",
                    {
                        "round": round_index,
                        "outcome": decision.outcome,
                        "reason": decision.reason,
                    },
                )
        context = archive.candidates[archive.served].texts["rules"]
        (cli.output / "selected_rules.md").write_text(context)
        selected_hash = hashlib.sha256(context.encode()).hexdigest()
        runner.phase = "test"
        pairs = []
        for index in test_indices:
            none, _ = runner.execute(index, "", "none")
            gepa, _ = runner.execute(index, context, "reef_gepa")
            pairs.append(
                {
                    "index": index,
                    "none": none["reward"],
                    "reef_gepa": gepa["reward"],
                    "delta": gepa["reward"] - none["reward"],
                }
            )
            write_json(cli.output / "paired_results.json", pairs)
        summary = {
            "status": "complete",
            "task": cli.task,
            "n": len(pairs),
            "none_mean": statistics.mean(p["none"] for p in pairs),
            "reef_gepa_mean": statistics.mean(p["reef_gepa"] for p in pairs),
            "mean_delta": statistics.mean(p["delta"] for p in pairs),
            "wins_ties_losses": [
                sum(p["delta"] > 1e-9 for p in pairs),
                sum(abs(p["delta"]) <= 1e-9 for p in pairs),
                sum(p["delta"] < -1e-9 for p in pairs),
            ],
            "selected_candidate": archive.served,
            "rules_sha256": selected_hash,
            "proposals": len(archive.proposals),
            "candidate_count": len(archive.candidates),
            "episode_calls": runner.count,
        }
        write_json(cli.output / "summary.json", summary)
        write_json(cli.output / "status.json", summary)
        (cli.output / "REPORT.md").write_text(
            "# REEF-GEPA on CLBench\n\n```json\n"
            + json.dumps(summary, indent=2)
            + "\n```\n\nOriginal Reef GEPA proposer, Pareto archive and strict validation selector. "
            "Frozen Qwen3-4B; training indices 0–2, validation 3–5, test starts at 12. "
            "Rules freeze before testing. Development and test costs are in episodes.jsonl; reflection calls are separate. "
            "This tests prompt optimization, not the Reef service or online parameter training.\n"
        )
        print(json.dumps(summary), flush=True)
    except Exception:
        write_json(
            cli.output / "status.json",
            {"status": "failed", "traceback": traceback.format_exc()},
        )
        raise


if __name__ == "__main__":
    main()
