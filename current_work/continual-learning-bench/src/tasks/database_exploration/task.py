"""Database exploration continual learning task.

Systems answer natural-language questions about an unknown SQLite database.
For each question the system can issue exploratory SQL queries before submitting
a final answer.  The continual learning signal is the reduction in exploratory
queries over time as the system learns the schema, data distributions, and quirks.
"""

from __future__ import annotations

import json
import os
import random
import re
import shutil
import sqlite3
import tempfile
import time
from pathlib import Path
from typing import Any, ClassVar, Optional

from pydantic import BaseModel, Field

from ...interface import (
    ContinualLearningTask,
    EvalMetrics,
    InstanceOutcome,
    Observation,
    Query,
    Response,
    TaskAgentBrief,
    TaskResult,
    TaskStepResult,
)
from ...registry import register_task
from ..schedules import TaskScheduleSpec, get_task_schedule
from ..variants import TaskVariantSpec, get_task_variant, validate_variant_defaults
from .prompts import (
    ACTION_FIELD_DESCRIPTION,
    CONTENT_FIELD_DESCRIPTION,
    FEEDBACK_BUDGET_EXCEEDED,
    FEEDBACK_CORRECT,
    FEEDBACK_INCORRECT,
    FEEDBACK_TIMEOUT,
    QUERY_RESULT_TEMPLATE,
    QUESTION_TEMPLATE,
)

RESPONSE_TIMEOUT_SECONDS = 60.0

MAX_QUERY_RESULT_ROWS = 50

SQL_QUERY_TIMEOUT_SECONDS = 10

# Anchor to repo root so ``clbench setup`` and defaults work regardless of cwd.
_REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = _REPO_ROOT / "data" / "database_exploration"
DEFAULT_DB_PATH = DATA_DIR / "products.db"
DEFAULT_DRIFTED_DB_PATH = DATA_DIR / "products_drifted.db"
DEFAULT_QUESTIONS_PATH = DATA_DIR / "questions.json"


def _build_schema_drift_question_sequence(
    *,
    pre_questions: list[dict[str, Any]],
    post_questions: list[dict[str, Any]],
    pre_drift_count: int,
    post_drift_count: int,
    seed: int,
    run_index: int | None,
) -> list[dict[str, Any]]:
    """Return shuffled pre/post-drift questions while preserving stage boundary."""
    pre_shuffled = list(pre_questions[:pre_drift_count])
    post_shuffled = list(post_questions[:post_drift_count])
    run_off = (run_index or 0) * 100_003
    random.Random(seed + 410_917 + run_off).shuffle(pre_shuffled)
    random.Random(seed + 910_357 + run_off).shuffle(post_shuffled)
    return pre_shuffled + post_shuffled


class DatabaseAction(BaseModel):
    """Structured response for each interaction step."""

    action: str = Field(description=ACTION_FIELD_DESCRIPTION)
    content: str = Field(description=CONTENT_FIELD_DESCRIPTION)


@register_task("database_exploration")
class DatabaseExploration(ContinualLearningTask):
    """Continual learning over an unknown database.

    The system receives a sequence of natural-language questions about a SQLite
    database whose schema it does not know.  For each question it may issue SQL
    queries (QUERY) to explore before submitting a final answer (ANSWER).

    Primary metric: reduction in exploratory queries over time.
    """

    r_max = 1.0
    has_setup: ClassVar[bool] = True

    @classmethod
    def setup(cls, *, force: bool = False) -> None:
        """Download ``products.db`` and ``products_drifted.db`` from Hugging Face when missing.

        Uses :envvar:`CL_BENCHMARK_DB_REPO` (default ``continual-learning-bench/database-exploration``)
        and optional :envvar:`HF_TOKEN`. Existing files are left in place unless ``force`` is true.
        """
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as e:
            raise ImportError(
                "database_exploration setup requires huggingface-hub; run `uv sync` or "
                "`pip install huggingface-hub`."
            ) from e

        repo_id = (
            os.environ.get("CL_BENCHMARK_DB_REPO")
            or "continual-learning-bench/database-exploration"
        ).strip()
        token = os.environ.get("HF_TOKEN")
        dest = DATA_DIR
        dest.mkdir(parents=True, exist_ok=True)
        for name in ("products.db", "products_drifted.db"):
            out = dest / name
            if out.is_file() and not force:
                continue
            hf_hub_download(
                repo_id=repo_id,
                repo_type="dataset",
                filename=name,
                local_dir=str(dest),
                token=token,
            )
            if not out.is_file():
                raise FileNotFoundError(f"Expected {out} after Hugging Face download")

    def __init__(
        self,
        variant: Optional[str] = None,
        schedule: Optional[str] = None,
        run_index: Optional[int] = None,
        rollout_index: Optional[int] = None,
        db_path: str = str(DEFAULT_DB_PATH),
        questions_path: str = str(DEFAULT_QUESTIONS_PATH),
        num_questions: int = 30,
        num_instances: Optional[int] = None,
        max_queries_per_question: int = 15,
        seed: int = 42,
        response_timeout_seconds: float = RESPONSE_TIMEOUT_SECONDS,
    ):
        if num_instances is not None:
            num_questions = num_instances

        self.schedule = schedule
        self.rollout_schedule = schedule
        self.rollout_schedule_spec: TaskScheduleSpec | None = None
        self._run_index = run_index if run_index is not None else rollout_index

        if self.schedule and not variant:
            self.rollout_schedule_spec = get_task_schedule(
                "database_exploration", self.schedule
            )
            if self.rollout_schedule_spec.stages:
                stage = self.rollout_schedule_spec.stages[0]
                if stage.variant:
                    variant = stage.variant

        self.variant = variant
        self.variant_spec: TaskVariantSpec | None = None
        self.variant_display_name: str | None = None

        if self.variant:
            self.variant_spec = get_task_variant("database_exploration", self.variant)
            validate_variant_defaults(
                self.variant_spec,
                {
                    "db_path": db_path,
                    "questions_path": questions_path,
                    "num_questions": num_questions,
                    "max_queries_per_question": max_queries_per_question,
                    "seed": seed,
                },
                init_defaults={
                    "db_path": str(DEFAULT_DB_PATH),
                    "questions_path": str(DEFAULT_QUESTIONS_PATH),
                    "num_questions": 30,
                    "max_queries_per_question": 15,
                    "seed": 42,
                },
            )
            fixed = self.variant_spec.defaults
            db_path = str(fixed.get("db_path", db_path))
            questions_path = str(fixed.get("questions_path", questions_path))
            num_questions = int(fixed.get("num_questions", num_questions))
            max_queries_per_question = int(
                fixed.get("max_queries_per_question", max_queries_per_question)
            )
            seed = int(fixed.get("seed", seed))
            self.variant_display_name = self.variant_spec.display_name

        self.db_path = Path(db_path)
        self.questions_path = Path(questions_path)
        self.num_questions = num_questions
        self.num_instances = num_questions
        self.max_queries_per_question = max_queries_per_question
        self.seed = seed
        self.response_timeout_seconds = response_timeout_seconds

        # Schema drift config
        variant_config = self.variant_spec.config if self.variant_spec else {}
        self._schema_drift_enabled = bool(variant_config.get("schema_drift", False))
        self._pre_drift_count = int(variant_config.get("pre_drift_count", 0))
        self._post_drift_count = int(variant_config.get("post_drift_count", 0))
        self._post_drift_questions_path: Path | None = None
        if self._schema_drift_enabled:
            pdq = variant_config.get("post_drift_questions_path", "")
            if pdq:
                self._post_drift_questions_path = Path(pdq)

        # Runtime state
        self._conn: sqlite3.Connection | None = None
        self._temp_db_path: Path | None = None
        self._drift_applied: bool = False
        self._stage_change_notice: str | None = None
        self.instances: list[dict[str, Any]] = []
        self._current_question_idx: int = 0
        self._queries_this_question: int = 0
        self._question_history: list[dict[str, Any]] = []
        self._awaiting_action: bool = False
        self._instance_outcomes: list[InstanceOutcome] = []

    # ------------------------------------------------------------------
    # ContinualLearningTask interface
    # ------------------------------------------------------------------

    def build_canonical_run_state(self) -> None:
        """Load questions, open DB connection, and clear per-run state."""
        if self._schema_drift_enabled:
            fd, tmp_path = tempfile.mkstemp(suffix=".db")
            os.close(fd)
            shutil.copy2(str(self.db_path), tmp_path)
            self._temp_db_path = Path(tmp_path)
            self._conn = sqlite3.connect(str(self._temp_db_path))
        else:
            self._conn = sqlite3.connect(str(self.db_path))

        with open(self.questions_path) as f:
            pre_questions = json.load(f)

        if self._schema_drift_enabled and self._post_drift_questions_path:
            with open(self._post_drift_questions_path) as f:
                post_questions = json.load(f)
            self.instances = _build_schema_drift_question_sequence(
                pre_questions=pre_questions,
                post_questions=post_questions,
                pre_drift_count=self._pre_drift_count,
                post_drift_count=self._post_drift_count,
                seed=self.seed,
                run_index=self._run_index,
            )
        else:
            self.instances = pre_questions[: self.num_questions]

        self._current_question_idx = 0
        self._queries_this_question = 0
        self._question_history = []
        self._awaiting_action = True
        self._instance_outcomes = []
        self._drift_applied = False
        self._stage_change_notice = None

    def build_current_query(self) -> Query:
        """Return the query for the currently active question."""
        return self._build_question_query()

    def step(self, response: Response) -> TaskStepResult:
        """Process a system response and return observation + next query."""
        observation, done = self.process_response(response)

        if observation.instance_complete and not done:
            self._sync_stage_context()

        instance_outcome = None
        if observation.instance_complete and self._question_history:
            last = self._question_history[-1]
            step_regret = float(last["regret"])
            reward = 1 - step_regret / self.max_queries_per_question
            instance_outcome = InstanceOutcome(
                instance_id=str(last["question_id"]),
                instance_index=len(self._question_history) - 1,
                reward=round(reward, 4),
                success=last["correct"],
                raw_metric_name="exploration_regret",
                raw_metric_value=last.get("regret", last["num_queries"]),
                raw_metric_higher_is_better=False,
                metadata={
                    "num_queries": last["num_queries"],
                    "num_actions": last.get("num_actions"),
                    "regret": last.get("regret"),
                    "difficulty": last["difficulty"],
                },
            )
            self._instance_outcomes.append(instance_outcome)

        next_query = None
        if not done:
            next_query = self._build_question_query()

        return TaskStepResult(
            observation=observation,
            next_query=next_query,
            done=done,
            instance_outcome=instance_outcome,
        )

    def process_response(self, response: Response) -> tuple[Observation, bool]:
        if not self._awaiting_action:
            return Observation(content="No action expected."), True

        parsed: DatabaseAction = response.action
        action = parsed.action.strip().upper()

        response_metadata = response.metadata or {}
        if response_metadata.get("latency_timeout"):
            return self._handle_answer("", timed_out=True)

        if action == "QUERY":
            return self._handle_query(parsed.content)
        elif action == "ANSWER":
            return self._handle_answer(parsed.content)
        else:
            return Observation(
                content=(
                    f"Unknown action '{parsed.action}'. "
                    "Use QUERY to run SQL or ANSWER to submit your answer."
                ),
                instance_complete=False,
                metadata={"instance_complete": False},
            ), False

    def get_query(self) -> Query:
        return self._build_question_query()

    def evaluate(self) -> TaskResult:
        if self._conn:
            self._conn.close()
            self._conn = None
        if self._temp_db_path and self._temp_db_path.exists():
            self._temp_db_path.unlink()
            self._temp_db_path = None

        total_questions = len(self._question_history)
        correct = sum(1 for q in self._question_history if q["correct"])
        accuracy = correct / total_questions if total_questions else 0

        total_queries = sum(q["num_queries"] for q in self._question_history)
        avg_queries = total_queries / total_questions if total_questions else 0

        # Regret: each exploratory query is regret. Incorrect answers add
        # full budget as regret (the exploration was wasted).
        regret_curve: list[float] = []
        cumulative_regret = 0.0
        for q in self._question_history:
            if q["correct"]:
                step_regret = float(q.get("regret", q["num_queries"]))
            else:
                step_regret = float(self.max_queries_per_question)
            cumulative_regret += step_regret
            regret_curve.append(cumulative_regret)

        max_possible_regret = float(total_questions * self.max_queries_per_question)
        normalized_regret = (
            cumulative_regret / max_possible_regret if max_possible_regret > 0 else 0
        )

        # Score: fraction of max possible regret avoided (0 = max regret, 1 = optimal).
        # Equals mean(per-instance reward). Higher is always better.
        score = 1 - normalized_regret

        # Learning curve: first half vs second half
        mid = total_questions // 2
        if mid > 0:
            first_half = self._question_history[:mid]
            second_half = self._question_history[mid:]
            first_half_queries = sum(q["num_queries"] for q in first_half) / len(
                first_half
            )
            second_half_queries = sum(q["num_queries"] for q in second_half) / len(
                second_half
            )
            query_reduction = first_half_queries - second_half_queries

            first_half_accuracy = sum(1 for q in first_half if q["correct"]) / len(
                first_half
            )
            second_half_accuracy = sum(1 for q in second_half if q["correct"]) / len(
                second_half
            )
        else:
            first_half_queries = second_half_queries = avg_queries
            query_reduction = 0
            first_half_accuracy = second_half_accuracy = accuracy

        # Per-instance loss: queries used (or full budget if wrong)
        loss_curve: list[float] = []
        for q in self._question_history:
            if q["correct"]:
                loss_curve.append(float(q.get("regret", q["num_queries"])))
            else:
                loss_curve.append(float(self.max_queries_per_question))

        exploration_regret = float(
            sum(
                q.get("regret", q["num_queries"])
                for q in self._question_history
                if q["correct"]
            )
        )
        incorrect_regret = float(
            sum(
                self.max_queries_per_question
                for q in self._question_history
                if not q["correct"]
            )
        )

        eval_metrics = EvalMetrics(
            loss_curve=loss_curve,
            optimal_performance=float(total_questions),
            actual_performance=float(correct),
            extra={
                "exploration_regret": exploration_regret,
                "incorrect_regret": incorrect_regret,
                "cumulative_regret": cumulative_regret,
                "normalized_regret": normalized_regret,
            },
        )

        summary_parts = []
        if self.schedule:
            summary_parts.append(f"Schedule: {self.schedule}")
        if self.variant:
            summary_parts.append(
                f"Variant: {self.variant} ({self.variant_display_name or self.variant})"
            )
        summary_parts.extend(
            [
                f"Questions: {total_questions}  "
                f"Correct: {correct}  "
                f"Accuracy: {accuracy:.1%}",
                f"Total exploratory queries: {total_queries}  "
                f"Avg per question: {avg_queries:.1f}",
                "",
                "REGRET METRICS (Continual Learning):",
                f"  Cumulative regret: {cumulative_regret:.0f} "
                "(exploratory QUERY steps only; ANSWER not counted)",
                f"  Score: {score:.3f} (1 is perfect; 0 is worst-case regret)",
                f"  Normalized regret (diagnostic): {normalized_regret:.3f}",
                "",
                "LEARNING CURVE:",
                f"  First half avg queries: {first_half_queries:.1f}",
                f"  Second half avg queries: {second_half_queries:.1f}",
                f"  Query reduction: {query_reduction:+.1f}",
                f"  First half accuracy: {first_half_accuracy:.1%}",
                f"  Second half accuracy: {second_half_accuracy:.1%}",
            ]
        )

        metrics: dict[str, Any] = {
            "accuracy": accuracy,
            "correct": correct,
            "total_questions": total_questions,
            "total_queries": total_queries,
            "avg_queries_per_question": avg_queries,
            "first_half_avg_queries": first_half_queries,
            "second_half_avg_queries": second_half_queries,
            "query_reduction": query_reduction,
            "first_half_accuracy": first_half_accuracy,
            "second_half_accuracy": second_half_accuracy,
            "cumulative_regret": cumulative_regret,
            "normalized_regret": normalized_regret,
            "loss_curve": loss_curve,
            "question_history": self._question_history,
        }
        if self.variant:
            metrics["variant_id"] = self.variant
            metrics["variant_display_name"] = self.variant_display_name or self.variant
        if self.schedule:
            metrics["schedule_id"] = self.schedule
        # Same scale as per-instance rewards: 1 avoids all regret, 0 is worst case.
        metrics["score"] = score

        return TaskResult(
            metrics=metrics,
            summary="\n".join(summary_parts),
            eval_metrics=eval_metrics,
            instance_outcomes=list(self._instance_outcomes),
        )

    def get_agent_brief(self) -> TaskAgentBrief:
        return TaskAgentBrief(
            objective="Answer questions about an unknown database using as few exploratory SQL queries as possible.",
            instance_unit="One natural-language question about the database.",
            reward_definition=(
                "Per question, regret counts only exploratory QUERY steps "
                "(your final ANSWER is not counted). So regret = "
                "max(0, actions - 1) where actions = queries + 1. Wrong "
                "answers use the full query budget as regret. Reward is "
                "1 - regret / query_budget, so larger is better."
            ),
            completion_definition=(
                "An instance completes when you submit an answer "
                "or exhaust the query budget."
            ),
        )

    @property
    def name(self) -> str:
        return "database_exploration"

    @property
    def description(self) -> str:
        return (
            "Learn an unknown database schema through exploration, "
            "reducing exploratory queries over time"
        )

    def get_response_timeout_seconds(self, query: Query) -> float | None:
        if self.response_timeout_seconds <= 0:
            return None
        return self.response_timeout_seconds

    def get_timeout_fallback_response(
        self,
        query: Query,
        timeout_seconds: float,
        elapsed_seconds: float,
    ) -> Response:
        return Response(
            action=DatabaseAction(action="ANSWER", content=""),
            metadata={
                "latency_timeout": True,
                "timeout_seconds": timeout_seconds,
                "elapsed_seconds": elapsed_seconds,
            },
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _current_question(self) -> dict[str, Any]:
        return self.instances[self._current_question_idx]

    def _build_question_query(self) -> Query:
        q = self._current_question()
        question_num = self._current_question_idx + 1
        total = len(self.instances)
        budget = self.max_queries_per_question

        prompt = QUESTION_TEMPLATE.format(
            question_num=question_num,
            total=total,
            question_text=q["question"],
            budget=budget,
            queries_used=self._queries_this_question,
        )

        if self._stage_change_notice:
            prompt = f"{self._stage_change_notice}\n\n{prompt}"
            self._stage_change_notice = None

        live_db_path = (
            str(self._temp_db_path) if self._temp_db_path else str(self.db_path)
        )
        metadata = {
            "question_id": q["question_id"],
            "question_num": question_num,
            "difficulty": q.get("difficulty", "unknown"),
            "queries_used": self._queries_this_question,
            "query_budget": budget,
            "db_path": live_db_path,
        }

        return Query(
            prompt=prompt,
            response_schema=DatabaseAction,
            instance_id=str(q["question_id"]),
            instance_index=self._current_question_idx,
            metadata=metadata,
        )

    def _handle_query(self, sql: str) -> tuple[Observation, bool]:
        self._queries_this_question += 1
        budget = self.max_queries_per_question

        if self._queries_this_question > budget:
            return self._handle_answer(
                "",
                budget_exceeded=True,
            )

        result_text = self._execute_sql(sql)
        remaining = budget - self._queries_this_question

        content = QUERY_RESULT_TEMPLATE.format(
            queries_used=self._queries_this_question,
            budget=budget,
            remaining=remaining,
            result_text=result_text,
        )

        return Observation(
            content=content,
            instance_complete=False,
            metadata={"instance_complete": False},
        ), False

    def _handle_answer(
        self,
        answer: str,
        timed_out: bool = False,
        budget_exceeded: bool = False,
    ) -> tuple[Observation, bool]:
        q = self._current_question()
        correct = self._evaluate_answer(answer, q)

        # Actions = exploratory QUERY steps plus this ANSWER. On a correct
        # instance, regret is exploration only (ANSWER not penalized):
        # max(0, num_actions - 1). On failure, regret is the full query budget.
        num_actions = self._queries_this_question + 1
        if correct:
            step_regret_record = float(max(0, num_actions - 1))
        else:
            step_regret_record = float(self.max_queries_per_question)

        record = {
            "question_id": q["question_id"],
            "question": q["question"],
            "difficulty": q.get("difficulty", "unknown"),
            "submitted_answer": answer,
            "ground_truth": q["answer"],
            "correct": correct,
            "num_queries": self._queries_this_question,
            "num_actions": num_actions,
            "regret": step_regret_record,
            "timed_out": timed_out,
            "budget_exceeded": budget_exceeded,
        }
        self._question_history.append(record)

        question_num = self._current_question_idx + 1

        gt_sql = q.get("sql", "N/A").strip()

        if timed_out:
            feedback = FEEDBACK_TIMEOUT.format(
                question_num=question_num,
                ground_truth=q["answer"],
                ground_truth_sql=gt_sql,
            )
        elif budget_exceeded:
            feedback = FEEDBACK_BUDGET_EXCEEDED.format(
                question_num=question_num,
                ground_truth=q["answer"],
                ground_truth_sql=gt_sql,
            )
        elif correct:
            feedback = FEEDBACK_CORRECT.format(
                question_num=question_num,
                answer=answer,
                num_queries=self._queries_this_question,
            )
        else:
            feedback = FEEDBACK_INCORRECT.format(
                question_num=question_num,
                answer=answer,
                ground_truth=q["answer"],
                ground_truth_sql=gt_sql,
                num_queries=self._queries_this_question,
            )

        self._current_question_idx += 1
        self._queries_this_question = 0

        done = self._current_question_idx >= len(self.instances)
        self._awaiting_action = not done

        return Observation(
            content=feedback,
            instance_complete=True,
            metadata={"instance_complete": True},
        ), done

    def _execute_sql(self, sql: str) -> str:
        """Execute SQL against the database and return formatted results."""
        if self._conn is None:
            return "ERROR: Database connection not available."

        sql = sql.strip().rstrip(";").strip()

        deadline = time.monotonic() + SQL_QUERY_TIMEOUT_SECONDS
        self._conn.set_progress_handler(
            lambda: 1 if time.monotonic() > deadline else 0,
            1000,
        )
        try:
            # Handle .tables meta-command
            if sql.lower() == ".tables":
                rows = self._conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
                ).fetchall()
                return "\n".join(r[0] for r in rows)

            # Handle .schema meta-command
            if sql.lower().startswith(".schema"):
                parts = sql.split(maxsplit=1)
                if len(parts) > 1:
                    table = parts[1].strip()
                    rows = self._conn.execute(
                        "SELECT sql FROM sqlite_master WHERE name = ?",
                        (table,),
                    ).fetchall()
                else:
                    rows = self._conn.execute(
                        "SELECT sql FROM sqlite_master WHERE type='table'"
                    ).fetchall()
                return "\n\n".join(r[0] for r in rows if r[0])

            # Handle PRAGMA
            if sql.upper().startswith("PRAGMA"):
                try:
                    cursor = self._conn.execute(sql)
                    rows = cursor.fetchall()
                    if cursor.description:
                        cols = [d[0] for d in cursor.description]
                        return self._format_results(cols, rows)
                    return "(no results)"
                except sqlite3.OperationalError as e:
                    if "interrupted" in str(e).lower():
                        return (
                            f"ERROR: query exceeded {SQL_QUERY_TIMEOUT_SECONDS}s "
                            "timeout"
                        )
                    return f"ERROR: {e}"
                except Exception as e:
                    return f"ERROR: {e}"

            # Only allow read-only SQL
            first_word = sql.split()[0].upper() if sql.split() else ""
            if first_word not in (
                "SELECT",
                "WITH",
                "EXPLAIN",
                "PRAGMA",
            ):
                return (
                    f"ERROR: Only SELECT / WITH / EXPLAIN / PRAGMA statements "
                    f"are allowed. Got: {first_word}"
                )

            try:
                cursor = self._conn.execute(sql)
                rows = cursor.fetchmany(MAX_QUERY_RESULT_ROWS + 1)
                cols = [d[0] for d in cursor.description] if cursor.description else []
                truncated = len(rows) > MAX_QUERY_RESULT_ROWS
                if truncated:
                    rows = rows[:MAX_QUERY_RESULT_ROWS]
                result = self._format_results(cols, rows)
                if truncated:
                    result += f"\n... (showing first {MAX_QUERY_RESULT_ROWS} rows)"
                return result
            except sqlite3.OperationalError as e:
                if "interrupted" in str(e).lower():
                    return f"ERROR: query exceeded {SQL_QUERY_TIMEOUT_SECONDS}s timeout"
                return f"ERROR: {e}"
            except Exception as e:
                return f"ERROR: {e}"
        finally:
            self._conn.set_progress_handler(None, 0)

    def _format_results(self, columns: list[str], rows: list[tuple]) -> str:
        if not rows:
            return "(no results)"
        if not columns:
            return "\n".join(str(r) for r in rows)

        def _clean(v: object) -> str:
            if v is None:
                return "NULL"
            # Collapse embedded newlines/tabs so each row stays on one line.
            return (
                str(v)
                .replace("\r\n", " ")
                .replace("\n", " ")
                .replace("\r", " ")
                .replace("\t", " ")
            )

        str_rows = [[_clean(v) for v in row] for row in rows]
        widths = [len(c) for c in columns]
        for row in str_rows:
            for i, val in enumerate(row):
                widths[i] = max(widths[i], len(val))

        header = " | ".join(c.ljust(w) for c, w in zip(columns, widths))
        separator = "-+-".join("-" * w for w in widths)
        data_lines = [
            " | ".join(v.ljust(w) for v, w in zip(row, widths)) for row in str_rows
        ]

        return "\n".join([header, separator, *data_lines])

    # ------------------------------------------------------------------
    # Schema drift helpers
    # ------------------------------------------------------------------

    def _sync_stage_context(self) -> None:
        """Check if we've crossed the stage boundary and swap to drifted DB."""
        if not self._schema_drift_enabled or self._drift_applied:
            return
        if self._current_question_idx >= self._pre_drift_count:
            self._swap_to_drifted_db()
            self._drift_applied = True
            self._stage_change_notice = (
                "NOTICE: The live database schema or contents may have "
                "changed since your earlier exploration. Inspect current "
                "metadata and data before relying on prior assumptions."
            )

    def _swap_to_drifted_db(self) -> None:
        """Close the pre-drift connection and reopen against products_drifted.db.

        ``products_drifted.db`` is downloaded by ``clbench setup
        database_exploration`` and is the canonical post-migration database.
        Each worker gets its own writable temp copy so concurrent runs do not
        share state — matching the per-worker isolation already used for the
        pre-drift DB.
        """
        if not DEFAULT_DRIFTED_DB_PATH.is_file():
            raise FileNotFoundError(
                f"products_drifted.db not found at {DEFAULT_DRIFTED_DB_PATH}. "
                "Run `clbench setup database_exploration` to download it."
            )

        if self._conn is not None:
            self._conn.close()
            self._conn = None
        if self._temp_db_path is not None and self._temp_db_path.exists():
            self._temp_db_path.unlink()
            self._temp_db_path = None

        fd, tmp_path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        shutil.copy2(str(DEFAULT_DRIFTED_DB_PATH), tmp_path)
        self._temp_db_path = Path(tmp_path)
        self._conn = sqlite3.connect(str(self._temp_db_path))

    def _evaluate_answer(self, submitted: str, question: dict[str, Any]) -> bool:
        """Compare the submitted answer against ground truth."""
        if not submitted or not submitted.strip():
            return False

        submitted_clean = submitted.strip()
        expected = str(question["answer"]).strip()
        answer_type = question.get("answer_type", "text")
        tolerance = question.get("tolerance", 0)

        if answer_type == "integer":
            try:
                nums = re.findall(r"-?\d+", submitted_clean)
                if not nums:
                    return False
                return int(nums[0]) == int(expected)
            except (ValueError, IndexError):
                return False

        elif answer_type == "float":
            try:
                nums = re.findall(r"-?\d+\.?\d*", submitted_clean)
                if not nums:
                    return False
                submitted_val = float(nums[0])
                expected_val = float(expected)
                return abs(submitted_val - expected_val) <= max(
                    tolerance, abs(expected_val) * 0.01
                )
            except (ValueError, IndexError):
                return False

        else:
            return submitted_clean.lower() == expected.lower()
