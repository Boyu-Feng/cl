"""ACE system adapter for the continual learning benchmark.

This adapter keeps ACE's generator/reflector/curator roles, but maps them onto
the benchmark's response-only interface:
- `respond()` ingests any provided feedback, optionally reflects/curates at
  instance boundaries, and then generates the next action.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, ValidationError

from ...interface import (
    ContinualLearningSystem,
    Observation,
    Query,
    Response,
    observation_marks_instance_complete,
)
from ...registry import register_system
from ...usage import build_usage_event
from ...vendors.ace import Curator, Generator, Reflector
from ...vendors.ace.llm import timed_llm_call
from ...vendors.ace.playbook_utils import (
    extract_playbook_bullets,
    extract_json_from_text,
    get_next_global_id,
    get_playbook_stats,
    parse_playbook_line,
    update_bullet_counts,
)
from ...vendors.ace.utils import initialize_clients
from .artifacts import ensure_registered as ensure_ace_artifact_exporter_registered


ensure_ace_artifact_exporter_registered()

EMPTY_PLAYBOOK = """## STRATEGIES & INSIGHTS

## FORMULAS & CALCULATIONS

## CODE SNIPPETS & TEMPLATES

## COMMON MISTAKES TO AVOID

## PROBLEM-SOLVING HEURISTICS

## CONTEXT CLUES & INDICATORS

## OTHERS"""


@dataclass
class EpisodeTurn:
    """Single within-instance interaction stored until the boundary is ingested."""

    prompt: str
    query_metadata: Optional[dict[str, Any]]
    response_schema_name: str
    generator_response: str
    reasoning: str
    bullet_ids: list[str]
    action_payload: dict[str, Any]
    repair_attempted: bool
    intermediate_observation: Optional[str] = None
    intermediate_observation_metadata: Optional[dict[str, Any]] = None
    transition_prompt: Optional[str] = None
    transition_metadata: Optional[dict[str, Any]] = None
    terminal_observation: Optional[str] = None
    terminal_observation_metadata: Optional[dict[str, Any]] = None


def _parse_json_object(raw: str) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        value = extract_json_from_text(raw)
        if value is None:
            raise ValueError("Expected valid JSON from the ACE generator.") from exc
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object.")
    return value


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _parse_model_kwargs_json(raw: str, field_name: str) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{field_name} must be valid JSON.") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must decode to a JSON object.")
    return value


@register_system("ace")
class ACESystem(ContinualLearningSystem):
    """ACE-based system with learning triggered by feedback-rich `respond()` calls."""

    def __init__(
        self,
        model: str = "gpt-5",
        generator_model: str = "",
        reflector_model: str = "",
        curator_model: str = "",
        api_provider: str = "openai",
        initial_playbook_path: str = "",
        curate_every_n_updates: int = 1,
        generator_model_kwargs_json: str = "{}",
        reflector_model_kwargs_json: str = "{}",
        curator_model_kwargs_json: str = "{}",
        name: str = "ace",
    ):
        self._name = name
        self.model = model
        self.generator_model = generator_model or model
        self.reflector_model = reflector_model or model
        self.curator_model = curator_model or model
        self.api_provider = api_provider
        self.initial_playbook_path = initial_playbook_path
        self.curate_every_n_updates = max(1, curate_every_n_updates)
        self.generator_model_kwargs = _parse_model_kwargs_json(
            generator_model_kwargs_json, "generator_model_kwargs_json"
        )
        self.reflector_model_kwargs = _parse_model_kwargs_json(
            reflector_model_kwargs_json, "reflector_model_kwargs_json"
        )
        self.curator_model_kwargs = _parse_model_kwargs_json(
            curator_model_kwargs_json, "curator_model_kwargs_json"
        )

        self._initial_playbook = self._load_initial_playbook(initial_playbook_path)
        self.playbook = self._initial_playbook
        self.next_global_id = get_next_global_id(self.playbook)

        self._generator: Optional[Generator] = None
        self._reflector: Optional[Reflector] = None
        self._curator: Optional[Curator] = None

        self.episode_transcript: list[EpisodeTurn] = []
        self.playbook_snapshots: list[dict[str, Any]] = []
        self.interaction_count = 0
        self.update_count = 0
        self._last_update_reflection_ran = False
        self._last_update_curation_ran = False
        self._last_update_summary = ""
        self._latest_response_metadata: Optional[dict[str, Any]] = None
        self._latest_update_details: Optional[dict[str, Any]] = None
        self._last_ingested_feedback_id: Optional[int] = None

    def respond(self, query: Query) -> Response:
        self._ensure_initialized()

        if query.feedback is not None:
            self._ingest_feedback(query.feedback, next_query=query)

        self.interaction_count += 1

        action, generation_info = self._generate_action(query)
        self.episode_transcript.append(
            EpisodeTurn(
                prompt=query.prompt,
                query_metadata=self._copy_metadata(query.metadata),
                response_schema_name=query.response_schema.__name__,
                generator_response=generation_info["raw_response"],
                reasoning=generation_info.get("reasoning", ""),
                bullet_ids=generation_info["bullet_ids"],
                action_payload=self._serialize_action_payload(action),
                repair_attempted=generation_info["repair_attempted"],
            )
        )

        metadata = {
            "interaction_count": self.interaction_count,
            "system_type": "ace",
            "api_provider": self.api_provider,
            "generator_model": self.generator_model,
            "reflector_model": self.reflector_model,
            "curator_model": self.curator_model,
            "playbook_bullet_count": self._count_playbook_bullets(),
            "update_count": self.update_count,
            "reflection_ran": self._last_update_reflection_ran,
            "curation_ran": self._last_update_curation_ran,
            "current_episode_turn_count": len(self.episode_transcript),
            "bullet_ids_used": generation_info["bullet_ids"],
            "repair_attempted": generation_info["repair_attempted"],
            "last_update_summary": self._last_update_summary,
            "ace_playbook": self.playbook,
            "generator_reasoning": generation_info.get("reasoning", ""),
            "generator_raw_response": generation_info["raw_response"],
        }
        self._latest_response_metadata = metadata
        return Response(action=action, metadata=metadata)

    def observe(
        self, observation: Observation, next_query: Query | None = None
    ) -> None:
        """Ingest benchmark feedback immediately after each interaction."""
        self._ensure_initialized()
        self._ingest_feedback(observation, next_query=next_query)

    def reset(self) -> None:
        self.playbook = self._initial_playbook
        self.next_global_id = get_next_global_id(self.playbook)
        self.episode_transcript = []
        self.playbook_snapshots = []
        self.interaction_count = 0
        self.update_count = 0
        self._last_update_reflection_ran = False
        self._last_update_curation_ran = False
        self._last_update_summary = ""
        self._latest_response_metadata = None
        self._latest_update_details = None
        self._last_ingested_feedback_id = None
        self._generator = None
        self._reflector = None
        self._curator = None

    @property
    def name(self) -> str:
        return self._name

    def get_run_artifacts(self) -> Optional[dict[str, Any]]:
        return {
            "artifact_type": "ace",
            "initial_playbook": self._initial_playbook,
            "final_playbook": self.playbook,
            "playbook_snapshots": [
                dict(snapshot) for snapshot in self.playbook_snapshots
            ],
        }

    def _load_initial_playbook(self, path_str: str) -> str:
        if not path_str:
            return EMPTY_PLAYBOOK

        path = Path(path_str)
        if not path.exists():
            raise ValueError(f"Initial playbook path does not exist: {path}")
        return path.read_text(encoding="utf-8")

    def _ensure_initialized(self) -> None:
        if self._generator and self._reflector and self._curator:
            return

        generator_client, reflector_client, curator_client = initialize_clients(
            self.api_provider
        )
        self._generator = Generator(
            generator_client,
            self.api_provider,
            self.generator_model,
            model_kwargs=self.generator_model_kwargs,
        )
        self._reflector = Reflector(
            reflector_client,
            self.api_provider,
            self.reflector_model,
            model_kwargs=self.reflector_model_kwargs,
        )
        self._curator = Curator(
            curator_client,
            self.api_provider,
            self.curator_model,
            model_kwargs=self.curator_model_kwargs,
        )

    def _generate_action(self, query: Query) -> tuple[BaseModel, dict[str, Any]]:
        if query.response_schema is None:
            raise ValueError("ACE action turns require a response schema.")

        schema_json = json.dumps(query.response_schema.model_json_schema(), indent=2)
        base_context = self._build_generator_context(query)
        question = self._build_generator_question(query, schema_json)
        raw_response, bullet_ids, call_info = self._generator.generate(
            question=question,
            playbook=self.playbook,
            context=base_context,
            reflection="(empty)",
            use_json_mode=True,
            call_id=f"bench_respond_{self.interaction_count}",
            log_dir=None,
        )
        self._record_call_info_usage(call_info)
        bullet_ids = self._filter_known_bullet_ids(bullet_ids)

        try:
            action, reasoning = self._parse_generator_response(
                raw_response, query.response_schema
            )
            return action, {
                "raw_response": raw_response,
                "bullet_ids": bullet_ids,
                "reasoning": reasoning,
                "repair_attempted": False,
            }
        except (ValueError, ValidationError) as exc:
            repaired_response = self._repair_generator_response(
                query=query,
                schema_json=schema_json,
                base_context=base_context,
                raw_response=raw_response,
                validation_error=str(exc),
            )
            repaired_bullet_ids = bullet_ids
            try:
                action, reasoning = self._parse_generator_response(
                    repaired_response, query.response_schema
                )
            except (ValueError, ValidationError) as repair_exc:
                raw_preview = raw_response.strip().replace("\n", "\\n")[:400]
                repair_preview = repaired_response.strip().replace("\n", "\\n")[:400]
                raise ValueError(
                    "Expected valid JSON from the ACE generator after repair. "
                    f"Initial response preview: {raw_preview!r}. "
                    f"Repair response preview: {repair_preview!r}."
                ) from repair_exc
            return action, {
                "raw_response": repaired_response,
                "bullet_ids": repaired_bullet_ids,
                "reasoning": reasoning,
                "repair_attempted": True,
            }

    def _build_generator_question(self, query: Query, schema_json: str) -> str:
        parts = [
            "Solve the current benchmark interaction.",
            "The `final_answer` field in your JSON response must be a JSON object",
            "that matches the required response schema exactly.",
            "Do not serialize `final_answer` as a quoted string.",
            "Keep `reasoning` to at most 20 words. An empty string is allowed.",
            "Return exactly one JSON object with no markdown or surrounding prose.",
            "Required response schema:",
            schema_json,
            "Current interaction prompt:",
            query.prompt or "(missing prompt)",
        ]
        return "\n".join(parts)

    def _repair_generator_response(
        self,
        query: Query,
        schema_json: str,
        base_context: str,
        raw_response: str,
        validation_error: str,
    ) -> str:
        repair_prompt = "\n".join(
            [
                "Repair the benchmark action response.",
                "Return exactly one JSON object with these top-level fields:",
                '- "reasoning": short string of at most 20 words',
                '- "bullet_ids": array of strings',
                '- "final_answer": JSON object matching the required response schema exactly',
                "Do not include markdown, code fences, or any extra prose.",
                "Keep `reasoning` to at most 20 words. An empty string is allowed.",
                "If the previous response already chose an action, preserve that action and only fix formatting/schema issues.",
                "Required response schema:",
                schema_json,
                "Current interaction prompt:",
                query.prompt or "(missing prompt)",
                "Current interaction context:",
                base_context,
                "Validation error:",
                validation_error,
                "Previous invalid response:",
                raw_response,
            ]
        )
        repaired_response, repair_call_info = timed_llm_call(
            self._generator.api_client,
            self.api_provider,
            self.generator_model,
            repair_prompt,
            role="generator",
            call_id=f"bench_respond_{self.interaction_count}_repair",
            max_tokens=self._generator.max_tokens,
            log_dir=None,
            use_json_mode=True,
            extra_api_params=self.generator_model_kwargs,
        )
        self._record_call_info_usage(repair_call_info)
        return repaired_response

    def _build_generator_context(self, query: Query) -> str:
        parts: list[str] = []
        if self.episode_transcript:
            parts.append("Current episode transcript so far:")
            for idx, turn in enumerate(self.episode_transcript, start=1):
                parts.append(f"Turn {idx} prompt:\n{turn.prompt}")
                parts.append(
                    "Turn {idx} action returned:\n{payload}".format(
                        idx=idx,
                        payload=json.dumps(turn.action_payload, indent=2),
                    )
                )
                if turn.intermediate_observation is not None:
                    parts.append(
                        "Turn {idx} observation after action:\n{observation}".format(
                            idx=idx,
                            observation=turn.intermediate_observation,
                        )
                    )
        if query.metadata:
            parts.append("Current query metadata:")
            parts.append(json.dumps(query.metadata, indent=2, sort_keys=True))
        return "\n\n".join(parts) if parts else "(empty)"

    def _ingest_feedback(
        self, feedback: Optional[Observation], next_query: Optional[Query]
    ) -> None:
        if feedback is None:
            return
        if id(feedback) == self._last_ingested_feedback_id:
            return
        self._last_ingested_feedback_id = id(feedback)
        if not self.episode_transcript:
            self._last_update_reflection_ran = False
            self._last_update_curation_ran = False
            self._last_update_summary = (
                "Skipped feedback ingestion because the episode transcript was empty."
            )
            return

        is_terminal = observation_marks_instance_complete(feedback)
        if not is_terminal:
            last_turn = self.episode_transcript[-1]
            last_turn.intermediate_observation = feedback.content
            last_turn.intermediate_observation_metadata = self._copy_metadata(
                feedback.metadata
            )
            if next_query is not None:
                self._record_environment_transition(next_query)
            self._last_update_reflection_ran = False
            self._last_update_curation_ran = False
            self._last_update_summary = "Recorded intermediate feedback."
            return

        self._run_terminal_update(feedback)

    def _run_terminal_update(self, observation: Observation) -> None:
        bullet_ids = _dedupe_preserve_order(
            [
                bullet_id
                for turn in self.episode_transcript
                for bullet_id in turn.bullet_ids
            ]
        )
        self._record_terminal_observation(observation)
        reflection_question = self._build_reflection_question()
        reflection_predicted_answer = self._build_reflection_predicted_answer()
        reflection_trace = self._format_reflection_trace()
        bullets_used = extract_playbook_bullets(self.playbook, bullet_ids)

        playbook_before = self.playbook
        episode_turn_count = len(self.episode_transcript)

        (
            reflection_content,
            bullet_tags,
            reflection_call_info,
        ) = self._reflector.reflect(
            question=reflection_question,
            reasoning_trace=reflection_trace,
            predicted_answer=reflection_predicted_answer,
            ground_truth=None,
            environment_feedback=observation.content,
            bullets_used=bullets_used,
            use_ground_truth=False,
            use_json_mode=True,
            call_id=f"bench_update_{self.update_count + 1}_reflect",
            log_dir=None,
        )
        self._record_call_info_usage(reflection_call_info)
        self._last_update_reflection_ran = True

        if bullet_tags:
            self.playbook = update_bullet_counts(self.playbook, bullet_tags)

        self.update_count += 1

        curation_ran = False
        curator_operations: list[dict[str, Any]] = []
        if self.update_count % self.curate_every_n_updates == 0:
            (
                self.playbook,
                self.next_global_id,
                curator_operations,
                curator_call_info,
            ) = self._curator.curate(
                current_playbook=self.playbook,
                recent_reflection=reflection_content,
                question_context=self._format_question_context(),
                current_step=self.update_count,
                total_samples=self.update_count,
                token_budget=80000,
                playbook_stats=get_playbook_stats(self.playbook),
                use_ground_truth=False,
                use_json_mode=True,
                call_id=f"bench_update_{self.update_count}_curate",
                log_dir=None,
                next_global_id=self.next_global_id,
            )
            self._record_call_info_usage(curator_call_info)
            self.next_global_id = get_next_global_id(self.playbook)
            curation_ran = True

        self._last_update_curation_ran = curation_ran
        self._last_update_summary = (
            f"Reflected on {episode_turn_count} turn(s); "
            f"{'ran' if curation_ran else 'skipped'} curation."
        )
        self._latest_update_details = {
            "update_count": self.update_count,
            "episode_turn_count": episode_turn_count,
            "bullet_ids_used": bullet_ids,
            "reflection": {
                "raw_response": reflection_content,
                "bullet_tags": list(bullet_tags) if bullet_tags else [],
            },
            "curation": {
                "ran": curation_ran,
                "operations": list(curator_operations) if curation_ran else [],
            },
            "playbook_before": playbook_before,
            "playbook_after": self.playbook,
        }
        if self._latest_response_metadata is not None:
            self._latest_response_metadata["ace_terminal_update"] = (
                self._latest_update_details
            )
        self._record_playbook_snapshot()
        self.episode_transcript = []

    def _parse_generator_response(
        self, raw_response: str, response_schema: type[BaseModel]
    ) -> tuple[BaseModel, str]:
        payload = _parse_json_object(raw_response)
        reasoning = str(payload.get("reasoning", ""))
        final_answer = payload.get("final_answer")
        if final_answer is None:
            raise ValueError("Generator response did not include `final_answer`.")

        if isinstance(final_answer, str):
            try:
                action = response_schema.model_validate_json(final_answer)
            except (ValidationError, ValueError):
                try:
                    parsed_final_answer = json.loads(final_answer)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        "Generator `final_answer` was not valid JSON."
                    ) from exc
                action = response_schema.model_validate(parsed_final_answer)
        else:
            action = response_schema.model_validate(final_answer)

        return action, reasoning

    def _record_environment_transition(self, query: Query) -> None:
        if not self.episode_transcript:
            return
        previous_turn = self.episode_transcript[-1]
        if previous_turn.transition_prompt is not None:
            return
        previous_turn.transition_prompt = query.prompt
        previous_turn.transition_metadata = self._copy_metadata(query.metadata)

    def _record_terminal_observation(self, observation: Observation) -> None:
        if not self.episode_transcript:
            return
        last_turn = self.episode_transcript[-1]
        last_turn.terminal_observation = observation.content
        last_turn.terminal_observation_metadata = self._copy_metadata(
            observation.metadata
        )

    def _build_reflection_question(self) -> str:
        first_turn = self.episode_transcript[0]
        last_turn = self.episode_transcript[-1]
        parts = [
            "Benchmark episode to analyze.",
            f"Total turns: {len(self.episode_transcript)}",
            f"Initial response schema: {first_turn.response_schema_name}",
            "Initial prompt:",
            first_turn.prompt,
        ]
        if first_turn.query_metadata:
            parts.extend(
                [
                    "Initial query metadata:",
                    json.dumps(first_turn.query_metadata, indent=2, sort_keys=True),
                ]
            )
        if last_turn.prompt != first_turn.prompt:
            parts.extend(["Final decision prompt:", last_turn.prompt])
        return "\n".join(parts)

    def _build_reflection_predicted_answer(self) -> str:
        payload = {
            "episode_type": "trajectory",
            "num_turns": len(self.episode_transcript),
            "actions": [
                {
                    "turn": idx,
                    "response_schema_name": turn.response_schema_name,
                    "action": turn.action_payload,
                    "repair_attempted": turn.repair_attempted,
                    "bullet_ids": turn.bullet_ids,
                }
                for idx, turn in enumerate(self.episode_transcript, start=1)
            ],
        }
        return json.dumps(payload, indent=2)

    def _format_reflection_trace(self) -> str:
        trace: list[str] = []
        for idx, turn in enumerate(self.episode_transcript, start=1):
            trace.append(f"Turn {idx} prompt:\n{turn.prompt}")
            if turn.query_metadata:
                trace.append(
                    "Turn {idx} query metadata:\n{metadata}".format(
                        idx=idx,
                        metadata=json.dumps(
                            turn.query_metadata, indent=2, sort_keys=True
                        ),
                    )
                )
            trace.append(
                f"Turn {idx} reasoning:\n{turn.reasoning or '(none provided)'}"
            )
            trace.append(
                "Turn {idx} action:\n{payload}".format(
                    idx=idx,
                    payload=json.dumps(turn.action_payload, indent=2),
                )
            )
            trace.append(
                "Turn {idx} bullet_ids: {bullet_ids}".format(
                    idx=idx,
                    bullet_ids=turn.bullet_ids or [],
                )
            )
            if turn.intermediate_observation is not None:
                trace.append(
                    "Turn {idx} intermediate observation:\n{observation}".format(
                        idx=idx,
                        observation=turn.intermediate_observation,
                    )
                )
            if turn.intermediate_observation_metadata:
                trace.append(
                    "Turn {idx} intermediate observation metadata:\n{metadata}".format(
                        idx=idx,
                        metadata=json.dumps(
                            turn.intermediate_observation_metadata,
                            indent=2,
                            sort_keys=True,
                        ),
                    )
                )
            if turn.transition_prompt is not None:
                trace.append(
                    "Turn {idx} next decision prompt:\n{prompt}".format(
                        idx=idx,
                        prompt=turn.transition_prompt,
                    )
                )
            if turn.transition_metadata:
                trace.append(
                    "Turn {idx} next decision metadata:\n{metadata}".format(
                        idx=idx,
                        metadata=json.dumps(
                            turn.transition_metadata, indent=2, sort_keys=True
                        ),
                    )
                )
            if turn.terminal_observation is not None:
                trace.append(
                    "Turn {idx} terminal observation:\n{observation}".format(
                        idx=idx,
                        observation=turn.terminal_observation,
                    )
                )
            if turn.terminal_observation_metadata:
                trace.append(
                    "Turn {idx} terminal observation metadata:\n{metadata}".format(
                        idx=idx,
                        metadata=json.dumps(
                            turn.terminal_observation_metadata,
                            indent=2,
                            sort_keys=True,
                        ),
                    )
                )
        return "\n\n".join(trace)

    def _format_question_context(self) -> str:
        if not self.episode_transcript:
            return "Benchmark episode summary:\n(empty)"

        last_turn = self.episode_transcript[-1]
        turn_summaries = [
            "Turn {idx}:\nPrompt:\n{prompt}\nAction:\n{action}{observation}".format(
                idx=idx,
                prompt=turn.prompt,
                action=json.dumps(turn.action_payload, sort_keys=True),
                observation=(
                    "\nIntermediate observation:\n" + turn.intermediate_observation
                    if turn.intermediate_observation is not None
                    else ""
                ),
            )
            for idx, turn in enumerate(self.episode_transcript, start=1)
        ]
        return (
            "Benchmark episode summary:\n"
            f"Total turns: {len(self.episode_transcript)}\n\n"
            "Trajectory:\n"
            + "\n\n".join(turn_summaries)
            + "\n\nFinal feedback:\n"
            + (last_turn.terminal_observation or "(missing)")
        )

    def _count_playbook_bullets(self) -> int:
        count = 0
        for line in self.playbook.splitlines():
            if parse_playbook_line(line):
                count += 1
        return count

    def _filter_known_bullet_ids(self, bullet_ids: list[str]) -> list[str]:
        known_ids = {
            parsed["id"]
            for line in self.playbook.splitlines()
            if (parsed := parse_playbook_line(line))
        }
        return [bullet_id for bullet_id in bullet_ids if bullet_id in known_ids]

    def _copy_metadata(
        self, metadata: Optional[dict[str, Any]]
    ) -> Optional[dict[str, Any]]:
        if metadata is None:
            return None
        return copy.deepcopy(metadata)

    def _serialize_action_payload(self, action: BaseModel) -> dict[str, Any]:
        return action.model_dump()

    def _record_call_info_usage(self, call_info: Optional[dict[str, Any]]) -> None:
        if not call_info:
            return
        prompt_tokens = call_info.get("prompt_num_tokens")
        response_tokens = call_info.get("response_num_tokens")
        if prompt_tokens is None and response_tokens is None:
            return
        self.record_usage_event(
            build_usage_event(
                model=str(call_info.get("model", self.model)),
                provider=call_info.get("provider") or self.api_provider,
                input_tokens=int(prompt_tokens or 0),
                output_tokens=int(response_tokens or 0),
                total_tokens=call_info.get("total_num_tokens"),
                call_type="completion",
                metadata={
                    "role": call_info.get("role"),
                    "call_id": call_info.get("call_id"),
                },
            )
        )

    def _record_playbook_snapshot(self) -> None:
        self.playbook_snapshots.append(
            {
                "update_count": self.update_count,
                "interaction_count": self.interaction_count,
                "playbook_bullet_count": self._count_playbook_bullets(),
                "reflection_ran": self._last_update_reflection_ran,
                "curation_ran": self._last_update_curation_ran,
                "summary": self._last_update_summary,
                "playbook": self.playbook,
            }
        )
