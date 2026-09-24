"""clbench registration bridge for the TTCL online-LoRA implementation."""

from __future__ import annotations

import json
from typing import Any

from pydantic import ValidationError

from ..interface import (
    ContinualLearningSystem,
    Observation,
    Query,
    Response,
    observation_marks_instance_complete,
)
from ..registry import register_system


@register_system("lora_online")
class LoRAOnlineSystem(ContinualLearningSystem):
    """Use ``/home/fengboyu/cl/ttcl`` as a live-learning clbench system."""

    supports_baseline = False
    parallel_safe = False

    def __init__(
        self,
        model_path: str,
        output_dir: str = "results/lora_online",
        device: str = "cuda:0",
        dtype: str = "bfloat16",
        update_every: int = 8,
        replay_trajectories: int = 32,
        learning_rate: float = 2e-5,
        train_epochs: int = 1,
        max_seq_length: int = 2048,
        max_new_tokens: int = 256,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        target_modules: str = "q_proj,k_proj,v_proj,o_proj",
        adapter_path: str | None = None,
        name: str = "lora_online",
    ) -> None:
        try:
            from ttcl.online_lora import OnlineLoRAMemory
        except ImportError as exc:
            raise ImportError(
                "Cannot import TTCL. Start clbench with "
                "PYTHONPATH=/home/fengboyu/cl:$PYTHONPATH"
            ) from exc

        self._name = name
        self.memory = OnlineLoRAMemory(
            model_path=model_path,
            output_dir=output_dir,
            device=device,
            dtype=dtype,
            update_every=update_every,
            replay_trajectories=replay_trajectories,
            learning_rate=learning_rate,
            train_epochs=train_epochs,
            max_seq_length=max_seq_length,
            max_new_tokens=max_new_tokens,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=tuple(item.strip() for item in target_modules.split(",") if item.strip()),
            adapter_path=adapter_path,
        )
        self._trajectory: list[dict[str, Any]] = []
        self._last_response: dict[str, Any] | None = None

    def respond(self, query: Query) -> Response:
        schema_json = json.dumps(query.response_schema.model_json_schema(), ensure_ascii=False)
        structured_prompt = (
            f"{query.prompt}\n\n"
            "Return ONLY one valid JSON object. Do not use YAML, Markdown, or explanations. "
            f"The JSON must validate against this JSON schema: {schema_json}"
        )
        raw_response = self.memory.respond(structured_prompt)
        action = self._parse_action(raw_response, query.response_schema)
        self._last_response = {
            "prompt": query.prompt,
            "response": action.model_dump_json(),
        }
        self.memory.record_step(query.prompt, action.model_dump_json())
        self._trajectory.append(self._last_response)
        return Response(
            action=action,
            metadata={
                "system_type": "lora_online",
                "trajectory_count": self.memory.trajectory_count,
                "update_count": self.memory.update_count,
            },
        )

    @staticmethod
    def _parse_action(raw_response: str, schema: type[Any]) -> Any:
        candidates = [raw_response.strip()]
        start, end = raw_response.find("{"), raw_response.rfind("}")
        if start >= 0 and end > start:
            candidates.append(raw_response[start : end + 1])
        for candidate in candidates:
            try:
                return schema.model_validate_json(candidate)
            except (ValidationError, ValueError, json.JSONDecodeError):
                continue
        raise ValueError(f"Model output is not valid {schema.__name__} JSON: {raw_response[:300]!r}")

    def observe(self, observation: Observation, next_query: Query | None = None) -> None:
        if self._last_response is not None:
            self._last_response["result"] = observation.content
            if observation.metadata:
                self._last_response["result_metadata"] = observation.metadata
            self._last_response = None
        if observation_marks_instance_complete(observation):
            if self._trajectory:
                self.memory.observe_trajectory(self._trajectory)
            self._trajectory = []

    def reset(self) -> None:
        self._trajectory = []
        self._last_response = None
        self.memory.reset()

    @property
    def name(self) -> str:
        return self._name

    def get_run_artifacts(self) -> dict[str, Any]:
        return {
            "artifact_type": "lora_online",
            "trajectory_count": self.memory.trajectory_count,
            "update_count": self.memory.update_count,
            "latest_checkpoint": (
                str(self.memory.output_dir / f"update-{self.memory.update_count}")
                if self.memory.update_count
                else None
            ),
        }
