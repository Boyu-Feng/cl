"""Delta-Mem system for CLBench stateless/stateful reward and gain runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from ...interface import ContinualLearningSystem, Observation, Query, Response, observation_marks_instance_complete
from ...registry import register_system


@register_system("deltamem")
class DeltaMemSystem(ContinualLearningSystem):
    """Load one Delta-Mem adapter; state persists across instances in rollout runs.

    CLBench's baseline runner constructs this system in an isolated process per
    instance, which provides the paper's stateless (sl) condition automatically.
    The normal rollout keeps one object, so the online Delta-Mem matrix carries
    information from instance t to t+1.
    """

    supports_baseline = True
    parallel_safe = False

    def __init__(self, model_path: str, delta_adapter: str, device: str = "cuda:0",
                 dtype: str = "bfloat16", attn_implementation: str = "flash_attention_2",
                 max_new_tokens: int = 256, output_dir: str = "results/deltamem",
                 name: str = "deltamem", device_map: str = "single") -> None:
        from deltamem.runtime.session import DeltaMemChatSession, load_delta_mem_chat_model
        self._name = name
        self.max_new_tokens = max_new_tokens
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        model, tokenizer = load_delta_mem_chat_model(
            model_path=model_path, device=device, dtype=dtype,
            attn_implementation=attn_implementation, adapter_dir=delta_adapter,
            device_map=device_map,
        )
        input_device = str(model.get_input_embeddings().weight.device)
        self.device_map = {key: str(value) for key, value in model.hf_device_map.items()}
        print(f"Delta-Mem device map: {json.dumps(self.device_map)}", flush=True)
        self.session = DeltaMemChatSession(model=model, tokenizer=tokenizer, device=input_device)
        self._last: dict[str, Any] | None = None
        self._events: list[dict[str, Any]] = []
        self._instance = 0

    def respond(self, query: Query) -> Response:
        schema = json.dumps(query.response_schema.model_json_schema(), ensure_ascii=False)
        prompt = (f"{query.prompt}\n\nReturn ONLY one valid JSON object matching this schema. "
                  f"Do not return Markdown or explanations. JSON schema: {schema}")
        result = self.session.generate_reply(prompt, max_new_tokens=self.max_new_tokens,
                                             do_sample=False, write_enabled=True,
                                             include_debug=True)
        raw = str(result["assistant_display"])
        action = self._parse(raw, query.response_schema)
        self._last = {"instance_id": query.instance_id, "instance_index": query.instance_index,
                      "prompt": query.prompt, "response": action.model_dump(mode="json")}
        return Response(action=action, metadata={"system_type": "deltamem",
            "delta_state_stats": result.get("state_stats", {}),
            "turn_stats": result.get("turn_stats", {})})

    def observe(self, observation: Observation, next_query: Query | None = None) -> None:
        if self._last is None:
            return
        event = dict(self._last)
        event["observation"] = observation.content
        event["observation_metadata"] = observation.metadata or {}
        event["delta_state_stats"] = self.session.state_stats()
        self._events.append(event)
        self._last = None
        if observation_marks_instance_complete(observation):
            self._instance += 1
            path = self.output_dir / "process.jsonl"
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"instance": self._instance, **event}, ensure_ascii=False) + "\n")

    def reset(self) -> None:
        self.session.reset()
        self._last = None
        self._events = []
        self._instance = 0

    @staticmethod
    def _parse(raw: str, schema: type[Any]) -> Any:
        candidates = [raw.strip()]
        start, end = raw.find("{"), raw.rfind("}")
        if start >= 0 and end > start:
            candidates.append(raw[start:end + 1])
        for candidate in candidates:
            try:
                return schema.model_validate_json(candidate)
            except (ValidationError, ValueError, json.JSONDecodeError):
                pass
        raise ValueError(f"Model output is not valid {schema.__name__}: {raw[:400]!r}")

    @property
    def name(self) -> str:
        return self._name

    def get_run_artifacts(self) -> dict[str, Any]:
        return {"artifact_type": "deltamem", "process_trace": str(self.output_dir / "process.jsonl"),
                "device_map": self.device_map,
                "instances_observed": self._instance, "final_delta_state_stats": self.session.state_stats()}
