"""
Reflector agent for ACE system.
Analyzes generator outputs and provides feedback on bullet usage.
"""

import json
from typing import Any, Dict, List, Optional, Tuple

from ..playbook_utils import extract_json_from_text
from ..prompts.reflector import REFLECTOR_PROMPT, REFLECTOR_PROMPT_NO_GT
from ..llm import timed_llm_call


class Reflector:
    """
    Reflector agent that analyzes the generator's reasoning and tags
    bullets as helpful, harmful, or neutral.
    """

    def __init__(
        self,
        api_client,
        api_provider,
        model: str,
        max_tokens: int = 4096,
        model_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the Reflector agent.

        Args:
            api_client: OpenAI client for LLM calls
            api_provider: API provider for LLM calls
            model: Model name to use for reflection
            max_tokens: Maximum tokens for reflection
        """
        self.api_client = api_client
        self.api_provider = api_provider
        self.model = model
        self.max_tokens = max_tokens
        self.model_kwargs = model_kwargs or {}

    def reflect(
        self,
        question: str,
        reasoning_trace: str,
        predicted_answer: str,
        ground_truth: Optional[str],
        environment_feedback: str,
        bullets_used: str,
        use_ground_truth: bool = True,
        use_json_mode: bool = False,
        call_id: str = "reflect",
        log_dir: Optional[str] = None,
    ) -> Tuple[str, List[Dict[str, str]], Dict[str, Any]]:
        """
        Analyze the generator's output and tag bullets.

        Args:
            question: The original question
            reasoning_trace: The generator's reasoning
            predicted_answer: The generator's predicted answer
            ground_truth: The ground truth answer (if available)
            environment_feedback: Feedback about correctness
            bullets_used: String representation of bullets used
            use_ground_truth: Whether to use ground truth in reflection
            use_json_mode: Whether to use JSON mode
            call_id: Unique identifier for this call
            log_dir: Directory for logging

        Returns:
            Tuple of (reflection_content, bullet_tags, call_info)
        """
        # Select the appropriate prompt
        if use_ground_truth and ground_truth:
            prompt = REFLECTOR_PROMPT.format(
                question,
                reasoning_trace,
                predicted_answer,
                ground_truth,
                environment_feedback,
                bullets_used,
            )
        else:
            prompt = REFLECTOR_PROMPT_NO_GT.format(
                question,
                reasoning_trace,
                predicted_answer,
                environment_feedback,
                bullets_used,
            )

        response, call_info = timed_llm_call(
            self.api_client,
            self.api_provider,
            self.model,
            prompt,
            role="reflector",
            call_id=call_id,
            max_tokens=self.max_tokens,
            log_dir=log_dir,
            use_json_mode=use_json_mode,
            extra_api_params=self.model_kwargs,
        )

        try:
            reflection_payload = self._extract_and_validate_reflection(response)
        except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            print(f"Warning: Failed to parse reflector JSON response: {exc}")
            reflection_payload = {"bullet_tags": []}

        return response, reflection_payload.get("bullet_tags", []), call_info

    def _extract_and_validate_reflection(self, response: str) -> Dict[str, Any]:
        payload = extract_json_from_text(response)
        if not payload:
            raise ValueError("Failed to extract valid JSON from reflector response")
        if not isinstance(payload, dict):
            raise ValueError("Reflector response must be a JSON object")

        required_string_fields = [
            "reasoning",
            "error_identification",
            "root_cause_analysis",
            "correct_approach",
            "key_insight",
        ]
        for field in required_string_fields:
            if field not in payload:
                raise ValueError(f"Reflector JSON missing required '{field}' field")
            if not isinstance(payload[field], str):
                raise ValueError(f"Reflector '{field}' field must be a string")

        bullet_tags = payload.get("bullet_tags", [])
        if not isinstance(bullet_tags, list):
            raise ValueError("Reflector 'bullet_tags' field must be a list")
        for i, tag in enumerate(bullet_tags):
            if not isinstance(tag, dict):
                raise ValueError(f"Bullet tag {i} must be a dictionary")
            if "id" not in tag or "tag" not in tag:
                raise ValueError(
                    f"Bullet tag {i} must include both 'id' and 'tag' fields"
                )
            if not isinstance(tag["id"], str) or not isinstance(tag["tag"], str):
                raise ValueError("Bullet tag 'id' and 'tag' fields must be strings")

        return payload
