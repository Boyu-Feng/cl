"""
Mem0-inspired vector memory system for continual learning benchmark.

Uses the mem0 SDK (``mem0ai``) for memory management:
- LLM-based fact extraction from conversations
- Vector store with embedding-based semantic search
- Automatic conflict resolution and deduplication
- Memory retrieval injected into context for each query
"""

import logging
import os
import tempfile
from typing import Any

import litellm
from pydantic import BaseModel

# Disable mem0 telemetry to avoid a second qdrant lock on ~/.mem0
os.environ.setdefault("MEM0_TELEMETRY", "false")

from ...interface import (
    ContinualLearningSystem,
    Observation,
    Query,
    Response,
    observation_marks_instance_complete,
)
from ...registry import register_system
from ..utils import (
    TokenBudgetTracker,
    completion_with_structured_output,
)

logger = logging.getLogger(__name__)

litellm.drop_params = True

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

RESPONSE_SYSTEM_PROMPT = """\
You are an agent working on a continual learning benchmark task. You have \
access to a memory store of facts extracted from past interactions. Use \
these memories to inform your response to the current query.

Memories are retrieved by relevance and may come from previous task \
instances. Apply any relevant patterns, strategies, or facts you find."""


# ---------------------------------------------------------------------------
# System
# ---------------------------------------------------------------------------


@register_system("mem0")
class Mem0VectorSystem(ContinualLearningSystem):
    """
    Mem0 SDK-backed vector memory system.

    After each observation the full interaction (query + response + feedback)
    is fed to ``mem0.Memory.add()`` which extracts, deduplicates, and stores
    factual memories.  At query time, ``mem0.Memory.search()`` retrieves the
    most relevant memories which are injected into the prompt.
    """

    def __init__(
        self,
        model: str = "openai/gpt-5",
        extraction_model: str = "",
        embedding_model: str = "text-embedding-3-small",
        system_prompt: str = "",
        top_k: int = 10,
        user_id: str = "benchmark_user",
    ):
        """
        Args:
            model: LiteLLM model for task responses.
            extraction_model: LiteLLM model for mem0 memory extraction
                (defaults to model).
            embedding_model: Embedding model for mem0 vector store.
            system_prompt: Additional system prompt content.
            top_k: Number of memories to retrieve per query.
            user_id: User scope for mem0 memory operations.
        """
        self.model = model
        self.extraction_model = extraction_model or model
        self.embedding_model = embedding_model
        self.user_system_prompt = system_prompt
        self.top_k = top_k
        self.user_id = user_id

        # Conversation context (within current instance)
        self.messages: list[dict[str, str]] = []
        self._token_budget = TokenBudgetTracker()

        # Tracking
        self.interaction_count: int = 0
        self._last_query: Query | None = None
        self._last_response: Response | None = None
        # Last completed turn within the current instance, used to enrich the
        # retrieval key on the next query. Cleared at instance boundaries.
        self._last_turn_prompt: str | None = None
        self._last_turn_response: str | None = None
        self._last_turn_feedback: str | None = None

        # Each instance gets its own temp dir for qdrant so parallel
        # rollouts don't collide on the default /tmp/qdrant path.
        self._storage_dir: str | None = None
        self._mem0 = self._create_mem0()

    def _create_mem0(self):
        """Create a fresh mem0 Memory instance with isolated storage."""
        from mem0 import Memory

        # Clean up previous storage dir if it exists
        self._cleanup_storage()
        self._storage_dir = tempfile.mkdtemp(prefix="mem0_")

        config = {
            "llm": {
                "provider": "litellm",
                "config": {
                    "model": self.extraction_model,
                    "temperature": 0.0,
                },
            },
            "embedder": {
                "provider": "openai",
                "config": {
                    "model": self.embedding_model,
                },
            },
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "collection_name": "mem0_benchmark",
                    "path": self._storage_dir,
                },
            },
            "history_db_path": f"{self._storage_dir}/history.db",
        }
        return Memory.from_config(config)

    def _cleanup_storage(self) -> None:
        """Remove the temporary qdrant storage directory."""
        if self._storage_dir is not None:
            import shutil

            try:
                shutil.rmtree(self._storage_dir, ignore_errors=True)
            except Exception:
                pass
            self._storage_dir = None

    def respond(self, query: Query) -> Response:
        self.interaction_count += 1

        # Retrieve relevant memories via mem0. The retrieval key combines the
        # previous turn (prompt + response + feedback) within this instance
        # with the current query, so search reflects in-progress context
        # without ballooning to the full instance history.
        search_query = self._build_search_query(query.prompt)
        search_results = self._mem0.search(
            search_query,
            filters={"user_id": self.user_id},
            limit=self.top_k,
        )
        memories = search_results.get("results", [])

        # Build prompt with memories
        parts = []
        if memories:
            mem_lines = []
            for m in memories:
                mem_text = m.get("memory", "")
                if mem_text:
                    mem_lines.append(f"- {mem_text}")
            if mem_lines:
                parts.append(
                    "=== RELEVANT MEMORIES ===\n"
                    + "\n".join(mem_lines)
                    + "\n========================="
                )

        parts.append(query.prompt)
        query_content = "\n\n".join(parts)

        self._add_message("user", query_content)

        # Build LLM messages
        llm_messages = self.messages.copy()
        sys_parts = [RESPONSE_SYSTEM_PROMPT]
        if self.user_system_prompt:
            sys_parts.append(self.user_system_prompt)
        llm_messages.insert(0, {"role": "system", "content": "\n\n".join(sys_parts)})

        parsed, usage_event = completion_with_structured_output(
            model=self.model,
            messages=llm_messages,
            response_schema=query.response_schema,
        )
        assistant_record = parsed.model_dump_json()

        if usage_event is not None:
            self._note_prompt_token_usage(usage_event.input_tokens)
            self.record_usage_event(usage_event)

        self._add_message("assistant", assistant_record)

        retrieved_memories = [m.get("memory", "") for m in memories if m.get("memory")]
        response = Response(
            action=parsed,
            metadata={
                "interaction_count": self.interaction_count,
                "system_type": "mem0",
                "model": self.model,
                "retrieved_count": len(retrieved_memories),
                "retrieved_memories": retrieved_memories,
            },
        )
        self._last_query = query
        self._last_response = response
        return response

    def observe(
        self, observation: Observation, next_query: Query | None = None
    ) -> None:
        instance_complete = observation_marks_instance_complete(observation)
        content = observation.content.strip()

        # Store the interaction in mem0
        if self._last_query and self._last_response:
            action = self._last_response.action
            action_str = (
                action.model_dump_json()
                if isinstance(action, BaseModel)
                else str(action)
            )
            messages = [
                {"role": "user", "content": self._last_query.prompt},
                {"role": "assistant", "content": action_str},
            ]
            if content:
                messages.append({"role": "user", "content": f"FEEDBACK: {content}"})
            try:
                self._mem0.add(messages, user_id=self.user_id)
            except Exception:
                logger.warning("mem0 add failed, skipping", exc_info=True)

            # Remember this turn so the next query's retrieval key can include
            # the previous prompt/response/feedback from the same instance.
            self._last_turn_prompt = self._last_query.prompt
            self._last_turn_response = action_str
            self._last_turn_feedback = content or None

        # Add feedback to conversation context
        if content:
            self._add_message("user", f"FEEDBACK: {content}")

        # Clear per-instance conversation context at boundaries
        if instance_complete:
            self.messages = []
            self._last_turn_prompt = None
            self._last_turn_response = None
            self._last_turn_feedback = None

    def reset(self) -> None:
        self._mem0 = self._create_mem0()
        self.messages = []
        self._token_budget.reset()
        self.interaction_count = 0
        self._last_query = None
        self._last_response = None
        self._last_turn_prompt = None
        self._last_turn_response = None
        self._last_turn_feedback = None

    @property
    def name(self) -> str:
        return "mem0"

    def get_run_artifacts(self) -> dict[str, Any] | None:
        try:
            all_memories = self._mem0.get_all(user_id=self.user_id)
            results = all_memories.get("results", [])
            return {
                "artifact_type": "mem0",
                "memories": [
                    {"id": m.get("id", ""), "memory": m.get("memory", "")}
                    for m in results
                ],
                "memory_count": len(results),
            }
        except Exception:
            return {"artifact_type": "mem0", "memories": [], "memory_count": 0}

    # ---- internal ----

    def _add_message(self, role: str, content: str) -> None:
        self.messages.append({"role": role, "content": content})

    def _build_search_query(self, current_prompt: str) -> str:
        if self._last_turn_prompt is None:
            return current_prompt
        parts = [
            f"PREVIOUS QUERY: {self._last_turn_prompt}",
            f"PREVIOUS RESPONSE: {self._last_turn_response or ''}",
        ]
        if self._last_turn_feedback:
            parts.append(f"PREVIOUS FEEDBACK: {self._last_turn_feedback}")
        parts.append(f"CURRENT QUERY: {current_prompt}")
        return "\n".join(parts)

    def _estimate_message_tokens(self, messages: list[dict[str, str]]) -> int:
        if not messages:
            return 0
        try:
            return litellm.token_counter(model=self.model, messages=messages)
        except Exception:
            total = sum(4 + len(m.get("content", "")) // 4 for m in messages) + 2
            return total

    def _count_message_tokens(self, messages: list[dict[str, str]]) -> int:
        return self._token_budget.count(messages, self._estimate_message_tokens)

    def _note_prompt_token_usage(self, input_tokens: int | None) -> None:
        self._token_budget.note_usage(
            messages=self.messages,
            input_tokens=input_tokens,
            estimate_fn=self._estimate_message_tokens,
        )
