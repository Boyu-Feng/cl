"""Codex CLI system for continual learning benchmark.

Runs codex inside a Docker container for full environment isolation.
Conversation resume uses `codex exec resume` whenever a prior Codex
conversation id exists. File memory is existing `.codex/memories` contents plus
any configured workspace file pattern snapshotted after each successful turn.
"""

from __future__ import annotations

import json
import logging
import os
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Any, Optional

from ...interface import (
    ContinualLearningSystem,
    Observation,
    Query,
    Response,
    observation_marks_instance_complete,
)
from ...registry import register_system
from ...usage import build_usage_event
from ..common import (
    CONTAINER_WORKSPACE,
    DEFAULT_DOCKER_IMAGE,
    cleanup_run_workspace,
    create_run_workspace,
    docker_exec,
    load_skill_prompt_prefix,
    merge_memory_snapshots,
    normalize_other_memory_file_patterns,
    read_memory_snapshot,
    read_other_memory_files,
    start_docker_container,
    stop_docker_container,
)
from ..utils.structured_output import (
    extract_json,
    schema_to_prompt_instruction,
    validate_with_coercion,
)
from .artifacts import ensure_registered as ensure_artifact_exporter_registered


ensure_artifact_exporter_registered()

logger = logging.getLogger(__name__)

_CONTAINER_HOME = f"{CONTAINER_WORKSPACE}/.codex"
_MEMORY_DIR = ".codex/memories"

_EMPTY_TOKENS: dict[str, int] = {
    "input_tokens": 0,
    "output_tokens": 0,
    "reasoning_tokens": 0,
    "cached_input_tokens": 0,
}

_DOCKER_IMAGE = DEFAULT_DOCKER_IMAGE

_REASONING_EFFORT_VALUES = frozenset({"none", "minimal", "low", "medium", "high"})


def _parse_events(stdout: str) -> list[dict[str, Any]]:
    """Parse JSONL lines from codex stdout, skipping blank/malformed lines."""
    events: list[dict[str, Any]] = []
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            logger.warning("Skipping malformed JSONL line: %.120s", line)
    return events


def _extract_assistant_text(events: list[dict[str, Any]]) -> str:
    """Extract the last assistant message text from parsed codex events.

    Raises ValueError if no assistant message is found.
    """
    last_text: str | None = None
    last_event: dict[str, Any] | None = None
    for event in events:
        if event.get("type") != "turn.completed":
            last_event = event
            continue
        if not last_event:
            raise ValueError("No last_event found")
        payload = last_event.get("item", {})

        last_text = payload.get("text", "")

    if last_text is None:
        raise ValueError("No assistant message found in codex output")
    return last_text


def _extract_token_usage(events: list[dict[str, Any]]) -> dict[str, int]:
    """Extract the latest usage payload from one Codex invocation."""
    usage, _ = _extract_token_usage_with_source(events)
    return usage


def _extract_token_usage_with_source(
    events: list[dict[str, Any]],
) -> tuple[dict[str, int], bool]:
    """Return latest usage payload and whether it came from ``last_usage``.

    Current Codex JSONL emits ``turn.completed.usage`` from
    ``ThreadTokenUsage.total``.  If Codex adds per-call ``last_usage`` later, we
    can use that directly and skip benchmark-side deltas.
    """
    latest = dict(_EMPTY_TOKENS)
    completed_count = 0
    latest_used_last_usage = False
    for idx, event in enumerate(events):
        if event.get("type") != "turn.completed":
            continue
        completed_count += 1
        usage = event.get("last_usage")
        used_last_usage = isinstance(usage, dict)
        if not used_last_usage:
            usage = event.get("usage", {}) or {}
        input_tokens = int(usage.get("input_tokens", 0) or 0)
        output_tokens = int(usage.get("output_tokens", 0) or 0)
        reasoning_tokens = int(
            usage.get("reasoning_tokens", usage.get("reasoning_output_tokens", 0)) or 0
        )
        cached_input_tokens = int(usage.get("cached_input_tokens", 0) or 0)
        logger.debug(
            "Codex usage event[%d] turn.completed #%d raw_usage=%r parsed=(input=%d, output=%d, reasoning=%d, cached_input=%d)",
            idx,
            completed_count,
            usage,
            input_tokens,
            output_tokens,
            reasoning_tokens,
            cached_input_tokens,
        )
        latest["input_tokens"] = input_tokens
        latest["cached_input_tokens"] = cached_input_tokens
        latest["output_tokens"] = output_tokens
        latest["reasoning_tokens"] = reasoning_tokens
        latest_used_last_usage = used_last_usage
    logger.debug(
        "Codex token usage from latest of %d turn.completed events: input=%d output=%d reasoning=%d cached_input=%d total_io=%d uncached_input=%d source=%s",
        completed_count,
        latest["input_tokens"],
        latest["output_tokens"],
        latest["reasoning_tokens"],
        latest["cached_input_tokens"],
        latest["input_tokens"] + latest["output_tokens"],
        max(0, latest["input_tokens"] - latest["cached_input_tokens"]),
        "last_usage" if latest_used_last_usage else "usage",
    )
    return latest, latest_used_last_usage


def _token_usage_delta(
    current: dict[str, int], previous: dict[str, int] | None
) -> dict[str, int]:
    """Convert cumulative Codex usage totals into one-call usage."""
    if previous is None:
        return dict(current)
    delta: dict[str, int] = {}
    for key in _EMPTY_TOKENS:
        current_value = current.get(key, 0)
        previous_value = previous.get(key, 0)
        delta[key] = (
            current_value - previous_value
            if current_value >= previous_value
            else current_value
        )
    return delta


def _extract_cumulative_token_usage(events: list[dict[str, Any]]) -> dict[str, int]:
    """Extract latest cumulative ``usage`` payload from Codex JSONL events."""
    latest = dict(_EMPTY_TOKENS)
    for event in events:
        if event.get("type") != "turn.completed":
            continue
        usage = event.get("usage", {}) or {}
        latest["input_tokens"] = int(usage.get("input_tokens", 0) or 0)
        latest["output_tokens"] = int(usage.get("output_tokens", 0) or 0)
        latest["reasoning_tokens"] = int(
            usage.get("reasoning_tokens", usage.get("reasoning_output_tokens", 0)) or 0
        )
        latest["cached_input_tokens"] = int(usage.get("cached_input_tokens", 0) or 0)
    return latest


def _sum_token_usages(usages: list[dict[str, int]]) -> dict[str, int]:
    """Combine multiple Codex invocations within one benchmark interaction."""
    return {key: sum(usage.get(key, 0) for usage in usages) for key in _EMPTY_TOKENS}


def _debug_log_pricing_details(
    *,
    model: str,
    provider: str,
    token_usage: dict[str, int],
    usage_event: Any,
) -> None:
    """Emit detailed Codex cost diagnostics without affecting pricing behavior."""
    input_tokens = token_usage.get("input_tokens", 0)
    output_tokens = token_usage.get("output_tokens", 0)
    cached_input_tokens = token_usage.get("cached_input_tokens", 0)
    reasoning_tokens = token_usage.get("reasoning_tokens", 0)
    uncached_input_tokens = max(0, input_tokens - cached_input_tokens)

    logger.debug(
        "Codex cost inputs: model=%s provider=%s input=%d output=%d reasoning=%d cached_input=%d uncached_input=%d",
        model,
        provider,
        input_tokens,
        output_tokens,
        reasoning_tokens,
        cached_input_tokens,
        uncached_input_tokens,
    )
    logger.debug(
        "Codex usage event pricing result: cost_usd=%r pricing_source=%r pricing_error=%r total_tokens=%r cache_creation_input_tokens=%r",
        getattr(usage_event, "cost_usd", None),
        getattr(usage_event, "pricing_source", None),
        getattr(usage_event, "pricing_error", None),
        getattr(usage_event, "total_tokens", None),
        getattr(usage_event, "cache_creation_input_tokens", None),
    )

    try:
        import litellm
    except Exception as exc:  # pragma: no cover - diagnostic only
        logger.debug(
            "Codex pricing table debug unavailable: failed to import litellm: %s", exc
        )
        return

    candidate_keys = [model, model.split("/", 1)[-1], f"{provider}/{model}"]
    seen: set[str] = set()
    for key in candidate_keys:
        if key in seen:
            continue
        seen.add(key)
        pricing = getattr(litellm, "model_cost", {}).get(key)
        logger.debug(
            "Codex LiteLLM pricing lookup key=%r found=%s entry=%r",
            key,
            bool(pricing),
            pricing,
        )
        if not pricing:
            continue

        input_rate = pricing.get("input_cost_per_token") or 0.0
        output_rate = pricing.get("output_cost_per_token") or 0.0
        cache_read_rate = pricing.get("cache_read_input_token_cost", input_rate)
        if cache_read_rate is None:
            cache_read_rate = input_rate
        explicit_estimate = (
            uncached_input_tokens * input_rate
            + cached_input_tokens * cache_read_rate
            + output_tokens * output_rate
        )
        logger.debug(
            "Codex explicit pricing estimate for key=%r: uncached_input(%d)*%.12g + cached_input(%d)*%.12g + output(%d)*%.12g = %.12g",
            key,
            uncached_input_tokens,
            input_rate,
            cached_input_tokens,
            cache_read_rate,
            output_tokens,
            output_rate,
            explicit_estimate,
        )


def _json_candidates(text: str) -> list[str]:
    """Return JSON-like candidates from an assistant message, most recent last."""
    stripped = text.strip()
    if not stripped:
        return []

    candidates: list[str] = [stripped]

    if "```" in stripped:
        parts = stripped.split("```")
        for part in parts[1::2]:
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            if part.startswith("{") or part.startswith("["):
                candidates.append(part)

    decoder = json.JSONDecoder()
    for idx, char in enumerate(stripped):
        if char not in "{[":
            continue
        try:
            _, end = decoder.raw_decode(stripped[idx:])
        except json.JSONDecodeError:
            continue
        candidates.append(stripped[idx : idx + end])

    deduped: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        deduped.append(candidate)
    return deduped


def _parse_action_text(text: str, schema: type[Any]) -> Any:
    """Parse assistant text into the target schema, preferring the latest JSON."""
    last_error: Exception | None = None
    for candidate in reversed(_json_candidates(text)):
        try:
            return validate_with_coercion(candidate, schema)
        except Exception as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    return validate_with_coercion(extract_json(text), schema)


def _extract_session_ids(events: list[dict[str, Any]]) -> list[str]:
    """Return Codex conversation/thread ids observed in JSONL event order."""
    session_ids: list[str] = []
    seen: set[str] = set()
    for event in events:
        raw_id: Any = None
        if event.get("type") == "session_meta":
            payload = event.get("payload")
            if isinstance(payload, dict):
                raw_id = payload.get("id")
        elif event.get("type") == "thread.started":
            raw_id = event.get("thread_id")

        if raw_id is None:
            continue
        session_id = str(raw_id)
        if not session_id or session_id in seen:
            continue
        seen.add(session_id)
        session_ids.append(session_id)
    return session_ids


@register_system("codex")
class CodexSystem(ContinualLearningSystem):
    """Codex CLI system running inside a Docker container."""

    def __init__(
        self,
        model: str = "gpt-5.4-mini",
        name: str = "codex",
        timeout: int = 300,
        use_oauth: bool = False,
        docker_image: str = _DOCKER_IMAGE,
        version: str = "0.125.0",
        single_conversation: bool = True,
        disable_native_memories: bool = False,
        preamble_file: Optional[str] = None,
        skill: Optional[str] = None,
        invocation_type: str = "skill",
        other_memory_files: Optional[list[str]] = None,
        reasoning_effort: str = "none",
        model_auto_compact_token_limit: int | None = None,
    ):
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not use_oauth and not api_key:
            raise ValueError("OPENAI_API_KEY must be set when use_oauth is False")

        if (
            reasoning_effort is not None
            and reasoning_effort not in _REASONING_EFFORT_VALUES
        ):
            raise ValueError(
                f"reasoning_effort must be one of {sorted(_REASONING_EFFORT_VALUES)} or None, "
                f"got {reasoning_effort!r}"
            )

        if model_auto_compact_token_limit is not None and (
            isinstance(model_auto_compact_token_limit, bool)
            or not isinstance(model_auto_compact_token_limit, int)
            or model_auto_compact_token_limit < 1
        ):
            raise ValueError(
                "model_auto_compact_token_limit must be a positive int or None, "
                f"got {model_auto_compact_token_limit!r}"
            )

        if not isinstance(single_conversation, bool):
            raise ValueError(
                f"single_conversation must be a bool, got {single_conversation!r}"
            )

        if not isinstance(disable_native_memories, bool):
            raise ValueError(
                "disable_native_memories must be a bool, "
                f"got {disable_native_memories!r}"
            )

        self._name = name
        self._model = model
        self._timeout = timeout
        self._use_oauth = use_oauth
        self._api_key = api_key
        self._single_conversation = single_conversation
        self._disable_native_memories = disable_native_memories
        self._reasoning_effort = reasoning_effort
        self._model_auto_compact_token_limit = model_auto_compact_token_limit
        self._preamble = self._read_preamble_file(preamble_file)
        self._skill = skill
        self._invocation_type = invocation_type
        self._skill_prompt_prefix = ""
        self._other_memory_files = normalize_other_memory_file_patterns(
            other_memory_files
        )

        self._tmp_dir = create_run_workspace("codex_bench")

        if use_oauth:
            src_auth = Path.home() / ".codex" / "auth.json"
            if not src_auth.exists():
                raise FileNotFoundError(f"OAuth auth.json not found at {src_auth}")
            shutil.copy2(src_auth, Path(self._tmp_dir) / "auth.json")

        self._initialize_workspace()

        self._container_id: str | None = None
        self._docker_image = docker_image
        self._version = version
        self._conversation_id: str | None = None
        self._interaction_count: int = 0
        self._jsonl_log: list[dict[str, Any]] = []
        self._cumulative_tokens: dict[str, int] = dict(_EMPTY_TOKENS)
        self._last_cumulative_token_usage_by_conversation: dict[
            str, dict[str, int]
        ] = {}
        self._memory_history: list[dict[str, Any]] = []
        self._conversation_session_ids: set[str] = set()
        self._conversation_history: list[dict[str, Any]] = []
        self._instance_conversations: dict[
            tuple[str | None, int | None], dict[str, Any]
        ] = {}
        self._pending_feedback: str | None = None

        self._start_container()

    def _read_preamble_file(self, preamble_file: Optional[str]) -> str | None:
        """Read optional markdown pre-amble content from the host filesystem."""
        if preamble_file is None:
            return None
        path = Path(preamble_file).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"preamble_file does not exist: {path}")
        return path.read_text()

    def _initialize_workspace(self) -> None:
        """Create Codex home and install optional native skill files."""
        home = Path(self._tmp_dir) / ".codex"
        home.mkdir(parents=True, exist_ok=True)
        self._skill_prompt_prefix = load_skill_prompt_prefix(
            self._skill,
            agent_name="codex",
            workspace_dir=self._tmp_dir,
            invocation_type=self._invocation_type,
        )

    def _start_container(self) -> None:
        """Start a persistent Docker container with codex installed."""
        self._container_id = start_docker_container(
            name_prefix="codex_bench",
            host_workspace=self._tmp_dir,
            docker_image=self._docker_image,
            env={
                "OPENAI_API_KEY": self._api_key,
                "CODEX_HOME": _CONTAINER_HOME,
            },
            logger=logger,
        )

        # Install ca-certificates (slim images lack them) and codex
        logger.info("Installing ca-certificates + codex@%s...", self._version)
        result = self._docker_exec(
            "apt-get update -qq && apt-get install -y -qq ca-certificates > /dev/null 2>&1 && "
            f"npm install -g @openai/codex@{self._version}",
            timeout=120,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Failed to install codex: {result.stderr}")
        logger.info("Codex installed")

        # Symlink node/codex to /usr/local/bin for consistent PATH
        self._docker_exec(
            "for bin in node codex; do"
            '  BIN_PATH="$(which "$bin" 2>/dev/null || true)";'
            '  if [ -n "$BIN_PATH" ] && [ "$BIN_PATH" != "/usr/local/bin/$bin" ]; then'
            '    ln -sf "$BIN_PATH" "/usr/local/bin/$bin";'
            "  fi;"
            " done",
            timeout=10,
        )

        # Verify codex is accessible
        verify = self._docker_exec("codex --version", timeout=10)
        logger.info("Verified: %s", verify.stdout.strip() or verify.stderr.strip())

        # Write auth.json — codex requires this file, not just the env var
        if self._use_oauth:
            self._docker_exec(
                f"mkdir -p {_CONTAINER_HOME} && "
                f"cp {CONTAINER_WORKSPACE}/auth.json {_CONTAINER_HOME}/auth.json",
                timeout=10,
            )
        else:
            self._docker_exec(
                f"mkdir -p {_CONTAINER_HOME} && "
                f"echo '{{\"OPENAI_API_KEY\": \"'$OPENAI_API_KEY'\"}}' > {_CONTAINER_HOME}/auth.json",
                timeout=10,
            )
        logger.info("Auth configured, container ready")

    def _docker_exec(
        self,
        command: str,
        *,
        timeout: int | None = None,
    ) -> subprocess.CompletedProcess[str]:
        """Run a shell command inside the container."""
        return docker_exec(
            self._container_id,
            command,
            timeout=timeout,
            default_timeout=self._timeout,
        )

    def _read_memory_snapshot(self) -> dict[str, str]:
        """Read all present file memory into one relpath -> contents snapshot."""
        memory_files, _, _ = merge_memory_snapshots(
            self._read_native_memory_files(),
            self._read_watched_memory_files(),
            native_source="codex_memory_file",
        )
        return memory_files

    def _read_native_memory_files(self) -> dict[str, str]:
        """Read files from Codex's native memory directory when present."""
        return read_memory_snapshot(
            [Path(self._tmp_dir) / _MEMORY_DIR],
            logger=logger,
        )

    def _read_watched_memory_files(self) -> dict[str, str]:
        """Read configured workspace memory files when present."""
        return read_other_memory_files(
            self._tmp_dir,
            self._other_memory_files,
            logger=logger,
        )

    @property
    def _sessions_dir(self) -> Path:
        """Host-side Codex sessions directory for this run workspace."""
        return Path(self._tmp_dir) / ".codex" / "sessions"

    def _observed_session_ids(self) -> set[str]:
        """Return all Codex session ids observed or currently selected."""
        session_ids = set(self._conversation_session_ids)
        if self._conversation_id is not None:
            session_ids.add(str(self._conversation_id))
        return session_ids

    def _read_conversation_jsonl_files(self) -> dict[str, str]:
        """Read native Codex session JSONL files for observed conversations."""
        session_ids = self._observed_session_ids()
        if not session_ids:
            return {}

        sessions_dir = self._sessions_dir
        if not sessions_dir.exists():
            return {}

        files: dict[str, str] = {}
        workspace_dir = Path(self._tmp_dir)
        for path in sorted(sessions_dir.rglob("*.jsonl")):
            if not path.is_file() or not self._session_file_matches_ids(
                path,
                session_ids,
            ):
                continue
            try:
                rel = path.relative_to(workspace_dir).as_posix()
            except ValueError:
                logger.warning("Skipping Codex JSONL outside workspace: %s", path)
                continue
            try:
                files[rel] = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError) as exc:
                logger.warning("Could not read Codex session JSONL %s: %s", path, exc)
        return files

    @staticmethod
    def _session_file_matches_ids(path: Path, session_ids: set[str]) -> bool:
        """Return whether a Codex rollout JSONL belongs to an observed id."""
        stem = path.stem
        if stem in session_ids or any(stem.endswith(f"-{sid}") for sid in session_ids):
            return True

        try:
            with path.open("r", encoding="utf-8") as fh:
                first_line = fh.readline()
        except (OSError, UnicodeDecodeError) as exc:
            logger.warning("Could not inspect Codex session JSONL %s: %s", path, exc)
            return False

        try:
            first_event = json.loads(first_line) if first_line.strip() else {}
        except json.JSONDecodeError:
            return False
        payload = first_event.get("payload") if isinstance(first_event, dict) else None
        if not isinstance(payload, dict):
            return False
        payload_id = payload.get("id")
        return payload_id is not None and str(payload_id) in session_ids

    @staticmethod
    def _summarize_conversation_files(
        files: dict[str, str],
    ) -> tuple[list[str], dict[str, int]]:
        """Return stable file names and character counts for Codex JSONL files."""
        file_names = sorted(files)
        file_sizes = {name: len(files[name]) for name in file_names}
        return file_names, file_sizes

    def _record_conversation_snapshot(
        self,
        query: Query,
        *,
        step: int,
    ) -> dict[str, Any]:
        """Capture latest native Codex session files for an instance."""
        conversation_files = self._read_conversation_jsonl_files()
        file_names, file_sizes = self._summarize_conversation_files(conversation_files)
        session_ids = sorted(self._observed_session_ids())
        summary = {
            "step": step,
            "instance_id": query.instance_id,
            "instance_index": query.instance_index,
            "conversation_id": self._conversation_id,
            "session_ids": session_ids,
            "conversation_jsonl_files": file_names,
            "conversation_jsonl_file_sizes": file_sizes,
        }
        self._conversation_history.append(summary)
        self._instance_conversations[(query.instance_id, query.instance_index)] = {
            **summary,
            "conversation_jsonl_files": conversation_files,
        }
        return summary

    def _sorted_instance_conversations(self) -> list[dict[str, Any]]:
        """Return latest per-instance Codex conversation snapshots in task order."""

        def sort_key(snapshot: dict[str, Any]) -> tuple[int, int | str, str, int]:
            instance_index = snapshot.get("instance_index")
            if isinstance(instance_index, int):
                primary: tuple[int, int | str] = (0, instance_index)
            else:
                primary = (1, str(snapshot.get("instance_id") or ""))
            return (
                *primary,
                str(snapshot.get("instance_id") or ""),
                int(snapshot.get("step") or 0),
            )

        return sorted(self._instance_conversations.values(), key=sort_key)

    def _build_command(self, prompt: str) -> str:
        """Build the codex exec command string for docker exec.

        The prompt is passed as a single positional argument. Only the prompt
        itself is shell-escaped with shlex.quote.
        """
        options = [
            "--model",
            self._model,
            "--json",
            "--dangerously-bypass-approvals-and-sandbox",
            "--skip-git-repo-check",
        ]
        if self._reasoning_effort is not None:
            options.extend(["-c", f'model_reasoning_effort="{self._reasoning_effort}"'])
        if self._model_auto_compact_token_limit is not None:
            options.extend(
                [
                    "-c",
                    f"model_auto_compact_token_limit={self._model_auto_compact_token_limit}",
                ]
            )
        prefix = ["codex"]
        if not self._disable_native_memories:
            prefix.extend(["--enable", "memories"])
            options.extend(
                [
                    "-c",
                    "memories.use_memories=true",
                    "-c",
                    "memories.generate_memories=true",
                ]
            )
        conversation_id = self._conversation_id
        if conversation_id is not None:
            parts = [*prefix, "exec", "resume", *options, conversation_id]
        else:
            # Fresh instance-start turns are still persistent so later turns in
            # the same benchmark instance can resume this newly-created session.
            parts = [*prefix, "exec", *options]

        return " ".join([*parts, "--", shlex.quote(prompt)])

    def _build_prompt(self, query: Query, *, include_skill: bool = True) -> str:
        parts: list[str] = []
        if self._preamble:
            parts.append(f"PRE-AMBLE:\n{self._preamble}")
        if self._pending_feedback:
            parts.append(f"FEEDBACK FROM PREVIOUS ACTION: {self._pending_feedback}\n")
        parts.append(query.prompt)
        parts.append(schema_to_prompt_instruction(query.response_schema))
        prompt = "\n".join(parts)
        if include_skill:
            return f"{self._skill_prompt_prefix}{prompt}"
        return prompt

    def _prepare_conversation(self, query: Query) -> dict[str, Any]:
        """Apply conversation policy and return metadata for this turn."""
        previous_conversation_id = self._conversation_id
        resumed_this_turn = self._conversation_id is not None
        return {
            "mode": "single_conversation"
            if self._single_conversation
            else "per_instance_conversation",
            "single_conversation": self._single_conversation,
            "instance_id": query.instance_id,
            "instance_index": query.instance_index,
            "invocation_mode": "resume" if resumed_this_turn else "fresh_persistent",
            "resumed_this_turn": resumed_this_turn,
            "conversation_id_before": previous_conversation_id,
            "conversation_id_for_turn": self._conversation_id,
        }

    def _run_turn(self, prompt: str) -> list[dict[str, Any]]:
        cmd = self._build_command(prompt)

        logger.info(
            "Calling codex (interaction %d, prompt %d chars)",
            self._interaction_count + 1,
            len(prompt),
        )

        result = self._docker_exec(cmd, timeout=self._timeout)
        events = _parse_events(result.stdout)

        if result.returncode != 0:
            has_completed_turn = any(e.get("type") == "turn.completed" for e in events)
            has_assistant_message = any(
                e.get("type") == "item.completed"
                and (e.get("item") or {}).get("type") == "agent_message"
                for e in events
            )
            if has_completed_turn and has_assistant_message:
                logger.warning(
                    "Codex exited non-zero after a complete turn; continuing. stderr=%s",
                    result.stderr.strip()[:500],
                )
            else:
                detail = result.stderr.strip() or result.stdout.strip()
                logger.error(
                    "Codex failed (rc=%d): %s", result.returncode, detail[:500]
                )
                raise RuntimeError(
                    f"Codex exited with code {result.returncode}: {detail}"
                )

        logger.info("Codex returned %d events", len(events))
        self._jsonl_log.extend(events)

        session_ids = _extract_session_ids(events)
        self._conversation_session_ids.update(session_ids)
        if session_ids:
            self._conversation_id = session_ids[-1]

        return events

    def _usage_for_invocation(self, events: list[dict[str, Any]]) -> dict[str, int]:
        """Return per-call usage for one Codex invocation.

        Codex currently exposes cumulative thread totals in ``turn.completed.usage``.
        Resume output can also replay prior ``turn.completed`` events, so we keep
        the latest total per conversation and delta all token buckets.
        """
        usage, is_last_usage = _extract_token_usage_with_source(events)
        conversation_id = self._conversation_id
        if is_last_usage:
            cumulative_usage = _extract_cumulative_token_usage(events)
            if conversation_id is not None:
                self._last_cumulative_token_usage_by_conversation[conversation_id] = (
                    cumulative_usage
                )
            return usage

        if conversation_id is None:
            return usage

        previous = self._last_cumulative_token_usage_by_conversation.get(
            conversation_id
        )
        self._last_cumulative_token_usage_by_conversation[conversation_id] = dict(usage)
        return _token_usage_delta(usage, previous)

    def _build_repair_prompt(
        self,
        original_prompt: str,
        assistant_text: str,
        validation_error: str,
    ) -> str:
        return "\n".join(
            [
                "Your previous response did not match the required benchmark schema.",
                f"Validation error: {validation_error}",
                "Previous response:",
                assistant_text,
                "Return ONLY a corrected JSON object for the same task.",
                "Original prompt:",
                original_prompt,
            ]
        )

    def _run_action_turn(
        self,
        prompt: str,
        response_schema: type[Any],
        *,
        repair_original_prompt: str,
    ) -> tuple[Any, list[dict[str, int]], bool]:
        events = self._run_turn(prompt)
        assistant_text = _extract_assistant_text(events)
        invocation_usages = [self._usage_for_invocation(events)]

        try:
            action = _parse_action_text(assistant_text, response_schema)
            return action, invocation_usages, False
        except Exception as exc:
            repair_prompt = self._build_repair_prompt(
                repair_original_prompt,
                assistant_text,
                str(exc),
            )
            repair_events = self._run_turn(repair_prompt)
            repaired_text = _extract_assistant_text(repair_events)
            action = _parse_action_text(repaired_text, response_schema)
            invocation_usages.append(self._usage_for_invocation(repair_events))
            return action, invocation_usages, True

    def _record_turn_usage(
        self, invocation_usages: list[dict[str, int]]
    ) -> dict[str, int]:
        token_usage = _sum_token_usages(invocation_usages)
        logger.debug(
            "Codex respond token usage before cumulative update: current=%r per_invocation=%r cumulative_before=%r",
            token_usage,
            invocation_usages,
            self._cumulative_tokens,
        )
        self._add_to_cumulative_tokens(token_usage)
        self._record_usage_events(invocation_usages)
        logger.info(
            "Response parsed (in=%d out=%d reasoning=%d)",
            token_usage["input_tokens"],
            token_usage["output_tokens"],
            token_usage["reasoning_tokens"],
        )
        return token_usage

    def _add_to_cumulative_tokens(self, token_usage: dict[str, int]) -> None:
        for key in self._cumulative_tokens:
            before = self._cumulative_tokens[key]
            self._cumulative_tokens[key] += token_usage[key]
            logger.debug(
                "Codex cumulative token update: %s %d + %d = %d",
                key,
                before,
                token_usage[key],
                self._cumulative_tokens[key],
            )

    def _record_usage_events(self, invocation_usages: list[dict[str, int]]) -> None:
        for usage in invocation_usages:
            self._record_usage_event(usage)

    def _record_usage_event(self, usage: dict[str, int]) -> None:
        usage_event = build_usage_event(
            model=self._model,
            provider="openai",
            input_tokens=usage["input_tokens"],
            output_tokens=usage["output_tokens"],
            reasoning_tokens=usage["reasoning_tokens"],
            cached_input_tokens=usage["cached_input_tokens"],
            call_type="completion",
        )
        _debug_log_pricing_details(
            model=self._model,
            provider="openai",
            token_usage=usage,
            usage_event=usage_event,
        )
        self.record_usage_event(usage_event)

    def _record_memory_snapshot(self) -> dict[str, str]:
        memory_snapshot = self._read_memory_snapshot()
        if memory_snapshot or self._other_memory_files:
            self._memory_history.append(
                {
                    "step": self._interaction_count + 1,
                    "files": memory_snapshot,
                }
            )
            self._log_memory_snapshot(memory_snapshot)
        return memory_snapshot

    @staticmethod
    def _log_memory_snapshot(memory_files: dict[str, str]) -> None:
        if memory_files:
            logger.info(
                "Memory after step: %d file(s), %d chars",
                len(memory_files),
                sum(len(value) for value in memory_files.values()),
            )
            return
        logger.info("Memory after step: (empty)")

    def _finish_successful_interaction(self) -> None:
        self._pending_feedback = None
        self._interaction_count += 1

    def _build_response_metadata(
        self,
        *,
        continuity: dict[str, Any],
        token_usage: dict[str, int],
        repair_attempted: bool,
        memory_snapshot: dict[str, str],
        conversation_snapshot: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "interaction_count": self._interaction_count,
            "system_type": "codex",
            "model": self._model,
            "version": self._version,
            "conversation_id": self._conversation_id,
            "continuity": {
                **continuity,
                "conversation_id_after": self._conversation_id,
            },
            "token_usage": token_usage,
            "cumulative_tokens": dict(self._cumulative_tokens),
            "repair_attempted": repair_attempted,
            "memory_files": memory_snapshot,
            "codex_conversation_session_ids": conversation_snapshot["session_ids"],
            "codex_conversation_jsonl_files": conversation_snapshot[
                "conversation_jsonl_files"
            ],
            "codex_conversation_jsonl_file_sizes": conversation_snapshot[
                "conversation_jsonl_file_sizes"
            ],
        }

    def respond(self, query: Query) -> Response:
        continuity = self._prepare_conversation(query)
        prompt_without_skill = self._build_prompt(query, include_skill=False)
        prompt = self._build_prompt(
            query,
            include_skill=continuity["invocation_mode"] == "fresh_persistent",
        )
        action, invocation_usages, repair_attempted = self._run_action_turn(
            prompt,
            query.response_schema,
            repair_original_prompt=prompt_without_skill,
        )
        token_usage = self._record_turn_usage(invocation_usages)
        memory_snapshot = self._record_memory_snapshot()
        conversation_snapshot = self._record_conversation_snapshot(
            query,
            step=self._interaction_count + 1,
        )
        self._finish_successful_interaction()

        return Response(
            action=action,
            metadata=self._build_response_metadata(
                continuity=continuity,
                token_usage=token_usage,
                repair_attempted=repair_attempted,
                memory_snapshot=memory_snapshot,
                conversation_snapshot=conversation_snapshot,
            ),
        )

    def observe(
        self, observation: Observation, next_query: Optional[Query] = None
    ) -> None:
        _ = next_query
        if not self._single_conversation and observation_marks_instance_complete(
            observation
        ):
            self._conversation_id = None
            self._last_cumulative_token_usage_by_conversation = {}
        content = observation.content.strip()
        if not content:
            return
        self._pending_feedback = content

    def reset(self) -> None:
        self._clear_interaction_state()
        self._clear_conversation_state()
        self._clear_persistent_workspace_state()
        self._rebuild_workspace_state()

    def _clear_interaction_state(self) -> None:
        self._interaction_count = 0
        self._jsonl_log = []
        self._cumulative_tokens = dict(_EMPTY_TOKENS)
        self._memory_history = []
        self._conversation_history = []
        self._instance_conversations = {}
        self._pending_feedback = None

    def _clear_conversation_state(self) -> None:
        self._conversation_id = None
        self._conversation_session_ids = set()
        self._last_cumulative_token_usage_by_conversation = {}

    def _clear_persistent_workspace_state(self) -> None:
        workspace = Path(self._tmp_dir)
        self._delete_state_dirs(workspace)
        self._delete_watched_memory_files(workspace)

    @staticmethod
    def _delete_state_dirs(workspace: Path) -> None:
        shutil.rmtree(workspace / ".codex" / "sessions", ignore_errors=True)
        shutil.rmtree(workspace / _MEMORY_DIR, ignore_errors=True)

    def _delete_watched_memory_files(self, workspace: Path) -> None:
        for relpath in self._read_watched_memory_files():
            path = workspace / relpath
            if path.is_file():
                path.unlink()

    def _rebuild_workspace_state(self) -> None:
        self._initialize_workspace()

    def get_run_artifacts(self) -> dict[str, Any]:
        memory_snapshot = self._read_memory_snapshot()
        conversation_jsonl_files = self._read_conversation_jsonl_files()
        return {
            "artifact_type": "codex",
            "memory_files": memory_snapshot,
            "memory_history": list(self._memory_history),
            "conversation_id": self._conversation_id,
            "codex_conversation_session_ids": sorted(self._observed_session_ids()),
            "codex_conversation_jsonl_files": conversation_jsonl_files,
            "codex_conversation_history": list(self._conversation_history),
            "codex_instance_conversations": self._sorted_instance_conversations(),
            "version": self._version,
            "interaction_count": self._interaction_count,
            "cumulative_tokens": dict(self._cumulative_tokens),
            "jsonl_events": self._jsonl_log,
        }

    @property
    def name(self) -> str:
        return self._name

    def _stop_container(self) -> None:
        if self._container_id is not None:
            stop_docker_container(self._container_id)
            self._container_id = None

    def __del__(self) -> None:
        if hasattr(self, "_container_id"):
            self._stop_container()
        if hasattr(self, "_tmp_dir"):
            cleanup_run_workspace(self._tmp_dir)
