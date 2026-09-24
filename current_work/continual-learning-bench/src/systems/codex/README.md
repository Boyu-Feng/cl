# Codex

Runs the OpenAI Codex CLI inside a per-run Docker container similar to how the
[Harbor Framework](https://github.com/harbor-framework/harbor/blob/main/src/harbor/agents/installed/codex.py) does it. Each `respond()` invokes `codex exec` inside that container, parses JSONL events from stdout, and validates the assistant's latest JSON-like response against the benchmark response schema.

## Runtime

- Registered system name: `codex`.
- Default model: `gpt-5.4-mini`.
- Default CLI package: `@openai/codex@0.125.0`.
- Default Docker image: `node:22-slim`.
- Host workspace: `~/.cache/codex_bench/run_*`.
- Container workspace: `/workspace`.
- `CODEX_HOME`: `/workspace/.codex`.

Container startup, `docker exec`, cleanup, watched-file snapshotting, memory
directory reads, and session JSONL reads use shared helpers or host workspace
paths. Codex-specific logic remains in `system.py`: CLI installation, auth
setup, command construction, JSONL parsing, conversation policy, native session
capture, token usage accounting, and response repair.

## CLI Options

Every turn runs `codex exec` with:

- `--enable memories`, unless `disable_native_memories=True`
- `--model <model>`
- `--json`
- `--dangerously-bypass-approvals-and-sandbox`
- `--skip-git-repo-check`

When native memories are enabled, the command also sets
`memories.use_memories=true` and `memories.generate_memories=true`.

`reasoning_effort` accepts `None`, `"none"`, `"minimal"`, `"low"`, `"medium"`,
or `"high"`. The default is `"none"`, which is passed as
`-c model_reasoning_effort="none"`. Set it to `None` to omit the override.

`model_auto_compact_token_limit`, when set, must be a positive integer and is
passed through as `-c model_auto_compact_token_limit=<value>`.

## Prompt

Each turn builds a prompt from:

- Optional preamble text loaded from `preamble_file`, prefixed as `PRE-AMBLE:`.
- Pending feedback captured by `observe()`, prefixed as
  `FEEDBACK FROM PREVIOUS ACTION:`.
- The current query prompt.
- A schema instruction generated from the query response schema.

If `skill` is configured, it may point to a single markdown file, a skill
folder containing `SKILL.md`, or that `SKILL.md` directly. With the default
`invocation_type="skill"`, it is installed at `.codex/skills/<name>/` and the
prompt is prefixed only with `$<name>`. The skill body is not inlined in this
mode. With `invocation_type="prepend"`, the `SKILL.md` body is inlined at the
start of the prompt instead. Skill invocation and prepended skill text are used
only on a fresh Codex conversation turn; resume and repair turns omit both.

`single_conversation` defaults to `True`. When it is `True`, Codex keeps using
one conversation id across instance changes until `reset()` is called. When it
is `False`, Codex still resumes within a benchmark instance, but new instance
identities start fresh.

`disable_native_memories` defaults to `False`. Set it to `True` only to omit
the Codex native memories CLI flags. The adapter still watches `.codex/memories`
either way.

`other_memory_files`, when set, is a list of relative workspace file paths,
directory paths, or glob patterns. Matched files are snapshotted into
`memory_files`.

## Memory

Codex native memories are enabled by default through the CLI. The adapter reads
any files that exist under `.codex/memories`, then merges those files with the
configured `other_memory_files` watch list. The merged snapshot is reported only
as `memory_files` and appended to `memory_history`; there are no separate
`memory_backend`, `memory_dir`, or `other_memory_files` output keys.

## Resume

Codex conversation resume uses `codex exec resume <conversation_id>`.

- Once a Codex conversation id exists, later calls use
  `codex exec resume <conversation_id>`.
- Fresh calls are persistent, not `--ephemeral`, so later turns can resume the
  newly-created conversation.
- `single_conversation` decides whether a conversation carries across new
  instance boundaries.
- `reset()` is a hard reset: it clears conversation ids, token deltas, pending
  feedback, memory history, Codex session files, `.codex/memories`, and matched
  `other_memory_files`.

## Auth

By default, the system requires `OPENAI_API_KEY` in the environment. The Docker
container receives the key as an environment variable, then writes
`/workspace/.codex/auth.json` because Codex expects that file.

## Response Parsing And Usage

The adapter extracts the latest assistant message from Codex JSONL events,
prefers the latest valid JSON-like candidate in that text, and validates it
against the response schema. If validation fails, it sends one repair prompt and
parses the repair response.

Input and cached-input token usage is summed from all `turn.completed` events for
the main response and any repair response. Codex reports generated-token counts
cumulatively, so output and reasoning tokens are recorded as the delta from the
previous completed turn. The recorded usage event uses provider `openai` and
includes input, output, reasoning, and cached-input token counts.

## Artifacts

Codex persists native session rollouts under `CODEX_HOME/sessions`, which maps
to `<run-workspace>/.codex/sessions` on the host (not the user's global
`~/.codex` during Docker-isolated runs). The adapter tracks the observed Codex
conversation/thread ids from stdout and exports only matching session JSONL
files for the current benchmark run.

`get_run_artifacts()` returns:

- `artifact_type="codex"`
- `memory_files`
- `memory_history`
- `conversation_id`
- `codex_conversation_session_ids`
- `codex_conversation_jsonl_files`: final native Codex session JSONL contents
- `codex_conversation_history`
- `codex_instance_conversations`
- `version`
- `interaction_count`
- `cumulative_tokens`
- raw `jsonl_events`

The artifact exporter writes captured session JSONL files under
`codex_conversations/final/` and per-instance snapshots under
`instances/instance_####/codex_conversations/`.
