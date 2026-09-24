# Claude Code

Runs the Claude Code CLI (`@anthropic-ai/claude-code`) inside a per-run Docker
container in the same manner as the [Harbor Framework](https://github.com/harbor-framework/harbor/blob/main/src/harbor/agents/installed/claude_code.py). Each `respond()` invokes `claude -p` with stream JSON output, parses
assistant text from the stream, and validates the latest JSON-like response
against the benchmark response schema.

## Runtime

- Registered system name: `claude`.
- Default model: `claude-sonnet-4-6`.
- Default CLI package: `@anthropic-ai/claude-code@2.1.119`.
- Default Docker image: `node:22-slim`.
- Host workspace: `~/.cache/claude_bench/run_*`.
- Container workspace: `/workspace`.
- `CLAUDE_CONFIG_DIR`: `/workspace/.claude`.

Container startup, `docker exec`, cleanup, seed-memory copying, and file-memory
snapshotting use shared helpers from `src/systems/common.py`. Claude-specific
logic remains in `system.py`: CLI installation, Claude CLI argument building,
stream parsing, conversation policy, token usage accounting, memory snapshotting,
and response repair.

## Prompt

Each turn builds a prompt from:

- Optional preamble text loaded from `preamble_file`, prefixed as `PRE-AMBLE:`.
- Pending feedback captured by `observe()`, prefixed as
  `FEEDBACK FROM PREVIOUS ACTION:`.
- The current query prompt.
- A schema instruction generated from the query response schema.
- The formatted `memory_instruction`, with `{memory_dir}` set to
  `/workspace/.claude/projects/-workspace/memory`.

If `skill` is configured, it may point to a single markdown file, a skill
folder containing `SKILL.md`, or that `SKILL.md` directly. With the default
`invocation_type="skill"`, it is installed at `.claude/skills/<name>/` and the
prompt is encoded as Claude Code command tags:
`<command-message><name></command-message>`,
`<command-name>/<name></command-name>`, and
`<command-args>...benchmark prompt...</command-args>`. With
`invocation_type="prepend"`, the `SKILL.md` body is inlined at the start of the
prompt instead. Skill invocation and prepended skill text are used only on a
fresh Claude conversation turn; resume and repair turns omit both.

`single_conversation` defaults to `True`. When it is `True`, Claude keeps using
one conversation id across instance changes until `reset()` is called. When it
is `False`, Claude still resumes within a benchmark instance, but new instance
identities start fresh.

`other_memory_files`, when set, is a list of relative workspace file paths,
directory paths, or glob patterns. Matched files are treated as memory files and
snapshotted into `memory_files` alongside Claude's native memory files. If an
additional file has the same relative path as a native Claude memory file, the
native key is preserved and the additional file is written under an
`other_memory/` key in the merged view.

## Memory

Claude file memory is always initialized. The canonical host memory path is
`<workspace>/.claude/projects/-workspace/memory`, and the canonical artifact
memory dir is `.claude/projects/-workspace/memory`.

`seed_memory_dir`, when set, must point to an existing directory. Its direct
contents are copied into the canonical memory directory during workspace
initialization.

After each successful turn, the adapter snapshots files from:

- `.claude/projects/-workspace/memory`
- Any other `*/memory` directories under `.claude/projects`

If multiple Claude memory directories contain the same relative path, later
directories in sorted project order overwrite earlier entries in the native
snapshot. Native files and opt-in files are reported only in the merged
`memory_files` snapshot and appended to `memory_history`; there are no separate
`memory_backend`, `memory_dir`, `native_memory_files`, `other_memory_files`, or
source-map output keys.

## Continuation

Claude conversation continuity uses Claude Code session state.

- Once a Claude session id exists, later calls use `--resume <session_id>`.
- `single_conversation` decides whether a conversation carries across new
  instance boundaries.
- If a Claude invocation exits non-zero, the adapter retries once. The retry uses
  `--resume <session_id>` whenever a current session id exists; otherwise it
  retries fresh.
- If response-schema validation fails, the repair prompt uses
  `--resume <session_id>` whenever a current session id exists; otherwise it runs
  fresh.
- `reset()` is a hard reset: it clears conversation ids, pending feedback,
  memory history, Claude project session/memory files, and matched
  `other_memory_files`, then rebuilds workspace scaffolding and configured
  skills.

## Auth And Container Environment

The system requires an Anthropic API key from either the `api_key` constructor
argument or `ANTHROPIC_API_KEY` in the environment. The Docker container receives:

- `ANTHROPIC_API_KEY`
- `CLAUDE_CONFIG_DIR=/workspace/.claude`
- `IS_SANDBOX=1`
- `DISABLE_AUTOUPDATER=1`
- `CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1`

The CLI is invoked with `--permission-mode bypassPermissions`; the Docker
container is the benchmark sandbox.

## CLI Options

Every turn starts with these Claude CLI arguments:

- `claude`
- `-p`
- `--output-format stream-json`
- `--verbose`
- `--permission-mode bypassPermissions`
- `--model <model>`

`reasoning_effort` accepts `None`, `"low"`, `"medium"`, `"high"`, `"xhigh"`,
or `"max"`. The default is `"low"`, emitted as `--effort low`. `None` omits the
`--effort` option.

The optional `max_turns`, `allowed_tools`, and `disallowed_tools` constructor
arguments are passed through as `--max-turns`, `--allowedTools`, and
`--disallowedTools`.

## Response Parsing And Usage

The adapter concatenates text blocks from all assistant stream events, prefers
the latest valid JSON-like candidate, and validates it against the response
schema. If validation fails, it sends one repair prompt and parses the repair
response.

Usage is summed from all `result` stream events for the main response and any
repair response. Claude Code reports per-invocation output tokens even when
continuing a session, so output tokens are summed directly. Input tokens include
`input_tokens`, `cache_read_input_tokens`, and `cache_creation_input_tokens`;
cached input tokens track only `cache_read_input_tokens`. When every `result`
event includes Claude Code's `total_cost_usd`, that provider-reported cost is
used as the recorded cost. Otherwise the recorded usage event falls back to
LiteLLM pricing with provider `anthropic` and the parsed input, output,
cache-read, and cache-creation token counts.

## Artifacts

`get_run_artifacts()` returns:

- `artifact_type="claude"`
- `memory_files` (merged native Claude memory plus opt-in additional files)
- `memory_history`
- `conversation_id`
- `claude_conversation_session_ids`
- `claude_conversation_jsonl_files`: final native Claude Code conversation
  transcript JSONL files from `.claude/projects/**/<session_id>.jsonl`; files
  whose stem is not an observed main conversation session id are not exported,
  so subagent JSONLs stay separate from the benchmark conversation artifacts
- `claude_conversation_history`: per-turn lists and sizes for those conversation
  JSONL files
- `claude_instance_conversations`: latest conversation JSONL snapshot per
  `Query.instance_index`, exported under `instances/instance_####/` in sidecar
  artifacts
- `version`
- `interaction_count`
- `cumulative_tokens`
- raw `jsonl_events`
