# Mem0

Vector-memory system backed by the [`mem0ai`](https://github.com/mem0ai/mem0) SDK. After every turn, the (query, response, feedback) triple is fed to `Memory.add()`, which uses an LLM to extract atomic facts, deduplicate them, and store them in a Qdrant vector index. Before each turn, `Memory.search()` retrieves the top-k most relevant memories and injects them into the prompt.

## What the LM sees

Each turn, the LM (default `openai/gpt-5` via LiteLLM) gets:

- A system prompt explaining it has access to a memory store of past facts.
- A `=== RELEVANT MEMORIES ===` block with the top-k retrieved memories (default 10).
- The current query prompt.
- The running message history *within the current instance*. At instance boundaries the in-instance history is cleared — only mem0's extracted facts persist across instances.

Retrieval is keyed on the current prompt plus the previous turn's prompt/response/feedback (when available), so search reflects in-progress instance context without ballooning to the full instance history.

## Additional integration notes

- Uses the upstream `mem0ai` SDK directly; fact extraction and dedup are handled by mem0's own LLM pipeline.
- Each run gets its own temp Qdrant directory (`tempfile.mkdtemp(prefix="mem0_")`) so parallel rollouts don't collide on the default `/tmp/qdrant` path. `MEM0_TELEMETRY` is disabled to avoid a second lock on `~/.mem0`.
- The extraction LLM defaults to the same model as the response LLM but can be overridden via `extraction_model`. Embeddings default to OpenAI `text-embedding-3-small`.
- Final memory contents are exported as run artifacts via `Memory.get_all()`.
