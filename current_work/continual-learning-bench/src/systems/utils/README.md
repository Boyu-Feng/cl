# README.md

Shared provider and system utilities.

- Keep provider detection, native state, structured-output retries, and token budgets here.
- Preserve refusal classification and exception cause chains.
- Add provider-specific coercion at this layer, not inside individual benchmark systems, when multiple systems need it.
- Avoid logging secrets, hidden reasoning content, or full prompts at normal log levels.
