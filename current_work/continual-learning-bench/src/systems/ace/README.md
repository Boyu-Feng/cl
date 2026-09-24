# ACE

ACE (Agentic Context Engineering) keeps a structured, sectioned **playbook** of bullet-form strategies, formulas, code snippets, mistakes, and heuristics. The playbook is grown over time by three roles:

- **Generator** — picks an action for the current query, conditioned on the playbook.
- **Reflector** — reviews the trajectory and feedback to tag bullets as helpful/harmful and propose new ones.
- **Curator** — periodically rewrites the playbook (add/edit/delete bullets) based on the reflection.

## What the LM sees

- **Generator**: the current playbook, the in-progress episode transcript (prior turns + intermediate observations), and a JSON-schema action prompt. Returns `{reasoning, bullet_ids, final_answer}`. History across instances is *not* shown — learning carries over only via the playbook.
- **Reflector** (ran once when instance completes): the full episode trace, the bullets that were used, and the terminal observation. Returns reflection text + bullet usage tags.
- **Curator** (ran when `curate_every_n_updates` number of instances complete): the current playbook + recent reflection. Returns add/edit/delete operations applied to the playbook.

## Reference

This system is based on the [ACE agent](https://github.com/ace-agent/ace).

## Differences from the original ACE

The implementation reuses the upstream ACE Generator/Reflector/Curator and playbook utilities (vendored under `src/vendors/ace/`). The adapter wraps them to fit the benchmark's `respond(query) → action` interface:

- One `respond()` call per turn produces a schema-validated action; reflection/curation only fire at instance boundaries (when an `Observation` marks the instance complete).
- Ground-truth answers are not used in reflection — only the environment feedback the benchmark provides.
- A repair pass re-prompts the generator if its first JSON output fails schema validation.
- Playbook snapshots and the final playbook are exported as run artifacts.
