"""Task-independent, model-authored bank with transactional updates."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import re

PROMPT_PATH = Path(__file__).with_name("trajectory_extraction_prompt.md")
UPDATE_PROMPT = PROMPT_PATH.read_text()


def encode(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def render(entries):
    if not entries:
        return ""
    return (
        "Model-written experience bank from completed earlier interactions. "
        "These notes can be wrong. Apply only relevant entries whose scope matches "
        "the current task; verify hypotheses with current tools. Current instructions "
        "and current observations take precedence. Do not blindly copy old answers.\n"
        + encode(entries)
    )


def parse_update(raw):
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines[-1].strip() == "```":
            text = "\n".join(lines[1:-1])
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("Expected one JSON object")
    return value


class TrajectoryBank:
    def __init__(self, max_entries=8, max_chars=9000, max_tokens=2048):
        self.entries = []
        self.last_observed = 0
        self.version = 0
        self.max_entries, self.max_chars, self.max_tokens = (
            max_entries,
            max_chars,
            max_tokens,
        )

    def context(self):
        return render(self.entries)

    def state_dict(self):
        return {
            "entries": copy.deepcopy(self.entries),
            "last_observed": self.last_observed,
            "version": self.version,
        }

    def payload(self, episode):
        if episode["episode"] != self.last_observed + 1:
            raise ValueError("Episodes must be processed once in chronological order")
        reward = episode["reward"]
        if reward is not None and (
            not isinstance(reward, (int, float)) or not math.isfinite(reward)
        ):
            raise ValueError("Reward must be finite or null")
        if not episode["completed"] and reward is not None:
            raise ValueError("An incomplete episode cannot have a fabricated reward")
        return {
            "existing_bank": copy.deepcopy(self.entries),
            "trajectory": copy.deepcopy(episode),
            "limits": {
                "max_entries": self.max_entries,
                "max_bank_characters": self.max_chars,
                "max_rendered_bank_tokens": self.max_tokens,
                "max_operations": 4,
            },
        }

    def candidate(self, update, episode, count_tokens):
        """Validate structure/provenance/budget, not semantic truth or utility."""
        if set(update) != {
            "trajectory_summary",
            "reward_interpretation",
            "decision",
            "operations",
        }:
            raise ValueError("Use exactly the four documented top-level fields")
        for key in ("trajectory_summary", "reward_interpretation"):
            if not isinstance(update[key], str) or not update[key].strip():
                raise ValueError(f"{key} must be nonempty text")
        operations = update["operations"]
        if not isinstance(operations, list) or len(operations) > 4:
            raise ValueError("operations must be a list of at most four changes")
        if update["decision"] == "KEEP":
            if operations:
                raise ValueError("KEEP requires empty operations")
            return copy.deepcopy(self.entries)
        if update["decision"] not in {"KEEP", "UPDATE"}:
            raise ValueError(
                "Top-level decision must be KEEP or UPDATE; ADD/REVISE/REMOVE belong in operations[].op"
            )
        if not operations:
            raise ValueError("UPDATE requires at least one operation")
        allowed = {(episode["episode"], step["step"]) for step in episode["steps"]}
        for entry in self.entries:
            for ref in entry["evidence"]:
                allowed.update((ref["episode"], s) for s in ref["steps"])
        entries = {entry["id"]: copy.deepcopy(entry) for entry in self.entries}
        touched = set()
        for operation in operations:
            if not isinstance(operation, dict):
                raise ValueError("Each operation must be an object")
            op, identity = operation.get("op"), operation.get("id")
            if not isinstance(identity, str) or not re.fullmatch(
                r"E[1-9][0-9]*", identity
            ):
                raise ValueError("IDs must have the form E1, E2, ...")
            if identity in touched:
                raise ValueError("Change each ID at most once per update")
            touched.add(identity)
            if (
                not isinstance(operation.get("reason"), str)
                or not operation["reason"].strip()
            ):
                raise ValueError("Each operation requires a reason")
            if op not in {"ADD", "REVISE", "REMOVE"}:
                raise ValueError("Unknown operation")
            if op == "ADD" and identity in entries:
                raise ValueError("ADD requires an unused ID")
            if op != "ADD" and identity not in entries:
                raise ValueError("REVISE/REMOVE requires an existing ID")
            expected = {"op", "id", "reason"} | ({"entry"} if op != "REMOVE" else set())
            if set(operation) != expected:
                raise ValueError(f"Wrong operation fields for {op}")
            if op == "REMOVE":
                del entries[identity]
                continue
            entry = copy.deepcopy(operation["entry"])
            fields = {
                "type",
                "title",
                "scope",
                "lesson",
                "application",
                "limitations",
                "evidence",
            }
            if not isinstance(entry, dict) or set(entry) != fields:
                raise ValueError("Entry must contain exactly the documented fields")
            for field in fields - {"evidence"}:
                if not isinstance(entry[field], str) or not entry[field].strip():
                    raise ValueError(f"Entry {field} must be nonempty text")
            if entry["type"] not in {"fact", "procedure", "hypothesis"}:
                raise ValueError("Unknown experience type")
            if not isinstance(entry["evidence"], list) or not entry["evidence"]:
                raise ValueError("At least one evidence citation is required")
            for ref in entry["evidence"]:
                if not isinstance(ref, dict) or set(ref) != {"episode", "steps"}:
                    raise ValueError("Evidence must specify episode and steps")
                if (
                    type(ref["episode"]) is not int
                    or not isinstance(ref["steps"], list)
                    or not ref["steps"]
                ):
                    raise ValueError(
                        "Evidence episode and nonempty steps must be integers"
                    )
                if any(
                    type(s) is not int or (ref["episode"], s) not in allowed
                    for s in ref["steps"]
                ):
                    raise ValueError(
                        "Evidence references an unavailable step or a future episode"
                    )
            entry["id"] = identity
            entries[identity] = entry
        result = list(entries.values())
        if len(result) > self.max_entries or len(encode(result)) > self.max_chars:
            raise ValueError(
                "Bank exceeds entry/character limit; merge or remove instead"
            )
        if count_tokens(render(result)) > self.max_tokens:
            raise ValueError("Bank exceeds token limit; shorten or merge instead")
        return result

    def update(self, episode, generate, seed, count_tokens, output_tokens=4096):
        payload = self.payload(episode)
        before = self.state_dict()
        messages = [
            {"role": "system", "content": UPDATE_PROMPT},
            {"role": "user", "content": encode(payload)},
        ]
        attempts, accepted, parsed = [], False, None
        error = None
        for retry in range(2):
            record = {
                "attempt": retry,
                "messages": copy.deepcopy(messages),
                "seed": seed + retry,
            }
            try:
                completion = generate(
                    messages,
                    seed + retry,
                    max_new_tokens=output_tokens,
                    temperature=0.0,
                )
                record["completion"] = completion
                if completion["finish_reason"] != "stop":
                    raise ValueError(
                        "Writer output was truncated; return a shorter complete JSON object"
                    )
                proposed = parse_update(completion["raw_response"])
                # Packaging-only compatibility: preserve every proposed operation
                # and entry verbatim when the model repeats its sole operation
                # type in the outer decision field.
                outer = proposed.get("decision")
                operations = proposed.get("operations")
                if (
                    outer in {"ADD", "REVISE", "REMOVE"}
                    and isinstance(operations, list)
                    and operations
                    and all(
                        isinstance(op, dict) and op.get("op") == outer
                        for op in operations
                    )
                ):
                    record["packaging_repair"] = {
                        "field": "decision",
                        "from": outer,
                        "to": "UPDATE",
                    }
                    proposed["decision"] = "UPDATE"
                candidate = self.candidate(proposed, episode, count_tokens)
                self.entries = candidate
                self.version += int(proposed["decision"] == "UPDATE")
                accepted, parsed, error = True, proposed, None
                attempts.append(record)
                break
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                record["error"] = error
                attempts.append(record)
                if "completion" not in record:
                    break  # A generation/context failure is not fixed by a longer prompt.
                messages = [
                    {"role": "system", "content": UPDATE_PROMPT},
                    {
                        "role": "user",
                        "content": encode(payload)
                        + "\nYour previous response was rejected: "
                        + error
                        + "\nReturn a valid, concise update for the same data. KEEP is allowed. Do not invent evidence.",
                    },
                ]
        self.last_observed = episode["episode"]
        return {
            "episode": episode["episode"],
            "accepted": accepted,
            "error": error,
            "decision": parsed["decision"] if parsed else "REJECTED",
            "parsed_update": parsed,
            "attempts": attempts,
            "bank_before": before,
            "bank_after": self.state_dict(),
            "semantic_validation": False,
        }
