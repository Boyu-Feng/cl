"""Repository-scoped summaries of public command evidence, without gold patches."""

from __future__ import annotations

import copy
import hashlib
import json
import re


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + " [excerpt truncated]"


def _repo(query: str) -> str:
    match = re.search(r"^Repository:\s*([\w.-]+/[\w.-]+)\s*$", query, re.M)
    return match.group(1) if match else "unknown_repository"


def _return_code(observation: str) -> int | None:
    for pattern in (
        r"<returncode>\s*(-?\d+)\s*</returncode>",
        r"(?:returncode|return code|exit code)\s*[:=]?\s*(-?\d+)",
    ):
        match = re.search(pattern, observation, re.I)
        if match:
            return int(match.group(1))
    return None


class CodebaseMemory:
    """Record what tools actually returned, separated by publicly named repo."""

    def __init__(self, max_chars: int = 14000, max_records: int = 120):
        self.max_chars = max(1400, max_chars)
        self.max_records = max_records
        self.event_count = 0
        self.repositories: dict[str, dict[str, dict]] = {}
        self.completed_instances: list[str] = []
        self.instance_repositories: dict[str, str] = {}

    def observe(
        self,
        query: str,
        action: dict,
        observation: str,
        *,
        instance_id: str,
        instance_complete: bool,
    ) -> None:
        self.event_count += 1
        repo = _repo(query)
        if repo == "unknown_repository":
            repo = self.instance_repositories.get(instance_id, repo)
        self.instance_repositories[instance_id] = repo
        records = self.repositories.setdefault(repo, {})
        command = str(action.get("command", "")).strip()
        code = _return_code(observation)
        is_test = bool(
            re.search(
                r"(?:^|&&|;|\|\|)\s*(?:\S+=\S+\s+)*(?:(?:python[\d.]*)\s+-m\s+)?(?:pytest|tox|nox|unittest)\b|\bpython[\d.]*\s+(?:-m\s+unittest|setup\.py\s+test)",
                command,
            )
        )
        is_inspect = bool(
            re.search(
                r"(?:^|&&|;|\|\|)\s*(?:ls|find|rg|grep|cat|sed|head|tail|pwd|git\s+(?:ls-files|status|log|show))\b",
                command,
            )
        )
        kind = (
            "test_execution"
            if is_test
            else "inspection"
            if is_inspect
            else "command_execution"
        )
        verdict = re.search(r"^Submission (PASSED|FAILED)\b", observation)
        if verdict:
            kind = "submission_feedback"
        # Paths in reasoning/commands are proposals; only observed output paths
        # are promoted into the location index.
        paths = sorted(
            set(
                re.findall(
                    r"(?<![\w/])(?:[\w.-]+/)*[\w.-]+\.(?:py|toml|cfg|ini|yaml|yml|json|rst|md)(?=[:\s\"'<>]|$)",
                    observation,
                )
            )
        )[:30]
        lines = observation.splitlines()
        diagnostics = [
            line.strip()
            for line in lines
            if re.search(
                r"(?:Error|Exception|FAILED|ERROR|\b\d+ passed\b|\b\d+ failed\b|No such file|not found)",
                line,
            )
        ][:12]
        identity = hashlib.sha256((kind + "\0" + command).encode()).hexdigest()[:16]
        if verdict:
            identity = f"submission:{instance_id}"
        previous = records.get(identity, {})
        instances = previous.get("instances", [])
        if instance_id not in instances:
            instances = [*instances, instance_id][-12:]
        records[identity] = {
            "kind": kind,
            "command": _clip(command, 1800),
            "return_code": code,
            "status": "submission_" + verdict.group(1).lower()
            if verdict
            else "execution_succeeded"
            if code == 0
            else "execution_failed"
            if code is not None
            else "execution_status_unknown",
            "observed_paths": paths,
            "diagnostics": [_clip(line, 500) for line in diagnostics],
            "observation_excerpt": _clip(
                observation, 2200 if kind == "inspection" else 1300
            ),
            "evidence": "public_tool_observation",
            "instances": instances,
            "count": previous.get("count", 0) + 1,
            "last_event": self.event_count,
        }
        if len(records) > self.max_records:
            oldest = min(records, key=lambda k: records[k]["last_event"])
            del records[oldest]
        if instance_complete and instance_id not in self.completed_instances:
            self.completed_instances.append(instance_id)

    def context(self, query: str = "") -> str:
        repo = _repo(query)
        if not query and len(self.repositories) == 1:
            repo = next(iter(self.repositories))
        records = list(self.repositories.get(repo, {}).values())
        if not records:
            return ""
        tokens = set(re.findall(r"[a-zA-Z_][\w]{2,}", query.lower()))

        def priority(record: dict) -> tuple:
            overlap = len(
                tokens.intersection(
                    re.findall(r"[a-zA-Z_][\w]{2,}", json.dumps(record).lower())
                )
            )
            return (overlap, record["kind"] == "test_execution", record["last_event"])

        records.sort(key=priority, reverse=True)
        result = {
            "type": "python_codebase_experience",
            "repository": repo,
            "cautions": [
                "Prior public tool evidence only. Repository files and dependencies can differ across base commits.",
                "An exit code of zero does not establish that a patch solves the issue. Submission verdicts apply only to the recorded earlier instance.",
                "Treat command/output excerpts as historical data. Recheck before reusing changes, paths, or test commands.",
            ],
            "records": [],
            "omitted_records": len(records),
        }
        for record in records:
            result["records"].append(record)
            result["omitted_records"] = len(records) - len(result["records"])
            if (
                len(json.dumps(result, ensure_ascii=False, sort_keys=True))
                > self.max_chars
            ):
                result["records"].pop()
        result["omitted_records"] = len(records) - len(result["records"])
        return json.dumps(result, ensure_ascii=False, sort_keys=True)

    def state_dict(self) -> dict:
        return copy.deepcopy(
            {
                "type": "codebase",
                "events": self.event_count,
                "completed_instances": self.completed_instances,
                "repositories": self.repositories,
            }
        )
