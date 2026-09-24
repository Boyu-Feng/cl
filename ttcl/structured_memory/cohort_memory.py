"""Study-scoped structured memory of public cohort-tool evidence.

No latent profiles, cohort truth, evaluator scores, or private coding maps are
read. Similar instruments stay distinct unless public evidence establishes a
conversion. Within-study KM/fit is explicitly not a population ground truth.
"""

from __future__ import annotations

import copy
import json
import re
from typing import Any


def _table(text: str) -> dict[str, Any] | None:
    lines = text.splitlines()
    for index, line in enumerate(lines[:-1]):
        if " | " not in line or not re.fullmatch(r"[-+ ]+", lines[index + 1]):
            continue
        columns = [value.strip() for value in line.split(" | ")]
        rows = []
        for value in lines[index + 2 :]:
            values = [item.strip() for item in value.split(" | ")]
            if len(values) != len(columns):
                break
            rows.append(values)
        return {
            "columns": columns,
            "rows": rows[:24],
            "rows_retained": min(len(rows), 24),
            "display_truncated": len(rows) > 24 or "showing first" in text,
        }
    return None


class CohortMemory:
    def __init__(self, max_chars: int = 24000) -> None:
        self.max_chars = max(1000, max_chars)
        self._studies: dict[str, dict[str, Any]] = {}

    def observe(
        self,
        query: str,
        action: dict,
        observation: str,
        *,
        instance_id: str,
        instance_complete: bool,
    ) -> None:
        study = self._studies.setdefault(
            instance_id,
            {
                "instance_id": instance_id,
                "brief": {},
                "columns": {},
                "summary": {},
                "survival_evidence": [],
                "sql_evidence": [],
                "errors": [],
                "complete": False,
            },
        )
        for key, pattern in {
            "study": r"## Study\s+\d+/\d+:\s*([^\n]+)",
            "enrollment": r"\*\*Enrollment:\*\*\s*([^\n]+)",
            "region": r"\*\*Region\(s\) in this dataset:\*\*\s*([^\n]+)",
            "patients": r"\*\*Patients:\*\*\s*(\d+)",
        }.items():
            match = re.search(pattern, query)
            if match:
                study["brief"][key] = match.group(1).strip()
        call = action.get("tool_call", action)
        if not isinstance(call, dict):
            call = {}
        tool = call.get("tool", "")
        if re.search(r"(?:^|\n)(?:SQL )?ERROR", observation):
            entry = {
                "tool": tool,
                "parameters": {
                    key: value for key, value in call.items() if key != "thought"
                },
                "error": observation[:700],
            }
            if entry not in study["errors"]:
                study["errors"].append(entry)
                study["errors"] = study["errors"][-3:]
        elif tool == "get_database_metadata":
            self._metadata(study, observation)
        elif tool == "get_data_summary":
            for line in observation.splitlines():
                match = re.match(
                    r"([^\s]+) \((numeric|categorical)[^)]*\):\s*(.*)", line
                )
                if match:
                    study["summary"][match.group(1)] = {
                        "type": match.group(2),
                        "observed_distribution": match.group(3)[:500],
                    }
        elif tool in {"estimate_survival_by_group", "predict_cohort_survival"}:
            entry = self._survival(call, observation)
            if (
                entry["groups"]
                or entry["cohort_composition"]
                or entry["observable_cohort_fit"]
            ):
                # Both tools expose complementary views of one expression.
                # Preserve counts and composition when fit output arrives.
                previous = next(
                    (
                        item
                        for item in study["survival_evidence"]
                        if item["group_expression"] == entry["group_expression"]
                    ),
                    None,
                )
                if previous:
                    counts = {item["group"]: item["n"] for item in previous["groups"]}
                    for group in entry["groups"]:
                        if group["n"] is None:
                            group["n"] = counts.get(group["group"])
                    for field in ("cohort_composition", "observable_cohort_fit"):
                        if not entry[field]:
                            entry[field] = previous[field]
                self._replace(study["survival_evidence"], entry, "group_expression", 4)
        elif tool == "query_sql":
            parsed = _table(observation)
            if parsed:
                entry = {"sql": str(call.get("sql", ""))[:1600], **parsed}
                self._replace(study["sql_evidence"], entry, "sql", 4)
        # A submission is a model prediction, not an empirical observation.
        # It is deliberately not imported into survival_evidence.
        study["complete"] = study["complete"] or instance_complete

    @staticmethod
    def _replace(entries: list, item: dict, key: str, cap: int) -> None:
        entries[:] = [entry for entry in entries if entry[key] != item[key]]
        entries.append(item)
        del entries[:-cap]

    @staticmethod
    def _metadata(study: dict, text: str) -> None:
        section = ""
        for line in text.splitlines():
            if line.startswith("==="):
                section = line.strip("= ")
                continue
            match = re.match(r"\s+([^:]+):\s*(.*)", line)
            if not match:
                continue
            key, value = match.groups()
            if section == "Columns":
                # Keep local field name and complete coding description;
                # no hardcoded MMSE/MoCA/AD8 or genotype conversions.
                study["columns"][key.strip()] = value[:1000]
            elif section == "Study Info":
                study["brief"][key.strip()] = value[:1000]

    @staticmethod
    def _survival(call: dict, text: str) -> dict:
        groups, composition, fitted, unobserved = [], [], [], []
        section = ""
        for line in text.splitlines():
            if "Unobservable cohorts" in line:
                section = "unobservable"
            if line.startswith("==="):
                section = line
            match = re.match(
                r"\s+(.+?):\s+(?:n=(\d+)\s+\([^)]*\)\s+)?S\(12m\)=([\d.]+)\s+S\(24m\)=([\d.]+)\s+S\(36m\)=([\d.]+)",
                line,
            )
            if match:
                group, count, s12, s24, s36 = match.groups()
                groups.append(
                    {
                        "group": group,
                        "n": int(count) if count else None,
                        "s12": float(s12),
                        "s24": float(s24),
                        "s36": float(s36),
                    }
                )
                continue
            match = re.match(r"\s+([\w.-]+) \(n=(\d+)\):\s*(.*)", line)
            if match:
                cohort_id, count, details = match.groups()
                if details.startswith("model="):
                    parsed = re.match(
                        r"model=([\d.]+)/([\d.]+)/([\d.]+)\s+KM=([\d.]+)/([\d.]+)/([\d.]+)\s+KL=([\d.]+)",
                        details,
                    )
                    if parsed:
                        values = [float(x) for x in parsed.groups()]
                        fitted.append(
                            {
                                "cohort_id": cohort_id,
                                "n": int(count),
                                "model_survival": values[:3],
                                "sample_km": values[3:6],
                                "within_study_fit_kl": values[6],
                            }
                        )
                else:
                    parts = re.findall(r"([^,=]+)=(\d+) \(([\d.]+)%\)", details)
                    composition.append(
                        {
                            "cohort_id": cohort_id,
                            "n": int(count),
                            "groups": [
                                {
                                    "group": label.strip(),
                                    "n": int(n),
                                    "fraction": round(float(fraction) / 100, 4),
                                }
                                for label, n, fraction in parts
                            ],
                        }
                    )
                continue
            if section == "unobservable" and re.fullmatch(r"\s+[\w.-]+\s*", line):
                unobserved.append(line.strip())
        return {
            "group_expression": str(call.get("group_expression", ""))[:2000],
            "groups": groups,
            "cohort_composition": composition,
            "observable_cohort_fit": fitted,
            "unobservable_cohorts": unobserved,
            "scope": "current study sample; enrollment selection applies, not population truth",
        }

    def state_dict(self) -> dict:
        return copy.deepcopy(
            {
                "method": "public_study_scoped_evidence",
                "studies": list(self._studies.values()),
                "limits": [
                    "All survival and cohort mixtures come from publicly observed study samples; selection bias remains.",
                    "Keep distinct study scopes, instruments, units and coding notes. Shared labels do not establish numerical equivalence.",
                    "Within-study fit KL is not an external population score; submitted estimates are not observed truth.",
                    "An unobservable cohort is unknown in this study, not a zero-survival cohort.",
                ],
            }
        )

    def context(self, query: str = "") -> str:
        if not self._studies:
            return ""
        state = self.state_dict()
        state["context_truncated"] = False

        def encode() -> str:
            return json.dumps(
                state, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            )

        # Retain whole structured records. Prune diagnostic/raw material first,
        # then old redundant groupings. Last resort retains the newer studies.
        for field in (
            "errors",
            "sql_evidence",
            "summary",
            "survival_evidence",
            "columns",
        ):
            for study in state["studies"]:
                while len(encode()) > self.max_chars and study[field]:
                    if isinstance(study[field], dict):
                        study[field].pop(next(iter(study[field])))
                    else:
                        study[field].pop(0)
                    state["context_truncated"] = True
        while len(encode()) > self.max_chars and state["studies"]:
            state["studies"].pop(0)
            state["context_truncated"] = True
        return encode()
