"""Task-specific experience ablations, derived only from public interactions.

These are engineered baselines, not learned writers. No benchmark data imports.
"""

from __future__ import annotations

import copy
import json
import re
from collections import Counter

from ttcl.structured_memory.database_memory import DatabaseMemory, _question
from ttcl.structured_memory.cohort_memory import CohortMemory
from ttcl.structured_memory.poker_memory import _line, _AUTO_MARKER


def encode(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def words(text):
    stop = {
        "the",
        "and",
        "for",
        "with",
        "from",
        "have",
        "this",
        "that",
        "were",
        "what",
        "how",
        "many",
        "you",
        "your",
        "are",
        "was",
        "not",
        "only",
        "study",
        "question",
    }
    return set(re.findall(r"[a-zA-Z_][\w]{2,}", text.lower())) - stop


def bounded(header, records, limit):
    result = {**header, "records": [], "omitted_records": len(records)}
    for record in records:
        result["records"].append(record)
        result["omitted_records"] -= 1
        if len(encode(result)) > limit:
            result["records"].pop()
            result["omitted_records"] += 1
    return encode(result)


class FocusedDatabaseMemory(DatabaseMemory):
    """Compact schema registry plus question-linked query/verification cases."""

    def context(self, query=""):
        if not self.records:
            return ""
        pending_drift = self._drift_notice(query)
        current = [r for r in self.records.values() if r["schema_epoch"] == self.epoch]
        if pending_drift:
            return encode(
                {
                    "type": "database_procedures",
                    "schema_changed": True,
                    "instruction": "Prior table/column/value assumptions need revalidation. Inspect the current schema before reusing them.",
                    "records": [],
                }
            )
        relevant = words(_question(query))
        schemas, examples, errors = [], [], []
        feedback = {
            r["instances"][-1]: r for r in current if r["kind"] == "answer_feedback"
        }
        for r in current:
            source = {
                "source_episodes": r["instances"],
                "schema_epoch": r["schema_epoch"],
            }
            if r["kind"] == "table_inventory":
                schemas.append(
                    {"kind": "table_inventory", "tables": r["tables"], **source}
                )
            elif r["kind"] == "schema":
                item = {"kind": "schema", **source}
                if r.get("columns"):
                    item.update(
                        table=r["table"],
                        columns=[
                            {k: c[k] for k in ("name", "type", "pk") if k in c}
                            for c in r["columns"]
                        ],
                    )
                else:
                    item.update(
                        {
                            k: r[k]
                            for k in ("catalog_objects", "ddl", "raw_excerpt")
                            if k in r
                        }
                    )
                schemas.append(item)
            elif r["kind"] == "executed_query":
                last_source = r["instances"][-1]
                verdict = feedback.get(last_source, {}).get("verdict", "unknown")
                examples.append(
                    {
                        "kind": "query_case",
                        "question": r["question"],
                        "sql": r["sql"],
                        "tables": r["referenced_tables"],
                        "result_columns": r["result_columns"],
                        "sample_rows": r["sample_rows"],
                        "result_excerpt": r.get("result_excerpt", ""),
                        "episode_verdict": verdict,
                        "verification_scope": "SQL executed; verdict is for the episode's final answer, not each query",
                        **source,
                    }
                )
            elif r["kind"] == "sql_error":
                errors.append(
                    {
                        "kind": "failed_query",
                        "sql": r["sql"],
                        "error": r["error"],
                        **source,
                    }
                )
            elif r["kind"] == "answer_feedback":
                examples.append(
                    {
                        "kind": "answer_case",
                        "question": r["question"],
                        "submitted_answer": r["submitted_answer"],
                        "verdict": r["verdict"],
                        **(
                            {
                                "public_correct_answer": r[
                                    "publicly_revealed_correct_answer"
                                ]
                            }
                            if "publicly_revealed_correct_answer" in r
                            else {}
                        ),
                        **source,
                    }
                )
        # Relevance precedes recency. Raw data samples help establish units/codes;
        # field names and successful execution alone do not establish semantics.
        examples.sort(
            key=lambda r: (
                len(relevant & words(encode(r))),
                r.get("episode_verdict") == "correct",
            ),
            reverse=True,
        )
        errors.sort(key=lambda r: len(relevant & words(encode(r))), reverse=True)
        return bounded(
            {
                "type": "database_procedures",
                "schema_epoch": self.epoch,
                "use": "Reuse observed schemas and relevant SQL shapes to save exploratory queries. Read sample values to verify units and encodings. Adapt literals and filters to the CURRENT question; past answers are not current answers. A zero result does not validate a conversion. Never infer a dataset's category solely from a table suffix.",
                "verification": "Only public final-answer verdicts are available. Revalidate joins, meanings, and changed schemas.",
            },
            schemas + examples[:16] + errors[:5],
            self.max_chars,
        )


class FocusedCohortMemory(CohortMemory):
    """Retrieve scoped analysis recipes, with their prerequisites and evidence."""

    def context(self, query=""):
        if not self._studies:
            return ""
        target = words(query)
        studies = sorted(
            self._studies.values(),
            key=lambda s: len(target & words(encode(s["brief"]))),
            reverse=True,
        )
        records = []
        for study in studies:
            evidence = sorted(
                study["survival_evidence"],
                key=lambda e: (
                    bool(e["observable_cohort_fit"]),
                    len(target & words(e["group_expression"])),
                ),
                reverse=True,
            )
            for e in evidence[:2]:
                cols = {
                    k: v
                    for k, v in study["columns"].items()
                    if k in re.findall(r"\b\w+\b", e["group_expression"])
                }
                records.append(
                    {
                        "kind": "scoped_analysis_recipe",
                        "source_episode": study["instance_id"],
                        "study_scope": study["brief"],
                        "required_column_definitions": cols,
                        "group_expression": e["group_expression"],
                        "observed_groups": e["groups"],
                        "sample_validation": e["observable_cohort_fit"],
                        "cohort_composition": e["cohort_composition"],
                        "unobservable_cohorts": e["unobservable_cohorts"],
                        "scope": e["scope"],
                    }
                )
            if not evidence:
                records.append(
                    {
                        "kind": "study_dictionary",
                        "source_episode": study["instance_id"],
                        "study_scope": study["brief"],
                        "column_definitions": study["columns"],
                    }
                )
        return bounded(
            {
                "type": "cohort_analysis_recipes",
                "use": "Start by checking current column definitions and enrollment. A prior group expression is a reusable analysis proposal only if its required variables and encodings match. Evaluate it on CURRENT study tools before using its estimates. Distinct instruments remain distinct; do not copy previous numeric survival estimates to a different population. Unobservable cohorts are unknown, not zero. Sample fit does not establish population accuracy.",
            },
            records,
            self.max_chars,
        )


class ConditionalPokerMemory:
    """Count public responses to our raises, preserving opportunities and scope.

    The first opponent action following an accepted raise responds on the previous
    street even if the next prompt is on a later street. Remaining batched actions
    are left unattributed. Missing terminal actions are not inferred from profit.
    """

    def __init__(self, max_chars=16000):
        self.max_chars = max_chars
        self.profiles = {}
        self.current = None
        self.previous = None
        self.seen_queries = set()
        self.completed = set()

    def observe(self, query, action, observation, *, instance_id, instance_complete):
        if instance_id in self.completed:
            return
        hand = re.search(r"^Hand #(\d+) - ([A-Z_]+)", query, re.M)
        name = _line(query, "Opponent")
        if not hand or not name:
            return
        if instance_id != self.current:
            self.current, self.previous = instance_id, None
            self.seen_queries = set()
        p = self.profiles.setdefault(
            name, {"completed_hands": 0, "responses": {}, "revealed_showdowns": []}
        )
        fingerprint = query[hand.start() :]
        if fingerprint not in self.seen_queries:
            self.seen_queries.add(fingerprint)
            actions = (
                _line(query, "Opponent's actions") or _line(query, "Opponent's action")
            ).split("->")
            actions = [a.strip() for a in actions if a.strip()]
            if (
                self.previous
                and self.previous["name"] == name
                and self.previous["action"] == "RAISE"
            ):
                key = self.previous["condition"]
                bucket = p["responses"][key]
                if actions and actions[0] in {"CALL", "RAISE", "FOLD", "ALL_IN"}:
                    bucket["observed_responses"] += 1
                    counts = bucket["counts"]
                    counts[actions[0]] = counts.get(actions[0], 0) + 1
                # A response is processed only once, including invalid retries.
                self.previous = None
        if observation.startswith("Invalid poker action:"):
            return
        chosen = str(action.get("action", "")).upper()
        self.previous = {"name": name, "action": chosen}
        if chosen == "RAISE":
            pot_match = re.search(r"^Pot: (\d+) chips", query, re.M)
            pot = int(pot_match.group(1)) if pot_match else 0
            amount = action.get("amount")
            ratio = (
                amount / pot if isinstance(amount, (int, float)) and pot > 0 else None
            )
            size = (
                "unknown"
                if ratio is None
                else "raise_to/pot<=0.5"
                if ratio <= 0.5
                else "raise_to/pot<=1"
                if ratio <= 1
                else "raise_to/pot>1"
            )
            key = hand.group(2) + "/" + size
            bucket = p["responses"].setdefault(
                key,
                {"our_raise_opportunities": 0, "observed_responses": 0, "counts": {}},
            )
            bucket["our_raise_opportunities"] += 1
            self.previous["condition"] = key
        if instance_complete:
            self.completed.add(instance_id)
            p["completed_hands"] += 1
            primary = observation.split(_AUTO_MARKER)[0]
            if "SHOWDOWN:" in primary and _line(primary, "Opponent's hand"):
                p["revealed_showdowns"].append(
                    {
                        "source_episode": instance_id,
                        "board": _line(primary, "Board"),
                        "our_hand": _line(primary, "Your hand"),
                        "opponent_hand": _line(primary, "Opponent's hand"),
                    }
                )
                p["revealed_showdowns"] = p["revealed_showdowns"][-3:]
            self.previous = None

    def state_dict(self):
        return copy.deepcopy(
            {"type": "conditional_poker_observations", "opponents": self.profiles}
        )

    def context(self, query=""):
        name = _line(query, "Opponent")
        p = self.profiles.get(name)
        if not p:
            return ""
        totals = Counter()
        opportunities = observed = 0
        records = []
        for key, row in sorted(p["responses"].items()):
            totals.update(row["counts"])
            opportunities += row["our_raise_opportunities"]
            observed += row["observed_responses"]
            records.append(
                {
                    "condition": key,
                    **row,
                    "missing_response_count": row["our_raise_opportunities"]
                    - row["observed_responses"],
                }
            )
        return bounded(
            {
                "type": "opponent_response_evidence",
                "opponent": name,
                "completed_hands": p["completed_hands"],
                "all_streets_after_our_raise": {
                    "opportunities": opportunities,
                    "observed": observed,
                    "counts": dict(totals),
                    "missing": opportunities - observed,
                },
                "use": "These are observed responses to our raises, not action values or estimates of opponent card strength. An observed call means that raise did not make the opponent fold. Missing terminal responses are unknown, not folds or calls. Small samples and selectively revealed showdowns limit inference. Judge the CURRENT cards and legal options separately. Bet-size bins use raise-to amount divided by the pot before our action, not a causal effect estimate.",
                "showdowns": p["revealed_showdowns"],
            },
            records,
            self.max_chars,
        )


FOCUSED = {
    "database_exploration": FocusedDatabaseMemory,
    "cohort_studies": FocusedCohortMemory,
    "exploitable_poker": ConditionalPokerMemory,
}


class ProceduralDatabaseMemory(DatabaseMemory):
    """Turn observed schemas/errors into inspectable next-query procedures.

    Probe expressions are generic SQL diagnostics over observed identifiers. Their
    results/units/category meanings are never filled in from task internals.
    """

    @staticmethod
    def identifier(name):
        return '"' + name.replace('"', '""') + '"'

    def context(self, query=""):
        if not self.records:
            return ""
        if self._drift_notice(query):
            return encode(
                {
                    "type": "database_procedure_memory",
                    "schema_changed": True,
                    "next_step": "Reinspect current schema before using old identifiers.",
                }
            )
        records = [r for r in self.records.values() if r["schema_epoch"] == self.epoch]
        schemas, inventory = {}, set()
        for r in records:
            if r["kind"] == "schema" and r.get("columns") and r.get("table"):
                schemas[r["table"]] = r
            if r["kind"] == "table_inventory":
                inventory.update(r["tables"])
            if r["kind"] == "schema":
                inventory.update(
                    o["name"] for o in r.get("catalog_objects", []) if o.get("name")
                )
        wanted = _question(query)
        tokens = words(wanted)
        date_needed = bool(
            re.search(
                r"\b(?:year|month|date|quarter|q[1-4]|20\d\d|since|before|after)\b",
                wanted,
                re.I,
            )
        )
        price_needed = bool(
            re.search(r"\b(?:price|cost|priced|expensive|dollars?)\b|\$", wanted, re.I)
        )
        entries = []
        for table, r in sorted(schemas.items()):
            columns = {
                c["name"]: c.get("type", "") for c in r["columns"] if "name" in c
            }
            t = self.identifier(table)
            entry = {
                "kind": "observed_schema",
                "table": table,
                "columns": columns,
                "source_episodes": r["instances"],
                "schema_epoch": self.epoch,
                "reuse": "These columns were actually observed. Reuse them; repeat PRAGMA only if the schema changed or evidence conflicts.",
            }
            probes = []
            for column in columns:
                c = self.identifier(column)
                if re.search(r"cat(?:egory)?|department|segment", column, re.I):
                    probes.append(
                        {
                            "purpose": "Resolve this table's category and exact filter literals; question wording is not a stored value.",
                            "sql": f"SELECT {c}, COUNT(*) AS n FROM {t} GROUP BY {c} LIMIT 8",
                        }
                    )
                if date_needed and re.search(r"(?:^ts$|time|date|_at$)", column, re.I):
                    probes.append(
                        {
                            "purpose": "Compare raw values and candidate date decodings; their units are not yet established. Choose a decoding only after checking its displayed results.",
                            "sql": f"SELECT {c} AS raw_value, datetime({c}, 'unixepoch') AS seconds_candidate, datetime({c}/1000.0, 'unixepoch') AS milliseconds_candidate, datetime({c}) AS text_candidate FROM {t} WHERE {c} IS NOT NULL LIMIT 3",
                        }
                    )
                if price_needed and re.search(r"prc|price|cost", column, re.I):
                    probes.append(
                        {
                            "purpose": "Inspect price magnitude before applying a monetary threshold; do not assume the unit from the name.",
                            "sql": f"SELECT MIN({c}) AS min_value, MAX({c}) AS max_value, AVG({c}) AS mean_value FROM {t}",
                        }
                    )
            entry["candidate_diagnostic_queries"] = probes
            entries.append(entry)
        errors = [
            {
                "kind": "observed_failed_sql",
                "sql": r["sql"],
                "error": r["error"],
                "source_episodes": r["instances"],
                "next_step": "Do not repeat the missing table/column assumption. Check the observed schema, including the alias-to-table mapping.",
            }
            for r in records
            if r["kind"] == "sql_error"
        ]
        errors.sort(key=lambda r: len(tokens & words(encode(r))), reverse=True)
        # Correctness belongs to the final answer, never every query in a trace.
        correct = {
            r["instances"][-1]
            for r in records
            if r["kind"] == "answer_feedback" and r["verdict"] == "correct"
        }
        examples = [
            {
                "kind": "observed_query_result",
                "sql": r["sql"],
                "question": r["question"],
                "columns": r["result_columns"],
                "rows": r["sample_rows"],
                "source_episodes": r["instances"],
                "episode_answer_correct": r["instances"][-1] in correct,
                "scope": "Executed query; not necessarily a correct solution to the current question",
            }
            for r in records
            if r["kind"] == "executed_query" and r["sample_rows"]
        ]
        examples.sort(
            key=lambda r: (r["episode_answer_correct"], len(tokens & words(encode(r)))),
            reverse=True,
        )
        # Put compact schema/probes first, then high-priority errors and useful
        # value samples. Wrong final answers are not provided as templates.
        return bounded(
            {
                "type": "database_procedure_memory",
                "observed_tables": sorted(inventory),
                "procedure": [
                    "Consult the observed schemas below before proposing any table or field name. A group suffix alone does not identify a product category.",
                    "Use relevant diagnostic queries ONLY for still-unknown category values, date encodings or units. These are proposed queries, not observations; execute within the current official query budget.",
                    "Inspect actual join-key values if a join is uncertain; matching-looking column names alone do not validate it.",
                    "Once meanings are verified, aggregate with filters adapted to the CURRENT question. Parenthesize OR date filters under an AND category filter.",
                    "A NULL/zero result can indicate a wrong filter or conversion. Check underlying values before accepting it as the answer. Submit the actual executed result, not a guess.",
                ],
            },
            entries + errors[:4] + examples[:8],
            self.max_chars,
        )
