"""Accumulate sales evidence from public messages, never evaluator internals.

No benchmark generator parameters, semantic family map, future actuals, or
task-owned files are consulted. Field aliases below are lexical interpretations
of headers actually returned to the agent; their exact spelling is retained.
"""

from __future__ import annotations

import copy
import csv
import io
import json
import math
import re
from collections import defaultdict
from typing import Any


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (ValueError, TypeError):
        return None
    return result if math.isfinite(result) else None


def _json_values(text: str) -> list[Any]:
    """Decode complete JSON values in tool wrappers/fenced public feedback."""
    decoder = json.JSONDecoder()
    values = []
    position = 0
    while position < len(text):
        match = re.search(r"[\[{]", text[position:])
        if not match:
            break
        start = position + match.start()
        try:
            value, end = decoder.raw_decode(text, start)
        except (ValueError, RecursionError):
            position = start + 1
            continue
        values.append(value)
        position = end
    return values


def _records(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return [row for row in value if isinstance(row, dict)]
    if isinstance(value, dict):
        # Only common display envelopes, never flatten arbitrary hidden keys.
        for key in ("data", "rows", "records", "furniture", "locations", "catalog"):
            if isinstance(value.get(key), list):
                return _records(value[key])
        return [value]
    return []


def _csv_records(text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if "," not in line or any(x in line for x in ("{", "}", "<", ">")):
            continue
        headers = next(csv.reader([line]), [])
        if not any(
            h.strip().lower() in {"year", "sale_year", "txn_year"} for h in headers
        ):
            continue
        if not any(
            h.strip().lower() in {"quantity", "units_sold", "items_sold"}
            for h in headers
        ):
            continue
        block = [line]
        for subsequent in lines[index + 1 :]:
            parsed = next(csv.reader([subsequent]), [])
            if len(parsed) != len(headers):
                break
            block.append(subsequent)
        rows.extend(csv.DictReader(io.StringIO("\n".join(block))))
    return rows


def _field(row: dict[str, Any], names: tuple[str, ...]) -> tuple[str | None, Any]:
    for name in names:
        if name in row and row[name] is not None:
            return name, row[name]
    return None, None


class SalesMemory:
    """Observed annual panels, public catalog links, and settled forecast errors."""

    def __init__(self, max_chars: int = 20000) -> None:
        self.max_chars = max(1000, max_chars)
        self._catalog: dict[str, dict[str, Any]] = {}
        self._locations: dict[str, dict[str, Any]] = {}
        self._observations: dict[str, dict[str, Any]] = {}
        self._settled: dict[str, dict[str, Any]] = {}
        self._schemas: dict[str, dict[str, Any]] = {}
        self._episodes: list[str] = []
        self._diagnostics: list[dict[str, Any]] = []

    def observe(
        self,
        query: str,
        action: dict,
        observation: str,
        *,
        instance_id: str,
        instance_complete: bool,
    ) -> None:
        if instance_id not in self._episodes:
            self._episodes.append(instance_id)
        # Public feedback for the elapsed year is supplied in the next prompt.
        for payload in _json_values(query):
            if (
                isinstance(payload, dict)
                and "feedback_year" in payload
                and isinstance(payload.get("entries"), list)
            ):
                self._feedback(payload, instance_id)
        command = str(action.get("command", ""))
        payloads = _json_values(observation)
        records = [row for value in payloads for row in _records(value)]
        records.extend(_csv_records(observation))
        for row in records:
            self._record(row, instance_id, command)
        for path, columns in re.findall(
            r"`(data/[^`]+\.csv)`\s*[—-]\s*columns:\s*([^\n]+)", query
        ):
            self._schemas[path] = {
                "columns": [c.strip() for c in columns.split(",")],
                "source_instance": instance_id,
            }
        if re.search(r"Traceback|Error:|No such file|KeyError", observation):
            entry = {
                "command": command[:500],
                "error": observation[:600],
                "instance": instance_id,
            }
            if entry not in self._diagnostics:
                self._diagnostics.append(entry)
                self._diagnostics = self._diagnostics[-8:]

    def _feedback(self, payload: dict, source: str) -> None:
        year = _number(payload.get("feedback_year"))
        if year is None or not year.is_integer():
            return
        for entry in payload["entries"]:
            if (
                not isinstance(entry, dict)
                or not isinstance(entry.get("locality"), str)
                or not isinstance(entry.get("furniture_name"), str)
            ):
                continue
            actual = _number(entry.get("actual"))
            predicted = _number(entry.get("predicted"))
            if actual is None or actual < 0:
                continue
            key = json.dumps([entry["locality"], entry["furniture_name"], int(year)])
            self._settled[key] = {
                "locality": entry["locality"],
                "product": entry["furniture_name"],
                "year": int(year),
                "actual": actual,
                "predicted": predicted,
                "signed_error": predicted - actual if predicted is not None else None,
                "source_instance": source,
                "evidence": "public_elapsed_year_feedback",
            }

    def _record(self, row: dict, source: str, command: str) -> None:
        product_key, product_id = _field(
            row, ("furniture_id", "product_id", "item_code")
        )
        name_key, name = _field(row, ("furniture_name", "product_name", "description"))
        location_key, location_id = _field(
            row, ("location_id", "store_id", "branch_id")
        )
        _, locality = _field(row, ("locality", "city", "branch_city"))
        year_key, raw_year = _field(row, ("year", "sale_year", "txn_year"))
        count_key, raw_count = _field(row, ("items_sold", "quantity", "units_sold"))
        if product_id is not None and isinstance(name, str) and name:
            _, category = _field(
                row, ("furniture_type", "category", "item_category", "type")
            )
            _, price = _field(row, ("furniture_price", "retail_price", "price"))
            # Join only ID/name pairs that co-occurred in an actual public row.
            self._catalog[f"{product_key}:{product_id}"] = {
                "id": str(product_id),
                "id_field": product_key,
                "name": name,
                "name_field": name_key,
                "category": category,
                "price": _number(price),
                "source_instance": source,
            }
        if location_id is not None and isinstance(locality, str):
            self._locations[f"{location_key}:{location_id}"] = {
                "id": str(location_id),
                "id_field": location_key,
                "locality": locality,
                "source_instance": source,
            }
        year, count = _number(raw_year), _number(raw_count)
        if year is None or not year.is_integer() or count is None or count < 0:
            return
        if product_id is None and not isinstance(name, str):
            return
        if location_id is None and not isinstance(locality, str):
            return
        # Agent-produced output is evidence of what the tool printed, not
        # certified ground truth; provenance and source command stay explicit.
        row_key = json.dumps(
            [source, product_key, product_id, name, location_id, locality, int(year)],
            sort_keys=True,
        )
        self._observations[row_key] = {
            "product_id": str(product_id) if product_id is not None else None,
            "product_id_field": product_key,
            "product_name": name,
            "location_id": str(location_id) if location_id is not None else None,
            "location_id_field": location_key,
            "locality": locality,
            "year": int(year),
            "items_sold": count,
            "observed_fields": [product_key or name_key, year_key, count_key],
            "source_instance": source,
            "source_command": command[:320],
        }

    def _panel(self) -> list[dict[str, Any]]:
        panel: dict[tuple, dict] = {}
        for row in self._observations.values():
            product = row["product_name"]
            linked_by_value = []
            if not product:
                match = self._catalog.get(
                    f"{row['product_id_field']}:{row['product_id']}"
                )
                if match is None:
                    candidates = [
                        item
                        for item in self._catalog.values()
                        if item["id"] == row["product_id"]
                    ]
                    if len({item["name"] for item in candidates}) == 1:
                        match = candidates[0]
                        linked_by_value.append(
                            "product: unique ID value in public catalog; header differs"
                        )
                product = (
                    match["name"]
                    if match
                    else f"{row['product_id_field']}:{row['product_id']}"
                )
            locality = row["locality"] or self._locations.get(
                f"{row['location_id_field']}:{row['location_id']}", {}
            ).get("locality")
            if locality is None:
                candidates = [
                    item
                    for item in self._locations.values()
                    if item["id"] == row["location_id"]
                ]
                if len({item["locality"] for item in candidates}) == 1:
                    locality = candidates[0]["locality"]
                    linked_by_value.append(
                        "location: unique ID value in public location catalog; header differs"
                    )
            locality = (
                locality
                or f"unresolved_{row['location_id_field']}:{row['location_id']}"
            )
            key = (str(locality), str(product), row["year"])
            point = {
                "locality": str(locality),
                "product": str(product),
                "year": row["year"],
                "items_sold": row["items_sold"],
                "source": "tool_display",
                "instance": row["source_instance"],
            }
            if linked_by_value:
                point["inferred_joins"] = linked_by_value
            if key in panel and panel[key]["items_sold"] != point["items_sold"]:
                point["conflicting_previous_value"] = panel[key]["items_sold"]
            panel[key] = point
        for row in self._settled.values():
            key = (row["locality"], row["product"], row["year"])
            panel[key] = {
                "locality": key[0],
                "product": key[1],
                "year": key[2],
                "items_sold": row["actual"],
                "source": row["evidence"],
                "instance": row["source_instance"],
            }
        return sorted(
            panel.values(),
            key=lambda row: (row["locality"], row["product"], row["year"]),
        )

    def state_dict(self) -> dict:
        panel = self._panel()
        series = defaultdict(list)
        for row in panel:
            series[(row["locality"], row["product"])].append(row)
        trends = []
        for (locality, product), points in sorted(series.items()):
            points = sorted(points, key=lambda row: row["year"])
            if len(points) < 2:
                continue
            first, last = points[0], points[-1]
            years = last["year"] - first["year"]
            if years <= 0 or first["items_sold"] <= 0:
                continue
            trends.append(
                {
                    "locality": locality,
                    "product": product,
                    "n_years_observed": len(points),
                    "first_year": first["year"],
                    "last_year": last["year"],
                    "first_count": first["items_sold"],
                    "last_count": last["items_sold"],
                    "endpoint_annual_growth": round(
                        (last["items_sold"] / first["items_sold"]) ** (1 / years) - 1, 6
                    ),
                }
            )
        residuals = defaultdict(list)
        for row in self._settled.values():
            if row["signed_error"] is not None:
                residuals[row["locality"]].append(row)
        errors = []
        for locality, rows in sorted(residuals.items()):
            total = sum(row["actual"] for row in rows)
            errors.append(
                {
                    "locality": locality,
                    "n_settled": len(rows),
                    "mean_signed_error": round(
                        sum(row["signed_error"] for row in rows) / len(rows), 4
                    ),
                    "wape": round(
                        sum(abs(row["signed_error"]) for row in rows) / total, 6
                    )
                    if total
                    else None,
                    "interpretation": "positive signed error means overprediction; elapsed years only",
                }
            )
        return copy.deepcopy(
            {
                "method": "public_sales_panel",
                "instances_seen": self._episodes,
                "catalog": list(self._catalog.values()),
                "observed_schemas": self._schemas,
                "annual_panel": panel,
                "observed_trends": trends,
                "settled_error_by_locality": errors,
                "recent_errors": self._diagnostics,
                "limits": [
                    "No future realized sales or latent demand parameters are available.",
                    "Tool displays may be agent-computed; source is retained, not certified as truth.",
                    "Endpoint growth is descriptive evidence, not a causal or shared-category law.",
                ],
            }
        )

    def context(self, query: str = "") -> str:
        if not self._episodes:
            return ""
        state = self.state_dict()
        state["context_truncated"] = False

        def encode() -> str:
            return json.dumps(
                state, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            )

        # Prune whole records so the result remains valid and old provenance
        # never becomes a dangling/truncated JSON fragment.
        for field in (
            "recent_errors",
            "catalog",
            "annual_panel",
            "observed_trends",
            "settled_error_by_locality",
        ):
            while len(encode()) > self.max_chars and state[field]:
                state[field].pop(0)
                state["context_truncated"] = True
        while len(encode()) > self.max_chars and state["observed_schemas"]:
            state["observed_schemas"].pop(next(iter(state["observed_schemas"])))
            state["context_truncated"] = True
        while len(encode()) > self.max_chars and state["instances_seen"]:
            state["instances_seen"].pop(0)
            state["context_truncated"] = True
        return encode()
