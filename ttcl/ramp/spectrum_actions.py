"""Optional public-candidate action space for spectrum occupancy selection.

The language model selects candidate IDs; a deterministic decoder copies public
measurement estimates into the benchmark report. This changes the action space
and uses external observation memory, so comparisons need the same action-space
and memory configuration in both frozen and adaptive runs.
"""

from __future__ import annotations

import copy
import json
import re
from dataclasses import asdict, dataclass

from ttcl.ramp.spectrum_memory import SpectrumMemory


@dataclass(frozen=True)
class SpectrumCandidate:
    candidate_id: int
    center_freq: float
    bandwidth: float
    estimated_power: float
    scan_count: int
    first_seen: int
    last_seen: int
    currently_active: bool
    center_range_mhz: tuple[float, float]
    width_range_mhz: tuple[float, float]


@dataclass(frozen=True)
class SpectrumActionCatalog:
    scan_number: int
    observed_scans: int
    band_mhz: tuple[float, float]
    max_candidates: int
    candidates: tuple[SpectrumCandidate, ...]
    omitted_candidates: int
    omitted_current_candidates: int

    def state_dict(self):
        """A fresh JSON-serializable snapshot for the experiment audit trail."""
        return asdict(self)


def build_candidate_catalog(
    memory: SpectrumMemory,
    current_observation,
    instance_id=None,
    *,
    max_candidates=None,
):
    """Include the current public scan without advancing persistent memory.

    All currently detected candidates are reserved a slot when they fit. If they
    exceed the bound, recurrence, measured power, then stable ID break ties.
    Remaining slots favor historically repeated and then recently seen evidence.
    These are catalogue capacity rules, not rules choosing the model's answer.
    """
    limit = memory.max_candidates if max_candidates is None else max_candidates
    if type(limit) is not int or limit < 1:
        raise ValueError("max_candidates must be a positive integer")
    if isinstance(current_observation, str):
        from ttcl.common.bsm import parse_scan_observation

        current_observation = parse_scan_observation(current_observation, instance_id)

    temporary = copy.deepcopy(memory)
    # A full persistent tracker must not evict new current peaks before applying
    # the explicit catalog budget. This does not change persistent tracking.
    temporary.max_tracks = max(
        temporary.max_tracks,
        len(memory.state_dict()["candidates"])
        + len(current_observation["detected_peaks"]),
    )
    temporary.observe(current_observation)
    state = temporary.state_dict()
    scan = state["last_scan"]
    candidates = [
        SpectrumCandidate(
            candidate_id=row["candidate_id"],
            center_freq=row["center_freq_mhz"],
            bandwidth=row["bandwidth_mhz"],
            estimated_power=row["mean_power_dbm"],
            scan_count=row["scan_count"],
            first_seen=row["first_seen"],
            last_seen=row["last_seen"],
            currently_active=row["last_seen"] == scan,
            center_range_mhz=tuple(row["center_range_mhz"]),
            width_range_mhz=tuple(row["width_range_mhz"]),
        )
        for row in state["candidates"]
    ]
    current = sorted(
        (row for row in candidates if row.currently_active),
        key=lambda row: (-row.scan_count, -row.estimated_power, row.candidate_id),
    )
    historic = sorted(
        (row for row in candidates if not row.currently_active),
        key=lambda row: (-row.scan_count, -row.last_seen, row.candidate_id),
    )
    selected = current[:limit]
    selected.extend(historic[: limit - len(selected)])
    # Display repeated candidates first without reassigning their stable IDs.
    selected.sort(
        key=lambda row: (
            -row.scan_count,
            not row.currently_active,
            row.center_freq,
            row.candidate_id,
        )
    )
    return SpectrumActionCatalog(
        scan_number=scan,
        observed_scans=state["observed_scans"],
        band_mhz=tuple(state["band_mhz"]),
        max_candidates=limit,
        candidates=tuple(selected),
        omitted_candidates=len(candidates) - len(selected),
        omitted_current_candidates=max(0, len(current) - limit),
    )


def render_selection_prompt(catalog: SpectrumActionCatalog):
    """A compact standalone question using only current and past public data."""
    lines = [
        "You monitor long-run spectrum occupancy. Select all credible transmitter "
        "candidates, including historically supported ones absent from the current scan. "
        "Momentary silence does not establish that a region is available. "
        "A singleton can be real or interference; recurrence and measurement consistency "
        "are evidence, not proof. Wrongly occupied and wrongly available regions have equal cost.",
        f"Current scan={catalog.scan_number}; observed scans={catalog.observed_scans}; "
        f"band={catalog.band_mhz[0]:g}-{catalog.band_mhz[1]:g} MHz.",
        "Catalog fields: id; center/width means and +/- maximum observed deviation "
        "in MHz; power mean dBm; n distinct scans; last scan seen; now=1 means "
        "matched a current peak. IDs are stable and need not be consecutive.",
    ]
    for row in catalog.candidates:
        center_spread = max(
            row.center_freq - row.center_range_mhz[0],
            row.center_range_mhz[1] - row.center_freq,
        )
        width_spread = max(
            row.bandwidth - row.width_range_mhz[0],
            row.width_range_mhz[1] - row.bandwidth,
        )
        lines.append(
            f"id={row.candidate_id} center={row.center_freq:.2f}+/-{center_spread:.2f} "
            f"width={row.bandwidth:.2f}+/-{width_spread:.2f} "
            f"power={row.estimated_power:.1f} n={row.scan_count} "
            f"last={row.last_seen} now={int(row.currently_active)}"
        )
    if not catalog.candidates:
        lines.append("No candidates have been observed.")
    if catalog.omitted_candidates:
        lines.append(
            f"Catalog capacity={catalog.max_candidates}; "
            f"{catalog.omitted_candidates} candidates omitted "
            f"({catalog.omitted_current_candidates} current). Current detections reserve "
            "slots; remaining slots favor repeated historical evidence. "
            "Omission is a capacity limit, not evidence that a transmitter is absent."
        )
    lines.append(
        'Return ONLY one JSON object with exactly this key: {"include":[integer IDs]}. '
        "Choose IDs from this catalog only, without duplicates. Include all candidates "
        "you judge to occupy spectrum over the long run. An empty list means no "
        "credible transmitter evidence. Do not output frequencies or report objects."
    )
    return "\n".join(lines)


def _unique_json_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"Duplicate JSON object key: {key}")
        value[key] = item
    return value


def decode_selection(raw, catalog: SpectrumActionCatalog, response_schema=None):
    """Strictly decode an include-list action into a benchmark ScanReport shape.

    The learned action is the original include-list JSON. Its deterministic
    numeric expansion is only the evaluator-facing report, not a teacher target.
    """
    if not isinstance(raw, str):
        raise ValueError("Selection response must be JSON text")
    raw = raw.strip()
    fence = re.fullmatch(r"```(?:json)?[ \t]*\r?\n(.*?)\r?\n```", raw, re.DOTALL)
    if fence is not None:
        raw = fence.group(1)
    try:
        value = json.loads(raw, object_pairs_hook=_unique_json_object)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("Selection response must be one valid JSON object") from exc
    if not isinstance(value, dict) or set(value) != {"include"}:
        raise ValueError("Selection object must contain exactly the include key")
    included = value["include"]
    if not isinstance(included, list):
        raise ValueError("include must be a list of candidate IDs")
    if any(type(uid) is not int or uid < 1 for uid in included):
        raise ValueError("Candidate IDs must be positive nonboolean integers")
    if len(included) > len(catalog.candidates) or len(set(included)) != len(included):
        raise ValueError("include exceeds catalog bounds or contains duplicate IDs")
    by_id = {row.candidate_id: row for row in catalog.candidates}
    if any(uid not in by_id for uid in included):
        raise ValueError("include contains an ID absent from this catalog")
    report = {
        "transmitters": [
            {
                "center_freq": by_id[uid].center_freq,
                "bandwidth": by_id[uid].bandwidth,
                "currently_active": by_id[uid].currently_active,
                "estimated_power": by_id[uid].estimated_power,
            }
            for uid in included
        ]
    }
    if response_schema is not None:
        return response_schema.model_validate(report)
    return report
