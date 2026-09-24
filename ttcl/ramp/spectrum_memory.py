"""Bounded external memory of public BSM measurements, without reward or labels.

This is an explicit text-memory extension, not parameter-only RAMP. Render the
memory before answering a scan, then observe that public scan exactly once.
Frozen and adaptive comparisons must use the same memory configuration.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class _Candidate:
    uid: int
    count: int
    first_seen: int
    last_seen: int
    center_sum: float
    width_sum: float
    power_sum: float
    center_min: float
    center_max: float
    width_min: float
    width_max: float

    @property
    def center(self):
        return self.center_sum / self.count

    @property
    def width(self):
        return self.width_sum / self.count

    def add(self, scan, center, width, power):
        self.count += 1
        self.last_seen = scan
        self.center_sum += center
        self.width_sum += width
        self.power_sum += power
        self.center_min = min(self.center_min, center)
        self.center_max = max(self.center_max, center)
        self.width_min = min(self.width_min, width)
        self.width_max = max(self.width_max, width)

    def snapshot(self):
        return {
            "candidate_id": self.uid,
            "center_freq_mhz": round(self.center, 4),
            "bandwidth_mhz": round(self.width, 4),
            "mean_power_dbm": round(self.power_sum / self.count, 4),
            "center_range_mhz": [self.center_min, self.center_max],
            "width_range_mhz": [self.width_min, self.width_max],
            "scan_count": self.count,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "status": (
                "repeated_candidate" if self.count >= 2 else "uncertain_single_scan"
            ),
        }


class SpectrumMemory:
    """Cluster earlier public peaks using fixed, explicitly configured tolerances.

    Matching is one-to-one within each scan. Repeated observations increase
    support, but are never labeled as confirmed truth or permanent occupancy.
    Distinct bandwidth modes beyond the width tolerance remain separate.
    """

    def __init__(
        self,
        center_tolerance_mhz=3.0,
        width_tolerance_mhz=5.0,
        max_candidates=24,
        max_tracks=128,
    ):
        for value in (center_tolerance_mhz, width_tolerance_mhz):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("Clustering tolerances must be finite and positive")
        if (
            type(max_candidates) is not int
            or type(max_tracks) is not int
            or not 1 <= max_candidates <= max_tracks
        ):
            raise ValueError("Require 1 <= max_candidates <= max_tracks")
        self.center_tolerance_mhz = center_tolerance_mhz
        self.width_tolerance_mhz = width_tolerance_mhz
        self.max_candidates = max_candidates
        self.max_tracks = max_tracks
        self._candidates = []
        self._next_uid = 1
        self._last_scan = 0
        self._observed_scans = 0
        self._band = None
        self._evicted = 0

    @staticmethod
    def _retention_key(candidate):
        return (-candidate.count, -candidate.last_seen, candidate.uid)

    def observe(self, observation, instance_id=None):
        """Accept an allowlisted observation dict or raw public query prompt.

        Extra fields are neither read nor retained. Prompt parsing is lazy to
        keep this module usable independently of benchmark/model dependencies.
        Nonmonotonic scan numbers fail rather than double-counting candidates.
        """
        if isinstance(observation, str):
            from ttcl.common.bsm import parse_scan_observation

            observation = parse_scan_observation(observation, instance_id)
        scan = observation["scan_number"]
        if type(scan) is not int or scan <= self._last_scan:
            raise ValueError("Observe each scan once in strictly increasing order")
        band = tuple(float(x) for x in observation["band_mhz"])
        if (
            len(band) != 2
            or not all(math.isfinite(x) for x in band)
            or band[0] >= band[1]
        ):
            raise ValueError("band_mhz must contain finite increasing endpoints")
        if self._band is not None and self._band != band:
            raise ValueError("Start a new SpectrumMemory when the monitored band changes")
        # Validate the entire input before mutating state.
        peaks = []
        for peak in observation["detected_peaks"]:
            center, width, power = (
                float(peak[name]) for name in ("freq_mhz", "width_mhz", "power_dbm")
            )
            if (
                not all(math.isfinite(x) for x in (center, width, power))
                or width <= 0
            ):
                raise ValueError("Peak measurements must be finite with positive width")
            peaks.append((center, width, power))
        peaks.sort()

        edges = []
        for index, (center, width, _) in enumerate(peaks):
            for candidate in self._candidates:
                dc = abs(center - candidate.center)
                dw = abs(width - candidate.width)
                if dc > self.center_tolerance_mhz or dw > self.width_tolerance_mhz:
                    continue
                # Avoid arbitrarily long chains of gradually shifting peaks.
                if (
                    max(center, candidate.center_max) - min(center, candidate.center_min)
                    > 2 * self.center_tolerance_mhz
                ):
                    continue
                if (
                    max(width, candidate.width_max) - min(width, candidate.width_min)
                    > 2 * self.width_tolerance_mhz
                ):
                    continue
                distance = dc / self.center_tolerance_mhz + dw / self.width_tolerance_mhz
                edges.append((distance, candidate.uid, index, candidate))
        matched_peaks, matched_candidates = set(), set()
        for _, uid, index, candidate in sorted(edges, key=lambda edge: edge[:3]):
            if index in matched_peaks or uid in matched_candidates:
                continue
            candidate.add(scan, *peaks[index])
            matched_peaks.add(index)
            matched_candidates.add(uid)
        for index, (center, width, power) in enumerate(peaks):
            if index in matched_peaks:
                continue
            self._candidates.append(
                _Candidate(
                    self._next_uid, 1, scan, scan, center, width, power,
                    center, center, width, width,
                )
            )
            self._next_uid += 1
        if len(self._candidates) > self.max_tracks:
            self._evicted += len(self._candidates) - self.max_tracks
            self._candidates = sorted(self._candidates, key=self._retention_key)[
                : self.max_tracks
            ]
        self._last_scan = scan
        self._observed_scans += 1
        self._band = band

    def render(self):
        """Return a compact snapshot containing only already-observed scans."""
        if not self._observed_scans:
            return ""
        chosen = sorted(self._candidates, key=self._retention_key)[: self.max_candidates]
        chosen.sort(key=lambda candidate: (candidate.center, candidate.width))
        lines = [
            f"Historical public-observation memory: {self._observed_scans} earlier scans, through scan {self._last_scan}.",
            "Each line is a candidate, not a confirmed transmitter. "
            "Use repeated evidence in the long-run report even when absent now. "
            "Singletons are uncertain. Infer current activity from the CURRENT scan.",
            "Center/width are means in MHz; +/- is maximum observed deviation. "
            "n counts distinct scans (n=1 uncertain, n>=2 repeated); last is last-seen scan.",
        ]
        for c in chosen:
            center_spread = max(c.center - c.center_min, c.center_max - c.center)
            width_spread = max(c.width - c.width_min, c.width_max - c.width)
            lines.append(
                f"- center={c.center:.2f}+/-{center_spread:.2f} "
                f"width={c.width:.2f}+/-{width_spread:.2f} n={c.count} last={c.last_seen}"
            )
        if not chosen:
            lines.append("No peaks observed in the previous scans.")
        omitted = len(self._candidates) - len(chosen)
        if omitted:
            lines.append(f"{omitted} tracked candidates omitted from this bounded summary.")
        return "\n".join(lines)

    def context(self):
        return self.render()

    def state_dict(self):
        return {
            "kind": "public_observation_text_memory",
            "center_tolerance_mhz": self.center_tolerance_mhz,
            "width_tolerance_mhz": self.width_tolerance_mhz,
            "max_candidates": self.max_candidates,
            "max_tracks": self.max_tracks,
            "observed_scans": self._observed_scans,
            "last_scan": self._last_scan,
            "band_mhz": list(self._band) if self._band is not None else None,
            "evicted_candidates": self._evicted,
            "candidates": [
                c.snapshot() for c in sorted(self._candidates, key=lambda c: c.uid)
            ],
        }
