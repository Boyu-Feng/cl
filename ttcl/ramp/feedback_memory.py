"""Bounded experience memory derived from public observations and scalar rewards.

Single responses are episodes, not correctness labels. Only within-scan
comparisons support preferences; single-ID changes support local attribution.
No model calls, evaluator internals, or cross-question reward baselines are used.
"""

from __future__ import annotations

from collections import defaultdict, deque
import copy
import itertools
import json
import math


def condition(row):
    return ("repeated" if row.scan_count >= 2 else "singleton") + "/" + (
        "present" if row.currently_active else "absent")


def selection_ids(raw):
    raw = raw.strip()
    if raw.startswith("```"):
        raw = "\n".join(raw.splitlines()[1:-1])
    return sorted(json.loads(raw)["include"])


class FeedbackMemory:
    def __init__(self, window=16, min_gap=0.01):
        if type(window) is not int or window < 1:
            raise ValueError("feedback window must be a positive integer")
        if not math.isfinite(min_gap) or min_gap <= 0:
            raise ValueError("feedback min_gap must be finite and positive")
        self.episodes = deque(maxlen=window)
        self.min_gap = min_gap
        self.observed_scans = 0
        self.reward_count = 0
        self.last_scan = 0

    def observe(self, *, scan, current, catalog, responses, reports, rewards):
        """Call once AFTER all answers and rewards; only allowlisted data enters."""
        if scan <= self.last_scan:
            raise ValueError("Feedback scans must be strictly increasing")
        if not rewards or len({len(responses), len(reports), len(rewards)}) != 1:
            raise ValueError("Feedback candidates must have equal nonzero lengths")
        if any(not math.isfinite(reward) for reward in rewards):
            raise ValueError("Feedback rewards must be finite")
        actions = []
        for raw, report, reward in zip(responses, reports, rewards):
            if report is None:
                action = None
            elif catalog is not None:
                action = {"include": selection_ids(raw)}
            else:
                action = {"regions_mhz": sorted(
                    [row.center_freq, row.bandwidth] for row in report.transmitters)}
            actions.append({"action": action, "reward": float(reward),
                            "valid": report is not None})
        valid = [i for i, action in enumerate(actions) if action["valid"]]
        preference = None
        if len(valid) >= 2:
            best = max(valid, key=lambda i: rewards[i])
            worst = min(valid, key=lambda i: rewards[i])
            if rewards[best] - rewards[worst] >= self.min_gap:
                preference = {"preferred": best, "disfavored": worst,
                              "reward_gap": rewards[best] - rewards[worst]}
        effects = []
        if catalog is not None:
            by_id = {row.candidate_id: row for row in catalog.candidates}
            seen_pairs = set()
            for i, j in itertools.combinations(valid, 2):
                a, b = (set(actions[k]["action"]["include"]) for k in (i, j))
                changed = a ^ b
                if len(changed) != 1:
                    continue
                uid = next(iter(changed))
                pair = tuple(sorted((tuple(sorted(a)), tuple(sorted(b)))))
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)
                delta = float(rewards[i] - rewards[j]) * (1 if uid in a else -1)
                if abs(delta) < self.min_gap:
                    continue
                row = by_id[uid]
                effects.append({"candidate_id": uid, "condition": condition(row),
                                "include_minus_exclude": delta,
                                "pair": [i, j]})
        public = None if current is None else {
            "scan_number": current["scan_number"],
            "peaks_mhz": [[p["freq_mhz"], p["width_mhz"]]
                          for p in current["detected_peaks"]],
        }
        episode = {"scan": scan, "public_observation": public,
                   "catalog": catalog.state_dict() if catalog is not None else None,
                   "actions": actions, "preference": preference, "local_effects": effects}
        self.episodes.append(episode)
        self.last_scan = scan
        self.observed_scans += 1
        self.reward_count += len(rewards)
        return copy.deepcopy(episode)

    def render(self, catalog=None):
        if not self.episodes:
            return ""
        lines = [
            f"Past scalar-reward experience (through scan {self.last_scan}; higher reward is better):",
            "Single-answer scores are outcomes, NOT correct/incorrect labels. Never compare "
            "scores across different scans to infer action quality. Past actions are evidence, "
            "not answers to copy. Reassess using current observations.",
        ]
        # Keep complete records on disk; bound the prompt independently.
        for episode in list(self.episodes)[-3:]:
            actions = episode["actions"]
            indices = [0]
            pref = episode["preference"]
            if pref:
                indices = list(dict.fromkeys([0, pref["preferred"], pref["disfavored"]]))
            public = episode["public_observation"]
            peaks = public["peaks_mhz"] if public else []
            lines.append(f"Scan {episode['scan']} observed peaks [center,width] MHz="
                         + json.dumps(peaks[:24], separators=(",", ":"))
                         + (f" (showing 24/{len(peaks)})" if len(peaks) > 24 else ""))
            for i in indices:
                item = actions[i]
                action = item["action"]
                if action is not None:
                    action = {key: values[:24] for key, values in action.items()}
                lines.append(f"  candidate {i}: action={json.dumps(action, separators=(',', ':'))} "
                             f"reward={item['reward']:.4f} valid={item['valid']}")
            if pref:
                lines.append(f"  SAME-SCAN preference: candidate {pref['preferred']} > "
                             f"{pref['disfavored']}, gap={pref['reward_gap']:.4f}. "
                             "This ranks these whole actions, not every individual choice.")
        if catalog is not None:
            grouped = defaultdict(list)
            for episode in self.episodes:
                for effect in episode["local_effects"]:
                    grouped[(effect["candidate_id"], effect["condition"])].append(
                        (episode["scan"], effect["include_minus_exclude"]))
            lines.append("Local evidence from same-scan pairs differing by exactly ONE ID. "
                         "Applicable only to the same tracked ID and evidence condition; "
                         "not a ground-truth label or a guaranteed future effect.")
            relevant = [(row, grouped[(row.candidate_id, condition(row))])
                        for row in catalog.candidates
                        if grouped[(row.candidate_id, condition(row))]]
            relevant.sort(key=lambda item: -max(scan for scan, _ in item[1]))
            for row, evidence in relevant[:6]:
                positive = sum(delta > 0 for _, delta in evidence)
                negative = len(evidence) - positive
                scans = sorted({scan for scan, _ in evidence})
                mean = sum(delta for _, delta in evidence) / len(evidence)
                lines.append(f"  ID {row.candidate_id} ({condition(row)}): including helped in "
                             f"{positive} comparisons, hurt in {negative}; "
                             f"mean include-minus-exclude reward={mean:+.4f}; source scans={scans}. "
                             "Comparisons from one scan are not independent trials.")
        return "\n".join(lines)

    def state_dict(self):
        return {"window": self.episodes.maxlen, "min_gap": self.min_gap,
                "observed_scans": self.observed_scans, "reward_count": self.reward_count,
                "last_scan": self.last_scan, "episodes": copy.deepcopy(list(self.episodes))}
