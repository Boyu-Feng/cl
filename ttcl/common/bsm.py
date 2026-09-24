"""Shared CLBench BSM paths and public response/observation parsing."""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / "current_work/continual-learning-bench"
sys.path.insert(0, str(BENCH))

def parse_report(raw, schema):
    """Accept a complete JSON report, including one wrapped in Markdown."""
    decoder = json.JSONDecoder()
    for pos, char in enumerate(raw):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(raw[pos:])
            return schema.model_validate(value)
        except ValueError:
            continue
    raise ValueError("No valid scan report in generated text")

def parse_scan_observation(prompt, instance_id):
    """Keep public measurements while dropping repeated task instructions."""
    scan_match = re.search(r"--- Scan (\d+)/(\d+) ---", prompt)
    scan_id_match = re.search(r"^\s*scan_id:\s*(\S+)", prompt, re.MULTILINE)
    noise_match = re.search(
        r"^\s*estimated_noise_floor_dbm:\s*(-?\d+(?:\.\d+)?)",
        prompt,
        re.MULTILINE,
    )
    band_match = re.search(
        r"^Band:\s*(-?\d+(?:\.\d+)?)-(-?\d+(?:\.\d+)?)\s+MHz",
        prompt,
        re.MULTILINE,
    )
    peak_pattern = re.compile(
        r"peak_id:\s*([^|\n]+)\s*\|\s*freq:\s*(-?\d+(?:\.\d+)?)\s*MHz"
        r"\s*\|\s*power:\s*(-?\d+(?:\.\d+)?)\s*dBm"
        r"\s*\|\s*width:\s*(\d+(?:\.\d+)?)\s*MHz"
    )
    if scan_match is None or scan_id_match is None or band_match is None:
        raise ValueError("Could not isolate the scan observation from CLBench prompt")
    peaks = [
        {
            "peak_id": match.group(1).strip(),
            "freq_mhz": float(match.group(2)),
            "power_dbm": float(match.group(3)),
            "width_mhz": float(match.group(4)),
        }
        for match in peak_pattern.finditer(prompt)
    ]
    return {
        "instance_id": instance_id,
        "scan_number": int(scan_match.group(1)),
        "scan_id": scan_id_match.group(1),
        "noise_floor_dbm": float(noise_match.group(1)) if noise_match else None,
        "band_mhz": [float(band_match.group(1)), float(band_match.group(2))],
        "detected_peaks": peaks,
    }
