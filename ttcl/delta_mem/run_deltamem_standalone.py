"""Standalone Delta-Mem evaluation for the frozen BSM JSONL corpus.

This runner does not import the CLBench package. It reads frozen scan rows,
uses Delta-Mem for generation, and reproduces the BSM long-run availability IoU.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


TTCL_ROOT = Path(__file__).resolve().parents[1]
DELTA_ROOT = TTCL_ROOT.parent / "current_work" / "delta-Mem"
if str(DELTA_ROOT) not in sys.path:
    sys.path.insert(0, str(DELTA_ROOT))

PROMPT_TEMPLATE = """You are a spectrum monitoring analyst. You monitor a radio frequency band containing multiple transmitter channels. On each scan, you receive a list of detected peaks from an automated detector and must produce a structured report of your current model of the band structure.

The number of transmitters in the band is unknown. Some transmitters may be active in the current scan, while others may be temporarily inactive and reappear later. Your goal is to infer the persistent set of transmitters that occupy spectrum over time, not just the peaks visible in the current scan.

Each detected peak is described by:
- freq_mhz: estimated center frequency
- power_dbm: estimated power level
- width_mhz: estimated bandwidth

Report ALL transmitters you believe exist in the band, including ones that are currently inactive but supported by evidence from earlier scans. Use currently visible peaks as evidence, but do not limit your report to only the peaks seen in the current scan. Judge availability by long-run occupancy structure.

Do not omit the transmitters field. If the current scan contains detected peaks, report each credible transmitter peak unless you have specific evidence that it is interference or noise. An empty transmitter list means you believe there is no credible transmitter evidence in the band.

Current scan metadata:
{metadata}

Detected peaks:
{peaks}

Return only a JSON object with this exact schema:
{{
  "transmitters": [
    {{"center_freq": <float>, "bandwidth": <float>, "currently_active": <boolean>, "estimated_power": <float>}}
  ]
}}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone Delta-Mem BSM evaluation.")
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--delta-adapter", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--num-trajectories", type=int, default=None)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mode", choices=["stateful", "stateless"], default="stateful")
    return parser.parse_args()


def _load_session():
    from deltamem.runtime.session import DeltaMemChatSession, load_delta_mem_chat_model

    return DeltaMemChatSession, load_delta_mem_chat_model


def _extract_json(raw: str) -> dict[str, Any]:
    start = raw.find("{")
    if start < 0:
        raise ValueError(f"No JSON object in model output: {raw[:300]!r}")
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(raw)):
        char = raw[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
        elif char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                value = json.loads(raw[start : index + 1])
                if not isinstance(value, dict) or not isinstance(value.get("transmitters"), list):
                    raise ValueError("Report must contain a transmitters list")
                return value
    raise ValueError(f"Unclosed JSON object in model output: {raw[:300]!r}")


def _normalize(intervals: list[tuple[float, float]], end: float) -> list[tuple[float, float]]:
    clipped = [(max(0.0, lo), min(end, hi)) for lo, hi in intervals]
    clipped = [(lo, hi) for lo, hi in clipped if hi > lo]
    clipped.sort()
    merged: list[tuple[float, float]] = []
    for lo, hi in clipped:
        if merged and lo <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
        else:
            merged.append((lo, hi))
    return merged


def _complement(occupied: list[tuple[float, float]], end: float) -> list[tuple[float, float]]:
    available: list[tuple[float, float]] = []
    cursor = 0.0
    for lo, hi in _normalize(occupied, end):
        if lo > cursor:
            available.append((cursor, lo))
        cursor = max(cursor, hi)
    if cursor < end:
        available.append((cursor, end))
    return available


def _measure(intervals: list[tuple[float, float]]) -> float:
    return sum(hi - lo for lo, hi in intervals)


def _intersect(left: list[tuple[float, float]], right: list[tuple[float, float]]) -> list[tuple[float, float]]:
    result: list[tuple[float, float]] = []
    i = j = 0
    while i < len(left) and j < len(right):
        lo = max(left[i][0], right[j][0])
        hi = min(left[i][1], right[j][1])
        if hi > lo:
            result.append((lo, hi))
        if left[i][1] <= right[j][1]:
            i += 1
        else:
            j += 1
    return result


def _channel_intervals(metadata: dict[str, Any]) -> tuple[list[tuple[float, float]], float]:
    first = metadata["stages"][0]["dgp_params"]
    width = float(first["band_width"])
    W = float(first["W"])
    G = float(first["G"])
    channels: dict[int, dict[str, Any]] = {}
    for stage in metadata["stages"]:
        params = stage["dgp_params"]
        for channel in params["channels"]:
            channels[int(channel["id"])] = channel
    occupied = []
    for channel in channels.values():
        center = float(channel.get("center_freq", channel.get("center_freq_override", channel["slot"] * (W + G) + W / 2)))
        bandwidth = float(channel.get("bandwidth", channel.get("bandwidth_override", W)))
        occupied.append((center - bandwidth / 2, center + bandwidth / 2))
    return _normalize(occupied, width), width


def _score(report: dict[str, Any], gt_occupied: list[tuple[float, float]], band_width: float) -> float:
    reported = []
    for item in report["transmitters"]:
        center = float(item["center_freq"])
        bandwidth = max(0.0, float(item["bandwidth"]))
        reported.append((center - bandwidth / 2, center + bandwidth / 2))
    gt_available = _complement(gt_occupied, band_width)
    reported_available = _complement(reported, band_width)
    overlap = _measure(_intersect(gt_available, reported_available))
    gt_measure = _measure(gt_available)
    reported_measure = _measure(reported_available)
    union = gt_measure + reported_measure - overlap
    return 1.0 if union == 0.0 else overlap / union


def _prompt(row: dict[str, Any]) -> str:
    metadata = {
        key: row[key]
        for key in ("scan_idx", "stage_idx", "variant_id", "scan_id", "sensor_id", "detector_version", "integration_time_ms", "estimated_noise_floor_dbm")
        if key in row
    }
    return PROMPT_TEMPLATE.format(
        metadata=json.dumps(metadata, ensure_ascii=False),
        peaks=json.dumps(row.get("detected_peaks", []), ensure_ascii=False),
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    DeltaMemChatSession, load_model = _load_session()
    data_path = Path(args.data_path).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in data_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = rows[: args.num_trajectories] if args.num_trajectories else rows
    gt_occupied, band_width = _channel_intervals(json.loads(data_path.with_name(data_path.stem + "_metadata.json").read_text(encoding="utf-8")))
    model, tokenizer = load_model(model_path=args.base_model, device=args.device, dtype=args.dtype, attn_implementation=args.attn_implementation, adapter_dir=args.delta_adapter)
    session = DeltaMemChatSession(model=model, tokenizer=tokenizer, device=args.device)
    progress_path = output_dir / "progress.json"
    snapshot_dir = output_dir / "session_snapshot"
    records: list[dict[str, Any]] = []
    if args.resume and progress_path.is_file():
        saved = json.loads(progress_path.read_text(encoding="utf-8"))
        records = list(saved.get("trajectory_records", []))[: len(rows)]
        if args.mode == "stateful" and records and snapshot_dir.is_dir():
            session.load_snapshot_dir(snapshot_dir)
        print(f"resumed={len(records)}/{len(rows)} mode={args.mode}", flush=True)

    for row in rows[len(records) :]:
        if args.mode == "stateless":
            session.reset()
        result = session.generate_reply(_prompt(row), max_new_tokens=args.max_new_tokens, do_sample=False, write_enabled=True)
        report = _extract_json(str(result["assistant_display"]))
        reward = _score(report, gt_occupied, band_width)
        records.append({"trajectory": len(records) + 1, "scan_idx": row["scan_idx"], "scan_id": row.get("scan_id", ""), "reward": round(reward, 4), "report": report})
        scores = [float(item["reward"]) for item in records]
        progress_path.write_text(json.dumps({"mode": args.mode, "completed": len(records), "total": len(rows), "last_reward": reward, "running_mean_score": sum(scores) / len(scores), "trajectory_records": records}, indent=2, ensure_ascii=False), encoding="utf-8")
        if args.mode == "stateful":
            session.save_snapshot(snapshot_dir)
        state_stats = session.state_stats()
        print(f"mode={args.mode} trajectory={len(records)}/{len(rows)} reward={reward:.4f} running_mean={sum(scores) / len(scores):.4f} delta_state_norm={state_stats.get('mean_state_norm', 0.0):.4f}", flush=True)

    scores = [float(item["reward"]) for item in records]
    half = max(1, len(scores) // 2)
    first = sum(scores[:half]) / half if scores else 0.0
    second = sum(scores[half:]) / max(1, len(scores) - half) if scores else 0.0
    payload = {"base_model": args.base_model, "delta_adapter": args.delta_adapter, "data_path": str(data_path), "mode": args.mode, "num_trajectories": len(records), "mean_score": sum(scores) / len(scores) if scores else 0.0, "score_curve": scores, "first_half_score": first, "second_half_score": second, "learning_delta": second - first, "trajectory_records": records}
    (output_dir / "metrics.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def main() -> None:
    print(json.dumps(run(parse_args()), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
