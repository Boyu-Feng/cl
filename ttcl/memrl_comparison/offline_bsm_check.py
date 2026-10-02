"""Score a public-scan ledger on BSM without an LLM or hidden-state access.

The policy reads only the current prompt and prior prompts. The official
ground truth is passed to the scorer after the action has been constructed.
This isolates whether missing persistent state can explain low BSM scores.
"""
from __future__ import annotations

import argparse
import hashlib
import math
from pathlib import Path
import statistics

from ttcl.icl_mem0_comparison.protocol import save
from .improved_cl import ImprovedCLMemory, PEAK


def run(output: Path, seed: int = 42):
    from src.tasks.blind_spectrum_monitoring.task import (
        BlindSpectrumMonitoringTask, ScanReport, Transmitter, _score_report,
    )
    if output.exists():
        raise FileExistsError(output)
    ledger = ImprovedCLMemory.__new__(ImprovedCLMemory)
    ledger.scan_index = 0
    ledger.scan_registry = []
    rows = []
    for index in range(90):
        task = BlindSpectrumMonitoringTask(seed=seed, schedule='default',
                                           response_timeout_seconds=0)
        prompt = task.reset_baseline_instance(index).prompt
        current = [Transmitter(center_freq=float(x['freq']),
                               bandwidth=float(x['width']), currently_active=True,
                               estimated_power=float(x['power']))
                   for x in PEAK.finditer(prompt)]
        persistent = current[:]
        minimum = max(2, math.ceil(.05 * ledger.scan_index))
        for item in ledger.scan_registry:
            if item['count'] < minimum:
                continue
            if any(abs(t.center_freq - item['center']) <= max(2.5, .25 * item['width'])
                   for t in persistent):
                continue
            persistent.append(Transmitter(center_freq=item['center'],
                                          bandwidth=item['width'],
                                          currently_active=False, estimated_power=-50.))
        # The scorer is called only after both actions are fixed.
        scorer_args = dict(all_latent_channel_defs=task._get_all_latent_channel_defs(),
                           W=task.W, G=task.G, band_width=task.band_width)
        reactive = _score_report(ScanReport(transmitters=current), **scorer_args)['score']
        stateful = _score_report(ScanReport(transmitters=persistent), **scorer_args)['score']
        rows.append(dict(index=index, public_prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                         prior_scan_count=ledger.scan_index,
                         current_peaks=len(current), added_historical=len(persistent)-len(current),
                         reactive_reward=reactive, stateful_reward=stateful))
        ledger._record_scan(prompt)
    suffix = rows[18:]
    summary = dict(n=len(suffix), reactive_mean=statistics.fmean(r['reactive_reward'] for r in suffix),
                   stateful_mean=statistics.fmean(r['stateful_reward'] for r in suffix),
                   wins=sum(r['stateful_reward'] > r['reactive_reward'] for r in suffix),
                   losses=sum(r['stateful_reward'] < r['reactive_reward'] for r in suffix),
                   ties=sum(r['stateful_reward'] == r['reactive_reward'] for r in suffix))
    save(output, dict(protocol='public_scan_ledger_v2', seed=seed,
                      note=__doc__, summary=summary, rows=rows))
    print(summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    run(args.output.resolve(), args.seed)


if __name__ == '__main__':
    main()
