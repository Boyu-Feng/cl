"""Write a readable report from a completed paired typed-grounded run."""
from __future__ import annotations

import argparse
from pathlib import Path
import time

from ttcl.icl_mem0_comparison.protocol import read


def report(output: Path):
    status = read(output / 'status.json')
    summary = read(output / 'summary.json')
    if status['phase'] not in {'complete', 'finished_with_failures'}:
        raise ValueError('Evaluation is still running')
    lines = ['# Typed grounded-evidence MemRL evaluation', '',
             f"Status: **{status['phase']}**; recorded cells: "
             f"{summary['recorded_cells']}/{summary['expected_cells']}; "
             f"failed cells: {summary['failed_cells']}.", '',
             'Both arms used the same frozen actor, task order, decoding and memory budget. '
             'Each arm built its memory chain online from empty. Only paired, valid cells '
             'were scored. ALFWorld uses first-attempt and within-three success; CLBench '
             'uses official reward on the last 80% of each domain.', '',
             '| Benchmark / task | Metric | Paired n | Native MemRL | Typed grounded evidence | Difference |',
             '| --- | --- | ---: | ---: | ---: | ---: |']
    for group, metrics in sorted(summary['aggregates'].items()):
        for name, item in metrics.items():
            base = item['means']['vanilla']
            new = item['means']['typed_grounded']
            if base is None or new is None:
                continue
            lines.append(f'| {group} | {name} | {item["n"]} | {base:.6f} | '
                         f'{new:.6f} | {new-base:+.6f} |')
    lines.extend(['', 'Coverage and failures by task:', '',
                  '| Benchmark / task / repeat | Valid paired | Expected paired | Failed cells |',
                  '| --- | ---: | ---: | ---: |'])
    for group, item in sorted(summary['results'].items()):
        lines.append(f'| {group} | {item["scored_pairs"]} | '
                     f'{item["expected_pairs"]} | {item["failed_cells"]} |')
    lines.extend(['', 'Cost on the same valid scored pairs:', '',
                  '| Benchmark / task / repeat | Native actor calls | Candidate actor calls | Native input tokens | Candidate input tokens |',
                  '| --- | ---: | ---: | ---: | ---: |'])
    for group, item in sorted(summary['results'].items()):
        native = item['costs']['vanilla']
        candidate = item['costs']['typed_grounded']
        lines.append(f'| {group} | {native["actor_calls"]} | '
                     f'{candidate["actor_calls"]} | '
                     f'{native["actor_input_tokens"]} | '
                     f'{candidate["actor_input_tokens"]} |')
    lines.extend(['', 'The method stores exact public inputs, unverified submitted actions, '
                  'observed feedback, and source hashes as separate fields. It retrieves '
                  'bounded events with one rule in both benchmarks and selects generic '
                  'type-derived operators for schema-valid CLBench actions. Extra '
                  'calls, failures, and negative transfer must be assessed alongside scores.'])
    target = output / 'REPORT.md'
    target.write_text('\n'.join(lines) + '\n')
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--wait', action='store_true')
    args = parser.parse_args()
    output = args.output.resolve()
    if args.wait:
        while read(output / 'status.json')['phase'] not in {'complete', 'finished_with_failures'}:
            time.sleep(30)
    print(report(output))


if __name__ == '__main__':
    main()
