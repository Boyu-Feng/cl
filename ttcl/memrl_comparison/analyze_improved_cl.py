"""Audit paired online CLBench memory runs and write a provisional/final report."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import statistics

from ttcl.icl_mem0_comparison.protocol import read, save, sha


RUNS = {'blind_spectrum_monitoring': 'bsm_action_v2',
        'cohort_studies': 'cohort_action_v3'}
COUNTS = {'blind_spectrum_monitoring': 90, 'cohort_studies': 20}


def _interval(cells, seed, expected_unique):
    by_index = {}
    for row in cells:
        by_index.setdefault(row['index'], []).append(row['delta'])
    # Partial prefixes are dominated by task order; report an uncertainty
    # interval only once every primary-range task index has been observed.
    if len(by_index) < expected_unique:
        return None
    rng = random.Random(seed)
    keys = list(by_index)
    values = []
    for _ in range(10000):
        picked = rng.choices(keys, k=len(keys))
        values.append(statistics.fmean(v for key in picked for v in by_index[key]))
    values.sort()
    return [values[250], values[9750]]


def analyze(root: Path, output: Path):
    results = {}
    for task, prefix in RUNS.items():
        n = COUNTS[task]
        all_pairs = []
        per_repeat = {}
        for repeat in (303, 404):
            run = root / f'20261001_{prefix}_{repeat}'
            design = read(run / 'design.json')
            if design['task'] != task or design['repeat'] != repeat or design['limit'] != n:
                raise ValueError(f'Run design mismatch: {run}')
            for name in ('improved_cl.py', 'evaluate_improved_cl.py', 'worker.py', 'memory.py'):
                if not (run / 'source' / name).exists():
                    raise ValueError(f'Frozen source missing: {run / name}')
            if sha(run / 'source' / 'improved_cl.py') != design['implementation_sha256']:
                raise ValueError(f'Frozen method source changed: {run}')
            if sha(run / 'source' / 'evaluate_improved_cl.py') != design['runner_sha256']:
                raise ValueError(f'Frozen runner source changed: {run}')
            progress_path = run / 'progress.json'
            progress = read(progress_path) if progress_path.exists() else {}
            paired = []
            for index in range(n):
                a = run / 'vanilla' / f'episode_{index+1:03d}' / 'row.json'
                b = run / 'public_evidence' / f'episode_{index+1:03d}' / 'row.json'
                if not a.exists() or not b.exists():
                    continue
                x, y = read(a), read(b)
                if x['initial_query_sha256'] != design['input_hashes'][index] or \
                   y['initial_query_sha256'] != design['input_hashes'][index]:
                    raise ValueError(f'Paired public input changed: {run}/{index+1}')
                if x['status'] != 'complete' or y['status'] != 'complete':
                    continue
                action_path = run / 'public_evidence' / f'episode_{index+1:03d}' / 'policy_action.json'
                if not action_path.exists():
                    raise ValueError(f'Policy action audit missing: {action_path}')
                action = read(action_path)
                pair = dict(index=index, repeat=repeat, vanilla=x['reward'],
                            public_evidence=y['reward'], delta=y['reward']-x['reward'],
                            action_change=(len(action['added_history']) if task == 'blind_spectrum_monitoring'
                                           else action['prior_report_count']))
                paired.append(pair)
                if index >= int(.2*n):
                    all_pairs.append(pair)
            per_repeat[str(repeat)] = dict(
                run_complete=(progress.get('completed') == progress.get('expected') == 2*n
                              and (run / 'rows.json').exists()),
                recorded_cells=progress.get('completed', 0), expected_cells=2*n,
                scored_pairs=len(paired),
                primary_pairs=sum(x['index'] >= int(.2*n) for x in paired))
        vals = all_pairs
        results[task] = dict(primary_n=len(vals), expected_primary_n=2*(n-int(.2*n)),
                             vanilla_mean=statistics.fmean(x['vanilla'] for x in vals) if vals else None,
                             public_evidence_mean=statistics.fmean(x['public_evidence'] for x in vals) if vals else None,
                             delta_mean=statistics.fmean(x['delta'] for x in vals) if vals else None,
                             delta_cluster_bootstrap_95=_interval(vals, 20261001,
                                                                  n-int(.2*n)),
                             wins=sum(x['delta']>0 for x in vals),
                             losses=sum(x['delta']<0 for x in vals),
                             ties=sum(x['delta']==0 for x in vals),
                             action_used=sum(x['action_change']>0 for x in vals),
                             repeats=per_repeat, primary_pairs=vals)
    save(output, results)
    lines = ['# CLBench 任务状态记忆：在线配对验证', '',
             '改进版与原版 MemRL 从空记忆独立在线演化；相同提示复用相同模型响应。'
             '新增结构化状态与动作投影只使用当前及此前的公开任务输入、工具反馈和此前的原始提交。'
             '原生 MemRL 检索、Q 更新和成功分支仍沿用原实验的官方奖励；'
             '因此这不是完全隐藏评分的 CLBench 提交。', '',
             '| 领域 | 已完成主范围配对 | 原版 MemRL | 改进版 | 差值 | 胜/负/平 |',
             '|---|---:|---:|---:|---:|---:|']
    for task, row in results.items():
        if row['primary_n']:
            lines.append(f"| {task} | {row['primary_n']}/{row['expected_primary_n']} |"
                         f" {row['vanilla_mean']:.6f} | {row['public_evidence_mean']:.6f} |"
                         f" {row['delta_mean']:+.6f} | {row['wins']}/{row['losses']}/{row['ties']} |")
        else:
            lines.append(f"| {task} | 0/{row['expected_primary_n']} | — | — | — | — |")
    lines += ['', '仅当各链 recorded_cells 达到 expected_cells 且 rows.json 已写入，才把在线运行视为完成；'
              '未计分配对单独保留，不补零。此前所有均值均为进行中的阶段值。'
              '离线机制检查见同目录 BSM_PUBLIC_LEDGER_OFFLINE_20261001.json'
              ' 与 COHORT_PUBLIC_ENSEMBLE_OFFLINE_20261001.json。',
              '实验输入、源码快照和每题最终动作分别见各运行目录的 design.json、source/ 与 policy_action.json。']
    (output.with_suffix('.md')).write_text('\n'.join(lines) + '\n')
    print(json.dumps({k:{key:v for key,v in row.items() if key != 'primary_pairs'}
                      for k,row in results.items()}, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    analyze(args.root.resolve(), args.output.resolve())


if __name__ == '__main__':
    main()
