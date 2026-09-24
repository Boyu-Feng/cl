"""Analyze the fixed pilot; paired intervals resample entire task sequences."""
from pathlib import Path
import argparse
import json
import numpy as np

from .core import read, save, digest


def analyze(root):
    plan = read(root / 'plan.json')
    arms = plan['arms']
    rows = {}
    for arm in arms:
        assert read(root / 'evaluation' / arm / 'status.json')['phase'] == 'complete'
        values = [json.loads(s) for s in (root / 'evaluation' / arm / 'scores.jsonl').read_text().splitlines()]
        rows[arm] = {(v['sequence'], v['repeat'], v['position']): v for v in values}
        assert len(rows[arm]) == 72
    keys = sorted(rows['none'])
    for arm in arms:
        assert sorted(rows[arm]) == keys
        for key in keys:
            for field in ['game', 'seed', 'initial_sha256']:
                assert rows[arm][key][field] == rows['none'][key][field]
    sequences = sorted({k[0] for k in keys})
    rng = np.random.default_rng(20260922)
    draws = rng.integers(0, len(sequences), (20000, len(sequences)))
    summary = {}
    for arm in arms:
        primary = [r for k, r in rows[arm].items() if k[2] > 0]
        summary[arm] = {'successes': int(sum(r['reward'] for r in primary)), 'n': len(primary),
            'rate': float(np.mean([r['reward'] for r in primary])),
            'by_position': {str(p): float(np.mean([r['reward'] for k, r in rows[arm].items() if k[2] == p]))
                            for p in range(3)},
            'by_family': {f: float(np.mean([r['reward'] for r in primary if r['family'] == f]))
                          for f in sorted({r['family'] for r in primary})},
            'mean_steps': float(np.mean([r['steps'] for r in primary]))}
    comparisons = {}
    for other in ['none', 'untrained', 'absolute', 'delta_reset']:
        differences = {k: rows['delta'][k]['reward'] - rows[other][k]['reward'] for k in keys if k[2] > 0}
        clusters = np.array([np.mean([v for k, v in differences.items() if k[0] == s]) for s in sequences])
        ci = np.quantile(clusters[draws].mean(axis=1), [.025, .975])
        comparisons[other] = {'delta_minus_other': float(clusters.mean()), 'cluster_bootstrap_95_ci': ci.tolist(),
                             'wins': sum(v > 0 for v in differences.values()),
                             'losses': sum(v < 0 for v in differences.values()),
                             'ties': sum(v == 0 for v in differences.values())}
    training = {a: read(root / 'training' / a / 'status.json') for a in ['delta', 'absolute']}
    fusion_clusters = np.array([np.mean([rows['delta'][k]['reward'] - rows['delta_reset'][k]['reward']
                        for k in keys if k[0] == s and k[2] == 2]) for s in sequences])
    fusion = {'position': 2, 'delta_minus_reset': float(fusion_clusters.mean()),
              'cluster_bootstrap_95_ci': np.quantile(fusion_clusters[draws].mean(axis=1), [.025,.975]).tolist(),
              'note': 'Only task 3 can benefit from fusing an existing document with a new trajectory.'}
    assert all(v['phase'] == 'complete' for v in training.values())
    for game, expected in read(root / 'data_hashes.json').items():
        assert digest((Path(plan['data_root']) / game).read_bytes()) == expected
    first_mismatches = {a: sum(rows[a][k]['reward'] != rows['none'][k]['reward']
                       for k in keys if k[2] == 0) for a in arms}
    result = {'primary_positions': [1, 2], 'summary': summary, 'paired_comparisons': comparisons,
              'training': training, 'fusion_third_task': fusion, 'first_task_disagreements_against_none': first_mismatches,
              'data_hashes_unchanged': True, 'bootstrap_unit': 'sequence, preserving both seeds and positions',
              'independent_eval_sequences': len(sequences), 'training_seeds': 1,
              'caveat': 'Small fixed pilot; intervals do not include training-seed uncertainty.'}
    save(root / 'analysis.json', result)
    lines = ['# ALFWorld 连续经验融合实验', '',
        '主指标只统计第 2、3 题：12 个未见环境任务序列 × 2 次采样 × 2 题，每组 48 次。', '',
        '| 方法 | 成功数 | 成功率 | 第1题 | 第2题 | 第3题 |', '|---|---:|---:|---:|---:|---:|']
    for arm, v in summary.items():
        p = v['by_position']
        lines.append(f"| {arm} | {v['successes']}/{v['n']} | {v['rate']:.1%} | {p['0']:.1%} | {p['1']:.1%} | {p['2']:.1%} |")
    lines += ['', '差值训练相对于各对照的配对差异（按整个序列 bootstrap，20,000 次）：', '',
              '| 对照 | 成功率差 | 95%区间 | 胜/负/平 |', '|---|---:|---:|---:|']
    for arm, v in comparisons.items():
        lo, hi = v['cluster_bootstrap_95_ci']
        lines.append(f"| {arm} | {100*v['delta_minus_other']:+.1f} pp | [{100*lo:+.1f}, {100*hi:+.1f}] pp | {v['wins']}/{v['losses']}/{v['ties']} |")
    lines += ['', '训练使用同一个冻结解题模型和两个全新 LoRA；只更新经验生成器。',
              f"第3题融合效应：delta − delta_reset = {100*fusion['delta_minus_reset']:+.1f} pp；前两题尚不能区分是否融合旧文档。",
              '差值组使用 R(下一题，有更新经验) − R(同一道下一题，无经验)，不做组内优势居中。',
              'absolute 使用下一题原始成功奖励；untrained 是同一提示下的未训练总结器；delta_reset 每次清空旧经验。',
              '任务失败记 0，基础设施错误中止。只有有经验分支的轨迹进入下一次经验更新。', '',
              '这是一个训练种子、每链仅两次经验更新的小规模实验，不能据此证明长程持续进化。',
              'ALFWorld 检验交互规划经验迁移；结果不能直接代表数学推理任务。',
              '固定采样种子并不保证不同推理批次下逐 token 完全一致；首题差异计数见 analysis.json。', '',
              '训练信号：']
    for arm, v in training.items():
        lines.append(f"- {arm}: {v['writer_examples']} 条，正/负/零优势 = {v['positive']}/{v['negative']}/{v['zero']}。")
    (root / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    analyze(p.parse_args().root)
