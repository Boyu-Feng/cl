"""Choose one common residual scale using audited exposed train-domain tasks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ttcl.trajectory_hyperlora.alfworld_online_from_empty import file_hash


def select(root: Path, output: Path):
    if output.exists():
        raise FileExistsError(output)
    candidates = []
    for scale in (1, 2, 4):
        rows = []
        for seed in (42, 43, 44):
            suffix = '_72' if scale == 1 else f'_scale{scale}'
            path = root / (f'alf_feedback_residual_dev12_balanced_seed{seed}'
                           f'{suffix}_audited_20261007.json')
            audit = json.loads(path.read_text())
            summary = audit['summary']
            if (summary['targets'] != 12 or summary['source_id'] != 3 or
                    summary['residual_scale'] != scale or
                    summary['old'] != 4 or
                    audit['protocol'].find('train-domain') < 0):
                raise ValueError(f'Unpaired or changed development audit {path}')
            rows.append({'seed': seed, 'score': summary['trained'],
                         'old_score': summary['old'],
                         'audit_sha256': file_hash(path)})
        candidates.append({'scale': scale,
                           'total': sum(row['score'] for row in rows),
                           'rows': rows})
    chosen = min(candidates, key=lambda row: (-row['total'], row['scale']))
    report = {'protocol': 'Common three-seed scale chosen only from previously exposed official train-domain dev12; ties choose smaller scale; no validation outcomes used',
              'candidates': candidates, 'chosen_scale': chosen['scale']}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path,
                        default=Path('results/trajectory_hyperlora'))
    parser.add_argument('--output', type=Path,
                        default=Path('results/trajectory_hyperlora/alf_feedback_residual_v2_scale_selected_20261007.json'))
    args = parser.parse_args()
    select(args.root, args.output)
