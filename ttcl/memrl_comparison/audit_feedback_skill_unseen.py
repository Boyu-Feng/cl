"""Replay the independent valid_unseen feedback-skill comparison."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

from ttcl.icl_mem0_comparison.protocol import read, save, sha
from .audit_feedback_skill_actor import audit_one
from .feedback_skill_actor import learn


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    design_path = root / 'design.json'
    design = read(design_path)
    plan = read(Path(design['plan_path']))
    previous = read(Path(design['previous_unseen_design_path']))
    if (design['schema'] != 'alf_feedback_skill_unseen_v1'
            or sha(Path(design['plan_path'])) != design['plan_sha256']
            or sha(Path(design['review_path'])) != design['review_sha256']
            or sha(Path(design['previous_unseen_design_path'])) != design['previous_unseen_design_sha256']
            or design['skill_table'] != learn(read(Path(design['review_path'])), Path(plan['alf']['data_root']))
            or len(design['cases']) != 6 or len(design['repeats']) != 2
            or design['arms'] != ['empty_skill', 'learned_skill']):
        raise ValueError('Frozen unseen design mismatch')
    excluded = {row['input_sha256'] for row in previous['cases']}
    excluded.update(row['input_sha256'] for row in read(Path(design['review_path']))['targets'])
    rows = []
    for case in design['cases']:
        if (not case['game'].startswith('json_2.1.1/valid_unseen/')
                or case['input_sha256'] in excluded or not case['reviewed']):
            raise ValueError('Unseen evaluation input leakage or review mismatch')
        for repeat in design['repeats']:
            for arm in design['arms']:
                path = root / case['family'] / case['input_sha256'][:12] / str(repeat) / arm / 'episode.json'
                if not path.exists():
                    raise ValueError(f'Missing unseen episode: {path}')
                rows.append(audit_one(design, case, repeat, arm, path))
    pairs = defaultdict(dict)
    for row in rows:
        pairs[(row['input_sha256'], row['repeat'])][row['arm']] = row
    if len(pairs) != 12 or any(set(v) != set(design['arms']) for v in pairs.values()):
        raise ValueError('Incomplete unseen pairs')
    outcome = {'win': 0, 'loss': 0, 'tie': 0}
    for pair in pairs.values():
        delta = pair['learned_skill']['reward'] - pair['empty_skill']['reward']
        outcome['win' if delta > 0 else 'loss' if delta < 0 else 'tie'] += 1
    totals = {arm: {metric: sum(row[metric] for row in rows if row['arm'] == arm)
                    for metric in ('reward', 'steps', 'invalid', 'actor_calls', 'learned_actions')}
              for arm in design['arms']}
    report = dict(schema='alf_feedback_skill_unseen_audit_v1', design_sha256=sha(design_path),
                  auditor_sha256=sha(Path(__file__)), rows=rows,
                  paired_outcomes=outcome, totals=totals,
                  caveat='Small independent valid_unseen subset, not all 134 games or CLBench')
    save(args.output, report)
    print(json.dumps({'paired_outcomes': outcome, 'totals': totals}), flush=True)


if __name__ == '__main__':
    main()
