"""Replay the frozen small valid_unseen state-progress ALFWorld evaluation."""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import json

from ttcl.icl_mem0_comparison.protocol import read, save, sha
from . import probe_state_progress_actor as pilot
from .probe_state_progress_holdout import extended_goal
from .audit_state_progress_actor import audit_one


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    design_path = root / 'design.json'
    design = read(design_path)
    if (design['schema'] != 'alf_state_progress_unseen_v1' or
            design['split'] != 'valid_unseen' or len(design['cases']) != 12 or
            design['repeats'] != [94601, 94602] or
            design['arms'] != ['native', 'state_progress'] or
            sha(Path(design['plan_path'])) != design['plan_sha256'] or
            sha(Path(design['review_path'])) != design['review_sha256'] or
            sha(Path(pilot.__file__)) != design['policy_sha256'] or
            sha(Path(__file__).with_name('probe_state_progress_unseen.py')) !=
            design['runner_sha256']):
        raise ValueError('Frozen valid_unseen design or source changed')
    pilot._goal = extended_goal
    rows = []
    paired = defaultdict(dict)
    for case in design['cases']:
        for repeat in design['repeats']:
            for arm in design['arms']:
                path = root / case['family'] / str(case['rank']) / str(repeat) / arm / 'episode.json'
                if not path.exists():
                    raise ValueError(f'Missing planned episode: {path}')
                audit_arm = 'native' if arm == 'state_progress' and not case['supported'] else arm
                row = audit_one(design, case, repeat, audit_arm, path)
                row.update(arm=arm, rank=case['rank'], game=case['game'],
                           supported=case['supported'])
                rows.append(row)
                paired[(case['game'],repeat)][arm] = row
    if len(rows) != 48 or len(paired) != 24:
        raise ValueError('Incomplete paired test subset')
    result = {'win':0,'loss':0,'tie':0}
    by_family = defaultdict(lambda:{'win':0,'loss':0,'tie':0})
    for (game, repeat), pair in paired.items():
        if set(pair) != {'native','state_progress'}:
            raise ValueError('Unpaired result')
        delta = pair['state_progress']['reward']-pair['native']['reward']
        outcome = 'win' if delta > 0 else 'loss' if delta < 0 else 'tie'
        result[outcome] += 1
        by_family[pair['native']['family']][outcome] += 1
    report = dict(schema='alf_state_progress_unseen_audit_v1',
                  design_sha256=sha(design_path), auditor_sha256=sha(Path(__file__)),
                  rows=rows, paired_outcomes=result, by_family=dict(by_family),
                  supported_games=sum(c['supported'] for c in design['cases']),
                  totals={arm:dict(reward=sum(r['reward'] for r in rows if r['arm']==arm),
                                   steps=sum(r['steps'] for r in rows if r['arm']==arm),
                                   invalid=sum(r['invalid'] for r in rows if r['arm']==arm),
                                   actor_calls=sum(r['actor_calls'] for r in rows if r['arm']==arm),
                                   rule_actions=sum(r['rule_actions'] for r in rows if r['arm']==arm))
                          for arm in design['arms']},
                  caveat='Small 12-game valid_unseen subset, two seeds; unsupported syntax fallback included; cannot substitute for all 134 official games or a full MemRL online chain')
    save(args.output, report)
    print(json.dumps({k:report[k] for k in ('paired_outcomes','by_family','supported_games','totals')},
                     ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
