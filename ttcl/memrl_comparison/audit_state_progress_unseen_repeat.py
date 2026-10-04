"""Official replay and stability accounting for selected unseen repeat arms."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

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
    if (design['schema'] != 'alf_state_progress_unseen_repeat_v1' or
            len(design['cases']) != 7 or
            sha(Path(design['source_plan'])) != design['source_plan_sha256'] or
            sha(Path(design['review_path'])) != design['review_sha256'] or
            sha(Path(pilot.__file__)) != design['policy_sha256'] or
            sha(Path(__file__).with_name('probe_state_progress_unseen_repeat.py')) !=
            design['runner_sha256']):
        raise ValueError('Frozen repeat source changed')
    pilot._goal = extended_goal
    rows = []
    original_wins_repeated = repeat_candidate_success = repeat_native_success = 0
    for case in design['cases']:
        arm_rows = {}
        for arm in design['arms']:
            path = root / case['family'] / str(case['rank']) / str(case['repeat']) / arm / 'episode.json'
            audited = audit_one(dict(plan_path=design['source_plan']), case, case['repeat'],
                                'state_progress' if arm == 'state_progress_repeat' else 'native', path)
            audited['arm'] = arm
            audited['game'] = case['game']
            rows.append(audited)
            arm_rows[arm] = audited
        c = arm_rows['state_progress_repeat']['reward']
        n = arm_rows['native_repeat']['reward']
        repeat_candidate_success += c
        repeat_native_success += n
        original_wins_repeated += c > n
    report = dict(schema='alf_state_progress_unseen_repeat_audit_v1',
                  design_sha256=sha(design_path), auditor_sha256=sha(Path(__file__)),
                  rows=rows, original_positive_pairs=7,
                  repeat_candidate_success=repeat_candidate_success,
                  repeat_native_success=repeat_native_success,
                  repeat_still_wins=original_wins_repeated,
                  caveat='Posthoc repeat controls on seven selected original wins; not a new unbiased estimate over all 24 original pairs')
    save(args.output, report)
    print(json.dumps({k:report[k] for k in ('original_positive_pairs',
                                           'repeat_candidate_success',
                                           'repeat_native_success',
                                           'repeat_still_wins')}), flush=True)


if __name__ == '__main__':
    main()
