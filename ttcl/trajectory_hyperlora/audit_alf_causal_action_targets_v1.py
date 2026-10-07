"""Rebuild every policy-conditional action target and check its binding."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from ttcl.trajectory_hyperlora.alfworld_online_from_empty import file_hash
from ttcl.trajectory_hyperlora.prepare_alf_causal_action_targets_v1 import expected


def audit(args):
    if args.audit_output.exists():
        raise FileExistsError(args.audit_output)
    data = json.loads(args.output.read_text())
    rows = expected(args)
    if (data['rows'] != rows or len(rows) != 42 or
            len({x['input_content_sha256'] for x in rows}) != 42 or
            data['intervention30_sha256'] != file_hash(args.intervention30) or
            data['intervention50_sha256'] != file_hash(args.intervention50)):
        raise ValueError('Changed causal action-target review')
    by_family = Counter(x['family'] for x in rows
                        if x['policy_advantage50'] == 1.)
    summary = {'reviewed_targets': 42,
        'positive_at_30': sum(x['policy_advantage30'] for x in rows),
        'positive_at_50': sum(x['policy_advantage50'] for x in rows),
        'positive_at_50_by_family': dict(sorted(by_family.items())),
        'policy_conditional_only': True}
    result = {'protocol': 'Independent recomputation of content-bound, policy-conditional action advantages',
        'targets_review_sha256': file_hash(args.output),
        'intervention30_sha256': file_hash(args.intervention30),
        'intervention50_sha256': file_hash(args.intervention50),
        'summary': summary}
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.write_text(json.dumps(result, ensure_ascii=False,
                                        indent=2)+'\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--first-divergence-review', type=Path, default=Path('data/annotations/alf_first_divergence42_reviewed_20261007.json'))
    parser.add_argument('--intervention30', type=Path, default=Path('results/trajectory_hyperlora/alf_own_only_first_divergence42_intervention_v2_20261007.json'))
    parser.add_argument('--audit30', type=Path, default=Path('results/trajectory_hyperlora/alf_own_only_first_divergence42_intervention_v2_audited_20261007.json'))
    parser.add_argument('--intervention50', type=Path, default=Path('results/trajectory_hyperlora/alf_own_only_first_divergence_budget50_v3_20261007.json'))
    parser.add_argument('--audit50', type=Path, default=Path('results/trajectory_hyperlora/alf_own_only_first_divergence_budget50_v3_audited_20261007.json'))
    parser.add_argument('--output', type=Path, default=Path('data/annotations/alf_own_only_causal_action42_reviewed_20261007.json'))
    parser.add_argument('--audit-output', type=Path, default=Path('results/trajectory_hyperlora/alf_own_only_causal_action42_audited_20261007.json'))
    audit(parser.parse_args())
