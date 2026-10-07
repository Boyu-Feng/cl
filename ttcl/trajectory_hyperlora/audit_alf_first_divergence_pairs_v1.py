"""Recompute weak first-divergence labels from original and replayed episodes."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from ttcl.trajectory_hyperlora.alfworld_online_from_empty import file_hash
from ttcl.trajectory_hyperlora.prepare_alf_first_divergence_pairs_v1 import build


def audit(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    bank = json.loads(args.own_bank.read_text())
    original = json.loads(args.report.read_text())
    pairs = json.loads(args.pairs.read_text())
    expected = build(bank, original, file_hash(args.own_bank),
                     file_hash(args.report))
    if (pairs['own_bank_sha256'] != file_hash(args.own_bank) or
            pairs['report_sha256'] != file_hash(args.report) or
            pairs['pairs'] != expected or len(expected) != 42 or
            len({x['input_content_sha256'] for x in expected}) != 42):
        raise ValueError('Changed first-divergence action binding')
    summary = {'pairs': 42,
        'first_action': sum(x['turn'] == 0 for x in expected),
        'families': dict(sorted(Counter(x['family'] for x in expected).items())),
        'both_actions_admissible': 42,
        'causal_action_advantage_established': False,
        'checkpoint_ancestor_exposed': True}
    result = {'protocol': 'Independent recomputation of weak action-preference content bindings; not causal reward evidence',
        'pairs_review_sha256': file_hash(args.pairs),
        'own_bank_sha256': file_hash(args.own_bank),
        'report_sha256': file_hash(args.report), 'summary': summary}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--own-bank', type=Path, default=Path('data/annotations/alf_own_only42_replay_reviewed_20261007.json'))
    parser.add_argument('--report', type=Path, default=Path('results/trajectory_hyperlora/alf_sibling_train240_taskpair_current1000_20261006.json'))
    parser.add_argument('--pairs', type=Path, default=Path('data/annotations/alf_first_divergence42_reviewed_20261007.json'))
    parser.add_argument('--output', type=Path, default=Path('results/trajectory_hyperlora/alf_first_divergence42_audited_20261007.json'))
    audit(parser.parse_args())
