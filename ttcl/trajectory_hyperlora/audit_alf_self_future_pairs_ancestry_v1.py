"""Expose ancestor-checkpoint target overlap of proposed self-history pairs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ttcl.trajectory_hyperlora.alfworld_online_from_empty import file_hash


def audit(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    plan = json.loads(args.pairs.read_text())
    direct = json.loads(args.direct_labels.read_text())
    ancestor = json.loads(args.ancestor_candidates.read_text())
    direct_games = {row['target_game'] for row in direct['tasks']}
    ancestor_games = {row['target_game'] for row in ancestor['rows']}
    train, dev = set(plan['train_games']), set(plan['dev_games'])
    if (not ancestor_games or train & dev or
            plan['checkpoint_labels_sha256'] !=
                file_hash(args.direct_labels)):
        raise ValueError('Changed lineage inputs')
    summary = {'train_games': len(train), 'dev_games': len(dev),
        'direct_label_overlap_train': len(train & direct_games),
        'direct_label_overlap_dev': len(dev & direct_games),
        'ancestor_candidate_overlap_train': len(train & ancestor_games),
        'ancestor_candidate_overlap_dev': len(dev & ancestor_games),
        'independent_of_complete_checkpoint':
            not bool((train | dev) & ancestor_games)}
    result = {'protocol': 'Read-only explicit ancestor candidate overlap check; any overlapped internal dev outcomes are not an independent test of the existing checkpoint',
        'pairs_review_sha256': file_hash(args.pairs),
        'direct_labels_sha256': file_hash(args.direct_labels),
        'ancestor_candidates_sha256': file_hash(args.ancestor_candidates),
        'summary': summary}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pairs', type=Path, default=Path(
        'data/annotations/alf_self_future_pairs_v2_checkpoint_disjoint_reviewed_20261007.json'))
    parser.add_argument('--direct-labels', type=Path, default=Path(
        'results/trajectory_hyperlora/alf_sibling_simple_train40_onpolicy_best_labels_20261006.json'))
    parser.add_argument('--ancestor-candidates', type=Path, default=Path(
        'results/trajectory_hyperlora/alf_sibling_train600_candidates_20261006.json'))
    parser.add_argument('--output', type=Path, default=Path(
        'results/trajectory_hyperlora/alf_self_future_pairs_v2_ancestry_audited_20261007.json'))
    audit(parser.parse_args())
