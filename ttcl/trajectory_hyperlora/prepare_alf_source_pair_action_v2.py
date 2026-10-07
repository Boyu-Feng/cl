"""Review two more source-pair action contrasts per variable-reward target.

Ranks one and two are fixed by earliest divergence, then source IDs. The
previous rank-zero cases and their intervention rewards do not select pairs.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from ttcl.trajectory_hyperlora.alfworld_online_from_empty import digest, file_hash
from ttcl.trajectory_hyperlora.prepare_alf_source_pair_action_v1 import expected as first_expected


def expected(args):
    first = json.loads(args.first_review.read_text())
    if first['rows'] != first_expected(args):
        raise ValueError('Changed rank-zero source-pair review')
    review = json.loads(args.review.read_text())
    report = json.loads(args.report.read_text())
    audit = json.loads(args.audit.read_text())
    if (report['review_sha256'] != file_hash(args.review) or
            audit['raw_report_sha256'] != file_hash(args.report) or
            audit['review_sha256'] != file_hash(args.review) or
            report['checkpoint_sha256'] != file_hash(args.checkpoint) or
            report['failures'] or len(review['sources']) != 8 or
            len(review['targets']) != 18 or len(report['pairs']) != 144):
        raise ValueError('Changed audited train source-reward matrix')
    by_target = defaultdict(dict)
    for observed, binding in zip(report['pairs'], review['pair_bindings'], strict=True):
        if (observed['input_content_sha256'] != binding['input_content_sha256'] or
                observed['episode']['status'] != 'complete'):
            raise ValueError('Changed source-target binding')
        by_target[observed['target_id']][observed['source_id']] = observed
    new_rows = []
    for first_row in first['rows']:
        target = review['targets'][first_row['target_id']]
        arms = by_target[target['target_id']]
        winners = [x for x in arms.values() if x['episode']['reward'] == 1.]
        losers = [x for x in arms.values() if x['episode']['reward'] == 0.]
        candidates = []
        for winner in winners:
            for loser in losers:
                good = winner['episode']['trajectory']
                bad = loser['episode']['trajectory']
                turn = next((j for j, (a, b) in enumerate(zip(good, bad))
                             if a['command'] != b['command']), None)
                if turn is None:
                    continue
                if (good[:turn] != bad[:turn] or
                        winner['episode']['initial_observation'] !=
                            loser['episode']['initial_observation']):
                    raise ValueError('Changed shared state before divergence')
                candidates.append((turn, winner['source_id'], loser['source_id'],
                                   winner, loser))
        candidates.sort(key=lambda x: x[:3])
        if (len(candidates) < 3 or
                candidates[0][:3] != (first_row['turn'],
                    first_row['winner_source_id'],
                    first_row['loser_source_id']) or
                file_hash(args.data_root / target['game']) !=
                    target['game_sha256']):
            raise ValueError('Changed source-pair ranking or target game')
        for rank in (1, 2):
            turn, win_id, lose_id, winner, loser = candidates[rank]
            good = winner['episode']['trajectory']
            bad = loser['episode']['trajectory']
            content = {'selection_rank': rank,
                'first_review_sha256': file_hash(args.first_review),
                'target_id': target['target_id'],
                'target_game_sha256': target['game_sha256'],
                'winner_source_id': win_id,
                'loser_source_id': lose_id,
                'winner_pair_binding': winner['input_content_sha256'],
                'loser_pair_binding': loser['input_content_sha256'],
                'winner_episode_sha256': digest(winner['episode']),
                'loser_episode_sha256': digest(loser['episode']),
                'shared_prefix_sha256': digest(good[:turn]),
                'turn': turn,
                'positive_action': good[turn]['command'],
                'negative_action': bad[turn]['command']}
            new_rows.append({'target_game': target['game'],
                'family': target['family'], **content,
                'input_content_sha256': digest(content)})
    return new_rows


def prepare(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    rows = expected(args)
    if len(rows) != 20:
        raise ValueError('Expected two fresh source-pair bindings per target')
    result = {'protocol': 'Additional twenty new content-bound train source-pair action candidates, fixed ranks one and two per variable-reward target; no prior intervention rewards used',
        'first_review_sha256': file_hash(args.first_review),
        'review_sha256': file_hash(args.review),
        'report_sha256': file_hash(args.report),
        'audit_sha256': file_hash(args.audit),
        'checkpoint_sha256': file_hash(args.checkpoint),
        'rows': rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({'reviewed_candidates': len(rows),
                      'turns': [x['turn'] for x in rows]}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--first-review', type=Path, default=Path('data/annotations/alf_source_pair_first_action10_reviewed_20261007.json'))
    parser.add_argument('--review', type=Path, default=Path('data/annotations/alf_own_success_future_reward_8x18_reviewed_20261007.json'))
    parser.add_argument('--report', type=Path, default=Path('results/trajectory_hyperlora/alf_own_success_future_reward_8x18_20261007.json'))
    parser.add_argument('--audit', type=Path, default=Path('results/trajectory_hyperlora/alf_own_success_future_reward_8x18_audited_20261007.json'))
    parser.add_argument('--checkpoint', type=Path, default=Path('results/trajectory_hyperlora/alf_sibling_simple40_onpolicy600_20261006.pt'))
    parser.add_argument('--data-root', type=Path, default=Path('ttcl/data/alfworld_delta'))
    parser.add_argument('--output', type=Path, default=Path('data/annotations/alf_source_pair_next20_reviewed_20261007.json'))
    prepare(parser.parse_args())
