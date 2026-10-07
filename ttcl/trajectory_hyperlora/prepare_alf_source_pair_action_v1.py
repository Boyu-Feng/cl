"""Review source-dependent first-action contrasts from audited own-history rewards.

The reward matrix is training data. A successful and failed history are
selected deterministically per varying target; the action contrast is a
candidate until a fixed-policy intervention is independently replayed.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from ttcl.trajectory_hyperlora.alfworld_online_from_empty import digest, file_hash


def expected(args):
    review = json.loads(args.review.read_text())
    report = json.loads(args.report.read_text())
    audit = json.loads(args.audit.read_text())
    if (report['review_sha256'] != file_hash(args.review) or
            audit['review_sha256'] != file_hash(args.review) or
            audit['raw_report_sha256'] != file_hash(args.report) or
            audit['checkpoint_sha256'] != file_hash(args.checkpoint) or
            report['checkpoint_sha256'] != file_hash(args.checkpoint) or
            report['failures'] or len(review['sources']) != 8 or
            len(review['targets']) != 18 or len(report['pairs']) != 144 or
            len(review['pair_bindings']) != 144 or
            report['max_steps'] != review['max_steps'] or
            report['max_new_tokens'] != review['max_new_tokens'] or
            report['actor_history_turns'] != review['actor_history_turns'] or
            report['loop_guard_max'] != review['loop_guard_max']):
        raise ValueError('Changed or incomplete audited source-reward matrix')
    by_target = defaultdict(dict)
    for observed, binding in zip(report['pairs'], review['pair_bindings'], strict=True):
        if (observed['source_id'] != binding['source_id'] or
                observed['target_id'] != binding['target_id'] or
                observed['input_content_sha256'] != binding['input_content_sha256'] or
                observed['episode']['status'] != 'complete'):
            raise ValueError('Changed source-target content binding')
        by_target[observed['target_id']][observed['source_id']] = observed
    rows = []
    for target in review['targets']:
        target_id = target['target_id']
        arms = by_target[target_id]
        if len(arms) != 8:
            raise ValueError('Incomplete source arms')
        winners = [x for x in arms.values() if x['episode']['reward'] == 1.]
        losers = [x for x in arms.values() if x['episode']['reward'] == 0.]
        if not winners or not losers:
            continue
        candidates = []
        for winner in winners:
            for loser in losers:
                good = winner['episode']['trajectory']
                bad = loser['episode']['trajectory']
                if (winner['episode']['initial_observation'] !=
                        loser['episode']['initial_observation']):
                    raise ValueError('Source arms have different target reset')
                turn = next((j for j, (a, b) in enumerate(zip(good, bad))
                             if a['command'] != b['command']), None)
                if turn is None:
                    continue
                candidates.append((turn, winner['source_id'], loser['source_id'],
                                   winner, loser))
        if not candidates:
            raise ValueError('Reward varies without an action difference')
        turn, win_id, lose_id, winner, loser = min(candidates,
                                                   key=lambda x: x[:3])
        good = winner['episode']['trajectory']
        bad = loser['episode']['trajectory']
        if (good[:turn] != bad[:turn] or
                file_hash(args.data_root / target['game']) !=
                    target['game_sha256']):
            raise ValueError('Changed prefix or game content')
        content = {'target_id': target_id, 'target_game_sha256':
            target['game_sha256'], 'winner_source_id': win_id,
            'loser_source_id': lose_id,
            'winner_pair_binding': winner['input_content_sha256'],
            'loser_pair_binding': loser['input_content_sha256'],
            'winner_episode_sha256': digest(winner['episode']),
            'loser_episode_sha256': digest(loser['episode']),
            'shared_prefix_sha256': digest(good[:turn]),
            'turn': turn, 'positive_action': good[turn]['command'],
            'negative_action': bad[turn]['command']}
        rows.append({'target_game': target['game'],
            'family': target['family'], **content,
            'input_content_sha256': digest(content)})
    return rows


def prepare(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    rows = expected(args)
    result = {'protocol': 'New reviewed official-train source-dependent action contrasts: earliest first divergence per variable-reward target, deterministic tie-break by source IDs; candidates only until fixed-winning-policy intervention',
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
    parser.add_argument('--review', type=Path, default=Path('data/annotations/alf_own_success_future_reward_8x18_reviewed_20261007.json'))
    parser.add_argument('--report', type=Path, default=Path('results/trajectory_hyperlora/alf_own_success_future_reward_8x18_20261007.json'))
    parser.add_argument('--audit', type=Path, default=Path('results/trajectory_hyperlora/alf_own_success_future_reward_8x18_audited_20261007.json'))
    parser.add_argument('--checkpoint', type=Path, default=Path('results/trajectory_hyperlora/alf_sibling_simple40_onpolicy600_20261006.pt'))
    parser.add_argument('--data-root', type=Path, default=Path('ttcl/data/alfworld_delta'))
    parser.add_argument('--output', type=Path, default=Path('data/annotations/alf_source_pair_first_action10_reviewed_20261007.json'))
    prepare(parser.parse_args())
