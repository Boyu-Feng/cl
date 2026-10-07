"""Independently verify source-pair action candidates in original ALFWorld games."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from ttcl.experience_evolution.environment import make_env
from ttcl.trajectory_hyperlora.alfworld_online_from_empty import file_hash
from ttcl.trajectory_hyperlora.prepare_alf_source_pair_action_v1 import expected


def audit(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    review = json.loads(args.review.read_text())
    candidate = json.loads(args.candidates.read_text())
    if (candidate['rows'] != expected(args) or
            candidate['review_sha256'] != file_hash(args.review) or
            candidate['report_sha256'] != file_hash(args.report) or
            candidate['audit_sha256'] != file_hash(args.audit) or
            candidate['checkpoint_sha256'] != file_hash(args.checkpoint)):
        raise ValueError('Changed candidate review or lineage')
    report = json.loads(args.report.read_text())
    pair = {(x['target_id'], x['source_id']): x for x in report['pairs']}
    results = []
    for row in candidate['rows']:
        win = pair[row['target_id'], row['winner_source_id']]['episode']
        lose = pair[row['target_id'], row['loser_source_id']]['episode']
        game = args.data_root / row['target_game']
        env = make_env(game)
        try:
            state = env.reset()
            if str(state['feedback']) != win['initial_observation']:
                raise ValueError('Changed game reset')
            for step in win['trajectory'][:row['turn']]:
                if step['command'] not in state['admissible_commands']:
                    raise ValueError('Reviewed common-prefix action inadmissible')
                state, _, done = env.step(step['command'])
                if (str(state['feedback']) != step['observation'] or
                        bool(state['won']) != step['won'] or done):
                    raise ValueError('Common prefix no longer replays')
            available = list(state['admissible_commands'])
            if (row['positive_action'] not in available or
                    row['negative_action'] not in available or
                    row['positive_action'] == row['negative_action'] or
                    win['reward'] != 1. or lose['reward'] != 0.):
                raise ValueError('Invalid paired action or terminal label')
            results.append({'input_content_sha256': row['input_content_sha256'],
                'target_id': row['target_id'], 'turn': row['turn'],
                'admissible_commands_sha256':
                    hashlib.sha256(json.dumps(available).encode()).hexdigest()})
        finally:
            env.close()
    result = {'protocol': 'Independent target-game reset and shared-prefix replay for source-dependent action candidates; terminal source rewards already audited, no action effect yet inferred',
        'candidates_sha256': file_hash(args.candidates),
        'report_sha256': file_hash(args.report),
        'summary': {'verified_candidates': len(results),
                    'failed_replays': 0}, 'rows': results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(result['summary']), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--review', type=Path, default=Path('data/annotations/alf_own_success_future_reward_8x18_reviewed_20261007.json'))
    parser.add_argument('--report', type=Path, default=Path('results/trajectory_hyperlora/alf_own_success_future_reward_8x18_20261007.json'))
    parser.add_argument('--audit', type=Path, default=Path('results/trajectory_hyperlora/alf_own_success_future_reward_8x18_audited_20261007.json'))
    parser.add_argument('--checkpoint', type=Path, default=Path('results/trajectory_hyperlora/alf_sibling_simple40_onpolicy600_20261006.pt'))
    parser.add_argument('--data-root', type=Path, default=Path('ttcl/data/alfworld_delta'))
    parser.add_argument('--candidates', type=Path, default=Path('data/annotations/alf_source_pair_first_action10_reviewed_20261007.json'))
    parser.add_argument('--output', type=Path, default=Path('results/trajectory_hyperlora/alf_source_pair_first_action10_audited_20261007.json'))
    audit(parser.parse_args())
