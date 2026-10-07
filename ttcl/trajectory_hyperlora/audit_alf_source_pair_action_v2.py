"""Review all twenty additional source-pair shared decision states."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from ttcl.experience_evolution.environment import make_env
from ttcl.trajectory_hyperlora.alfworld_online_from_empty import file_hash
from ttcl.trajectory_hyperlora.prepare_alf_source_pair_action_v2 import expected


def audit(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    reviewed = json.loads(args.candidates.read_text())
    rows = expected(args)
    report = json.loads(args.report.read_text())
    if (reviewed['rows'] != rows or len(rows) != 20 or
            reviewed['first_review_sha256'] != file_hash(args.first_review) or
            reviewed['report_sha256'] != file_hash(args.report) or
            reviewed['checkpoint_sha256'] != file_hash(args.checkpoint)):
        raise ValueError('Changed second source-pair review')
    arms = {(x['target_id'], x['source_id']): x for x in report['pairs']}
    checks = []
    for row in rows:
        winner = arms[row['target_id'], row['winner_source_id']]['episode']
        loser = arms[row['target_id'], row['loser_source_id']]['episode']
        env = make_env(args.data_root / row['target_game'])
        try:
            state = env.reset()
            if str(state['feedback']) != winner['initial_observation']:
                raise ValueError('Changed ALFWorld reset')
            for step in winner['trajectory'][:row['turn']]:
                if step['command'] not in state['admissible_commands']:
                    raise ValueError('Changed common-prefix admissibility')
                state, _, done = env.step(step['command'])
                if (str(state['feedback']) != step['observation'] or
                        bool(state['won']) != step['won'] or done):
                    raise ValueError('Changed common-prefix transition')
            admissible = list(state['admissible_commands'])
            if (row['positive_action'] not in admissible or
                    row['negative_action'] not in admissible or
                    row['positive_action'] == row['negative_action'] or
                    winner['reward'] != 1. or loser['reward'] != 0.):
                raise ValueError('Invalid source-pair actions')
            checks.append({'input_content_sha256': row['input_content_sha256'],
                'admissible_commands_sha256': hashlib.sha256(
                    json.dumps(admissible).encode()).hexdigest()})
        finally:
            env.close()
    result = {'protocol': 'Independent common-prefix original-environment replay and action admissibility for new source-pair contrasts',
        'candidates_sha256': file_hash(args.candidates),
        'report_sha256': file_hash(args.report),
        'summary': {'verified_candidates': len(checks),
                    'failed_replays': 0}, 'rows': checks}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(result['summary']), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--first-review', type=Path, default=Path('data/annotations/alf_source_pair_first_action10_reviewed_20261007.json'))
    parser.add_argument('--review', type=Path, default=Path('data/annotations/alf_own_success_future_reward_8x18_reviewed_20261007.json'))
    parser.add_argument('--report', type=Path, default=Path('results/trajectory_hyperlora/alf_own_success_future_reward_8x18_20261007.json'))
    parser.add_argument('--audit', type=Path, default=Path('results/trajectory_hyperlora/alf_own_success_future_reward_8x18_audited_20261007.json'))
    parser.add_argument('--checkpoint', type=Path, default=Path('results/trajectory_hyperlora/alf_sibling_simple40_onpolicy600_20261006.pt'))
    parser.add_argument('--data-root', type=Path, default=Path('ttcl/data/alfworld_delta'))
    parser.add_argument('--candidates', type=Path, default=Path('data/annotations/alf_source_pair_next20_reviewed_20261007.json'))
    parser.add_argument('--output', type=Path, default=Path('results/trajectory_hyperlora/alf_source_pair_next20_audited_20261007.json'))
    audit(parser.parse_args())
