"""Replay each source-conditioned forced-action continuation in ALFWorld."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ttcl.experience_evolution.environment import make_env
from ttcl.trajectory_hyperlora.alfworld_online_from_empty import digest, file_hash
from ttcl.trajectory_hyperlora.prepare_alf_source_pair_action_v1 import expected


def audit(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    candidates = json.loads(args.candidates.read_text())
    candidate_audit = json.loads(args.candidate_audit.read_text())
    report = json.loads(args.report.read_text())
    run = json.loads(args.intervention.read_text())
    if (candidates['rows'] != expected(args) or
            candidate_audit['candidates_sha256'] !=
                file_hash(args.candidates) or
            run['candidates_sha256'] != file_hash(args.candidates) or
            run['candidate_audit_sha256'] !=
                file_hash(args.candidate_audit) or
            run['matrix_sha256'] != file_hash(args.report) or
            run['checkpoint_sha256'] != file_hash(args.checkpoint) or
            run['failures'] or run['max_cases'] != 10 or
            len(run['rows']) != 10):
        raise ValueError('Changed or incomplete source-pair intervention')
    arms = {(x['target_id'], x['source_id']): x for x in report['pairs']}
    critical = []
    for original, observed in zip(candidates['rows'], run['rows'], strict=True):
        good = arms[original['target_id'],
                    original['winner_source_id']]['episode']
        bad = arms[original['target_id'],
                   original['loser_source_id']]['episode']
        forced = observed['forced_episode']
        if (observed['input_content_sha256'] !=
                original['input_content_sha256'] or
            observed['target_id'] != original['target_id'] or
            observed['winner_source_id'] !=
                original['winner_source_id'] or
            observed['loser_source_id'] != original['loser_source_id'] or
            observed['turn'] != original['turn'] or
            observed['reference_episode_sha256'] != digest(good) or
            observed['reference_reward'] != 1. or
            observed['loser_history_reward'] != bad['reward'] or
            bad['reward'] != 0. or
            observed['forced_reward'] != forced['reward'] or
            forced['status'] != 'complete' or
            forced['steps'] != len(forced['trajectory']) or
            forced['steps'] > 50 or
            forced['invalid_commands'] != 0 or
            forced['trajectory'][:original['turn']] !=
                good['trajectory'][:original['turn']] or
            forced['trajectory'][original['turn']]['command'] !=
                original['negative_action']):
            raise ValueError('Changed candidate binding or actor rollout')
        env = make_env(args.data_root / original['target_game'])
        try:
            state = env.reset()
            if str(state['feedback']) != forced['initial_observation']:
                raise ValueError('Changed game reset')
            for index, step in enumerate(forced['trajectory']):
                available = list(state['admissible_commands'])
                if (step['turn'] != index or
                        step['command'] not in available or
                        step['valid'] is not True):
                    raise ValueError('Invalid counterfactual action')
                state, _, done = env.step(step['command'])
                if (str(state['feedback']) != step['observation'] or
                        bool(state['won']) != step['won'] or
                        done and index != forced['steps']-1):
                    raise ValueError('Counterfactual transition mismatch')
            if float(bool(state['won'])) != forced['reward']:
                raise ValueError('Counterfactual reward mismatch')
        finally:
            env.close()
        if forced['reward'] == 0.:
            critical.append(original['input_content_sha256'])
    result = {'protocol': 'Independent full environment replay of all source-conditioned action interventions; labels remain local to frozen winning-history policy',
        'intervention_sha256': file_hash(args.intervention),
        'candidates_sha256': file_hash(args.candidates),
        'critical_bindings': critical,
        'summary': {'candidates': 10,
            'forced_action_loses': len(critical),
            'forced_action_still_wins': 10-len(critical),
            'failed_replays': 0}}
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
    parser.add_argument('--candidate-audit', type=Path, default=Path('results/trajectory_hyperlora/alf_source_pair_first_action10_audited_20261007.json'))
    parser.add_argument('--intervention', type=Path, default=Path('results/trajectory_hyperlora/alf_source_pair_action_intervention10_v1_20261007.json'))
    parser.add_argument('--output', type=Path, default=Path('results/trajectory_hyperlora/alf_source_pair_action_intervention10_v1_audited_20261007.json'))
    audit(parser.parse_args())
