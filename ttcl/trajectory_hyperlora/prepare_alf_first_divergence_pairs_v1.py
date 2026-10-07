"""Bind first shared-prefix action divergences to reviewed ALFWorld trajectories."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ttcl.trajectory_hyperlora.alfworld_online_from_empty import digest, file_hash


def build(own_bank: dict, original: dict, bank_hash: str, report_hash: str):
    games = {row['game']: row for row in original['games']}
    output = []
    for row in own_bank['rows']:
        game = games[row['game']]
        base = game['arms']['base']
        own = game['arms']['own']
        if (base['reward'] != 0 or own['reward'] != 1 or
                base['status'] != 'complete' or own['status'] != 'complete'):
            raise ValueError('Changed failure/success provenance')
        bt, ot = base['trajectory'], own['trajectory']
        turn = 0
        while (turn < min(len(bt), len(ot)) and
               bt[turn]['command'] == ot[turn]['command']):
            turn += 1
        if turn >= min(len(bt), len(ot)):
            raise ValueError('Missing first divergent action')
        reviewed = row['reviewed_targets'][turn]
        positive, negative = ot[turn]['command'], bt[turn]['command']
        if (reviewed['target_action'] != positive or
                positive == negative or
                positive not in reviewed['admissible_commands'] or
                negative not in reviewed['admissible_commands']):
            raise ValueError('Not a shared admissible branch point')
        content = {'bank_sha256': bank_hash, 'report_sha256': report_hash,
            'game_sha256': row['game_sha256'],
            'own_episode_sha256': row['episode_sha256'],
            'base_episode_sha256': digest(base),
            'target_action_input_sha256': reviewed['input_content_sha256'],
            'prefix_commands': [x['command'] for x in ot[:turn]],
            'turn': turn, 'observation': reviewed['observation'],
            'admissible_commands': reviewed['admissible_commands'],
            'positive_action': positive, 'negative_action': negative,
            'positive_reward': own['reward'], 'negative_reward': base['reward']}
        output.append({'game': row['game'], 'family': row['family'],
            **content, 'input_content_sha256': digest(content)})
    return output


def prepare(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    bank = json.loads(args.own_bank.read_text())
    original = json.loads(args.report.read_text())
    bank_audit = json.loads(args.bank_audit.read_text())
    if (bank_audit['review_sha256'] != file_hash(args.own_bank) or
            bank['source_arm'] != 'own' or
            len(bank['rows']) != 42 or bank['failures'] or
            len(original['failures']) != 0):
        raise ValueError('Changed source episode review')
    pairs = build(bank, original, file_hash(args.own_bank),
                  file_hash(args.report))
    result = {'protocol': 'Official train shared-prefix first action divergence between own-LoRA-success and paired-base-failure episodes; weak preference labels, not causal one-step advantages; target arm used expert-sibling source; checkpoint ancestor-exposed',
        'own_bank_sha256': file_hash(args.own_bank),
        'bank_audit_sha256': file_hash(args.bank_audit),
        'report_sha256': file_hash(args.report),
        'pairs': pairs}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({'pairs': len(pairs),
        'first_action': sum(x['turn'] == 0 for x in pairs)}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--own-bank', type=Path, default=Path('data/annotations/alf_own_only42_replay_reviewed_20261007.json'))
    parser.add_argument('--bank-audit', type=Path, default=Path('results/trajectory_hyperlora/alf_own_only42_replay_reviewed_audited_20261007.json'))
    parser.add_argument('--report', type=Path, default=Path('results/trajectory_hyperlora/alf_sibling_train240_taskpair_current1000_20261006.json'))
    parser.add_argument('--output', type=Path, default=Path('data/annotations/alf_first_divergence42_reviewed_20261007.json'))
    prepare(parser.parse_args())
