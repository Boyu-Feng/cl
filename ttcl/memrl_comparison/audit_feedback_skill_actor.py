"""Replay and verify the frozen ALFWorld feedback-skill comparison."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

from ttcl.alfworld_comparison.environment import make_env
from ttcl.experience_evolution.core import digest, seed
from ttcl.experience_evolution.environment import ACTOR_SYSTEM, clean_command
from ttcl.icl_mem0_comparison.protocol import read, save, sha
from .feedback_skill_actor import choose_action, learn, public_state


def audit_one(design: dict, case: dict, repeat: int, arm: str, path: Path) -> dict:
    episode = read(path)
    game = Path(read(Path(design['plan_path']))['alf']['data_root']) / case['game']
    if (sha(game) != case['input_sha256'] or episode['game'] != case['game']
            or episode['memory'] != '' or episode['memory_sha256'] != digest('')
            or episode['seed'] != seed(repeat, case['game'], 0, 'actor')
            or episode['steps'] != len(episode['trajectory']) == len(episode['generations'])
            or not 1 <= episode['steps'] <= 50 or episode['status'] != 'complete'):
        raise ValueError(f'Episode header mismatch: {path}')
    env = make_env(game)
    try:
        state = env.reset()
        if (str(state['feedback']) != episode['initial_observation']
                or hashlib.sha256(str(state['feedback']).encode()).hexdigest() != case['initial_observation_sha256']
                or digest(json.dumps(list(state['admissible_commands']))) != episode['initial_commands_sha256']):
            raise ValueError(f'Official reset mismatch: {path}')
        messages = [{'role': 'system', 'content': ACTOR_SYSTEM}]
        invalid = calls = learned_actions = 0
        table = design['skill_table'] if arm == 'learned_skill' else {'skills': {}}
        for turn, (row, generation) in enumerate(zip(episode['trajectory'], episode['generations'])):
            available = list(state['admissible_commands'])
            messages.append({'role': 'user', 'content': str(state['feedback']) +
                             '\nAvailable commands:\n' + '\n'.join(available)})
            if generation['seed'] != seed(episode['seed'], turn):
                raise ValueError(f'Actor seed mismatch: {path}:{turn}')
            tracker = public_state(case['public_task'], messages)
            command, reason = choose_action(tracker, available, table)
            if generation.get('public_state') != tracker or generation.get('skill_reason') != reason:
                raise ValueError(f'Public policy state mismatch: {path}:{turn}')
            if command is not None:
                if (generation['finish_reason'] != 'public_feedback_skill'
                        or generation['text'] != command or row['action'] != command
                        or generation['prompt_sha256'] != digest(json.dumps(messages, ensure_ascii=False))):
                    raise ValueError(f'Public policy action mismatch: {path}:{turn}')
            else:
                augmented = [dict(m) for m in messages]
                augmented[-1]['content'] += ('\n\nCurrent goal: ' + case['public_task'] +
                    '\nVisited locations: ' + (', '.join(tracker['visited'][-20:]) or 'none') +
                    '\nChoose a listed command that advances the unfinished goal. '
                    'Search a new location when possible; revisit one when carrying the goal object.')
                if generation['prompt_sha256'] != digest(json.dumps(augmented, ensure_ascii=False)):
                    raise ValueError(f'Actor prompt mismatch: {path}:{turn}')
                calls += 1
            learned_actions += reason.startswith('learned_')
            if row['action'] != clean_command(generation['text'], available):
                raise ValueError(f'Command cleaning mismatch: {path}:{turn}')
            valid = row['action'] in available
            if row['valid_command'] != valid:
                raise ValueError(f'Command validity mismatch: {path}:{turn}')
            invalid += not valid
            state, _, done = env.step(row['action'])
            if row['observation'] != str(state['feedback']):
                raise ValueError(f'Official feedback mismatch: {path}:{turn}')
            messages.append({'role': 'assistant', 'content': generation['text']})
            if (done or state['won']) and turn + 1 != episode['steps']:
                raise ValueError(f'Environment finished early: {path}:{turn}')
        if (episode['reward'] != float(bool(state['won']))
                or episode['termination'] != ('success' if state['won'] else 'budget_or_environment_done')
                or (not state['won'] and episode['steps'] < 50 and not done)):
            raise ValueError(f'Official reward mismatch: {path}')
        return dict(family=case['family'], game=case['game'], input_sha256=case['input_sha256'],
                    repeat=repeat, arm=arm, reward=episode['reward'], steps=episode['steps'],
                    invalid=invalid, actor_calls=calls, learned_actions=learned_actions,
                    episode_sha256=sha(path))
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    design_path = root / 'design.json'
    design = read(design_path)
    if (design['schema'] != 'alf_feedback_skill_pilot_v1'
            or sha(Path(design['plan_path'])) != design['plan_sha256']
            or sha(Path(design['review_path'])) != design['review_sha256']
            or sha(Path(design['cases_path'])) != design['cases_sha256']
            or design['arms'] != ['empty_skill', 'learned_skill']
            or design['skill_table'] != learn(read(Path(design['review_path'])),
                Path(read(Path(design['plan_path']))['alf']['data_root']))):
        raise ValueError('Frozen design mismatch')
    rows = []
    for case in design['cases']:
        for repeat in design['repeats']:
            for arm in design['arms']:
                path = root / case['family'] / case['input_sha256'][:12] / str(repeat) / arm / 'episode.json'
                if not path.exists():
                    raise ValueError(f'Missing planned episode: {path}')
                rows.append(audit_one(design, case, repeat, arm, path))
    pairs = defaultdict(dict)
    for row in rows:
        pairs[(row['input_sha256'], row['repeat'])][row['arm']] = row
    if len(pairs) != len(design['cases']) * len(design['repeats']) or any(set(v) != set(design['arms']) for v in pairs.values()):
        raise ValueError('Incomplete pairs')
    outcome = {'win': 0, 'loss': 0, 'tie': 0}
    for pair in pairs.values():
        delta = pair['learned_skill']['reward'] - pair['empty_skill']['reward']
        outcome['win' if delta > 0 else 'loss' if delta < 0 else 'tie'] += 1
    totals = {arm: {metric: sum(row[metric] for row in rows if row['arm'] == arm)
                    for metric in ('reward', 'steps', 'invalid', 'actor_calls', 'learned_actions')}
              for arm in design['arms']}
    report = dict(schema='alf_feedback_skill_audit_v1', design_sha256=sha(design_path),
                  auditor_sha256=sha(Path(__file__)), rows=rows,
                  paired_outcomes=outcome, totals=totals,
                  caveat='Small official-train development pilot; no valid_unseen or CLBench confirmation')
    save(args.output, report)
    print(json.dumps({'paired_outcomes': outcome, 'totals': totals}), flush=True)


if __name__ == '__main__':
    main()
