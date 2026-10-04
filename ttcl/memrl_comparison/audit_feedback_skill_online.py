"""Replay every episode and verify causal ordering of online skill updates."""
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
from .probe_feedback_skill_online import ARMS, REPEATS, SkillStore


def audit_episode(design: dict, case: dict, repeat: int, before: dict, path: Path) -> dict:
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
        calls = invalid = skill_actions = 0
        for turn, (row, generation) in enumerate(zip(episode['trajectory'], episode['generations'])):
            available = list(state['admissible_commands'])
            messages.append({'role': 'user', 'content': str(state['feedback']) +
                             '\nAvailable commands:\n' + '\n'.join(available)})
            if generation['seed'] != seed(episode['seed'], turn):
                raise ValueError(f'Actor seed mismatch: {path}:{turn}')
            tracker = public_state(case['public_task'], messages)
            command, reason = choose_action(tracker, available, before)
            if generation.get('public_state') != tracker or generation.get('skill_reason') != reason:
                raise ValueError(f'Public policy state mismatch: {path}:{turn}')
            if command is not None:
                if (generation['finish_reason'] != 'public_feedback_skill'
                        or generation['text'] != command or row['action'] != command
                        or generation['prompt_sha256'] != digest(json.dumps(messages, ensure_ascii=False))):
                    raise ValueError(f'Public rule mismatch: {path}:{turn}')
            else:
                augmented = [dict(m) for m in messages]
                augmented[-1]['content'] += ('\n\nCurrent goal: ' + case['public_task'] +
                    '\nVisited locations: ' + (', '.join(tracker['visited'][-20:]) or 'none') +
                    '\nChoose a listed command that advances the unfinished goal. '
                    'Search a new location when possible; revisit one when carrying the goal object.')
                if generation['prompt_sha256'] != digest(json.dumps(augmented, ensure_ascii=False)):
                    raise ValueError(f'Actor prompt mismatch: {path}:{turn}')
                calls += 1
            skill_actions += reason.startswith('learned_')
            if row['action'] != clean_command(generation['text'], available):
                raise ValueError(f'Command cleaning mismatch: {path}:{turn}')
            valid = row['action'] in available
            if row['valid_command'] != valid:
                raise ValueError(f'Validity mismatch: {path}:{turn}')
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
        return dict(reward=episode['reward'], steps=episode['steps'],
                    invalid=invalid, actor_calls=calls, skill_actions=skill_actions,
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
    review = read(Path(design['review_path']))
    data_root = Path(read(Path(design['plan_path']))['alf']['data_root'])
    if (design['schema'] != 'alf_feedback_skill_online_v1'
            or sha(Path(design['plan_path'])) != design['plan_sha256']
            or sha(Path(design['review_path'])) != design['review_sha256']
            or design['initial_trained_skill_table'] != learn(review, data_root)
            or design['arms'] != list(ARMS) or design['repeats'] != list(REPEATS)
            or len(design['cases']) != 9 or any(sha(Path(row['path'])) != row['sha256']
                                                 for row in design['exclude_designs'])):
        raise ValueError('Frozen online design mismatch')
    excluded = {target['input_sha256'] for target in review['targets']}
    for row in design['exclude_designs']:
        excluded.update(case['input_sha256'] for case in read(Path(row['path']))['cases'])
    rows = []
    for repeat in REPEATS:
        stores = {arm: SkillStore(review if arm.startswith('trained_') else None) for arm in ARMS}
        for index, case in enumerate(design['cases']):
            if (not case['game'].startswith('json_2.1.1/valid_unseen/')
                    or not case['reviewed'] or case['input_sha256'] in excluded):
                raise ValueError('Evaluation input review or split mismatch')
            for arm in ARMS:
                target = root / str(repeat) / f'{index:02d}_{case["input_sha256"][:12]}' / arm
                before = stores[arm].snapshot()
                if read(target / 'skill_before.json') != before:
                    raise ValueError('Skill table used future experience')
                details = audit_episode(design, case, repeat, before, target / 'episode.json')
                episode = read(target / 'episode.json')
                if arm.endswith('_online'):
                    annotation = stores[arm].observe(case['input_sha256'], episode)
                    if read(target / 'reviewed_update.json') != annotation:
                        raise ValueError('Feedback annotation mismatch')
                if read(target / 'skill_after.json') != stores[arm].snapshot():
                    raise ValueError('Online update order mismatch')
                rows.append(dict(index=index, family=case['family'],
                    input_sha256=case['input_sha256'], repeat=repeat, arm=arm,
                    skill_types_before={verb: [skill['tool'] for skill in skills]
                        for verb, skills in before['skills'].items()},
                    skill_types_after={verb: [skill['tool'] for skill in skills]
                        for verb, skills in stores[arm].snapshot()['skills'].items()},
                    **details))
    pairs = defaultdict(dict)
    for row in rows:
        pairs[(row['input_sha256'], row['repeat'])][row['arm']] = row
    if len(pairs) != 18 or any(set(pair) != set(ARMS) for pair in pairs.values()):
        raise ValueError('Unpaired online arms')
    comparisons = {}
    for frozen, online in [('empty_frozen', 'empty_online'), ('trained_frozen', 'trained_online')]:
        outcomes = {'win': 0, 'loss': 0, 'tie': 0}
        for pair in pairs.values():
            delta = pair[online]['reward'] - pair[frozen]['reward']
            outcomes['win' if delta > 0 else 'loss' if delta < 0 else 'tie'] += 1
        comparisons[f'{online}_vs_{frozen}'] = outcomes
    totals = {arm: {metric: sum(row[metric] for row in rows if row['arm'] == arm)
                    for metric in ('reward', 'steps', 'invalid', 'actor_calls', 'skill_actions')}
              for arm in ARMS}
    report = dict(schema='alf_feedback_skill_online_audit_v1', design_sha256=sha(design_path),
        auditor_sha256=sha(Path(__file__)), rows=rows, comparisons=comparisons, totals=totals,
        caveat='Transductive valid_unseen diagnostic: earlier test feedback is used on later test games, so these are not standard held-out scores')
    save(args.output, report)
    print(json.dumps({'comparisons': comparisons, 'totals': totals}), flush=True)


if __name__ == '__main__':
    main()
