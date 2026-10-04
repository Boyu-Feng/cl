"""Replay the pickup-controlled online experience comparison."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re

from ttcl.alfworld_comparison.environment import make_env
from ttcl.experience_evolution.core import digest, seed
from ttcl.experience_evolution.environment import ACTOR_SYSTEM, clean_command
from ttcl.icl_mem0_comparison.protocol import read, save, sha
from .feedback_skill_actor import public_state
from .probe_pickup_trajectory_memory import ARMS, REPEATS
from .probe_trajectory_experience_actor import relevant_experience, snapshot_memory
from .trajectory_writer_learning import ClaimMemory, teacher_operations


def audit_episode(design: dict, case: dict, repeat: int, arm: str,
                  memory: ClaimMemory, path: Path) -> dict:
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
        invalid = activated = pickup = calls = 0
        for turn, (row, generation) in enumerate(zip(episode['trajectory'], episode['generations'])):
            available = list(state['admissible_commands'])
            messages.append({'role': 'user', 'content': str(state['feedback']) +
                             '\nAvailable commands:\n' + '\n'.join(available)})
            if generation['seed'] != seed(episode['seed'], turn):
                raise ValueError(f'Seed mismatch: {path}:{turn}')
            tracker = public_state(case['public_task'], messages)
            if generation.get('pickup_state') != tracker:
                raise ValueError(f'Pickup state mismatch: {path}:{turn}')
            target = rf'{re.escape(tracker["target"])} \d+'
            choice = next((action for action in available
                if (match := re.fullmatch(rf'take ({target}) from \w+ \d+', action))
                and match.group(1) not in tracker['held'] and match.group(1) not in tracker['moved']), None)
            prompt = [dict(message) for message in messages]
            if choice is not None:
                if (generation['finish_reason'] != 'public_goal_pickup'
                        or generation['text'] != choice or row['action'] != choice
                        or generation.get('experience_decision') != {'reason': 'pickup_first'}
                        or generation.get('experience_text_sha256') != digest('')):
                    raise ValueError(f'Pickup action mismatch: {path}:{turn}')
                pickup += 1
            else:
                prompt[-1]['content'] += ('\n\nCurrent goal: ' + case['public_task'] +
                    '\nVisited locations: ' + (', '.join(tracker['visited'][-20:]) or 'none') +
                    '\nChoose a listed command that advances the unfinished goal. '
                    'Search a new location when possible; revisit one when carrying the goal object.')
                text, decision = (relevant_experience(case['public_task'], messages, memory)
                    if arm == 'pickup_online' else ('', dict(reason='frozen_empty')))
                if generation.get('experience_decision') != decision or generation.get('experience_text_sha256') != digest(text):
                    raise ValueError(f'Experience prompt decision mismatch: {path}:{turn}')
                if text:
                    prompt[-1]['content'] += ('\nRelevant experience from completed earlier tasks '
                        '(apply only if its condition matches):\n' + text)
                    activated += 1
                calls += 1
            if generation['prompt_sha256'] != digest(json.dumps(prompt, ensure_ascii=False)):
                raise ValueError(f'Prompt hash mismatch: {path}:{turn}')
            if row['action'] != clean_command(generation['text'], available):
                raise ValueError(f'Command mismatch: {path}:{turn}')
            valid = row['action'] in available
            if valid != row['valid_command']:
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
        return dict(reward=episode['reward'], steps=episode['steps'], invalid=invalid,
                    actor_calls=calls, pickup_actions=pickup, activated_turns=activated,
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
    dataset = read(Path(design['dataset_path']))
    if (design['schema'] != 'alf_pickup_trajectory_memory_online_v1'
            or sha(Path(design['plan_path'])) != design['plan_sha256']
            or sha(Path(design['dataset_path'])) != design['dataset_sha256']
            or any(sha(Path(row['path'])) != row['sha256'] for row in design['excluded_designs'])
            or len(design['cases']) != 9 or design['arms'] != list(ARMS)
            or design['repeats'] != list(REPEATS)):
        raise ValueError('Frozen pickup design mismatch')
    excluded = {row['input_sha256'] for split in dataset['rows'] for row in dataset['rows'][split]}
    excluded.update(dataset['frozen_pilot_excluded_inputs'])
    for design_row in design['excluded_designs']:
        excluded.update(case['input_sha256'] for case in read(Path(design_row['path']))['cases'])
    rows = []
    for repeat in REPEATS:
        memory = ClaimMemory()
        for index, case in enumerate(design['cases']):
            if (not case['game'].startswith('json_2.1.1/train/')
                    or case['input_sha256'] in excluded or not case['reviewed']):
                raise ValueError('Train input leakage or review mismatch')
            for arm in ARMS:
                target = root / str(repeat) / f'{index:02d}_{case["input_sha256"][:12]}' / arm
                before = snapshot_memory(memory) if arm == 'pickup_online' else {}
                if read(target / 'memory_before.json') != before:
                    raise ValueError('Memory read future trajectory')
                details = audit_episode(design, case, repeat, arm, memory, target / 'episode.json')
                if arm == 'pickup_online':
                    episode = read(target / 'episode.json')
                    operations = teacher_operations(memory, episode)
                    result = memory.apply(operations, episode, case['input_sha256'])
                    annotation = dict(schema='alf_trajectory_claim_update_v1',
                        game=case['game'], input_sha256=case['input_sha256'],
                        source_episode_sha256=sha(target / 'episode.json'),
                        operations=operations, applied=result['applied'], reviewed=True)
                    if result['rejected'] or read(target / 'reviewed_update.json') != annotation:
                        raise ValueError('New trajectory update mismatch')
                after = snapshot_memory(memory) if arm == 'pickup_online' else {}
                if read(target / 'memory_after.json') != after:
                    raise ValueError('Memory chain continuity mismatch')
                rows.append(dict(index=index, family=case['family'], input_sha256=case['input_sha256'],
                                 repeat=repeat, arm=arm, **details))
    pairs = defaultdict(dict)
    for row in rows:
        pairs[(row['input_sha256'], row['repeat'])][row['arm']] = row
    if len(pairs) != 18 or any(set(pair) != set(ARMS) for pair in pairs.values()):
        raise ValueError('Incomplete paired episodes')
    outcomes = {'win': 0, 'loss': 0, 'tie': 0}
    for pair in pairs.values():
        delta = pair['pickup_online']['reward'] - pair['pickup_frozen']['reward']
        outcomes['win' if delta > 0 else 'loss' if delta < 0 else 'tie'] += 1
    totals = {arm: {metric: sum(row[metric] for row in rows if row['arm'] == arm)
                    for metric in ('reward', 'steps', 'invalid', 'actor_calls', 'pickup_actions', 'activated_turns')}
              for arm in ARMS}
    report = dict(schema='alf_pickup_trajectory_memory_audit_v1',
        design_sha256=sha(design_path), auditor_sha256=sha(Path(__file__)),
        rows=rows, outcomes=outcomes, totals=totals,
        caveat='Nine official-train tasks, two independent seeds; online source-specific experience, no trained writer or valid_unseen result')
    save(args.output, report)
    print(json.dumps({'outcomes': outcomes, 'totals': totals}), flush=True)


if __name__ == '__main__':
    main()
