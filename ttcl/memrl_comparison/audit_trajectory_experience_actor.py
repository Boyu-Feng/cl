"""Official replay and no-future-leakage audit for online trajectory memory."""
from __future__ import annotations

import argparse
import copy
from collections import defaultdict
import hashlib
import json
from pathlib import Path

from ttcl.alfworld_comparison.environment import make_env
from ttcl.experience_evolution.core import digest, seed
from ttcl.experience_evolution.environment import ACTOR_SYSTEM, clean_command
from ttcl.icl_mem0_comparison.protocol import read, save, sha
from .probe_trajectory_experience_actor import (ARMS, REPEATS, learned_memory,
    relevant_experience, snapshot_memory)
from .trajectory_writer_learning import ClaimMemory, teacher_operations


def audit_episode(design: dict, case: dict, repeat: int, arm: str,
                  memory: ClaimMemory | None, path: Path) -> dict:
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
        invalid = activated = 0
        for turn, (row, generation) in enumerate(zip(episode['trajectory'], episode['generations'])):
            available = list(state['admissible_commands'])
            messages.append({'role': 'user', 'content': str(state['feedback']) +
                             '\nAvailable commands:\n' + '\n'.join(available)})
            if generation['seed'] != seed(episode['seed'], turn):
                raise ValueError(f'Actor seed mismatch: {path}:{turn}')
            prompt = [dict(message) for message in messages]
            if memory is None:
                if 'experience_decision' in generation:
                    raise ValueError('Native actor has experience metadata')
            else:
                text, decision = relevant_experience(case['public_task'], messages, memory)
                if (generation.get('experience_decision') != decision or
                        generation.get('experience_text_sha256') != digest(text)):
                    raise ValueError(f'Claim retrieval changed: {path}:{turn}')
                if text:
                    prompt[-1]['content'] += ('\n\nRelevant experience from completed earlier tasks '
                        '(apply only if its stated condition matches this task):\n' + text)
                    activated += 1
            if generation['prompt_sha256'] != digest(json.dumps(prompt, ensure_ascii=False)):
                raise ValueError(f'Actor prompt changed: {path}:{turn}')
            if row['action'] != clean_command(generation['text'], available):
                raise ValueError(f'Cleaned action changed: {path}:{turn}')
            valid = row['action'] in available
            if row['valid_command'] != valid:
                raise ValueError(f'Validity mismatch: {path}:{turn}')
            invalid += not valid
            state, _, done = env.step(row['action'])
            if row['observation'] != str(state['feedback']):
                raise ValueError(f'Official feedback changed: {path}:{turn}')
            messages.append({'role': 'assistant', 'content': generation['text']})
            if (done or state['won']) and turn + 1 != episode['steps']:
                raise ValueError(f'Environment finished early: {path}:{turn}')
        if (episode['reward'] != float(bool(state['won']))
                or episode['termination'] != ('success' if state['won'] else 'budget_or_environment_done')
                or (not state['won'] and episode['steps'] < 50 and not done)):
            raise ValueError(f'Official reward changed: {path}')
        return dict(reward=episode['reward'], steps=episode['steps'],
                    invalid=invalid, actor_calls=episode['steps'], activated_turns=activated,
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
    dataset_path = Path(design['dataset_path'])
    dataset = read(dataset_path)
    initial, sources = learned_memory(dataset)
    if (design['schema'] != 'alf_trajectory_experience_online_pilot_v1'
            or sha(Path(design['plan_path'])) != design['plan_sha256']
            or sha(dataset_path) != design['dataset_sha256']
            or design['source_game_sha256'] != sources
            or design['claim_memory'] != snapshot_memory(initial)
            or len(design['cases']) != 9 or design['arms'] != list(ARMS)
            or design['repeats'] != list(REPEATS)
            or any(sha(Path(row['path'])) != row['sha256'] for row in design['excluded_designs'])):
        raise ValueError('Frozen online experience design changed')
    excluded = set(sources)
    excluded.update(row['input_sha256'] for row in dataset['rows']['validation'])
    excluded.update(dataset['frozen_pilot_excluded_inputs'])
    for row in design['excluded_designs']:
        excluded.update(case['input_sha256'] for case in read(Path(row['path']))['cases'])
    rows = []
    for repeat in REPEATS:
        stores = {'online_claims': ClaimMemory(),
                  'pretrained_frozen': copy.deepcopy(initial),
                  'pretrained_online': copy.deepcopy(initial)}
        for index, case in enumerate(design['cases']):
            if (not case['game'].startswith('json_2.1.1/train/')
                    or case['input_sha256'] in excluded or not case['reviewed']):
                raise ValueError('Train input review or source split changed')
            for arm in ARMS:
                target = root / str(repeat) / f'{index:02d}_{case["input_sha256"][:12]}' / arm
                memory = None if arm == 'native' else stores[arm]
                if memory is not None and read(target / 'memory_before.json') != snapshot_memory(memory):
                    raise ValueError('Memory used future trajectory information')
                details = audit_episode(design, case, repeat, arm, memory, target / 'episode.json')
                if arm in ('online_claims', 'pretrained_online'):
                    episode = read(target / 'episode.json')
                    operations = teacher_operations(memory, episode)
                    result = memory.apply(operations, episode, case['input_sha256'])
                    annotation = dict(schema='alf_trajectory_claim_update_v1', game=case['game'],
                        input_sha256=case['input_sha256'],
                        source_episode_sha256=sha(target / 'episode.json'),
                        operations=operations, applied=result['applied'], reviewed=True)
                    if result['rejected'] or read(target / 'reviewed_update.json') != annotation:
                        raise ValueError('New trajectory annotation changed')
                if memory is not None and read(target / 'memory_after.json') != snapshot_memory(memory):
                    raise ValueError('Memory update continuity changed')
                rows.append(dict(index=index, family=case['family'],
                    input_sha256=case['input_sha256'], repeat=repeat, arm=arm, **details))
    pairs = defaultdict(dict)
    for row in rows:
        pairs[(row['input_sha256'], row['repeat'])][row['arm']] = row
    if len(pairs) != 18 or any(set(pair) != set(ARMS) for pair in pairs.values()):
        raise ValueError('Incomplete paired evaluation')
    comparisons = {}
    for base, candidate in [('native', 'online_claims'), ('pretrained_frozen', 'pretrained_online')]:
        outcome = {'win': 0, 'loss': 0, 'tie': 0}
        for pair in pairs.values():
            delta = pair[candidate]['reward'] - pair[base]['reward']
            outcome['win' if delta > 0 else 'loss' if delta < 0 else 'tie'] += 1
        comparisons[f'{candidate}_vs_{base}'] = outcome
    totals = {arm: {metric: sum(row[metric] for row in rows if row['arm'] == arm)
                    for metric in ('reward', 'steps', 'invalid', 'actor_calls', 'activated_turns')}
              for arm in ARMS}
    report = dict(schema='alf_trajectory_experience_online_audit_v1',
        design_sha256=sha(design_path), auditor_sha256=sha(Path(__file__)),
        rows=rows, comparisons=comparisons, totals=totals,
        caveat='Official-train online memory pilot; exact claim updates, but no trained writer or valid_unseen claim')
    save(args.output, report)
    print(json.dumps({'comparisons': comparisons, 'totals': totals}), flush=True)


if __name__ == '__main__':
    main()
