"""Official replay, frozen-MemRL retrieval, and no-future-claim audit."""
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
from .credit_probe import memory_arms
from .probe_memrl_claim_overlay import (ARMS, EXPERIENCE_PREFIX, REPEATS,
                                        design_for)
from .probe_trajectory_experience_actor import relevant_experience, snapshot_memory
from .trajectory_writer_learning import ClaimMemory, teacher_operations


def audit_episode(design: dict, case: dict, repeat: int, arm: str,
                  memory: ClaimMemory | None, path: Path) -> dict:
    episode = read(path)
    source_plan = read(Path(design['origin']) / 'plan.json')
    game = Path(source_plan['alf']['data_root']) / case['game']
    if (sha(game) != case['input_sha256'] or episode['game'] != case['game']
            or episode['memory'] != case['memrl_context']
            or episode['memory_sha256'] != digest(case['memrl_context'])
            or episode['seed'] != seed(repeat, case['game'], 0, 'actor')
            or episode['steps'] != len(episode['trajectory']) == len(episode['generations'])
            or not 1 <= episode['steps'] <= 50 or episode['status'] != 'complete'):
        raise ValueError(f'Episode header mismatch: {path}')
    env = make_env(game)
    try:
        state = env.reset()
        if (str(state['feedback']) != episode['initial_observation'] or
                hashlib.sha256(str(state['feedback']).encode()).hexdigest() !=
                case['initial_observation_sha256'] or
                digest(json.dumps(list(state['admissible_commands']))) !=
                episode['initial_commands_sha256']):
            raise ValueError(f'Official reset mismatch: {path}')
        system = ACTOR_SYSTEM + '\n\nPast experience:\n' + case['memrl_context']
        messages = [dict(role='system', content=system)]
        invalid = activated = no_effect = 0
        for turn, (row, generation) in enumerate(zip(episode['trajectory'], episode['generations'])):
            available = list(state['admissible_commands'])
            messages.append(dict(role='user', content=str(state['feedback']) +
                '\nAvailable commands:\n' + '\n'.join(available)))
            if generation['seed'] != seed(episode['seed'], turn):
                raise ValueError(f'Actor seed mismatch: {path}:{turn}')
            prompt = [dict(message) for message in messages]
            if memory is None:
                if 'experience_decision' in generation:
                    raise ValueError('Native MemRL arm has claim metadata')
            else:
                text, decision = relevant_experience(case['public_task'], messages, memory)
                if (generation.get('experience_decision') != decision or
                        generation.get('experience_text_sha256') != digest(text)):
                    raise ValueError(f'Claim decision changed: {path}:{turn}')
                if text:
                    prompt[-1]['content'] += EXPERIENCE_PREFIX + text
                    activated += 1
            if generation['prompt_sha256'] != digest(json.dumps(prompt, ensure_ascii=False)):
                raise ValueError(f'Prompt changed: {path}:{turn}')
            if row['action'] != clean_command(generation['text'], available):
                raise ValueError(f'Cleaned command changed: {path}:{turn}')
            valid = row['action'] in available
            if row['valid_command'] != valid:
                raise ValueError(f'Command validity changed: {path}:{turn}')
            invalid += not valid
            state, _, done = env.step(row['action'])
            if row['observation'] != str(state['feedback']):
                raise ValueError(f'Official feedback changed: {path}:{turn}')
            no_effect += 'Nothing happens.' in row['observation']
            messages.append(dict(role='assistant', content=generation['text']))
            if (done or state['won']) and turn + 1 != episode['steps']:
                raise ValueError(f'Environment finished early: {path}:{turn}')
        if (episode['reward'] != float(bool(state['won'])) or
                episode['termination'] != ('success' if state['won'] else 'budget_or_environment_done') or
                (not state['won'] and episode['steps'] < 50 and not done)):
            raise ValueError(f'Official reward changed: {path}')
        return dict(reward=episode['reward'], steps=episode['steps'], invalid=invalid,
                    no_effect=no_effect, activated_turns=activated, episode_sha256=sha(path))
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
    expected, static = design_for(Path(design['origin']), Path(design['dataset_path']),
                                  design['url'])
    if design != expected or design['arms'] != list(ARMS) or design['repeats'] != list(REPEATS):
        raise ValueError('Frozen MemRL claim design changed')
    if (sha(Path(design['origin']) / 'design.json') != design['origin_design_sha256'] or
            sha(Path(design['origin']) / 'plan.json') != design['origin_plan_sha256'] or
            sha(Path(design['dataset_path'])) != design['dataset_sha256'] or
            snapshot_memory(static) != design['static_memory']):
        raise ValueError('MemRL or claim source changed')
    for case in design['cases']:
        spec, arms = memory_arms(Path(design['origin']), case['case'])
        if (spec['source_input_sha256'] != case['input_sha256'] or
                spec['original_memrl_row_sha256'] != case['source_row_sha256'] or
                spec['retrieval_sha256'] != case['retrieval_sha256'] or
                spec['snapshot_sha256'] != case['snapshot_sha256'] or
                arms['full'] != case['memrl_context'] or
                digest(arms['full']) != case['context_sha256']):
            raise ValueError('Original MemRL retrieval changed')
    rows = []
    for repeat in REPEATS:
        online = ClaimMemory()
        for index, case in enumerate(design['cases']):
            for arm in ARMS:
                memory = static if arm == 'memrl_static' else online if arm == 'memrl_online' else None
                target = root / str(repeat) / f'{index:02d}_{case["input_sha256"][:12]}' / arm
                before = snapshot_memory(memory) if memory is not None else {}
                if read(target / 'memory_before.json') != before:
                    raise ValueError('Claim memory read future episode')
                details = audit_episode(design, case, repeat, arm, memory,
                                        target / 'episode.json')
                if arm == 'memrl_online':
                    episode = read(target / 'episode.json')
                    operations = teacher_operations(online, episode)
                    result = online.apply(operations, episode, case['input_sha256'])
                    annotation = dict(schema='alf_memrl_claim_update_v1', game=case['game'],
                        input_sha256=case['input_sha256'],
                        source_episode_sha256=sha(target / 'episode.json'),
                        operations=operations, applied=result['applied'], reviewed=True)
                    if result['rejected'] or read(target / 'reviewed_update.json') != annotation:
                        raise ValueError('Online claim source changed')
                after = snapshot_memory(memory) if memory is not None else {}
                if read(target / 'memory_after.json') != after:
                    raise ValueError('Claim memory continuity changed')
                rows.append(dict(index=index, family=case['family'],
                    input_sha256=case['input_sha256'], repeat=repeat, arm=arm, **details))
    pairs = defaultdict(dict)
    for row in rows:
        pairs[(row['input_sha256'], row['repeat'])][row['arm']] = row
    if len(pairs) != 18 or any(set(pair) != set(ARMS) for pair in pairs.values()):
        raise ValueError('Incomplete paired evaluation')
    outcomes = {}
    for arm in ('memrl_static', 'memrl_online'):
        result = dict(win=0, loss=0, tie=0)
        for pair in pairs.values():
            delta = pair[arm]['reward'] - pair['memrl']['reward']
            result['win' if delta > 0 else 'loss' if delta < 0 else 'tie'] += 1
        outcomes[arm] = result
    totals = {arm: {metric: sum(row[metric] for row in rows if row['arm'] == arm)
                    for metric in ('reward', 'steps', 'invalid', 'no_effect', 'activated_turns')}
              for arm in ARMS}
    report = dict(schema='alf_memrl_stage_claim_overlay_audit_v1',
        design_sha256=sha(design_path), auditor_sha256=sha(Path(__file__)),
        rows=rows, outcomes=outcomes, totals=totals,
        caveat='Fixed original MemRL retrieval/Q snapshot; evidence-claim overlay on official train; not full online MemRL or valid_unseen')
    save(args.output, report)
    print(json.dumps(dict(outcomes=outcomes, totals=totals)), flush=True)


if __name__ == '__main__':
    main()
