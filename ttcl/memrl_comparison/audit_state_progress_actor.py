"""Replay official ALFWorld episodes from the exploratory state-progress pilot."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

from ttcl.alfworld_comparison.environment import make_env
from ttcl.experience_evolution.core import digest, seed
from ttcl.experience_evolution.environment import ACTOR_SYSTEM
from ttcl.icl_mem0_comparison.protocol import read, save, sha
from .probe_state_progress_actor import select_public_action


def audit_one(design: dict, case: dict, repeat: int, arm: str, path: Path) -> dict:
    episode = read(path)
    source_plan = read(Path(design['plan_path']))
    game = Path(source_plan['alf']['data_root']) / case['game']
    if (sha(game) != case['input_sha256'] or episode['game'] != case['game'] or
            episode['memory'] != '' or episode['memory_sha256'] != digest('') or
            episode['seed'] != seed(repeat, case['game'], 0, 'actor') or
            not 1 <= episode['steps'] <= 50 or
            episode['steps'] != len(episode['trajectory']) == len(episode['generations']) or
            episode['status'] != 'complete'):
        raise ValueError(f'Frozen episode header changed: {path}')
    env = make_env(game)
    try:
        state = env.reset()
        if (str(state['feedback']) != episode['initial_observation'] or
                hashlib.sha256(str(state['feedback']).encode()).hexdigest() !=
                case['initial_observation_sha256'] or
                digest(json.dumps(list(state['admissible_commands']))) !=
                episode['initial_commands_sha256']):
            raise ValueError(f'Official reset mismatch: {path}')
        messages = [{'role':'system', 'content':ACTOR_SYSTEM}]
        invalid = rules = calls = 0
        for turn, (row, generation) in enumerate(zip(episode['trajectory'], episode['generations'])):
            available = list(state['admissible_commands'])
            messages.append({'role':'user', 'content':str(state['feedback']) +
                             '\nAvailable commands:\n' + '\n'.join(available)})
            if generation['seed'] != seed(episode['seed'], turn):
                raise ValueError(f'Actor seed mismatch at {path}:{turn}')
            if arm == 'state_progress':
                choice, public_state = select_public_action(case['public_task'], messages, available)
                if generation.get('public_state') != public_state:
                    raise ValueError(f'Public tracker mismatch at {path}:{turn}')
                if choice is not None:
                    if (generation['finish_reason'] != 'public_state_rule' or
                            generation['text'] != choice or row['action'] != choice or
                            generation['prompt_sha256'] != digest(json.dumps(messages, ensure_ascii=False))):
                        raise ValueError(f'State rule mismatch at {path}:{turn}')
                    rules += 1
                else:
                    augmented = [dict(m) for m in messages]
                    augmented[-1]['content'] += ('\n\nCurrent goal: ' + case['public_task'] +
                        '\nVisited locations: ' + (', '.join(public_state['visited'][-20:]) or 'none') +
                        '\nChoose a listed command that advances the unfinished goal. '
                        'Search a new location when possible; revisit one when carrying the goal object.')
                    if generation['prompt_sha256'] != digest(json.dumps(augmented, ensure_ascii=False)):
                        raise ValueError(f'Actor prompt mismatch at {path}:{turn}')
                    calls += 1
            else:
                if (generation.get('public_state') is not None or
                        generation['prompt_sha256'] != digest(json.dumps(messages, ensure_ascii=False))):
                    raise ValueError(f'Native prompt mismatch at {path}:{turn}')
                calls += 1
            valid = row['action'] in available
            if row['valid_command'] != valid:
                raise ValueError(f'Valid-command flag mismatch at {path}:{turn}')
            invalid += not valid
            state, _, done = env.step(row['action'])
            if row['observation'] != str(state['feedback']):
                raise ValueError(f'Official feedback mismatch at {path}:{turn}')
            messages.append({'role':'assistant', 'content':generation['text']})
            if (done or state['won']) and turn + 1 != episode['steps']:
                raise ValueError(f'Finished early at {path}:{turn}')
        if (episode['reward'] != float(bool(state['won'])) or
                episode['termination'] != ('success' if state['won'] else 'budget_or_environment_done') or
                (not state['won'] and episode['steps'] < 50 and not done)):
            raise ValueError(f'Official reward or termination mismatch: {path}')
        return dict(family=case['family'], repeat=repeat, arm=arm,
                    reward=episode['reward'], steps=episode['steps'],
                    invalid=invalid, actor_calls=calls, rule_actions=rules,
                    episode_sha256=sha(path))
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--roots', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rows = []
    designs = []
    for root in args.roots:
        root = root.resolve()
        design_path = root / 'design.json'
        design = read(design_path)
        if (design['schema'] != 'alf_state_progress_pilot_v1' or
                sha(Path(design['plan_path'])) != design['plan_sha256'] or
                sha(Path(design['review_path'])) != design['review_sha256'] or
                len(design['cases']) != 6 or design['arms'] != ['native', 'state_progress']):
            raise ValueError(f'Frozen design mismatch: {root}')
        designs.append(dict(path=str(design_path), sha256=sha(design_path)))
        for case in design['cases']:
            for repeat in design['repeats']:
                for arm in design['arms']:
                    path = root / case['family'] / str(repeat) / arm / 'episode.json'
                    if not path.exists():
                        raise ValueError(f'Missing planned episode: {path}')
                    rows.append(audit_one(design, case, repeat, arm, path))
    paired = defaultdict(dict)
    for row in rows:
        paired[(row['family'], row['repeat'])][row['arm']] = row
    if any(set(v) != {'native', 'state_progress'} for v in paired.values()):
        raise ValueError('Unpaired rows')
    outcome = {'win':0,'loss':0,'tie':0}
    by_family = defaultdict(lambda:{'win':0,'loss':0,'tie':0})
    for (family, repeat), pair in paired.items():
        delta = pair['state_progress']['reward'] - pair['native']['reward']
        label = 'win' if delta > 0 else 'loss' if delta < 0 else 'tie'
        outcome[label] += 1
        by_family[family][label] += 1
    report = dict(schema='alf_state_progress_audit_v1', designs=designs,
                  auditor_sha256=sha(Path(__file__)), rows=rows,
                  paired_outcomes=outcome, by_family=dict(by_family),
                  totals={arm:dict(reward=sum(r['reward'] for r in rows if r['arm']==arm),
                                   steps=sum(r['steps'] for r in rows if r['arm']==arm),
                                   invalid=sum(r['invalid'] for r in rows if r['arm']==arm),
                                   actor_calls=sum(r['actor_calls'] for r in rows if r['arm']==arm),
                                   rule_actions=sum(r['rule_actions'] for r in rows if r['arm']==arm))
                          for arm in ('native','state_progress')},
                  caveat='Six unique train games repeated across three seeds; exploratory, no independent valid_unseen or CLBench confirmation')
    save(args.output, report)
    print(json.dumps({k:report[k] for k in ('paired_outcomes','by_family','totals')},
                     ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
