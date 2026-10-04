"""Test one same-text experience activated only after seeing the goal object.

Two reviewed official ALFWorld train games were chosen posthoc because that
experience had opposite local solo effects. This tests a stage mechanism,
not an unbiased estimate of a general policy. No memory or Q update occurs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

from ttcl.alfworld_comparison.environment import make_env
from ttcl.experience_evolution.core import digest, seed
from ttcl.experience_evolution.environment import ACTOR_SYSTEM, clean_command
from ttcl.icl_mem0_comparison.protocol import Client, read, save, sha
from .annotate_claim_applicability_pilot import REVIEWS
from .credit_probe import memory_arms
from .probe_goal_slot_credit import _slots


MEMORY_TEXT_SHA256 = '004665ab100267d864585b7ed591d34c70d598f3b478bb30a2670a92ef665a3f'
REPEATS = (94001, 94002, 94003)
ARMS = ('always', 'delayed', 'none', 'always_repeat')


def design_for(sign_report: Path, output: Path, url: str) -> dict:
    source = read(sign_report)
    if source['schema'] != 'alf_credit_sign_transfer_audit_v1':
        raise ValueError('Wrong credit sign source')
    matching = [item for item in source['repeated_memories']
                if item['memory_text_sha256'] == MEMORY_TEXT_SHA256]
    if len(matching) != 1 or not matching[0]['opposite_signs'] or \
            len(matching[0]['instances']) != 2:
        raise ValueError('Expected two opposite-sign instances of one exact text')
    reviews_root = Path(__file__).parents[2] / 'data' / 'annotations'
    cases = []
    for index, item in enumerate(matching[0]['instances']):
        coalition = Path(item['output'])
        origin = Path(read(coalition / 'design.json')['origin'])
        spec, arms = memory_arms(origin, item['case'])
        if (item['memory_id'] != spec['ids'][item['memory_index']] or
                spec['memory_text_sha256'][item['memory_id']] !=
                MEMORY_TEXT_SHA256 or
                spec['source_input_sha256'] != item['input_sha256']):
            raise ValueError('Experience or public input changed')
        kind, review_name = REVIEWS[coalition.name]
        review_path = reviews_root / review_name
        reviews = read(review_path)['targets']
        found = [row for row in reviews if row['origin'] == kind and
                 row['case'] == item['case'] and row['reviewed'] and
                 row['input_sha256'] == item['input_sha256'] and
                 row['public_task'] == item['public_task']]
        if len(found) != 1:
            raise ValueError('Missing reviewed, content-bound official goal')
        obj, destination, operation = _slots(item['public_task'])
        source_row = origin / 'runs' / item['case'] / 'row.json'
        row = read(source_row)
        plan = read(origin / 'plan.json')
        if (row['game'] != item['game'] or
                sha(Path(plan['alf']['data_root']) / row['game']) !=
                item['input_sha256'] or
                plan['alf']['max_steps'] != 50):
            raise ValueError('Official game or action budget changed')
        cases.append(dict(index=index, origin=str(origin),
                          source_output=str(coalition),
                          source_design_sha256=sha(coalition / 'design.json'),
                          review_sha256=sha(review_path),
                          source_plan_sha256=sha(origin / 'plan.json'),
                          source_row_sha256=sha(source_row),
                          case=item['case'], game=row['game'],
                          game_sha256=item['input_sha256'],
                          public_task=item['public_task'],
                          goal_object=obj, goal_destination=destination,
                          goal_operation=operation,
                          memory_id=item['memory_id'],
                          memory_index=item['memory_index'],
                          memory_text_sha256=MEMORY_TEXT_SHA256,
                          memory_context=arms[f"only_{item['memory_index']}"],
                          local_shapley_sign=item['robust_shapley_sign']))
    if {case['local_shapley_sign'] for case in cases} != {-1, 1}:
        raise ValueError('Posthoc sign-flip selection changed')
    return dict(schema='alf_delayed_memory_activation_v1',
                sign_report_sha256=sha(sign_report),
                runner_sha256=sha(Path(__file__)),
                environment_adapter_sha256=sha(Path(__file__).parents[1] /
                                                 'alfworld_comparison' /
                                                 'environment.py'),
                cases=cases, repeats=list(REPEATS), arms=list(ARMS), url=url,
                budget='Two official train games; three fresh seeds; four arms each; at most 50 environment actions per arm; no memory updates',
                caveat='Posthoc selected exact-text sign flip. Delayed activation requires a public feedback phrase showing the goal object; no claim of general effect.')


def run_episode(plan: dict, client: Client, case: dict, repeat: int,
                arm: str, output: Path) -> dict:
    if arm not in ARMS or output.exists():
        raise ValueError('Invalid or existing stage episode')
    game = Path(plan['alf']['data_root']) / case['game']
    if sha(game) != case['game_sha256']:
        raise ValueError('Official game content changed')
    output.mkdir(parents=True)
    environment = make_env(game)
    actor_seed = seed(repeat, case['game'], 0, 'actor')
    target = re.compile(r'\b(?:a|an) ' + re.escape(case['goal_object']) +
                        r' \d+\b', re.I)
    eligible = case['memory_context']
    active = arm in ('always', 'always_repeat')
    messages = [{'role':'system', 'content':ACTOR_SYSTEM +
                 ('\n\nPast experience:\n' + eligible if active else '')}]
    episode = dict(game=case['game'], game_sha256=case['game_sha256'],
                   case=case['case'], arm=arm, seed=actor_seed,
                   repeat=repeat, eligible_memory=eligible,
                   eligible_memory_sha256=hashlib.sha256(eligible.encode()).hexdigest(),
                   goal_object=case['goal_object'],
                   trigger='post-action public feedback lists a numbered goal object',
                   trajectory=[], generations=[], reward=0.0, steps=0,
                   activation_turn=(1 if active else None),
                   status='failed')
    try:
        state = environment.reset()
        episode['initial_observation'] = str(state['feedback'])
        episode['initial_commands_sha256'] = digest(json.dumps(
            list(state['admissible_commands'])))
        if case['public_task'] not in episode['initial_observation']:
            raise ValueError('Reviewed goal differs from public reset')
        for turn in range(plan['alf']['max_steps']):
            feedback = str(state['feedback'])
            available = list(state['admissible_commands'])
            messages.append({'role':'user', 'content':feedback +
                             '\nAvailable commands:\n' + '\n'.join(available)})
            result = client.complete(messages, seed(actor_seed, turn),
                                     tokens=plan['alf']['actor_max_tokens'],
                                     temperature=plan['alf']['actor_temperature'],
                                     top_p=1.)
            completion = dict(text=result['raw_response'],
                              finish_reason=result['finish_reason'],
                              usage={'prompt_tokens':result['input_tokens'],
                                     'completion_tokens':result['output_tokens']},
                              seed=seed(actor_seed, turn),
                              prompt_sha256=digest(messages),
                              rendered_prompt_sha256=result['rendered_prompt_sha256'],
                              seconds=result['seconds'])
            command = clean_command(completion['text'], available)
            was_valid = command in available
            state, _, done = environment.step(command)
            messages.append({'role':'assistant', 'content':completion['text']})
            observed = str(state['feedback'])
            triggered = bool(target.search(observed))
            episode['generations'].append(completion)
            episode['trajectory'].append(dict(action=command,
                                              observation=observed,
                                              valid_command=was_valid,
                                              memory_active=active,
                                              trigger_after_action=triggered))
            episode.update(reward=float(bool(state['won'])), steps=turn + 1)
            if done or state['won'] or turn + 1 == plan['alf']['max_steps']:
                episode['status'] = 'complete'
                episode['termination'] = ('success' if state['won']
                                          else 'budget_or_environment_done')
                break
            if arm == 'delayed' and not active and triggered:
                active = True
                episode['activation_turn'] = turn + 2
                messages[0]['content'] += '\n\nPast experience:\n' + eligible
        save(output / 'episode.json', episode)
        return episode
    finally:
        environment.close()


def run(sign_report: Path, output: Path, url: str, prepare_only: bool) -> None:
    design = design_for(sign_report, output, url)
    output.mkdir(parents=True, exist_ok=True)
    path = output / 'design.json'
    if path.exists():
        if read(path) != design:
            raise ValueError('Frozen stage design changed')
    else:
        save(path, design)
    print(json.dumps(dict(design_sha256=sha(path),
                          cases=[(case['public_task'], case['local_shapley_sign'])
                                 for case in design['cases']])), flush=True)
    if prepare_only:
        return
    for case in design['cases']:
        plan = read(Path(case['origin']) / 'plan.json')
        plan['url'] = url
        plan['alf']['actor_url'] = url
        for repeat in REPEATS:
            for arm in ARMS:
                target = output / f"case_{case['index']}" / f'repeat_{repeat}' / arm
                episode_path = target / 'episode.json'
                if episode_path.exists():
                    continue
                episode = run_episode(plan, Client(plan, repeat), case,
                                      repeat, arm, target)
                print(json.dumps(dict(case=case['index'], repeat=repeat,
                                      arm=arm, reward=episode['reward'],
                                      steps=episode['steps'],
                                      activation_turn=episode['activation_turn'])),
                      flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sign-report', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--url', default='http://127.0.0.1:18559')
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    run(args.sign_report.resolve(), args.output.resolve(), args.url,
        args.prepare_only)


if __name__ == '__main__':
    main()
