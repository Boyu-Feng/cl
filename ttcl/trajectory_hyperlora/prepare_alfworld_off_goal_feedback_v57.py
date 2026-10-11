"""Bind replay-verified off-goal transformations; no invented correct actions.

These are task-irrelevant operation labels, not invalid-command labels or
claims that every action in a failed episode was wrong. Only public text is
used for the goal/object relation. No expert policy or hidden state is read.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

from ttcl.experience_evolution.environment import ACTOR_SYSTEM, make_env
from ttcl.trajectory_hyperlora.alfworld_online_from_empty import digest, file_hash


def off_goal(initial, command, feedback):
    if 'Your task is to:' not in initial:
        return None
    goal = initial.split('Your task is to:')[-1].strip()
    match = re.fullmatch(r'(clean|heat|cool) (\w+) (\d+) with (\w+ \d+)', command)
    if match is None:
        return None
    verb, actual_type, instance, receptacle = match.groups()
    adjective = {'clean':'clean', 'heat':'hot', 'cool':'cool'}[verb]
    expected = re.fullmatch(rf'{verb} some (\w+) and put it in (\w+)\.', goal)
    expected = expected or re.fullmatch(rf'put a {adjective} (\w+) in (\w+)\.', goal)
    if expected is None or expected.group(1) == actual_type:
        return None
    if feedback.strip() != f'You {verb} the {actual_type} {instance} using the {receptacle}.':
        return None
    return {'public_goal':goal, 'operation':verb, 'expected_object_type':expected.group(1),
            'actual_object_type':actual_type, 'label':'off_goal_transformation',
            'label_scope':'Operation succeeded on a different object type; no correct replacement action is asserted'}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--report',type=Path,default=Path('results/trajectory_hyperlora/alfworld_online_own_trajectory_v52b_stream240_20261010.json'))
    p.add_argument('--data-root',type=Path,default=Path('ttcl/data/alfworld_delta'))
    p.add_argument('--prefix-tasks',type=int,default=120)
    p.add_argument('--output',type=Path,default=Path('data/annotations/alfworld_off_goal_feedback_v57_prefix120_20261010.json'))
    args=p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report=json.loads(args.report.read_text())
    rows=report['rows'][:args.prefix_tasks]
    assert len(rows)==args.prefix_tasks and not report['failures']
    assert [r['index'] for r in rows]==list(range(args.prefix_tasks))
    targets=[]
    replayed=0
    for row in rows:
        ep=row['online']
        assert '/train/' in '/'+row['game'] and ep['status']=='complete'
        if ep['reward'] != 0:
            continue
        labels={i:off_goal(ep['initial_observation'],s['command'],s['observation'])
                for i,s in enumerate(ep['trajectory'])}
        if not any(labels.values()):
            continue
        game=args.data_root/row['game']
        env=make_env(game)
        episode_targets=[]
        try:
            state=env.reset()
            assert str(state['feedback'])==ep['initial_observation']
            messages=[{'role':'system','content':ACTOR_SYSTEM}]
            done=False
            for i,step in enumerate(ep['trajectory']):
                assert not done and step['valid'] and step['command'] in state['admissible_commands']
                current=str(state['feedback'])
                messages.append({'role':'user','content':current+'\nAvailable commands:\n'+'\n'.join(state['admissible_commands'])})
                if labels[i]:
                    content={'stream_index':row['index'],'game':row['game'],
                        'game_sha256':file_hash(game),'episode_sha256':digest(ep),'turn':i,
                        'current_observation':current,'target_messages':[dict(m) for m in messages],
                        'observed_action':step['command'],'feedback':step['observation'],
                        'supervision':labels[i],'runner_sha256':file_hash(Path(__file__))}
                    episode_targets.append({**content,'input_content_sha256':digest(content),'reviewed_target':True,
                        'review_note':'Exact replay of public observations, legal commands and feedback; explicit public-goal object-type mismatch'})
                state,_,done=env.step(step['command'])
                assert str(state['feedback'])==step['observation'] and bool(state['won'])==step['won']
                messages.append({'role':'assistant','content':step['response']})
            assert not state['won'] and ep['steps']==len(ep['trajectory'])
            assert ep['termination']=='budget_or_environment_done' and ep['invalid_commands']==0
        finally:
            env.close()
        targets.extend(episode_targets)
        replayed+=1
    result={'protocol':'Exposed TRAIN prefix only; replay-bound off-goal operation labels, no positive replacement actions; no training or evaluation performed',
        'prefix_tasks':args.prefix_tasks,'prefix_rows_sha256':digest(rows),
        'teacher_review_sha256':report['review_sha256'],'runner_sha256':file_hash(Path(__file__)),
        'replayed_failure_episodes':replayed,'targets':targets,'targets_sha256':digest(targets)}
    with args.output.open('x') as f:
        json.dump(result,f,ensure_ascii=False,indent=2)
        f.write('\n')
    print(json.dumps({'prefix_tasks':args.prefix_tasks,'replayed_failures':replayed,'off_goal_targets':len(targets)}),flush=True)


if __name__=='__main__':
    main()
