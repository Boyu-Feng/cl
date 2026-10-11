"""Review own-success causal prefix -> later action pairs for a new H objective.

This prepares data only. A target at zero-based step t sees source steps before
t-2; the actor retains its normal initial observation and last two turns.
Prefix status never uses the eventual episode outcome.
"""
import argparse
import json
from pathlib import Path

from ttcl.trajectory_hyperlora.alfworld_bootstrap_own_success_v43 import action_rows
from ttcl.trajectory_hyperlora.alfworld_online_from_empty import digest, file_hash
from ttcl.trajectory_hyperlora.train_alf_next_task import compact_messages


def prefix_records(initial_observation, completed_past_steps):
    records = []
    before = initial_observation
    for step in completed_past_steps:
        if step['won']:
            raise ValueError('Cannot predict a later action after episode termination')
        status = 'valid command' if step['valid'] else 'invalid command'
        records.append({'observation':before, 'action':step['command'],
                        'feedback':status+'. '+step['observation']})
        before = step['observation']
    return records


def prefix_text(records):
    if not records:
        raise ValueError('Empty earlier-history prefix')
    return 'Earlier observed steps of the current task:\n'+'\n'.join(
        f"Step {i+1}\nObservation: {r['observation']}\nAction: {r['action']}\nFeedback: {r['feedback']}"
        for i,r in enumerate(records))


def prepare(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    parent = json.loads(args.parent.read_text())
    audit = json.loads(args.parent.with_suffix('.audit.json').read_text())
    assert parent['complete'] and not parent['failures'] and audit['passed']
    assert audit['episodes_replayed']==240 and audit['report_sha256']==file_hash(args.parent)
    review = json.loads(args.parent_review.read_text())
    annotations = json.loads(args.parent_annotations.read_text())
    assert parent['review_sha256']==file_hash(args.parent_review)==annotations['eval_review_sha256']
    by_index = {r['stream_index']:r for r in annotations['annotations']}
    source_rows = parent['rows'][:args.prefix_tasks]
    targets = []
    wins = []
    for row, binding in zip(source_rows,review['targets'][:args.prefix_tasks],strict=True):
        assert row['game']==binding['game'] and '/train/' in row['game']
        assert binding['input_content_sha256']==row['target_input_content_sha256']
        assert binding['game_sha256']==file_hash(args.data_root/row['game'])
        if row['episode']['reward']!=1:
            continue
        actions = action_rows(args.data_root/row['game'],row['episode'])
        annotation = by_index[row['index']]
        assert annotation['actions_sha256']==digest(actions)
        assert annotation['episode_sha256']==digest(row['episode'])
        wins.append(row['index'])
        for t, action in enumerate(actions):
            cutoff = t-2
            if cutoff<=0:
                continue
            past = row['episode']['trajectory'][:cutoff]
            records = prefix_records(row['episode']['initial_observation'],past)
            actor_messages = compact_messages(action['target_messages'],history_turns=2)
            content = {'stream_index':row['index'],'game':row['game'],
                       'game_sha256':binding['game_sha256'],'episode_sha256':annotation['episode_sha256'],
                       'parent_annotation_sha256':annotation['input_content_sha256'],
                       'target_step':t,'source_end_exclusive':cutoff,
                       'actor_history_first_step':cutoff,
                       'source_records':records,'source_records_sha256':digest(records),
                       'actor_messages':actor_messages,'target_action':action['target_action'],
                       'initial_observation':row['episode']['initial_observation'],
                       'current_observation':action['current_observation']}
            assert len(records)==cutoff and cutoff<t
            assert action['target_action']==row['episode']['trajectory'][t]['command']
            targets.append({**content,'input_content_sha256':digest(content),'reviewed_target':True})
    out = {'stage':'causal_prefix_training_inputs_prepared_not_trained',
           'scope':'Replay of earlier own-success trajectories; not new live collection',
           'parent_sha256':file_hash(args.parent),'parent_audit_sha256':file_hash(args.parent.with_suffix('.audit.json')),
           'parent_review_sha256':file_hash(args.parent_review),
           'parent_annotations_sha256':file_hash(args.parent_annotations),
           'runner_sha256':file_hash(Path(__file__)),
           'dependencies':{n:file_hash(Path(__file__).with_name(n)) for n in (
               'alfworld_bootstrap_own_success_v43.py','train_alf_next_task.py','alfworld_online_from_empty.py')},
           'prefix_tasks':args.prefix_tasks,'own_success_indices':wins,'action_target_count':len(targets),
           'actor_history_turns':2,'source_policy':'Only completed steps strictly older than actor recent window; no target action or future terminal status',
           'training_availability':'Each episode may train only after its own completion and success verification',
           'limitation':'Causality alone does not establish that source history is necessary or useful; needs source ablations and real future environment rewards',
           'targets':targets,'targets_sha256':digest(targets)}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as f:
        json.dump(out,f,ensure_ascii=False,indent=2)
        f.write('\n')
    print(json.dumps({'prefix_tasks':args.prefix_tasks,'own_wins':len(wins),'reviewed_action_targets':len(targets),'training_started':False}))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent',type=Path,default=Path('results/trajectory_hyperlora/alfworld_cold_joint_memory_v63_hybrid_20261010.json'))
    p.add_argument('--parent-review',type=Path,default=Path('data/annotations/alfworld_cold_joint_memory_v63_hybrid_20261010_review.json'))
    p.add_argument('--parent-annotations',type=Path,default=Path('data/annotations/alfworld_cold_joint_memory_v63_hybrid_20261010_wins.json'))
    p.add_argument('--data-root',type=Path,default=Path('ttcl/data/alfworld_delta'))
    p.add_argument('--prefix-tasks',type=int,choices=(120,),default=120)
    p.add_argument('--output',type=Path,default=Path('data/annotations/alfworld_causal_prefix_v70_train120_20261010.json'))
    prepare(p.parse_args())


if __name__=='__main__':main()
