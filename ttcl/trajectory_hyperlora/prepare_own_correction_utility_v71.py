"""Bind own train failures to pre-task state and saved candidate corrections.

Preparation only: no utility labels are invented from failure or retention loss.
"""
import argparse
import json
from pathlib import Path

from ttcl.trajectory_hyperlora.alfworld_online_from_empty import digest, file_hash
from ttcl.trajectory_hyperlora.alfworld_persistent_failure_delta_v36 import save


def prepare(args):
    if args.output.exists(): raise FileExistsError(args.output)
    parent=json.loads(args.parent.read_text())
    audit_path=args.parent.with_suffix('.audit.json')
    audit=json.loads(audit_path.read_text())
    assert parent['complete'] and not parent['failures'] and parent['mode']=='hybrid'
    assert audit['passed'] and audit['episodes_replayed']==240
    assert audit['report_sha256']==file_hash(args.parent)
    for name,value in parent['code_sha256'].items():
        assert file_hash(Path(__file__).with_name(name))==value
    targets=[];snapshot=None
    for row in parent['rows'][:120]:
        if row['adapter_active'] and row['episode']['reward']==0:
            assert snapshot and file_hash(Path(snapshot['path']))==snapshot['sha256']
            w=row['write'];source=parent['rows'][row['source_stream_index']]
            assert w['reviewed_target'] and w['H_sha256']==row['H_before_sha256']
            assert w['episode_sha256']==digest(row['episode'])
            assert file_hash(Path(w['delta_file']))==w['delta_file_sha256']
            assert source['index']<row['index'] and source['episode']['reward']==1
            assert digest(source['episode'])==row['source_episode_sha256']==w['own_success_source_episode_sha256']
            assert '/train/' in row['game']
            assert file_hash(args.data_root/row['game'])==w['game_sha256']
            content={'index':row['index'],'game':row['game'],'game_sha256':w['game_sha256'],
                'initial_observation':row['episode']['initial_observation'],
                'first_episode_sha256':digest(row['episode']),
                'source_stream_index':source['index'],'source_episode_sha256':digest(source['episode']),
                'pre_task_snapshot':snapshot,'H_before_sha256':row['H_before_sha256'],
                'R_before_sha256':row['R_before_sha256'],
                'candidate_delta_path':w['delta_file'],'candidate_delta_file_sha256':w['delta_file_sha256'],
                'candidate_delta_sha256':w['delta_sha256'],
                'parent_target_binding':row['target_input_content_sha256'],
                'parent_write_binding':w['input_content_sha256']}
            targets.append({**content,'input_content_sha256':digest(content),'reviewed_target':True})
        if 'snapshot' in row:snapshot=row['snapshot']
    assert len(targets)==34
    review={'stage':'candidate_bindings_prepared_no_utility_outcomes',
        'parent_sha256':file_hash(args.parent),'parent_audit_sha256':file_hash(audit_path),
        'preparer_sha256':file_hash(Path(__file__)), 'parent_code_sha256':parent['code_sha256'],
        'targets':targets,'targets_sha256':digest(targets),
        'selection':'All active first failures in audited own-stream first120 official train tasks',
        'teacher':'Saved same-current-H own-success minus own-failure DeltaB, not expert trajectories',
        'planned_arms':['unchanged_retry','failure_delta','matched_norm_random'],
        'planned_budget':{'max_steps_per_attempt':30,'max_new_tokens':64,'history_turns':2,'loop_guard_max':2},
        'planned_read':'Original dynamic B plus capped original R, then candidate correction capped to 25 percent of the current effective adapter norm',
        'model_updates_during_collection':0,'write_counterfactual_results_to_parent_stream':False,
        'label_rule':'Only completed audited paired environment rewards may define correction utility; neither initial failure nor retention guard acceptance is a utility label',
        'student_input_rule':'For pre-attempt prediction, only initial observation and earlier own sources; this task completed failure and delta are teacher-only information',
        'training_availability':'New utility data become available only after auxiliary retries complete; disclose extra interaction budget and give direct LoRA the same data',
        'limitations':['Only first-failure selected tasks; does not estimate damage from correcting originally successful tasks',
                      'Teacher rescues are second attempts, not evidence of improved student first attempts',
                      'No utility outcomes collected yet; not ready for training',
                      'Frozen replay snapshots, not a new live online stream']}
    save(args.output,review)
    print(json.dumps({'reviewed_failures':len(targets),'planned_new_attempts':3*len(targets),'utility_labels_available':0}))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent',type=Path,default=Path('results/trajectory_hyperlora/alfworld_cold_joint_memory_v63_hybrid_20261010.json'))
    p.add_argument('--data-root',type=Path,default=Path('ttcl/data/alfworld_delta'))
    p.add_argument('--output',type=Path,default=Path('data/annotations/alfworld_own_correction_utility_v71_20261010.json'))
    prepare(p.parse_args())


if __name__=='__main__':main()
