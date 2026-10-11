"""Replay cleaned own successes and create NEW content-bound action targets."""
import json
from pathlib import Path
from ttcl.trajectory_hyperlora.clean_success_loops_v87 import ROOT as CLEAN_ROOT,REVIEW as CLEAN_REVIEW,DATA,candidates
from ttcl.trajectory_hyperlora.prepare_transition_reader_v82 import PARENT
from ttcl.trajectory_hyperlora.alfworld_bootstrap_own_success_v43 import action_rows
from ttcl.trajectory_hyperlora.train_alf_next_task import compact_messages
from ttcl.trajectory_hyperlora.alfworld_online_from_empty import digest,file_hash
from ttcl.trajectory_hyperlora.alfworld_persistent_failure_delta_v36 import save
OUTPUT=Path('data/annotations/alfworld_cleaned_targets_v88_20261010.json')
AUDIT=CLEAN_ROOT/'annotation_audit.json'


def main():
    if OUTPUT.exists() or AUDIT.exists():raise FileExistsError('New annotation output already exists')
    report_path=CLEAN_ROOT/'report.json';report=json.loads(report_path.read_text());review=json.loads(CLEAN_REVIEW.read_text())
    parent=json.loads(PARENT.read_text());assert report['complete'] and not report['failures']
    assert report['review_sha256']==file_hash(CLEAN_REVIEW) and review['parent_sha256']==file_hash(PARENT)
    for n,h in review['dependencies'].items():assert file_hash(Path(__file__).with_name(n))==h
    assert review['environment_sha256']==file_hash(Path('ttcl/experience_evolution/environment.py'))
    assert len(report['rows'])==len(review['sources'])==28
    annotations=[];targets=[];wins=[];trials_checked=0;env_steps=0
    for row,source in zip(report['rows'],review['sources'],strict=True):
        assert all(row[k]==v for k,v in source.items()) and source['game_sha256']==file_hash(DATA/source['game'])
        original=parent['rows'][row['index']]['episode'];assert digest(original)==source['episode_sha256']
        current=original;attempted=set();accepted=0
        for trial in row['trials']:
            assert trial['parent_episode_sha256']==digest(current)
            choices=[(i,j) for i,j in candidates(current) if (digest(current),i,j) not in attempted];assert choices
            i,j=choices[0];assert (i,j)==(trial['delete_start'],trial['delete_end_exclusive'])
            attempted.add((digest(current),i,j))
            commands=[x['command'] for x in current['trajectory']];assert trial['candidate_commands']==commands[:i]+commands[j:]
            candidate=trial['candidate_episode'];keep=candidate['status']=='complete' and candidate['reward']==1 and candidate['steps']<current['steps']
            assert keep==trial['accepted'];trials_checked+=1
            if keep:current=candidate;accepted+=1
        assert len(row['trials'])<=20 and row['accepted_deletions']==accepted
        assert digest(current)==row['new_episode_sha256']==digest(row['episode']) and current['reward']==1
        assert current['steps']==row['cleaned_steps']<=row['original_steps']==original['steps']
        if len(row['trials'])<20:
            assert not [(i,j) for i,j in candidates(current) if (digest(current),i,j) not in attempted]
        # Fresh environment reconstruction produces all messages and labels.
        actions=action_rows(DATA/row['game'],current);env_steps+=len(actions);wins.append(row['index'])
        annotation={'stream_index':row['index'],'game':row['game'],'game_sha256':row['game_sha256'],
            'original_episode_sha256':row['episode_sha256'],'episode_sha256':digest(current),'actions_sha256':digest(actions),
            'cleaning_report_sha256':file_hash(report_path)}
        annotation={**annotation,'input_content_sha256':digest(annotation),'reviewed_target':True};annotations.append(annotation)
        for step,action in enumerate(actions):
            if step<3:continue
            before=current['initial_observation'] if step==0 else current['trajectory'][step-1]['observation']
            assert before==action['current_observation'] and action['target_action']==current['trajectory'][step]['command']
            content={'stream_index':row['index'],'game':row['game'],'game_sha256':row['game_sha256'],
                'episode_sha256':digest(current),'parent_annotation_sha256':annotation['input_content_sha256'],
                'target_step':step,'actor_messages':compact_messages(action['target_messages'],history_turns=2),
                'target_action':action['target_action'],'initial_observation':current['initial_observation'],
                'current_observation':before,'observed_feedback_after_action':current['trajectory'][step]['observation'],
                'visible_prefix_sha256':digest(current['trajectory'][:step])}
            targets.append({**content,'input_content_sha256':digest(content),'reviewed_target':True})
        print(json.dumps({'index':row['index'],'sources_reviewed':len(wins),'new_targets':len(targets)}),flush=True)
    result={'cleaning_report_sha256':file_hash(report_path),'cleaning_review_sha256':file_hash(CLEAN_REVIEW),
        'runner_sha256':file_hash(Path(__file__)),'dependencies':{n:file_hash(Path(__file__).with_name(n)) for n in ['alfworld_bootstrap_own_success_v43.py','train_alf_next_task.py','clean_success_loops_v87.py']},
        'own_success_indices':wins,'annotations':annotations,'targets':targets,'targets_sha256':digest(targets),
        'actor_history_turns':2,'minimum_target_step':3,'annotation_environment_replays':28,'annotation_environment_steps':env_steps,
        'scope':'New labels/prompts reconstructed from cleaned trajectories through official environment replay, no labels inherited by history ID. Existing common bootstrap remains a separate unchanged training artifact.'}
    save(OUTPUT,result)
    save(AUDIT,{'passed':True,'report_sha256':file_hash(report_path),'new_annotation_sha256':file_hash(OUTPUT),
        'episodes_replayed':28,'candidate_chains_checked':trials_checked,'replay_env_steps':env_steps,
        'scope':'Reconstruct deletion decisions and final accepted chains, independently replay all final trajectories and generate new action labels. Rejected candidates are structurally checked, not independently rerun.'})
    print(json.dumps({'reviewed_sources':28,'new_targets':len(targets),'candidate_chains_checked':trials_checked}),flush=True)

if __name__=='__main__':main()
