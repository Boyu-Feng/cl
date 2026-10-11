"""Remove repeated segments from own TRAIN successes, verify official reward.

Extra environment trials are explicitly counted. No new model generations,
expert actions, dev/test trajectories, failure corrections or training updates.
Changed trajectories require newly bound action annotations before training.
"""
import json
from pathlib import Path
from ttcl.trajectory_hyperlora.prepare_transition_reader_v82 import build,PARENT,REVIEW as TRAIN_REVIEW
from ttcl.trajectory_hyperlora.alfworld_online_from_empty import digest,file_hash
from ttcl.trajectory_hyperlora.alfworld_persistent_failure_delta_v36 import save
from ttcl.trajectory_hyperlora.audit_my_online_update_v28_reward import replay
from ttcl.experience_evolution.environment import make_env
ROOT=Path('results/trajectory_hyperlora/alfworld_success_loop_cleaning_v87_20261010')
REVIEW=Path('data/annotations/alfworld_success_loop_cleaning_v87_20261010.json')
DATA=Path('ttcl/data/alfworld_delta')


def candidates(ep):
    steps=ep['trajectory'];before=[ep['initial_observation']]+[x['observation'] for x in steps[:-1]]
    return sorted([(i,j) for i in range(len(steps)) for j in range(i+1,len(steps))
        if before[i]==before[j] and steps[i]['command']==steps[j]['command']],key=lambda x:(-(x[1]-x[0]),x[0],x[1]))


def execute(game,commands,expected_initial):
    env=make_env(game);trajectory=[]
    try:
        state=env.reset();initial=str(state['feedback']);assert initial==expected_initial
        for command in commands:
            if command not in state['admissible_commands']:
                return {'status':'rejected_invalid_action','reward':0.,'steps':len(trajectory),'trajectory':trajectory,'invalid_command':command}
            state,_,done=env.step(command)
            trajectory.append({'turn':len(trajectory),'response':command,'command':command,'valid':True,
                'observation':str(state['feedback']),'won':bool(state['won'])})
            if done or state['won']:break
        return {'status':'complete','reward':float(bool(state['won'])),'steps':len(trajectory),'initial_observation':initial,
            'trajectory':trajectory,'invalid_commands':0,'termination':'success' if state['won'] else 'budget_or_environment_done'}
    finally:env.close()


def main():
    if ROOT.exists() or REVIEW.exists():raise FileExistsError('Use fresh frozen output')
    train,_=build();assert train==json.loads(TRAIN_REVIEW.read_text());parent=json.loads(PARENT.read_text())
    sources=[r for r in parent['rows'][:120] if r['episode']['reward']==1]
    assert len(sources)==28
    items=[{'index':r['index'],'game':r['game'],'game_sha256':file_hash(DATA/r['game']),
        'episode_sha256':digest(r['episode'])} for r in sources]
    review={'sources':items,'parent_sha256':file_hash(PARENT),'parent_audit_sha256':file_hash(PARENT.with_suffix('.audit.json')),
        'train_review_sha256':file_hash(TRAIN_REVIEW),'candidate_trial_cap_per_source':20,
        'selection':'Longest repeated observation+next-command segment first; accept only strictly shorter official success; recompute candidates after acceptance',
        'dependencies':{n:file_hash(Path(__file__).with_name(n)) for n in ['clean_success_loops_v87.py','prepare_transition_reader_v82.py','audit_my_online_update_v28_reward.py']},
        'environment_sha256':file_hash(Path('ttcl/experience_evolution/environment.py')),
        'training_updates':0,'model_generations':0,'scope':'Own train-prefix successes only; extra environment verification trials up to560 plus28 final replays. Output is new trajectory data, not inherited reviewed action labels.'}
    save(REVIEW,review);ROOT.mkdir();result={'complete':False,'review_sha256':file_hash(REVIEW),'rows':[],'failures':[]}
    try:
        for source,binding in zip(sources,items,strict=True):
            current=source['episode'];trials=[];accepted=0;attempted=set()
            while len(trials)<20:
                options=[(i,j) for i,j in candidates(current) if (digest(current),i,j) not in attempted]
                if not options:break
                i,j=options[0];before_hash=digest(current);attempted.add((before_hash,i,j))
                commands=[x['command'] for x in current['trajectory']];candidate_commands=commands[:i]+commands[j:]
                candidate=execute(DATA/source['game'],candidate_commands,current['initial_observation'])
                keep=candidate['status']=='complete' and candidate['reward']==1 and candidate['steps']<current['steps']
                trials.append({'parent_episode_sha256':before_hash,'delete_start':i,'delete_end_exclusive':j,
                    'candidate_commands':candidate_commands,'candidate_episode':candidate,'accepted':keep})
                if keep:current=candidate;accepted+=1
            # Every retained source is independently replayed, including unchanged ones.
            replay(DATA/source['game'],current)
            row={**binding,'original_steps':source['episode']['steps'],'cleaned_steps':current['steps'],
                'accepted_deletions':accepted,'trials':trials,'episode':current,'new_episode_sha256':digest(current),
                'needs_new_action_annotation':digest(current)!=binding['episode_sha256']}
            result['rows'].append(row)
            result['summary']={'sources':len(result['rows']),'shortened':sum(x['cleaned_steps']<x['original_steps'] for x in result['rows']),
                'original_steps':sum(x['original_steps'] for x in result['rows']),'cleaned_steps':sum(x['cleaned_steps'] for x in result['rows']),
                'candidate_trials':sum(len(x['trials']) for x in result['rows']),
                'candidate_env_steps':sum(t['candidate_episode']['steps'] for x in result['rows'] for t in x['trials']),
                'final_replay_episodes':len(result['rows']),'final_replay_env_steps':sum(x['cleaned_steps'] for x in result['rows'])}
            save(ROOT/'report.json',result);print(json.dumps({'index':source['index'],**result['summary']}),flush=True)
        result['complete']=True;save(ROOT/'report.json',result)
    except Exception as exc:
        result['failures'].append(repr(exc));save(ROOT/'report.json',result);raise

if __name__=='__main__':main()
