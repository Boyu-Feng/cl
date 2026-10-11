"""Independent provenance, causal update-chain, and scoring audit for v94b.

Does not regenerate neural outputs or repeat environment executions; verifies
saved environment traces, source bindings and temporal dependencies.
"""
import argparse,json
from pathlib import Path
from ttcl.trajectory_hyperlora.joint_domains_v93 import read,write,digest,filehash
from ttcl.trajectory_hyperlora.run_continual_xland_v94 import ROOT,ARMS,MILESTONES

def audit(folder,partial=False):
    plan=read(folder/'plan.json');recovery=read(folder/'recovery.json')
    old=Path(recovery['original_directory'])
    for f,h in recovery['source_hashes'].items():assert filehash(old/f)==h, f
    assert filehash(ROOT/'ttcl/trajectory_hyperlora/recover_continual_xland_v94.py')==recovery['recovery_code_sha256']
    for f,h in plan['code_freeze'].items():assert filehash(ROOT/'ttcl/trajectory_hyperlora'/f)==h,f
    collection=read(folder/'collection.json');assert collection==read(old/'collection.json')
    assert collection['complete']
    assert not ({t['ruleset_id'] for t in plan['train']}&{t['ruleset_id'] for t in plan['test']})
    def check_ep(ep):
        assert digest({k:v for k,v in ep.items() if k!='content_sha256'})==ep['content_sha256']
        traj=ep['trajectory'];assert 0<len(traj)<=plan['max_steps']
        assert ep['environment_replay_passed']
        assert ep['return']==sum(s['reward'] for s in traj)
        assert ep['positive']==any(s['reward']>0 for s in traj)
        for i,s in enumerate(traj):
            assert 0<=s['action']<6
            assert s['observation']==(ep['initial']['observation'] if i==0 else traj[i-1]['next_observation'])
            if i<len(traj)-1:assert not s['done']
    zero_hash=None
    for split,n,offset in [('train',4,940000),('test',7,1940000)]:
        assert len(collection[split])==8
        for ti,eps in enumerate(collection[split]):
            assert len(eps)==n
            for j,ep in enumerate(eps):
                check_ep(ep);assert ep['ruleset_id']==plan[split][ti]['ruleset_id']
                assert ep['seed']==offset+int(ep['ruleset_id'])*1009+j
                zero_hash=zero_hash or ep['b_before_sha256'];assert ep['b_before_sha256']==zero_hash
    history=read(folder/'training.json');targets=read(folder/'training_targets.json')['rows']
    if not partial:assert len(history)==96
    for i,r in enumerate(history):
        ti=i%8;assert r['update']==i+1 and r['task']==ti
        assert r['source_hashes']==[e['content_sha256'] for e in collection['train'][ti]]
        assert len(r['query_hashes'])==4
        valid={q['input_content_sha256'] for q in targets[ti]}
        assert set(r['query_hashes'])<=valid
    if not (folder/'evaluation.json').exists():
        assert partial
        return dict(passed=True,partial=True,training_updates=len(history),collection_episodes=88,evaluation_runs=0)
    report=read(folder/'evaluation.json');assert not report['failures']
    assert report['plan_sha256']==filehash(folder/'plan.json')
    assert report['trained_sha256']==filehash(folder/'trained.pt')
    if not partial:assert report['complete']
    tasks={t['task_id']:i for i,t in enumerate(plan['test'])}
    source_map={ep['content_sha256']:ep for g in collection['test'] for ep in g}
    source_map.update({r['episode']['content_sha256']:r['episode'] for r in report['onpolicy']})
    updates={}
    for u in report['updates']:
        key=(u['stage'],u['task_id'],u['arm'],u['after_episode']);assert key not in updates
        updates[key]=u;ti=tasks[u['task_id']];k=u['after_episode']
        assert 1<=k<=7
        if u['arm'] in ('hyper','reset','shifted'):
            if u['stage']=='shared':
                expected=collection['test'][(ti+1)%8 if u['arm']=='shifted' else ti][k-1]
            else:
                expected=next(r['episode'] for r in report['onpolicy'] if r['task_id']==u['task_id'] and r['arm']==u['arm'] and r['attempt_number']==k)
            assert u['source_hash']==expected['content_sha256'] and u['updates']==1
            if u['arm']=='reset':assert u['before']==zero_hash
        else:
            if u['stage']=='shared':eligible=collection['test'][ti][max(0,k-4):k]
            else:eligible=[r['episode'] for r in report['onpolicy'] if r['task_id']==u['task_id'] and r['arm']==u['arm'] and max(1,k-3)<=r['attempt_number']<=k]
            hashes={e['content_sha256'] for e in eligible}
            has_reward=any(e['positive'] for e in eligible)
            assert u['updates']==(8 if has_reward else 0)
            if not has_reward:assert u['before']==u['after']
            for q in u.get('targets',[]):
                assert q['episode_sha256'] in hashes
                ep=source_map[q['episode_sha256']];i=q['step'];s=ep['trajectory'][i]
                assert q['observation']==s['observation'] and q['action']==s['action']
                assert any(s['reward']>0 for s in ep['trajectory'][i:i+4])
                content={k:q[k] for k in ('observation','action','episode_sha256','step')}
                assert digest(content)==q['input_content_sha256'] and q['reviewed_target']
        if k>1 and u['arm']!='reset':
            assert u['before']==updates[(u['stage'],u['task_id'],u['arm'],k-1)]['after']
        elif k==1 and u['arm']!='online_prior':assert u['before']==zero_hash
    summary={}
    for stage in ('shared','onpolicy'):
        seen=set();summary[stage]={}
        for r in report[stage]:
            a=r['arm'];k=r['attempt_number'];task=r['task_id'];ep=r['episode']
            key=(task,a,k);assert key not in seen;seen.add(key);check_ep(ep)
            assert ep['ruleset_id']==plan['test'][tasks[task]]['ruleset_id']
            if stage=='shared':assert k-1 in MILESTONES and r['observed_episodes']==k-1
            assert ep['seed']==(2940000 if stage=='shared' else 3940000)+int(ep['ruleset_id'])*1009+k-1
            if a=='base' or (k==1 and a!='online_prior'):assert ep['b_before_sha256']==zero_hash
            elif k>1:assert ep['b_before_sha256']==updates[(stage,task,a,k-1)]['after']
            for metric in ('positive','return'):
                group=summary[stage].setdefault(a,{}).setdefault(str(k),dict(n=0,positive=0,return_sum=0))
                if metric=='positive':group['n']+=1;group['positive']+=int(ep['positive'])
                else:group['return_sum']+=ep['return']
        if not partial:assert len(seen)==192
    return dict(passed=True,partial=partial,training_updates=len(history),collection_episodes=88,evaluation_runs=len(report['shared'])+len(report['onpolicy']),updates_verified=len(updates),summary=summary,report_sha256=filehash(folder/'evaluation.json'),scope='Saved content, scoring and causal chain verification; existing environment replay assertions; no independent neural or environment rerun')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--partial',action='store_true');p.add_argument('--folder',type=Path,default=ROOT/'results/trajectory_hyperlora/continual_xland_v94b_20261010');a=p.parse_args()
    result=audit(a.folder,a.partial);write(a.folder/('partial_audit.json' if a.partial else 'audit.json'),result)
    print(json.dumps({k:v for k,v in result.items() if k!='summary'},ensure_ascii=False))
