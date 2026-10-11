"""Frozen single-attempt next24 test of training-screened parameter experience.

Five independent alternatives; no same-task retry or outcome-based selection.
Lexical retrieval is an explicit diagnostic, not a trained hypernetwork.
"""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import re
import torch
from ttcl.trajectory_hyperlora.evaluate_cross_task_updates_v79 import prepare as teacher_bindings, BOOT, TEACHERS, OUTPUT as TEACHER_REPORT, PARENT
from ttcl.trajectory_hyperlora.alfworld_online_from_empty import digest,file_hash
from ttcl.trajectory_hyperlora.alfworld_online_accumulate_lora_v1 import factors_hash
from ttcl.trajectory_hyperlora.alfworld_persistent_failure_delta_v36 import save
from ttcl.trajectory_hyperlora.alfworld_zero_shot_probe import load_agent,run_episode
from ttcl.trajectory_hyperlora.audit_my_online_update_v28_reward import replay

ARMS=('bootstrap','retrieved','rotated','random','static')
REVIEW=Path('data/annotations/alfworld_transfer_bank_v80_20261010.json')
OUTPUT=Path('results/trajectory_hyperlora/alfworld_transfer_bank_v80_20261010.json')
MODEL=Path('current_work/delta-Mem/model/Qwen3-4B-Instruct-2507')
DATA=Path('ttcl/data/alfworld_delta')
STATIC=BOOT.parent/'static.pt'


def similarities(query,documents):
    # Bag of visible initial-text words. No answer, family/path or reward input.
    tokenize=lambda s:Counter(re.findall(r'[a-z]+',s.lower()))
    docs=[tokenize(x) for x in documents]; q=tokenize(query)
    vocab=set().union(*docs,q)
    idf={w:1+math.log((1+len(docs))/(1+sum(w in d for d in docs))) for w in vocab}
    vector=lambda c:{w:(1+math.log(n))*idf[w] for w,n in c.items()}
    q=vector(q); qn=math.sqrt(sum(v*v for v in q.values()))
    values=[]
    for d in docs:
        d=vector(d);dn=math.sqrt(sum(v*v for v in d.values()))
        values.append(sum(v*d.get(w,0.) for w,v in q.items())/(qn*dn) if qn*dn else 0.)
    return values


def bindings():
    prior,_=teacher_bindings()
    result=json.loads(TEACHER_REPORT.read_text()); audit=json.loads(TEACHER_REPORT.with_suffix('.audit.json').read_text())
    assert audit['passed'] and audit['report_sha256']==file_hash(TEACHER_REPORT)
    selected=[r['source_index'] for r in result['rows'] if r['positive']]
    assert selected
    parent=json.loads(PARENT.read_text())
    bank=[]
    for i in selected:
        pair=next(p for p in prior['pairs'] if p['source_index']==i)
        artifact=TEACHERS/f'teacher_{i:03d}.pt'
        bank.append({'source_index':i,'artifact':str(artifact),'artifact_sha256':file_hash(artifact),
                     'source_binding':pair['source_binding'],
                     'initial':parent['rows'][i]['episode']['initial_observation']})
    schedule=json.loads(Path('data/annotations/alfworld_causal_prefix_v70_eval_20261010.json').read_text())
    targets=[]
    for original in schedule['targets']:
        i=original['index']; text=parent['rows'][i]['episode']['initial_observation']
        assert digest(text)==original['initial_observation_sha256']
        assert file_hash(DATA/original['game'])==original['game_sha256']
        assert i>=120 and all(b['source_index']<120 for b in bank)
        scores=similarities(text,[b['initial'] for b in bank])
        chosen=max(range(len(bank)),key=lambda k:scores[k])
        c={k:v for k,v in original.items() if k not in ('reviewed_target','input_content_sha256')}
        c.update(initial=text,selected_bank_index=chosen,rotated_bank_index=(chosen+1)%len(bank),similarities=scores)
        targets.append({**c,'input_content_sha256':digest(c),'reviewed_target':True})
    deps=['evaluate_transfer_bank_v80.py','evaluate_cross_task_updates_v79.py','alfworld_zero_shot_probe.py','direct_composition_pilot.py','audit_my_online_update_v28_reward.py']
    return {'bank':bank,'targets':targets,'arms':ARMS,'teacher_audit_sha256':file_hash(TEACHER_REPORT.with_suffix('.audit.json')),
        'bootstrap_sha256':file_hash(BOOT),'static_sha256':file_hash(STATIC),
        'model_weights':schedule['model_weights'],'model_config_sha256':file_hash(MODEL/'config.json'),
        'dependencies':{n:file_hash(Path(__file__).with_name(n)) for n in deps},
        'budget':{'bootstrap_steps':300,'candidate_steps':380,'static_additional_steps':400,'new_optimization_steps':0},
        'scope':'Frozen exposed-next24 single-attempt diagnostic; screened bank uses extra training-query feedback unavailable to static comparator, not a strict equal-information superiority test',
        'protocol':{'max_steps':30,'max_new_tokens':64,'history_turns':2,'loop_guard_max':2,'constrain_actions':True,'activation':'All arms active from first action'}}


def evaluate(device,review):
    if OUTPUT.exists():raise FileExistsError(OUTPUT)
    for p,h in review['model_weights'].items():assert file_hash(Path(p))==h
    agent,tok=load_agent(MODEL,BOOT.parent/'hyper.pt',device,.6)
    agent.requires_grad_(False);agent.eval();agent.set_source(None)
    boot=torch.load(BOOT,map_location=device,weights_only=True)
    static=torch.load(STATIC,map_location=device,weights_only=True)
    assert static['bootstrap_sha256']==review['bootstrap_sha256']
    states=[]
    for item in review['bank']:
        state=torch.load(item['artifact'],map_location=device,weights_only=True)
        assert state['A_sha256']==factors_hash(boot['A']) and state['bootstrap_sha256']==review['bootstrap_sha256']
        assert state['group_binding']==item['source_binding'];states.append(state)
    result={'complete':False,'review_sha256':file_hash(REVIEW),'rows':[],'failures':[]}
    try:
        for t in review['targets']:
            row={'index':t['index'],'game':t['game'],'input_content_sha256':t['input_content_sha256'],'arms':{}}
            for arm in ARMS:
                basis=static['A'] if arm=='static' else boot['A']
                with torch.no_grad():
                    for layer,a in zip(agent.adapters,basis,strict=True):layer.a.copy_(a.to(layer.a))
                values=static['B'] if arm=='static' else boot['B']
                if arm in ('retrieved','rotated','random'):
                    index=t['rotated_bank_index'] if arm=='rotated' else t['selected_bank_index']
                    delta=states[index]['random_delta' if arm=='random' else 'delta']
                    values=[b+d for b,d in zip(boot['B'],delta,strict=True)]
                ep=run_episode(agent,tok,DATA/t['game'],{},adapter=False,fixed_adapter=values,device=device,max_steps=30,max_new_tokens=64,constrain_actions=True,actor_history_turns=2,loop_guard_max=2)
                assert ep['status']=='complete' and digest(ep['initial_observation'])==t['initial_observation_sha256']
                row['arms'][arm]=ep;save(OUTPUT,{**result,'in_progress_row':row})
            result['rows'].append(row)
            result['summary']={'n':len(result['rows']),**{a:sum(r['arms'][a]['reward'] for r in result['rows']) for a in ARMS}}
            save(OUTPUT,result);print(json.dumps(result['summary']),flush=True)
        result['complete']=True;save(OUTPUT,result)
    except Exception as exc:
        result['failures'].append({'error':repr(exc),'partial_row':row});save(OUTPUT,result);raise


def audit(review):
    x=json.loads(OUTPUT.read_text());assert x['complete'] and not x['failures']
    assert x['review_sha256']==file_hash(REVIEW)
    for row,t in zip(x['rows'],review['targets'],strict=True):
        assert row['index']==t['index'] and row['game']==t['game'] and row['input_content_sha256']==t['input_content_sha256']
        assert set(row['arms'])==set(ARMS)
        for ep in row['arms'].values():
            assert ep['status']=='complete' and ep['steps']<=30
            assert digest(ep['initial_observation'])==t['initial_observation_sha256']
            replay(DATA/t['game'],ep)
    summary={'n':len(x['rows']),**{a:sum(r['arms'][a]['reward'] for r in x['rows']) for a in ARMS}}
    assert summary==x['summary']
    save(OUTPUT.with_suffix('.audit.json'),{'passed':True,'report_sha256':file_hash(OUTPUT),'summary':summary,
        'scope':'Reviewed source/target bindings and environment replay; no independent neural action regeneration',
        'episodes_replayed':len(x['rows'])*len(ARMS)})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('prepare','evaluate','audit'));p.add_argument('--device',default='cuda:3');args=p.parse_args()
    review=bindings()
    # JSON-normalize tuple arm names before exact binding comparison.
    review=json.loads(json.dumps(review))
    if args.command=='prepare':
        if REVIEW.exists():raise FileExistsError(REVIEW)
        save(REVIEW,review);print(json.dumps({'bank':len(review['bank']),'tasks':len(review['targets'])}))
    else:
        assert json.loads(REVIEW.read_text())==review
        if args.command=='evaluate':evaluate(args.device,review)
        else:audit(review)
