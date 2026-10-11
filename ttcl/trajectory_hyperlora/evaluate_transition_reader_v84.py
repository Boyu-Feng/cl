"""Two new single-attempt arms for semantic centered H; v82 supplies fixed controls."""
import argparse
import json
from pathlib import Path
import torch
from ttcl.trajectory_hyperlora.train_transition_reader_v84 import checked_cache, ROOT as CANDIDATE_ROOT
from ttcl.trajectory_hyperlora.prepare_transition_reader_v82 import REVIEW as TRAIN_REVIEW,BOOT
from ttcl.trajectory_hyperlora.cache_transition_reader_v82 import ROOT,MODEL,FIELDS,encode
from ttcl.trajectory_hyperlora.transition_reader_v84 import TransitionReader
from ttcl.trajectory_hyperlora.alfworld_online_from_empty import digest,file_hash
from ttcl.trajectory_hyperlora.alfworld_online_accumulate_lora_v1 import factors_hash
from ttcl.trajectory_hyperlora.alfworld_persistent_failure_delta_v36 import save
from ttcl.trajectory_hyperlora.alfworld_zero_shot_probe import load_agent,run_episode

ROOT=CANDIDATE_ROOT
REVIEW=Path('data/annotations/alfworld_transition_reader_v84_eval_20261010.json')
REFERENCE_REVIEW=Path('data/annotations/alfworld_transition_reader_v82_eval_20261010.json')
OUTPUT=Path('results/trajectory_hyperlora/alfworld_transition_reader_v84_eval_20261010.json')
DATA=Path('ttcl/data/alfworld_delta')
ARMS=('real','shifted')


def bindings():
    train,_,cache,_=checked_cache()
    report=json.loads((ROOT/'training.json').read_text());manifest=json.loads((ROOT/'training_manifest.json').read_text())
    assert report['complete'] and not report['failures']
    assert report['manifest_sha256']==file_hash(ROOT/'training_manifest.json')
    assert manifest['runner_sha256']==file_hash(Path(__file__).with_name('train_transition_reader_v84.py'))
    assert manifest['reader_sha256']==file_hash(Path(__file__).with_name('transition_reader_v84.py'))
    assert manifest['reader_parent_sha256']==file_hash(Path(__file__).with_name('transition_reader_v82.py'))
    assert manifest['top_k']==8 and manifest['temperature']==.05
    assert report['initial_reader_sha256']==file_hash(ROOT/'initial_reader.pt')
    for arm in ('hyper','static'):
        groups=[g for g in report['groups'] if g['arm']==arm]
        assert len(groups)==len(train['groups'])
        for g,t in zip(groups,train['groups'],strict=True):
            assert g['group_binding']==t['input_content_sha256'] and g['order_sha256']==digest(t['update_order'])
            assert g['steps']==len(t['update_order'])
        assert sum(g['steps'] for g in groups)==400
    for name in ('reader','static'):assert report[name+'_sha256']==file_hash(ROOT/(name+'.pt'))
    old=json.loads(Path('data/annotations/alfworld_causal_prefix_v70_eval_20261010.json').read_text())
    reference=json.loads(REFERENCE_REVIEW.read_text())
    assert reference['static_sha256']==report['static_sha256']
    prior=json.loads(Path('results/trajectory_hyperlora/alfworld_transition_reader_v82_20261010/training.json').read_text())
    assert prior['shared_A_sha256']==report['shared_A_sha256'] and prior['shared_B_sha256']==report['shared_B_sha256']
    targets=[]
    for t in old['targets']:
        assert file_hash(DATA/t['game'])==t['game_sha256']
        item={k:v for k,v in t.items() if k not in ('input_content_sha256','reviewed_target')}
        item['training_sha256']=file_hash(ROOT/'training.json')
        targets.append({**item,'input_content_sha256':digest(item),'reviewed_target':True})
    indices=train['evaluation_memory_indices'];assert indices and all(train['transitions'][k]['source_index']<120 for k in indices)
    return {'targets':targets,'memory_indices':indices,'memory_bindings':[train['transitions'][k]['input_content_sha256'] for k in indices],
        'reference_review_sha256':file_hash(REFERENCE_REVIEW),
        'training_sha256':file_hash(ROOT/'training.json'),'train_review_sha256':file_hash(TRAIN_REVIEW),
        'reader_sha256':file_hash(ROOT/'reader.pt'),'static_sha256':file_hash(ROOT/'static.pt'),
        'runner_sha256':file_hash(Path(__file__)),'rollout_sha256':file_hash(Path(__file__).with_name('alfworld_zero_shot_probe.py')),
        'model_weights':old['model_weights'],'arms':list(ARMS),'max_steps':30,'max_new_tokens':64,'history_turns':2,'loop_guard_max':2,
        'shifted':'Keep observation keys fixed; jointly rotate action and feedback value rows by one.',
        'scope':'Two new frozen-H alternatives on exposed next24; v82 exact common bootstrap/static/base controls will be joined only after both audits. No new control rollouts. Labels and optimizer budget unchanged; frozen cosine selection plus centered values, action CE only.'}


def evaluate(device,review):
    if OUTPUT.exists():raise FileExistsError(OUTPUT)
    for p,h in review['model_weights'].items():assert file_hash(Path(p))==h
    _,_,cache,_=checked_cache()
    agent,tok=load_agent(MODEL,BOOT.parent/'hyper.pt',device,.6)
    agent.requires_grad_(False);agent.eval();agent.set_source(None)
    state=torch.load(ROOT/'reader.pt',map_location=device,weights_only=True)
    static=torch.load(ROOT/'static.pt',map_location=device,weights_only=True)
    assert state['top_k']==8 and state['temperature']==.05
    reader=TransitionReader(state['feature_dim'],state['factor_shapes'],state['width'],state['top_k'],state['temperature']).to(device)
    reader.load_state_dict(state['state']);reader.requires_grad_(False);reader.eval()
    shared=state['shared_B'];bases=state['shared_A']
    assert state['bootstrap_sha256']==static['bootstrap_sha256']==file_hash(BOOT)
    memory=[]
    for field in FIELDS:memory.append(torch.cat([cache['values'][cache['rows'][k][field]] for k in review['memory_indices']],dim=0).to(device))
    obs,act,fb=memory;query_cache={}
    result={'complete':False,'review_sha256':file_hash(REVIEW),'rows':[],'failures':[]}
    try:
        for target in review['targets']:
            row={'index':target['index'],'game':target['game'],'input_content_sha256':target['input_content_sha256'],'arms':{}}
            for arm in ARMS:
                agent.set_source(None)
                with torch.no_grad():
                    for layer,a in zip(agent.adapters,static['A'] if arm=='static' else bases,strict=True):layer.a.copy_(a.to(layer.a))
                reads=[]
                def reader_hook(initial,current,available):
                    # Called before each action; returns no textual hint.
                    text='Task initial observation:\n'+initial+'\nObservation before action:\n'+current
                    key=digest(text)
                    if key not in query_cache:query_cache[key]=encode(agent,tok,text,device,512)
                    q,token_binding=query_cache[key]
                    with torch.no_grad():
                        values,info=reader.adapted_factors(q.to(device),obs,
                            act.roll(1,0) if arm=='shifted' else act,
                            fb.roll(1,0) if arm=='shifted' else fb,shared,bases)
                    for layer,b in zip(agent.adapters,values,strict=True):layer.b=b
                    weights=info['weights'].detach().cpu()[0]
                    reads.append({'query_text_sha256':key,**token_binding,'B_sha256':factors_hash(values),
                        'attention_weights':weights.tolist(),'clip_scale':float(info['scale']),
                        'effective_relative_norm':float(info['raw_effective_norm']*info['scale']/info['reference_effective_norm'])})
                    return ''
                fixed=shared if arm=='empty' else static['factors'] if arm=='static' else None
                episode=run_episode(agent,tok,DATA/target['game'],{},adapter=False,fixed_adapter=fixed,
                    evidence_provider=reader_hook if arm in ('real','shifted') else None,
                    device=device,max_steps=30,max_new_tokens=64,constrain_actions=True,actor_history_turns=2,loop_guard_max=2)
                assert episode['status']=='complete' and digest(episode['initial_observation'])==target['initial_observation_sha256']
                if arm in ('real','shifted'):assert len(reads)==episode['steps']
                episode['parameter_reads']=reads;row['arms'][arm]=episode
                save(OUTPUT,{**result,'in_progress_row':row})
            result['rows'].append(row)
            result['summary']={'n':len(result['rows']),**{a:sum(r['arms'][a]['reward'] for r in result['rows']) for a in ARMS}}
            save(OUTPUT,result);print(json.dumps(result['summary']),flush=True)
        torch.save(query_cache,ROOT/'eval_query_features.pt')
        result.update(complete=True,query_features_sha256=file_hash(ROOT/'eval_query_features.pt'));save(OUTPUT,result)
    except Exception as exc:
        result['failures'].append({'error':repr(exc),'partial_row':row});save(OUTPUT,result);raise

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('prepare','evaluate'));p.add_argument('--device',default='cuda:6');args=p.parse_args()
    review=bindings()
    if args.command=='prepare':
        if REVIEW.exists():raise FileExistsError(REVIEW)
        save(REVIEW,review)
    else:
        assert json.loads(REVIEW.read_text())==review
        evaluate(args.device,review)
