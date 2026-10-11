"""Isolate visible action-summary benefit before changing experience training."""
import argparse
import json
from pathlib import Path
import torch
from ttcl.trajectory_hyperlora.alfworld_zero_shot_probe import load_agent
from ttcl.trajectory_hyperlora.rollout_action_state_v85 import run_episode
from ttcl.trajectory_hyperlora.visible_action_state import action_state_text
from ttcl.trajectory_hyperlora.alfworld_online_from_empty import digest,file_hash
from ttcl.trajectory_hyperlora.alfworld_persistent_failure_delta_v36 import save
from ttcl.trajectory_hyperlora.audit_my_online_update_v28_reward import replay
from ttcl.trajectory_hyperlora.cache_transition_reader_v82 import MODEL
from ttcl.trajectory_hyperlora.prepare_transition_reader_v82 import BOOT
ROOT=Path('results/trajectory_hyperlora')
PARENT=ROOT/'alfworld_transition_reader_v82_eval_20261010.json'
PARENT_REVIEW=Path('data/annotations')/PARENT.name
OUTPUT=ROOT/'alfworld_action_state_v85_20261010.json'
REVIEW=Path('data/annotations')/OUTPUT.name
STATIC=ROOT/'alfworld_transition_reader_v82_20261010/static.pt'
DATA=Path('ttcl/data/alfworld_delta')
ARMS=('base','static')


def bindings():
    old=json.loads(PARENT.read_text());audit=json.loads(PARENT.with_suffix('.audit.json').read_text());prior=json.loads(PARENT_REVIEW.read_text())
    assert old['complete'] and not old['failures'] and audit['passed'] and audit['episodes_replayed']==120
    assert audit['report_sha256']==file_hash(PARENT) and old['review_sha256']==file_hash(PARENT_REVIEW)
    assert prior['static_sha256']==file_hash(STATIC)
    targets=[]
    for t in prior['targets']:
        assert file_hash(DATA/t['game'])==t['game_sha256']
        item={k:v for k,v in t.items() if k not in ('input_content_sha256','reviewed_target')}
        item['state_module_sha256']=file_hash(Path(__file__).with_name('visible_action_state.py'))
        targets.append({**item,'input_content_sha256':digest(item),'reviewed_target':True})
    return {'targets':targets,'parent_sha256':file_hash(PARENT),'parent_audit_sha256':file_hash(PARENT.with_suffix('.audit.json')),
        'parent_review_sha256':file_hash(PARENT_REVIEW),'static_sha256':file_hash(STATIC),'loader_checkpoint_sha256':file_hash(BOOT.parent/'hyper.pt'),
        'model_weights':prior['model_weights'],'arms':list(ARMS),'max_steps':30,'max_new_tokens':64,'history_turns':2,'loop_guard_max':2,
        'max_prompt_tokens':4096,'training_updates':0,
        'dependencies':{n:file_hash(Path(__file__).with_name(n)) for n in ['evaluate_action_state_v85.py','rollout_action_state_v85.py','visible_action_state.py','alfworld_zero_shot_probe.py','train_alf_next_task.py']},
        'scope':'Exposed24 diagnostic; current-attempt past action summary given equally to frozen base and fixed direct LoRA. Compared with audited same-checkpoint no-summary arms. Extra prompt tokens, no equal-compute claim; no hypernetwork or trajectory-training gain claim.'}


def evaluate(device):
    if OUTPUT.exists() or REVIEW.exists():raise FileExistsError('Use new result paths')
    review=bindings();save(REVIEW,review)
    for p,h in review['model_weights'].items():assert file_hash(Path(p))==h
    agent,tok=load_agent(MODEL,BOOT.parent/'hyper.pt',device,.6);agent.requires_grad_(False);agent.eval()
    static=torch.load(STATIC,map_location=device,weights_only=True)
    result={'complete':False,'review_sha256':file_hash(REVIEW),'rows':[],'failures':[]}
    try:
        for t in review['targets']:
            row={'index':t['index'],'game':t['game'],'input_content_sha256':t['input_content_sha256'],'arms':{}}
            for arm in ARMS:
                agent.set_source(None)
                with torch.no_grad():
                    for layer,a in zip(agent.adapters,static['A'],strict=True):layer.a.copy_(a.to(layer.a))
                ep=run_episode(agent,tok,DATA/t['game'],{},adapter=False,device=device,max_steps=30,max_new_tokens=64,
                    constrain_actions=True,fixed_adapter=static['factors'] if arm=='static' else None,actor_history_turns=2,loop_guard_max=2)
                assert ep['status']=='complete' and digest(ep['initial_observation'])==t['initial_observation_sha256']
                row['arms'][arm]=ep;save(OUTPUT,{**result,'in_progress_row':row})
            result['rows'].append(row);result['summary']={'n':len(result['rows']),**{a:sum(r['arms'][a]['reward'] for r in result['rows']) for a in ARMS}}
            save(OUTPUT,result);print(json.dumps(result['summary']),flush=True)
        result['complete']=True;save(OUTPUT,result)
    except Exception as exc:
        result['failures'].append(repr(exc));save(OUTPUT,result);raise


def audit():
    review=bindings();assert review==json.loads(REVIEW.read_text());d=json.loads(OUTPUT.read_text())
    assert d['complete'] and not d['failures'] and d['review_sha256']==file_hash(REVIEW)
    checked=0;reads=0
    for r,t in zip(d['rows'],review['targets'],strict=True):
        assert r['input_content_sha256']==t['input_content_sha256']
        for arm in ARMS:
            ep=r['arms'][arm];assert len(ep['state_reads'])==ep['steps']
            # Source-prefix and summary hashes are independent of tokenization.
            for i,s in enumerate(ep['state_reads']):
                prefix=ep['trajectory'][:i]
                assert s['prefix_sha256']==digest(prefix) and s['state_sha256']==digest(action_state_text(prefix))
                assert 0<s['prompt_tokens']<=4096;reads+=1
            replay(DATA/t['game'],ep);checked+=1
    assert checked==48
    summary={'n':24,**{a:sum(r['arms'][a]['reward'] for r in d['rows']) for a in ARMS}}
    assert summary==d['summary']
    old=json.loads(PARENT.read_text());pairs=list(zip(d['rows'],old['rows'],strict=True));comparisons={}
    for arm in ARMS:
        comparisons[arm]={'state_wins':sum(n['arms'][arm]['reward'] for n,o in pairs),'original_wins':sum(o['arms'][arm]['reward'] for n,o in pairs),
            'gains':[n['index'] for n,o in pairs if n['arms'][arm]['reward']>o['arms'][arm]['reward']],
            'losses':[n['index'] for n,o in pairs if n['arms'][arm]['reward']<o['arms'][arm]['reward']]}
    save(OUTPUT.with_suffix('.audit.json'),{'passed':True,'episodes_replayed':checked,'state_reads_checked':reads,'report_sha256':file_hash(OUTPUT),
        'review_sha256':file_hash(REVIEW),'summary':summary,'comparisons':comparisons,
        'scope':'Environment replay, summary visible-prefix hashes and recorded token budget. Does not regenerate neural actions or independently reconstruct actor prompt hashes.'})
    print(json.dumps(comparisons),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=('evaluate','audit'));p.add_argument('--device',default='cuda:6');a=p.parse_args()
    evaluate(a.device) if a.command=='evaluate' else audit()
