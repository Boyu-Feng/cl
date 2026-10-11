"""Measure actual utility of saved own-failure corrections on train tasks."""
import argparse
from contextlib import contextmanager
import json
from pathlib import Path

import torch

from ttcl.trajectory_hyperlora.alfworld_cold_joint_memory_v63 import (
    new_agent, read_memory, bounded_source_fields, records_from_episode,
    trainable_digest, factors_hash, bounded_readout, effective_norm,
    run_episode, digest, file_hash, save,
)
from ttcl.trajectory_hyperlora.audit_my_online_update_v28_reward import replay

ARMS=('unchanged_retry','failure_delta','matched_norm_random')


@contextmanager
def read_correction(agent,delta):
    original=agent.set_source
    def patched(source,*args,**kwargs):
        value=original(source,*args,**kwargs)
        if source is not None and delta is not None:
            applied,_=bounded_readout(agent,delta,[x.b for x in agent.adapters])
            for layer,b in zip(agent.adapters,applied,strict=True):
                layer.b=layer.b+b.to(layer.b)
        return value
    agent.set_source=patched
    try:yield
    finally:
        agent.set_source=original;original(None)


def random_control(agent,delta,index):
    generator=torch.Generator(device='cpu').manual_seed(710000+index)
    random=[torch.randn(x.shape,generator=generator,dtype=torch.float32) for x in delta]
    wanted=effective_norm(agent,delta);actual=effective_norm(agent,random)
    random=[x*(wanted/actual if actual else 0.) for x in random]
    assert abs(effective_norm(agent,random)-wanted)<=1e-5*max(1.,wanted)
    return random


def checked(args):
    review=json.loads(args.review.read_text());parent=json.loads(args.parent.read_text())
    assert review['parent_sha256']==file_hash(args.parent)
    assert review['preparer_sha256']==file_hash(Path(__file__).with_name('prepare_own_correction_utility_v71.py'))
    assert review['targets_sha256']==digest(review['targets'])
    for name,expected in review['parent_code_sha256'].items():
        assert file_hash(Path(__file__).with_name(name))==expected
    for t in review['targets']:
        assert t['reviewed_target']
        assert digest({k:v for k,v in t.items() if k not in ('reviewed_target','input_content_sha256')})==t['input_content_sha256']
        assert file_hash(args.data_root/t['game'])==t['game_sha256']
        assert file_hash(Path(t['pre_task_snapshot']['path']))==t['pre_task_snapshot']['sha256']
        assert file_hash(Path(t['candidate_delta_path']))==t['candidate_delta_file_sha256']
    return review,parent


def evaluate(args,review,parent):
    if args.output.exists():raise FileExistsError(args.output)
    torch.cuda.set_per_process_memory_fraction(.40,device=args.device)
    agent,tok=new_agent(args);agent.eval();params=dict(agent.named_parameters())
    result={'review_sha256':file_hash(args.review),'runner_sha256':file_hash(Path(__file__)),
            'rows':[],'failures':[],'complete':False,
            'scope':'Train-only frozen snapshot retries; not first-attempt gains or online policy updates'}
    for target in review['targets']:
        row={'index':target['index'],'game':target['game'],
             'input_content_sha256':target['input_content_sha256'],'arms':{}}
        try:
            state=torch.load(target['pre_task_snapshot']['path'],map_location='cpu',weights_only=True)
            with torch.no_grad():
                for name,value in state['trainable_state'].items():params[name].copy_(value.to(params[name]))
                for layer,a in zip(agent.adapters,state['A'],strict=True):
                    layer.a.copy_(a.to(layer.a));layer.a.requires_grad_(False)
            assert trainable_digest(agent)==target['H_before_sha256']
            assert factors_hash(state['R'])==target['R_before_sha256']
            delta=torch.load(target['candidate_delta_path'],map_location='cpu',weights_only=True)
            assert factors_hash(delta)==target['candidate_delta_sha256']
            random=random_control(agent,delta,target['index'])
            row['random_sha256']=factors_hash(random)
            row['delta_effective_norm']=effective_norm(agent,delta)
            row['random_effective_norm']=effective_norm(agent,random)
            source=parent['rows'][target['source_stream_index']]['episode']
            assert digest(source)==target['source_episode_sha256']
            fields=bounded_source_fields(agent,tok,records_from_episode(source),args.device)
            for arm,correction in zip(ARMS,(None,delta,random),strict=True):
                with read_memory(agent,state['R']),read_correction(agent,correction):
                    ep=run_episode(agent,tok,args.data_root/target['game'],fields,adapter=True,
                        device=args.device,max_steps=30,max_new_tokens=64,constrain_actions=True,
                        actor_history_turns=2,loop_guard_max=2)
                assert ep['status']=='complete' and ep['initial_observation']==target['initial_observation']
                row['arms'][arm]=ep;save(args.output,{**result,'in_progress_row':row})
            row['unchanged_episode_reproduced']=row['arms']['unchanged_retry']==parent['rows'][target['index']]['episode']
            assert trainable_digest(agent)==target['H_before_sha256']
            result['rows'].append(row)
            result['summary']={'n':len(result['rows']),**{a:sum(r['arms'][a]['reward'] for r in result['rows']) for a in ARMS},
                               'unchanged_reproduced':sum(r['unchanged_episode_reproduced'] for r in result['rows'])}
            save(args.output,result);print(json.dumps(result['summary']),flush=True)
        except Exception as exc:
            result['failures'].append({'index':target['index'],'error':repr(exc),'partial_row':row})
            save(args.output,result);raise
    result['complete']=True;save(args.output,result)


def audit(args,review,parent):
    result=json.loads(args.output.read_text())
    assert result['complete'] and not result['failures'] and len(result['rows'])==34
    assert result['runner_sha256']==file_hash(Path(__file__)) and result['review_sha256']==file_hash(args.review)
    labels=[]
    for row,target in zip(result['rows'],review['targets'],strict=True):
        assert row['index']==target['index'] and row['game']==target['game']
        assert row['input_content_sha256']==target['input_content_sha256']
        for arm in ARMS:
            ep=row['arms'][arm];assert ep['steps']<=30
            replay(args.data_root/row['game'],ep)
        reproduced=row['arms']['unchanged_retry']==parent['rows'][row['index']]['episode']
        assert row['unchanged_episode_reproduced']==reproduced
        labels.append({'index':row['index'],'target_binding':target['input_content_sha256'],
                       'unchanged_reproduced':reproduced,
                       'paired_gain':row['arms']['failure_delta']['reward']-row['arms']['unchanged_retry']['reward'],
                       'gain_over_random':row['arms']['failure_delta']['reward']-row['arms']['matched_norm_random']['reward'],
                       'retry_episodes_sha256':digest(row['arms'])})
    summary={'n':34,**{a:sum(r['arms'][a]['reward'] for r in result['rows']) for a in ARMS},
             'unchanged_reproduced':sum(r['unchanged_episode_reproduced'] for r in result['rows'])}
    assert summary==result['summary']
    save(args.output.with_suffix('.audit.json'),{'passed':True,'episodes_replayed':102,
         'summary':summary,'report_sha256':file_hash(args.output),'labels':labels,
         'scope':'Environment rewards and prebound inputs; not independent neural reproduction or full norm reconstruction',
         'training_ready':all(x['unchanged_reproduced'] for x in labels)})
    print(json.dumps(summary),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('evaluate','audit'))
    p.add_argument('--review',type=Path,default=Path('data/annotations/alfworld_own_correction_utility_v71_20261010.json'))
    p.add_argument('--parent',type=Path,default=Path('results/trajectory_hyperlora/alfworld_cold_joint_memory_v63_hybrid_20261010.json'))
    p.add_argument('--output',type=Path,default=Path('results/trajectory_hyperlora/alfworld_own_correction_utility_v71_20261010.json'))
    p.add_argument('--model',type=Path,default=Path('current_work/delta-Mem/model/Qwen3-4B-Instruct-2507'))
    p.add_argument('--data-root',type=Path,default=Path('ttcl/data/alfworld_delta'))
    p.add_argument('--device',default='cuda:6')
    args=p.parse_args();args.seed=42
    review,parent=checked(args)
    (evaluate if args.command=='evaluate' else audit)(args,review,parent)


if __name__=='__main__':main()
