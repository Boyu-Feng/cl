"""Next24 frozen-state test of utility-selected residual transfer."""
import argparse
from contextlib import contextmanager
import copy
import json
from pathlib import Path

import torch

from ttcl.trajectory_hyperlora.alfworld_cold_joint_memory_v63 import (
    new_agent, read_memory, bounded_source_fields, records_from_episode,
    run_episode, factors_hash, bounded_readout, effective_norm, digest, file_hash, save,
)
from ttcl.trajectory_hyperlora.contextual_alf_source import contextual_text_fields, task_context_text
from ttcl.trajectory_hyperlora.contextual_residual_memory_v72 import ContextualResidualMemory
from ttcl.trajectory_hyperlora.audit_my_online_update_v28_reward import replay

ARMS=('no_additional_residual','global_sum','contextual','shuffled_keys')


def family(game):return Path(game).parts[2].split('-')[0]


def bindings(args):
    bank=torch.load(args.bank,map_location='cpu',weights_only=True)
    meta=json.loads(args.bank.with_suffix('.json').read_text())
    assert meta['complete'] and meta['bank_sha256']==file_hash(args.bank)
    assert meta['builder_sha256']==file_hash(Path(__file__).with_name('build_contextual_residual_memory_v72.py'))
    assert meta['component_sha256']==file_hash(Path(__file__).with_name('contextual_residual_memory_v72.py'))
    parent=json.loads(args.parent.read_text())
    review_path=Path('data/annotations/alfworld_own_correction_utility_v71_20261010.json')
    review=json.loads(review_path.read_text());assert review['parent_sha256']==file_hash(args.parent)
    for name,value in review['parent_code_sha256'].items():assert file_hash(Path(__file__).with_name(name))==value
    for path,value in bank['encoder']['weights'].items():assert file_hash(Path(path))==value
    assert bank['encoder']['model_config_sha256']==file_hash(args.model/'config.json')
    assert bank['encoder_sha256']==digest(bank['encoder'])
    assert file_hash(Path(bank['prefix_snapshot']['path']))==bank['prefix_snapshot']['sha256']
    pool=[r for r in parent['rows'][:120] if r['episode']['reward']==1]
    targets=[]
    for row in parent['rows'][120:144]:
        sources=[r for r in pool if family(r['game'])==family(row['game'])]
        source=sources[-1] if len(sources)>=2 else None
        content={'index':row['index'],'game':row['game'],'game_sha256':file_hash(args.data_root/row['game']),
                 'source_index':source['index'] if source else None,
                 'source_episode_sha256':digest(source['episode']) if source else None,
                 'initial_observation_sha256':digest(row['episode']['initial_observation'])}
        targets.append({**content,'input_content_sha256':digest(content),'reviewed_target':True})
    fixed={'bank_sha256':file_hash(args.bank),'parent_sha256':file_hash(args.parent),
           'runner_sha256':file_hash(Path(__file__)),'targets':targets,'targets_sha256':digest(targets),
           'arms':list(ARMS),'max_steps':30,'max_new_tokens':64,'history_turns':2,'loop_guard_max':2,
           'training_updates':0,'source_pool':'Frozen first120 own successes; latest same-family source, minimum2',
           'key_policy':'Initial observation only, fixed per task','key_shuffle_degenerate':len(bank['entries'])<2,
           'scope':'Exposed next24 mechanism test; not full continual-learning result or fair new LoRA comparison'}
    return bank,parent,fixed


@contextmanager
def correction_reader(agent,arm,memory,key,task_index,trace):
    original=agent.set_source
    def patched(source,*args,**kwargs):
        value=original(source,*args,**kwargs)
        if source is not None and arm!='no_additional_residual':
            reference=[x.b for x in agent.adapters]
            if arm=='global_sum':
                eligible=[e for e in memory.entries if e['available_before_task']<=task_index and e['source_task']<task_index]
                total=[sum((e['delta'][i] for e in eligible),torch.zeros_like(b,device='cpu',dtype=torch.float32)) for i,b in enumerate(reference)]
                delta,info=bounded_readout(agent,total,reference)
            else:
                delta,info=memory.read(key=key,current_task=task_index,reference_factors=reference,
                    effective_norm=lambda x:effective_norm(agent,x),basis_sha256=factors_hash([x.a for x in agent.adapters]),
                    encoder_sha256=memory.encoder_sha256)
            if delta is not None:
                for layer,b in zip(agent.adapters,delta,strict=True):layer.b=layer.b+b.to(layer.b)
            trace.append(info)
        return value
    agent.set_source=patched
    try:yield
    finally:agent.set_source=original;original(None)


def evaluate(args,bank,parent,review):
    if args.output.exists():raise FileExistsError(args.output)
    torch.cuda.set_per_process_memory_fraction(.4,device=args.device)
    agent,tok=new_agent(args);agent.eval();params=dict(agent.named_parameters())
    state=torch.load(bank['prefix_snapshot']['path'],map_location='cpu',weights_only=True)
    with torch.no_grad():
        for n,v in state['trainable_state'].items():params[n].copy_(v.to(params[n]))
        for layer,a in zip(agent.adapters,state['A'],strict=True):layer.a.copy_(a.to(layer.a))
    agent.requires_grad_(False)
    assert factors_hash([x.a for x in agent.adapters])==bank['basis_sha256']
    memory=ContextualResidualMemory(basis_sha256=bank['basis_sha256'],encoder_sha256=bank['encoder_sha256'])
    memory.entries=bank['entries']
    shuffled=copy.deepcopy(memory)
    if shuffled.entries:
        keys=[e['key'].clone() for e in shuffled.entries]
        for i,e in enumerate(shuffled.entries):e['key']=keys[(i+1)%len(keys)]
    result={'complete':False,'rows':[],'failures':[],'review_sha256':file_hash(args.review)}
    for t in review['targets']:
        row={'index':t['index'],'game':t['game'],'input_content_sha256':t['input_content_sha256'],'arms':{}}
        try:
            active=t['source_index'] is not None
            fields={};key=None
            if active:
                source=parent['rows'][t['source_index']]['episode']
                assert digest(source)==t['source_episode_sha256']
                fields=bounded_source_fields(agent,tok,records_from_episode(source),args.device)
                # Read only visible initial text, never the target outcome.
                initial=parent['rows'][t['index']]['episode']['initial_observation']
                key=contextual_text_fields(agent,tok,task_context_text(initial,initial),args.device,2048,pooling='both')['pair_contextual']
            for arm in ARMS:
                trace=[]
                with read_memory(agent,state['R']),correction_reader(agent,arm,shuffled if arm=='shuffled_keys' else memory,key,t['index'],trace):
                    ep=run_episode(agent,tok,args.data_root/t['game'],fields,adapter=active,device=args.device,
                        max_steps=30,max_new_tokens=64,constrain_actions=True,actor_history_turns=2,loop_guard_max=2)
                assert ep['status']=='complete' and digest(ep['initial_observation'])==t['initial_observation_sha256']
                row['arms'][arm]={'episode':ep,'reads':trace}
                save(args.output,{**result,'in_progress_row':row})
            result['rows'].append(row)
            result['summary']={'n':len(result['rows']),**{a:sum(r['arms'][a]['episode']['reward'] for r in result['rows']) for a in ARMS}}
            save(args.output,result);print(json.dumps(result['summary']),flush=True)
        except Exception as exc:
            result['failures'].append({'index':t['index'],'error':repr(exc),'partial_row':row});save(args.output,result);raise
    result['complete']=True;save(args.output,result)


def audit(args,bank,parent,review):
    result=json.loads(args.output.read_text());assert result['complete'] and not result['failures']
    assert result['review_sha256']==file_hash(args.review) and len(result['rows'])==24
    for row,t in zip(result['rows'],review['targets'],strict=True):
        assert row['index']==t['index'] and row['game']==t['game'] and row['input_content_sha256']==t['input_content_sha256']
        for arm in ARMS:
            ep=row['arms'][arm]['episode'];assert ep['steps']<=30;replay(args.data_root/row['game'],ep)
    summary={'n':24,**{a:sum(r['arms'][a]['episode']['reward'] for r in result['rows']) for a in ARMS}}
    assert summary==result['summary']
    save(args.output.with_suffix('.audit.json'),{'passed':True,'episodes_replayed':96,'summary':summary,
         'report_sha256':file_hash(args.output),'review_sha256':file_hash(args.review),
         'scope':'Input bindings and environment replay; no independent neural regeneration',
         'key_shuffle_degenerate':review['key_shuffle_degenerate']})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('prepare','evaluate','audit'))
    p.add_argument('--bank',type=Path,default=Path('results/trajectory_hyperlora/contextual_residual_memory_v72_20261010.pt'))
    p.add_argument('--parent',type=Path,default=Path('results/trajectory_hyperlora/alfworld_cold_joint_memory_v63_hybrid_20261010.json'))
    p.add_argument('--review',type=Path,default=Path('data/annotations/contextual_residual_memory_v72_eval_20261010.json'))
    p.add_argument('--output',type=Path,default=Path('results/trajectory_hyperlora/contextual_residual_memory_v72_eval_20261010.json'))
    p.add_argument('--model',type=Path,default=Path('current_work/delta-Mem/model/Qwen3-4B-Instruct-2507'))
    p.add_argument('--data-root',type=Path,default=Path('ttcl/data/alfworld_delta'))
    p.add_argument('--device',default='cuda:6')
    args=p.parse_args();args.seed=42;bank,parent,review=bindings(args)
    if args.command=='prepare':
        if args.review.exists():raise FileExistsError(args.review)
        save(args.review,review)
    else:
        assert json.loads(args.review.read_text())==review
        (evaluate if args.command=='evaluate' else audit)(args,bank,parent,review)


if __name__=='__main__':main()
