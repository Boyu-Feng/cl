"""Exact shared own-success LoRA bootstrap, then H increment vs direct A+B.

Chronological replay mechanism experiment, not live online collection.
"""
import argparse
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from ttcl.trajectory_hyperlora.train_causal_prefix_v70 import checked,schedule,source_variants
from ttcl.trajectory_hyperlora.alfworld_bootstrap_own_success_v43 import new_agent
from ttcl.trajectory_hyperlora.alfworld_online_from_empty import file_hash,digest
from ttcl.trajectory_hyperlora.alfworld_persistent_failure_delta_v36 import save
from ttcl.trajectory_hyperlora.alfworld_online_accumulate_lora_v1 import factors_hash
from ttcl.trajectory_hyperlora.contextual_alf_source import contextual_text_fields,task_context_text
from ttcl.trajectory_hyperlora.train_alf_next_task import target_loss
from ttcl.trajectory_hyperlora.centered_experience_update_v75 import centered_factors

EMPTY='Earlier observed steps of the current task:\nNo earlier steps supplied.'


def train(args):
    if args.output.exists():raise FileExistsError(args.output)
    review=checked(args);groups=schedule(review);targets=review['targets']
    tok=AutoTokenizer.from_pretrained(args.model,local_files_only=True)
    prefixes=[];sources=[];queries=[]
    for row in targets:
        prompt=tok.apply_chat_template(row['actor_messages'],tokenize=False,add_generation_prompt=True)
        ids=tok(prompt,add_special_tokens=False).input_ids;assert len(ids)<=3000
        prefixes.append(ids);sources.append(source_variants(tok,row['source_records'])[0]['real'])
        queries.append(task_context_text(row['initial_observation'],row['current_observation']))
    dependencies={n:file_hash(Path(__file__).with_name(n)) for n in (
        'train_centered_experience_v75.py','centered_experience_update_v75.py','train_causal_prefix_v70.py',
        'prepare_causal_prefix_v70.py','alfworld_bootstrap_own_success_v43.py','direct_composition_pilot.py',
        'contextual_alf_source.py','train_alf_next_task.py','alfworld_failure_retry_residual_v31b.py')}
    manifest={'review_sha256':file_hash(args.review),'dependencies':dependencies,'schedule':groups,
        'source_texts_sha256':digest(sources),'actor_ids_sha256':digest(prefixes),
        'seed':42,'shared_bootstrap_steps':300,'continuation_steps_per_arm':sum(g['steps'] for g in groups[1:]),
        'bootstrap_lr':1e-4,'continuation_lr':1e-5,'max_relative':.25,'empty_text':EMPTY,
        'model_config_sha256':file_hash(args.model/'config.json'),
        'scope':'Chronological own-success replay; both arms share exactly one trained bootstrap, then identical target order',
        'hyper_trainables':'Fresh H only after bootstrap, shared A/B frozen','static_trainables':'A+B after bootstrap',
        'evaluation_plan':'Same next24 exposed tasks, true/empty/shifted centered H, continued A+B and base; activation from step3, one30-step attempt and guard2'}
    args.output.mkdir(parents=True);save(args.output/'manifest.json',manifest)
    torch.cuda.set_per_process_memory_fraction(.35,device=args.device)
    agent,tok=new_agent(args);agent.eval()
    original_names={n for n,p in agent.named_parameters() if p.requires_grad}
    named=dict(agent.named_parameters())
    agent.requires_grad_(False)
    for layer in agent.adapters:layer.a.requires_grad_(True)
    factors=torch.nn.ParameterList([torch.nn.Parameter(torch.zeros(1,x.base.out_features,x.rank,device=args.device)) for x in agent.adapters])
    result={'complete':False,'manifest_sha256':file_hash(args.output/'manifest.json'),'groups':[],'failures':[]}

    def parameters():return [p for p in agent.parameters() if p.requires_grad]+list(factors)

    def step(k,optimizer,arm,empty_fields=None,cache=None,shared=None):
        agent.eval();agent.set_source(None)
        if arm=='hyper':
            if k not in cache:
                cache[k]=(contextual_text_fields(agent,tok,sources[k],args.device,2048,pooling='both'),
                          contextual_text_fields(agent,tok,queries[k],args.device,2048,pooling='both'))
            fields,query=cache[k]
            agent.train();centered_factors(agent,fields,query,empty_fields,shared)
        else:
            agent.train()
            for layer,b in zip(agent.adapters,factors,strict=True):layer.b=b
        loss=target_loss(agent,tok,targets[k],prefixes[k],args.device)
        optimizer.zero_grad(set_to_none=True);loss.backward()
        trainables=[p for group in optimizer.param_groups for p in group['params']]
        torch.nn.utils.clip_grad_norm_(trainables,1.,error_if_nonfinite=True)
        optimizer.step();agent.set_source(None)
        if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
        return float(loss.detach())

    def record(arm,group,losses):
        result['groups'].append({'arm':arm,'stream_index':group['stream_index'],'steps':len(losses),
            'target_order_sha256':group['target_order_sha256'],'loss_mean':sum(losses)/len(losses) if losses else None})
        save(args.output/'training.json',result);print(json.dumps(result['groups'][-1]),flush=True)

    try:
        optimizer=torch.optim.AdamW(parameters(),lr=1e-4,weight_decay=.01)
        losses=[step(k,optimizer,'bootstrap') for k in groups[0]['target_indices']]
        record('bootstrap',groups[0],losses)
        bootstrap={'A':[x.a.detach().cpu().clone() for x in agent.adapters],
                   'B':[x.detach().cpu().clone() for x in factors],
                   'trainable_state':{n:named[n].detach().cpu().clone() for n in original_names}}
        torch.save(bootstrap,args.output/'bootstrap.pt')
        result['bootstrap_sha256']=file_hash(args.output/'bootstrap.pt')
        result['bootstrap_A_sha256']=factors_hash(bootstrap['A'])
        result['bootstrap_B_sha256']=factors_hash(bootstrap['B'])
        shared=[x.to(args.device) for x in bootstrap['B']]
        agent.requires_grad_(False)
        for n in original_names:named[n].requires_grad_(True)
        for layer in agent.adapters:layer.a.requires_grad_(False)
        factors=torch.nn.ParameterList()
        optimizer=torch.optim.AdamW([p for p in agent.parameters() if p.requires_grad],lr=1e-5,weight_decay=.01)
        agent.eval();empty=contextual_text_fields(agent,tok,EMPTY,args.device,2048,pooling='both');cache={}
        for group in groups[1:]:
            losses=[step(k,optimizer,'hyper',empty,cache,shared) for k in group['target_indices']]
            assert factors_hash([x.a for x in agent.adapters])==result['bootstrap_A_sha256']
            assert factors_hash(shared)==result['bootstrap_B_sha256']
            record('hyper',group,losses)
        checkpoint={'rank':4,'layers':2,'source_encoder':'contextual_lm','encoder_kind':'contextual',
            'relation_bottleneck':'none','context_mode':'none','context_strength':0.,'task_conditioned':True,
            'task_pair_pooling':'mean','task_context_scope':'current','contextual_source_max_tokens':2048,
            'trainable_state':{n:named[n].detach().cpu().clone() for n in original_names},
            'shared_B':bootstrap['B'],'shared_A':bootstrap['A'],'centered_experience':True,'empty_text':EMPTY,
            'manifest_sha256':result['manifest_sha256'],'bootstrap_sha256':result['bootstrap_sha256']}
        torch.save(checkpoint,args.output/'hyper.pt')
        # Restore the exact one-time shared bootstrap before direct continuation.
        agent.requires_grad_(False)
        with torch.no_grad():
            for n,v in bootstrap['trainable_state'].items():named[n].copy_(v.to(named[n]))
        for layer in agent.adapters:layer.a.requires_grad_(True)
        factors=torch.nn.ParameterList([torch.nn.Parameter(x.to(args.device).clone()) for x in bootstrap['B']])
        assert factors_hash([x.a for x in agent.adapters])==result['bootstrap_A_sha256']
        assert factors_hash(list(factors))==result['bootstrap_B_sha256']
        optimizer=torch.optim.AdamW(parameters(),lr=1e-5,weight_decay=.01)
        for group in groups[1:]:
            losses=[step(k,optimizer,'static') for k in group['target_indices']]
            record('static',group,losses)
        torch.save({'A':[x.a.detach().cpu().clone() for x in agent.adapters],
                    'factors':[x.detach().cpu().clone() for x in factors],
                    'manifest_sha256':result['manifest_sha256'],'bootstrap_sha256':result['bootstrap_sha256']},args.output/'static.pt')
        result.update(complete=True,hyper_sha256=file_hash(args.output/'hyper.pt'),static_sha256=file_hash(args.output/'static.pt'))
        save(args.output/'training.json',result)
    except Exception as exc:
        result['failures'].append(repr(exc));save(args.output/'training.json',result);raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--review',type=Path,default=Path('data/annotations/alfworld_causal_prefix_v70_train120_20261010.json'))
    p.add_argument('--output',type=Path,default=Path('results/trajectory_hyperlora/alfworld_centered_experience_v75_20261010'))
    p.add_argument('--model',type=Path,default=Path('current_work/delta-Mem/model/Qwen3-4B-Instruct-2507'))
    p.add_argument('--device',default='cuda:1')
    args=p.parse_args();args.seed=42;train(args)


if __name__=='__main__':main()
