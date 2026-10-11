"""Chronological joint-feedback training with matched static LoRA snapshots."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
from types import SimpleNamespace

import torch
from transformers import AutoTokenizer

from ttcl.trajectory_hyperlora.alfworld_online_from_empty import digest, file_hash
from ttcl.trajectory_hyperlora.alfworld_bootstrap_own_success_v43 import new_agent, examples
from ttcl.trajectory_hyperlora.alfworld_online_own_trajectory_v52b import fit_examples, bounded_source_fields, action_fields
from ttcl.trajectory_hyperlora.alfworld_online_shared_trajectory_v53 import static_train, a_digest, alignment_pair
from ttcl.trajectory_hyperlora.alfworld_persistent_failure_delta_v36 import save
from ttcl.trajectory_hyperlora.train_feedback_mixture_v59 import fit_feedback_mixture
from ttcl.trajectory_hyperlora.train_alf_next_task import query_ids, target_loss
from ttcl.trajectory_hyperlora.off_goal_unlikelihood_v58 import object_span_tokens


def parser():
    p=argparse.ArgumentParser()
    p.add_argument('--review',type=Path,default=Path('data/annotations/alfworld_feedback_stream_v60_20261010.json'))
    p.add_argument('--model',type=Path,default=Path('current_work/delta-Mem/model/Qwen3-4B-Instruct-2507'))
    p.add_argument('--data-root',type=Path,default=Path('ttcl/data/alfworld_delta'))
    p.add_argument('--output-dir',type=Path,default=Path('results/trajectory_hyperlora/alfworld_feedback_stream_v60_20261010'))
    p.add_argument('--hyper-device',default='cuda:3')
    p.add_argument('--static-device',default='cuda:6')
    return p


def checked(args):
    review=json.loads(args.review.read_text())
    assert review['runner_sha256']==file_hash(Path(__file__).with_name('prepare_alfworld_feedback_stream_v60.py'))
    for key in ('stream','probes','config'):
        assert review[key+'_sha256']==digest(review[key])
    assert len(review['stream'])==120 and len(review['probes'])==24
    latest={}
    for i,row in enumerate(review['stream']):
        assert row['index']==i and row['reviewed_target']
        assert row['input_content_sha256']==digest({k:v for k,v in row.items() if k not in ('input_content_sha256','reviewed_target')})
        assert row['source_index']==latest.get(row['family'])
        assert row['game_sha256']==file_hash(args.data_root/row['game'])
        assert bool(row['positive_actions'])==bool(row['reward'])
        latest[row['family']]=i
    for row in review['probes']:
        assert row['game_sha256']==file_hash(args.data_root/row['game'])
    return review


def positive_fit(agent,tok,opt,items,activate,params,device):
    agent.train(); losses=[]
    try:
        for item in items:
            activate(item)
            loss=target_loss(agent,tok,item['annotation'],item['prefix'],device)
            if not torch.isfinite(loss): raise ValueError('Nonfinite CE')
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True); opt.step()
            losses.append(float(loss.detach())); agent.set_source(None)
    finally:
        agent.set_source(None); agent.eval()
    return {'optimizer_steps':len(items),'positive_target_passes':len(items),
        'negative_target_passes':0,'positive_loss_mean':sum(losses)/len(losses)}


def main():
    args=parser().parse_args()
    if args.output_dir.exists(): raise FileExistsError(args.output_dir)
    review=checked(args); config=review['config']; stream=review['stream']
    tok=AutoTokenizer.from_pretrained(args.model,local_files_only=True)
    # Fail before allocating models if a declared input exceeds its budget.
    for row in stream:
        alignment_pair(row['source_records'],tok)
        for item in row['positive_actions']+row['negative_targets']:
            query_ids(tok,item,history_turns=2,max_prompt_tokens=3000)
        for item in row['negative_targets']: object_span_tokens(tok,item)
    args.output_dir.mkdir(parents=True)
    dependencies=['train_feedback_mixture_v59.py','off_goal_unlikelihood_v58.py',
        'alfworld_online_own_trajectory_v52b.py','alfworld_online_shared_trajectory_v53.py',
        'alfworld_bootstrap_own_success_v43.py','alfworld_zero_shot_probe.py',
        'train_alf_next_task.py','contextual_alf_source.py','direct_composition_pilot.py',
        'evaluate_alfworld_feedback_stream_v60.py','audit_alfworld_feedback_stream_v60.py']
    manifest={'review_sha256':file_hash(args.review),'trainer_sha256':file_hash(Path(__file__)),
        'model_config_sha256':file_hash(args.model/'config.json'),
        'dependencies':{name:file_hash(Path(__file__).with_name(name)) for name in dependencies},'config':config}
    save(args.output_dir/'manifest.json',manifest)
    for device in (args.hyper_device,args.static_device):
        torch.cuda.set_per_process_memory_fraction(.75,device=device)
    hyper,ht=new_agent(SimpleNamespace(seed=config['seed'],model=args.model,device=args.hyper_device))
    static,st=new_agent(SimpleNamespace(seed=config['seed'],model=args.model,device=args.static_device))
    assert a_digest(hyper)==a_digest(static)
    h_names={name for name,p in hyper.named_parameters() if p.requires_grad}
    factors=torch.nn.ParameterList([torch.nn.Parameter(torch.zeros(1,l.base.out_features,l.rank,device=args.static_device)) for l in static.adapters])
    hparams=[p for p in hyper.parameters() if p.requires_grad]
    sparams=[l.a for l in static.adapters]+list(factors)
    hopt=torch.optim.AdamW(hparams,lr=config['bootstrap_lr'],weight_decay=.01)
    sopt=torch.optim.AdamW(sparams,lr=config['bootstrap_lr'],weight_decay=.01)
    rng=random.Random(config['seed']); booted=False; wins=[]; last_positive=None; last_negative=None
    field_cache={}; action_cache={}; frozen_a=None
    report={'manifest_sha256':file_hash(args.output_dir/'manifest.json'),'rows':[],'bootstrap':None,'snapshots':[]}
    def snapshot(label,index):
        common={'rank':4,'layers':2,'review_sha256':file_hash(args.review),'prefix_length':index,
            'manifest_sha256':report['manifest_sha256']}
        hfile=args.output_dir/f'hyper_{label}.pt'; sfile=args.output_dir/f'static_{label}.pt'
        torch.save({**common,'source_encoder':'contextual_lm','encoder_kind':'contextual',
            'relation_bottleneck':'none','context_mode':'none','context_strength':0.,
            'task_conditioned':True,'task_pair_pooling':'mean','task_context_scope':'current',
            'contextual_source_max_tokens':2048,'trainable_state':{n:p.detach().cpu().clone() for n,p in hyper.named_parameters() if n in h_names}},hfile)
        torch.save({**common,'A':[l.a.detach().cpu().clone() for l in static.adapters],
            'factors':[p.detach().cpu().clone() for p in factors]},sfile)
        report['snapshots'].append({'label':label,'prefix_length':index,'hyper_sha256':file_hash(hfile),'static_sha256':file_hash(sfile)})
    def prepare_item(index,action,negative,event):
        source=stream[index]['source_index']
        assert source is not None and source<index<=event
        if source not in field_cache:
            field_cache[source]=bounded_source_fields(hyper,ht,stream[source]['source_records'],args.hyper_device)
        key=digest([index,action,negative])
        if key not in action_cache:
            prefix,query=action_fields(hyper,ht,action,stream[index]['source_records'][0]['observation'],args.hyper_device)
            action_cache[key]=(prefix,query)
        prefix,query=action_cache[key]
        binding=action['input_content_sha256'] if negative else digest([stream[index]['input_content_sha256'],action])
        return {'annotation':action,'prefix':prefix,'fields':field_cache[source],'query':query,
            'source_index':source,'available_after_index':index,'update_index':event,'input_content_sha256':binding}
    def activate_h(item): hyper.set_source(item['fields'],target_fields=item['query'])
    def activate_s(item):
        for layer,b in zip(static.adapters,factors,strict=True): layer.b=b
    for i,row in enumerate(stream):
        entry={'index':i,'input_content_sha256':row['input_content_sha256'],'hyper_steps':0,'static_steps':0}
        eligible=row['source_index'] is not None
        if row['positive_actions'] and eligible: last_positive=i
        if row['negative_targets'] and eligible: last_negative=i
        if not booted:
            if row['positive_actions']:
                wins.append({**row,'actions':row['positive_actions']})
                if len(wins)==8:
                    pairs=examples(wins)
                    hfit=fit_examples(hyper,ht,hopt,wins,pairs,300,random.Random(config['seed']),args.hyper_device)
                    sfit=static_train(static,st,factors,sopt,[a for _,_,a in pairs],300,random.Random(config['seed']),args.static_device)
                    report['bootstrap']={'index':i,'hyper':hfit,'static':sfit,'paired_actions':len(pairs)}
                    snapshot('bootstrap',i+1)
                    for layer in hyper.adapters: layer.a.requires_grad_(False)
                    frozen_a=a_digest(hyper); hparams=[p for p in hyper.parameters() if p.requires_grad]
                    hopt=torch.optim.AdamW(hparams,lr=config['online_lr'],weight_decay=.01)
                    sopt=torch.optim.AdamW(sparams,lr=config['online_lr'],weight_decay=.01)
                    booted=True
        elif eligible and (row['positive_actions'] or row['negative_targets']) and last_positive is not None:
            positives=stream[last_positive]['positive_actions']
            schedule=[]
            for _ in range(config['online_steps_per_event']):
                positive=prepare_item(last_positive,positives[rng.randrange(len(positives))],False,i)
                negative=None
                if last_negative is not None:
                    negatives=stream[last_negative]['negative_targets']
                    negative=prepare_item(last_negative,negatives[rng.randrange(len(negatives))],True,i)
                schedule.append((positive,negative))
            entry['schedule']=[{'positive_binding':p['input_content_sha256'],'positive_index':p['available_after_index'],
                'positive_source':p['source_index'],'negative_binding':n['input_content_sha256'] if n else None,
                'negative_index':n['available_after_index'] if n else None,'negative_source':n['source_index'] if n else None} for p,n in schedule]
            if last_negative is not None:
                hfit=fit_feedback_mixture(hyper,ht,hopt,schedule,device=args.hyper_device,activate=activate_h,parameters=hparams,negative_weight=config['negative_weight'])
                sfit=fit_feedback_mixture(static,st,sopt,schedule,device=args.static_device,activate=activate_s,parameters=sparams,negative_weight=config['negative_weight'])
            else:
                hfit=positive_fit(hyper,ht,hopt,[p for p,_ in schedule],activate_h,hparams,args.hyper_device)
                sfit=positive_fit(static,st,sopt,[p for p,_ in schedule],activate_s,sparams,args.static_device)
            entry.update(hyper_fit=hfit,static_fit=sfit,hyper_steps=hfit['optimizer_steps'],static_steps=sfit['optimizer_steps'])
            assert entry['hyper_steps']==entry['static_steps']==20
        if booted:
            assert a_digest(hyper)==frozen_a
            entry['hyper_A_sha256']=frozen_a
        report['rows'].append(entry)
        if i+1 in config['checkpoint_prefix_lengths']:
            assert booted; snapshot(str(i+1),i+1)
        save(args.output_dir/'training.json',report)
        if (i+1)%10==0:
            print(json.dumps({'prefix':i+1,'matched_online_steps':sum(x['hyper_steps'] for x in report['rows'])}),flush=True)
    assert len(report['rows'])==120 and len(report['snapshots'])==4
    save(args.output_dir/'completed.json',{'training_sha256':file_hash(args.output_dir/'training.json'),'manifest_sha256':report['manifest_sha256']})


if __name__=='__main__':
    main()
