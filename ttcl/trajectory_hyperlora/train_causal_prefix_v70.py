"""Matched chronological prefix-memory H and direct A+B LoRA training.

Fresh models; one replayed own-success stream. First eight wins bootstrap 300
steps, later eligible wins each add 20. The H A basis is frozen after bootstrap.
"""
import argparse
import gc
import json
import random
from pathlib import Path

import torch
from transformers import AutoTokenizer

from ttcl.trajectory_hyperlora.prepare_causal_prefix_v70 import prefix_text
from ttcl.trajectory_hyperlora.alfworld_bootstrap_own_success_v43 import new_agent
from ttcl.trajectory_hyperlora.alfworld_online_from_empty import digest,file_hash
from ttcl.trajectory_hyperlora.alfworld_online_accumulate_lora_v1 import factors_hash
from ttcl.trajectory_hyperlora.alfworld_persistent_failure_delta_v36 import save
from ttcl.trajectory_hyperlora.contextual_alf_source import contextual_text_fields,task_context_text
from ttcl.trajectory_hyperlora.train_alf_next_task import target_loss


def source_variants(tokenizer, records):
    """Retain identical records for real versus rotated-action controls."""
    chosen = [dict(x) for x in records]
    indices = list(range(len(chosen)))
    while chosen:
        actions = [x['action'] for x in chosen]
        shifted = [{**x,'action':actions[(i+1)%len(actions)]} for i,x in enumerate(chosen)]
        texts = {'real':prefix_text(chosen),'shifted':prefix_text(shifted)}
        counts = {k:len(tokenizer(v,add_special_tokens=False).input_ids) for k,v in texts.items()}
        if max(counts.values())<=2048:
            return texts,indices,counts
        if len(chosen)==1:
            raise ValueError('First history record exceeds encoder budget')
        chosen.pop(1);indices.pop(1)
    raise ValueError('No causal prefix available')


def checked(args):
    review = json.loads(args.review.read_text())
    assert review['stage']=='causal_prefix_training_inputs_prepared_not_trained'
    assert review['runner_sha256']==file_hash(Path(__file__).with_name('prepare_causal_prefix_v70.py'))
    for name,expected in review['dependencies'].items():
        assert file_hash(Path(__file__).with_name(name))==expected
    assert digest(review['targets'])==review['targets_sha256']
    for row in review['targets']:
        content = {k:v for k,v in row.items() if k not in ('reviewed_target','input_content_sha256')}
        assert row['reviewed_target'] and digest(content)==row['input_content_sha256']
        assert row['source_end_exclusive']==row['target_step']-2==len(row['source_records'])
        assert row['stream_index']<120
    return review


def schedule(review):
    wins = review['own_success_indices']
    assert wins==sorted(set(wins)) and len(wins)>=8
    targets=review['targets'];groups=[];rng=random.Random(42)
    for event,bootstrap in [(wins[7],True)]+[(i,False) for i in wins[8:]]:
        ids=[i for i,r in enumerate(targets) if r['stream_index']<=event] if bootstrap else [i for i,r in enumerate(targets) if r['stream_index']==event]
        budget=300 if bootstrap else 20
        selected=[];order=list(ids)
        for step in range(budget if ids else 0):
            if step%len(order)==0:rng.shuffle(order)
            selected.append(order[step%len(order)])
        groups.append({'stream_index':event,'bootstrap':bootstrap,'steps':len(selected),
                       'target_indices':selected,'target_order_sha256':digest(selected)})
    assert groups[0]['steps']==300
    return groups


def train_arm(args, arm, review, groups, prefixes, source_texts, queries, manifest_hash):
    agent,tok=new_agent(args);agent.eval()
    original_names={n for n,p in agent.named_parameters() if p.requires_grad}
    initial_a=factors_hash([x.a for x in agent.adapters])
    factors=None
    if arm=='static':
        agent.requires_grad_(False)
        for layer in agent.adapters:layer.a.requires_grad_(True)
        factors=torch.nn.ParameterList([torch.nn.Parameter(torch.zeros(1,x.base.out_features,x.rank,device=args.device)) for x in agent.adapters])
    def parameters():
        return [p for p in agent.parameters() if p.requires_grad]+([] if factors is None else list(factors))
    optimizer=torch.optim.AdamW(parameters(),lr=1e-4,weight_decay=.01)
    result={'arm':arm,'initial_A_sha256':initial_a,'manifest_sha256':manifest_hash,'groups':[],
            'initial_trainable_parameters':sum(p.numel() for p in parameters()),'complete':False}
    cache={};frozen_a=None
    try:
        for group in groups:
            losses=[]
            for k in group['target_indices']:
                row=review['targets'][k]
                if arm=='hyper' and k not in cache:
                    agent.eval();agent.set_source(None)
                    fields=contextual_text_fields(agent,tok,source_texts[k],args.device,2048,pooling='both')
                    query=contextual_text_fields(agent,tok,queries[k],args.device,2048,pooling='both')
                    cache[k]=(fields,query)
                agent.train()
                if arm=='hyper':
                    fields,query=cache[k];agent.set_source(fields,target_fields=query)
                else:
                    for layer,b in zip(agent.adapters,factors,strict=True):layer.b=b
                loss=target_loss(agent,tok,row,prefixes[k],args.device)
                optimizer.zero_grad(set_to_none=True);loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters(),1.,error_if_nonfinite=True)
                optimizer.step();agent.set_source(None)
                if not torch.isfinite(loss):raise ValueError('Nonfinite prefix training loss')
                losses.append(float(loss.detach()))
            agent.eval()
            if group['bootstrap']:
                if arm=='hyper':
                    for layer in agent.adapters:layer.a.requires_grad_(False)
                    frozen_a=factors_hash([x.a for x in agent.adapters])
                optimizer=torch.optim.AdamW(parameters(),lr=1e-5,weight_decay=.01)
                result['post_bootstrap_trainable_parameters']=sum(p.numel() for p in parameters())
            if frozen_a is not None:assert factors_hash([x.a for x in agent.adapters])==frozen_a
            result['groups'].append({k:v for k,v in group.items() if k!='target_indices'}|{'loss_mean':sum(losses)/len(losses) if losses else None})
            save(args.output_dir/f'{arm}_training.json',result)
            print(json.dumps({'arm':arm,'event':group['stream_index'],'steps':sum(x['steps'] for x in result['groups']),'loss_mean':result['groups'][-1]['loss_mean']}),flush=True)
        common={'rank':4,'layers':2,'review_sha256':file_hash(args.review),'manifest_sha256':manifest_hash,'training_steps':sum(g['steps'] for g in groups)}
        if arm=='hyper':
            checkpoint={**common,'source_encoder':'contextual_lm','encoder_kind':'contextual',
                        'relation_bottleneck':'none','context_mode':'none','context_strength':0.,
                        'task_conditioned':True,'task_pair_pooling':'mean','task_context_scope':'current',
                        'contextual_source_max_tokens':2048,
                        'trainable_state':{n:p.detach().cpu().clone() for n,p in agent.named_parameters() if n in original_names}}
        else:
            checkpoint={**common,'A':[x.a.detach().cpu().clone() for x in agent.adapters],
                        'factors':[x.detach().cpu().clone() for x in factors]}
        path=args.output_dir/f'{arm}.pt';torch.save(checkpoint,path)
        result.update(complete=True,checkpoint_sha256=file_hash(path),frozen_A_sha256=frozen_a)
        save(args.output_dir/f'{arm}_training.json',result)
        return result
    except Exception as exc:
        result['failure']=repr(exc);save(args.output_dir/f'{arm}_training.json',result);raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--review',type=Path,default=Path('data/annotations/alfworld_causal_prefix_v70_train120_20261010.json'))
    p.add_argument('--output-dir',type=Path,default=Path('results/trajectory_hyperlora/alfworld_causal_prefix_v70_20261010'))
    p.add_argument('--model',type=Path,default=Path('current_work/delta-Mem/model/Qwen3-4B-Instruct-2507'))
    p.add_argument('--device',default='cuda:1')
    args=p.parse_args();args.seed=42
    if args.output_dir.exists():raise FileExistsError(args.output_dir)
    review=checked(args);groups=schedule(review)
    tok=AutoTokenizer.from_pretrained(args.model,local_files_only=True)
    prefixes=[];source_texts=[];queries=[];encodings=[]
    for row in review['targets']:
        texts,indices,counts=source_variants(tok,row['source_records'])
        prompt=tok.apply_chat_template(row['actor_messages'],tokenize=False,add_generation_prompt=True)
        ids=tok(prompt,add_special_tokens=False).input_ids
        query=task_context_text(row['initial_observation'],row['current_observation'])
        assert len(ids)<=3000 and len(tok(query,add_special_tokens=False).input_ids)<=2048
        prefixes.append(ids);source_texts.append(texts['real']);queries.append(query)
        encodings.append({'input_content_sha256':row['input_content_sha256'],'retained_source_indices':indices,'source_token_counts':counts,'actor_tokens':len(ids)})
    names=['train_causal_prefix_v70.py','prepare_causal_prefix_v70.py','alfworld_bootstrap_own_success_v43.py',
           'alfworld_zero_shot_probe.py','contextual_alf_source.py','direct_composition_pilot.py','train_alf_next_task.py']
    manifest={'review_sha256':file_hash(args.review),'model_config_sha256':file_hash(args.model/'config.json'),
              'dependencies':{n:file_hash(Path(__file__).with_name(n)) for n in names},
              'seed':42,'schedule':groups,'encodings':encodings,'history_turns':2,
              'scope':'Fresh H and A+B on same chronological replay of own-success data; no new policy collection',
              'hyper_A_frozen_after_bootstrap':True,'static_A_continues_training':True,
              'evaluation_plan':'Exposed next 24 tasks (indices120:144), frozen models; activate at step>=3, compare true/empty/shifted old prefix H, direct static LoRA and base; all one30-step attempt, common guard2'}
    args.output_dir.mkdir(parents=True);save(args.output_dir/'manifest.json',manifest)
    torch.cuda.set_per_process_memory_fraction(.45,device=args.device)
    results=[]
    for arm in ('hyper','static'):
        results.append(train_arm(args,arm,review,groups,prefixes,source_texts,queries,file_hash(args.output_dir/'manifest.json')))
        gc.collect();torch.cuda.empty_cache()
    assert results[0]['initial_A_sha256']==results[1]['initial_A_sha256']
    assert [x['target_order_sha256'] for x in results[0]['groups']]==[x['target_order_sha256'] for x in results[1]['groups']]
    save(args.output_dir/'completed.json',{'complete':True,'training_steps_per_arm':sum(x['steps'] for x in groups),
         'initial_A_equal':True,'target_order_equal':True,'manifest_sha256':file_hash(args.output_dir/'manifest.json')})


if __name__=='__main__':main()
