"""Read-only source-format ablations of actual cold-run failure proposals.

Synthetic status labels and shifted actions are diagnostic inputs only. No
optimizer, new environment attempts, memory writes, or positive labels.
"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import torch

from ttcl.trajectory_hyperlora.alfworld_cold_joint_memory_v63 import (
    new_agent, trainable_digest, bounded_source_fields, source_factors,
    records_from_episode, file_hash, digest, save,
)
from ttcl.trajectory_hyperlora.contextual_alf_source import source_text


def variants(records):
    real = copy.deepcopy(records)
    neutral = copy.deepcopy(records)
    tail = neutral[-1]['feedback']
    if not tail.startswith('task not completed. '):
        raise ValueError('Expected an actual completed failure record')
    neutral[-1]['feedback'] = 'valid command. '+tail[len('task not completed. '):]
    shifted = copy.deepcopy(records)
    actions = [x['action'] for x in records]
    for row, action in zip(shifted, actions[1:]+actions[:1], strict=True):
        row['action'] = action
    return {'real': real, 'neutral_terminal': neutral, 'shifted_actions': shifted}


def common_failed_records(tokenizer, records):
    first, tail = records[:1], records[1:]
    while True:
        choices = variants(first+tail)
        counts = {k: len(tokenizer(source_text(v), add_special_tokens=False).input_ids) for k,v in choices.items()}
        if max(counts.values()) <= 2048:
            return choices, counts
        if not tail:
            raise ValueError('First diagnostic record exceeds source budget')
        tail = tail[1:]


def common_good_records(tokenizer, records):
    first, tail = records[:1], records[1:]
    while True:
        good = copy.deepcopy(first+tail)
        flipped = copy.deepcopy(good)
        last = flipped[-1]['feedback']
        if not last.startswith('task completed. '):
            raise ValueError('Expected an actual own-success source')
        flipped[-1]['feedback'] = 'task not completed. '+last[len('task completed. '):]
        choices = {'good':good, 'good_false_failure_tag':flipped}
        counts = {k:len(tokenizer(source_text(v),add_special_tokens=False).input_ids) for k,v in choices.items()}
        if max(counts.values()) <= 2048:
            return choices, counts
        if not tail:
            raise ValueError('First good record exceeds source budget')
        tail = tail[1:]


def geometry(agent):
    grams = [x.a.detach().double().cpu()@x.a.detach().double().cpu().T/(x.rank**2) for x in agent.adapters]
    def dot(left,right):
        return sum(float(((a[0].double().T@b[0].double())*g).sum()) for a,b,g in zip(left,right,grams,strict=True))
    def norm(x):
        return math.sqrt(max(0.,dot(x,x)))
    def cosine(a,b):
        denominator=norm(a)*norm(b)
        return dot(a,b)/denominator if denominator else None
    return norm,cosine


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--report',type=Path,default=Path('results/trajectory_hyperlora/alfworld_cold_joint_memory_v63_hybrid_20261010.json'))
    p.add_argument('--model',type=Path,default=Path('current_work/delta-Mem/model/Qwen3-4B-Instruct-2507'))
    p.add_argument('--output',type=Path,default=Path('results/trajectory_hyperlora/alfworld_joint_residual_content_v66_20261010.json'))
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--limit',type=int,default=6)
    args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    raw=args.report.read_bytes();report=json.loads(raw)
    selected=[r for r in report['rows'] if r['write'] is not None][:args.limit]
    if len(selected)!=args.limit:raise ValueError('Insufficient completed failure proposals')
    torch.cuda.set_per_process_memory_fraction(.45,device=args.device)
    agent,tok=new_agent(SimpleNamespace(seed=report['seed'],model=args.model,device=args.device))
    params=dict(agent.named_parameters());agent.eval()
    out={'report_snapshot_sha256':hashlib.sha256(raw).hexdigest(), 'snapshot_rows':len(report['rows']),
        'runner_sha256':file_hash(Path(__file__)), 'review_sha256':report['review_sha256'],
        'scope':'first completed actual failure proposals, raw un-clipped H output differences, no efficacy inference',
        'synthetic_sources_used_only_for_diagnosis':True,'rows':[]}
    for row in selected:
        i=row['index'];earlier=report['rows'][:i]
        previous=next(r for r in reversed(earlier) if 'snapshot' in r)['snapshot']
        assert file_hash(Path(previous['path']))==previous['sha256']
        state=torch.load(previous['path'],map_location='cpu',weights_only=True)
        with torch.no_grad():
            for name,value in state['trainable_state'].items():params[name].copy_(value.to(params[name]))
            for layer,a in zip(agent.adapters,state['A'],strict=True):
                layer.a.copy_(a.to(layer.a));layer.a.requires_grad_(False)
        assert trainable_digest(agent)==row['H_before_sha256']
        source=report['rows'][row['source_stream_index']]
        assert source['index']<i and source['episode']['reward']==1
        good_records=records_from_episode(source['episode'])
        failed_records=records_from_episode(row['episode'])
        bad_choices,bad_tokens=common_failed_records(tok,failed_records)
        good_choices,good_tokens=common_good_records(tok,good_records)
        choices={**good_choices,**bad_choices}
        outputs={}
        for name,records in choices.items():
            fields=bounded_source_fields(agent,tok,records,args.device)
            outputs[name]=source_factors(agent,tok,fields,row['episode']['initial_observation'],args.device)
        diffs={name:[g-b for g,b in zip(outputs['good'],value,strict=True)] for name,value in outputs.items() if name!='good'}
        norm,cosine=geometry(agent)
        full_norm=norm(diffs['real'])
        if full_norm==0:raise ValueError('Zero raw failure residual')
        metrics={name:{'effective_norm':norm(delta),'cosine_with_real':cosine(delta,diffs['real']),
            'change_relative_to_real':norm([x-y for x,y in zip(delta,diffs['real'],strict=True)])/full_norm}
            for name,delta in diffs.items()}
        item={'stream_index':i,'source_stream_index':source['index'],'H_sha256':row['H_before_sha256'],
            'model_snapshot':previous,'source_episode_sha256':digest(source['episode']),
            'failure_episode_sha256':digest(row['episode']),'records_retained':{k:len(v) for k,v in choices.items()},
            'source_tokens':{**good_tokens,**bad_tokens},'source_hashes':{k:digest(v) for k,v in choices.items()},
            'real_failed_records_original_count':len(failed_records),'metrics':metrics,
            'actual_write_step':row['write']['guard']['accepted_step']}
        out['rows'].append(item);print(json.dumps(item),flush=True)
    out['summary']={name:{key:sum(r['metrics'][name][key] for r in out['rows'])/len(out['rows'])
        for key in ('effective_norm','cosine_with_real','change_relative_to_real')}
        for name in ('real','neutral_terminal','shifted_actions','good_false_failure_tag')}
    save(args.output,out)
    print(json.dumps({'n':len(out['rows']),'summary':out['summary']}),flush=True)


if __name__=='__main__':main()
