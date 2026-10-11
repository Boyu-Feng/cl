"""Decompose source-dependent effective LoRA variation at fixed targets."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import torch
from ttcl.trajectory_hyperlora.alfworld_online_from_empty import file_hash, records_from_episode
from ttcl.trajectory_hyperlora.alfworld_online_own_trajectory_v52b import bounded_source_fields
from ttcl.trajectory_hyperlora.alfworld_zero_shot_probe import load_agent
from ttcl.trajectory_hyperlora.contextual_alf_source import contextual_text_fields, task_context_text
from ttcl.trajectory_hyperlora.diagnose_alfworld_source_dependence_v54 import effective_inner
from ttcl.trajectory_hyperlora.alfworld_persistent_failure_delta_v36 import save


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',type=Path,default=Path('results/trajectory_hyperlora/alfworld_bootstrap_own_success_v43b_hyper_20261010.pt'))
    p.add_argument('--output',type=Path,default=Path('results/trajectory_hyperlora/alfworld_source_variance_v54b_20261010.json'))
    args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    source=Path('results/trajectory_hyperlora/alfworld_online_own_trajectory_v52_stream240_20261010.json')
    family_path=Path('results/trajectory_hyperlora/alf_sibling_train240_taskpair_current1000_20261006.json')
    rows=json.loads(source.read_text())['rows']
    wins=[r for r in rows if r['online']['reward']==1]
    family_rows=json.loads(family_path.read_text())['games']
    families={r['game']:r['family'] for r in family_rows}
    targets={}
    for r in family_rows:
        if r['family'] not in targets and r['game'] not in {x['game'] for x in wins}:
            targets[r['family']]=r
    agent,tok=load_agent(Path('current_work/delta-Mem/model/Qwen3-4B-Instruct-2507'),args.checkpoint,'cuda:3',.7)
    fields=[bounded_source_fields(agent,tok,records_from_episode(r['online']),'cuda:3') for r in wins]
    result={'protocol':'Fixed target, vary past success source; weight geometry only, no training or environment outcome',
        'checkpoint_sha256':file_hash(args.checkpoint),'source_report_sha256':file_hash(source),
        'families_sha256':file_hash(family_path),'runner_sha256':file_hash(Path(__file__)),
        'source_indices':[r['index'] for r in wins],'targets':[],'head_spectra':[]}
    with torch.no_grad():
        for head in agent.b_heads:
            w=head.weight.float();ev=torch.linalg.eigvalsh(w.T@w).clamp_min(0).flip(0)
            result['head_spectra'].append({'top1_energy':float(ev[0]/ev.sum()),'top4_energy':float(ev[:4].sum()/ev.sum()),'stable_rank':float(ev.sum()/ev[0])})
        for family,target in sorted(targets.items()):
            initial=target['arms']['base']['initial_observation']
            query=contextual_text_fields(agent,tok,task_context_text(initial,initial),'cuda:3',2048,pooling='both')
            values=[]
            for f in fields:
                agent.set_source(f,target_fields=query)
                values.append([layer.b[0].float().clone() for layer in agent.adapters])
            groups=[]
            for scope,ids in [('all_families',list(range(len(wins)))),('same_family',[i for i,r in enumerate(wins) if families[r['game']]==family])]:
                if len(ids)<2:continue
                layers=[]
                for k,layer in enumerate(agent.adapters):
                    b=[values[i][k] for i in ids];mean=torch.stack(b).mean(0);a=layer.a.float()
                    total=torch.stack([effective_inner(x,x,a) for x in b]).mean()
                    variation=torch.stack([effective_inner(x-mean,x-mean,a) for x in b]).mean()
                    mean2=effective_inner(mean,mean,a)
                    cosine=torch.stack([effective_inner(x,mean,a)/(effective_inner(x,x,a)*mean2).clamp_min(1e-30).sqrt() for x in b])
                    layers.append({'variation_fraction_of_total_energy':float(variation/total.clamp_min(1e-30)),
                        'mean_cosine_to_average':float(cosine.mean()),'min_cosine_to_average':float(cosine.min())})
                groups.append({'source_scope':scope,'n_sources':len(ids),'layers':layers})
            result['targets'].append({'family':family,'game':target['game'],'groups':groups})
    agent.set_source(None)
    result['summary']={}
    for scope in ('all_families','same_family'):
        values=[v for t in result['targets'] for g in t['groups'] if g['source_scope']==scope for v in g['layers']]
        result['summary'][scope]={'target_layers':len(values),**{key:sum(v[key] for v in values)/len(values) for key in values[0]}}
    save(args.output,result)
    print(json.dumps({'head_spectra':result['head_spectra'],'summary':result['summary']},ensure_ascii=False))


if __name__=='__main__':main()
