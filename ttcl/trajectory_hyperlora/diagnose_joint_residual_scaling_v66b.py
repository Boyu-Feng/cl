"""Check whether actual failure differences mostly rescale existing LoRA.

Read-only follow-up to v66, with identical selected snapshots and sources.
"""
import json
import math
from pathlib import Path
from types import SimpleNamespace

import torch

from ttcl.trajectory_hyperlora.diagnose_joint_residual_content_v66 import (
    common_failed_records, common_good_records, geometry,
)
from ttcl.trajectory_hyperlora.alfworld_cold_joint_memory_v63 import (
    new_agent, trainable_digest, bounded_source_fields, source_factors,
    records_from_episode, file_hash, digest, save,
)


def main():
    root=Path('results/trajectory_hyperlora')
    source=root/'alfworld_joint_residual_content_v66_20261010.json'
    output=root/'alfworld_joint_residual_scaling_v66b_20261010.json'
    if output.exists():raise FileExistsError(output)
    diagnostic=json.loads(source.read_text())
    report=json.loads((root/'alfworld_cold_joint_memory_v63_hybrid_20261010.json').read_text())
    args=SimpleNamespace(seed=report['seed'],model=Path('current_work/delta-Mem/model/Qwen3-4B-Instruct-2507'),device='cuda:0')
    torch.cuda.set_per_process_memory_fraction(.45,device=args.device)
    agent,tok=new_agent(args);agent.eval();params=dict(agent.named_parameters())
    result={'v66_sha256':file_hash(source),'runner_sha256':file_hash(Path(__file__)),
        'scope':'effective-weight geometry only; no rollout, optimizer or persistent writes','rows':[]}
    for item in diagnostic['rows']:
        row=report['rows'][item['stream_index']]
        snap=item['model_snapshot'];assert file_hash(Path(snap['path']))==snap['sha256']
        state=torch.load(snap['path'],map_location='cpu',weights_only=True)
        with torch.no_grad():
            for name,value in state['trainable_state'].items():params[name].copy_(value.to(params[name]))
            for layer,a in zip(agent.adapters,state['A'],strict=True):
                layer.a.copy_(a.to(layer.a));layer.a.requires_grad_(False)
        assert trainable_digest(agent)==item['H_sha256']
        good,_=common_good_records(tok,records_from_episode(report['rows'][item['source_stream_index']]['episode']))
        bad,_=common_failed_records(tok,records_from_episode(row['episode']))
        choices={'good':good['good'],'real':bad['real'],'good_false_failure_tag':good['good_false_failure_tag']}
        values={}
        for name,records in choices.items():
            assert digest(records)==item['source_hashes'][name]
            fields=bounded_source_fields(agent,tok,records,args.device)
            values[name]=source_factors(agent,tok,fields,row['episode']['initial_observation'],args.device)
        norm,cosine=geometry(agent)
        metrics={}
        for name in ('real','good_false_failure_tag'):
            delta=[a-b for a,b in zip(values['good'],values[name],strict=True)]
            alignment=cosine(delta,values['good']);ratio=norm(delta)/norm(values['good'])
            metrics[name]={'cosine_with_existing_B':alignment,'norm_relative_to_existing_B':ratio,
                'least_squares_scalar_gain':alignment*ratio,
                'fraction_of_delta_squared_norm_parallel_to_existing_B':alignment**2,
                'orthogonal_norm_fraction':math.sqrt(max(0.,1-alignment**2))}
        result['rows'].append({'stream_index':item['stream_index'],'metrics':metrics})
    result['summary']={name:{key:sum(x['metrics'][name][key] for x in result['rows'])/len(result['rows'])
        for key in result['rows'][0]['metrics'][name]} for name in ('real','good_false_failure_tag')}
    save(output,result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
