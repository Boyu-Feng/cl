"""Synthetic real-model check; never counted as training or evaluation success."""
from pathlib import Path
import argparse
import gc
import torch
from .common import save,workspace,binding
from .model import FullPolicy


def run(output):
    torch.set_num_threads(8)
    output.mkdir(parents=True,exist_ok=False)
    base=workspace()/'current_work/delta-Mem/model/Qwen3-4B-Instruct-2507'
    if not base.is_dir():base=workspace()/'models/delta_mem/Qwen3-4B-Instruct-2507'
    model=FullPolicy(base)
    prompts=[[{'role':'user','content':'Write one short sentence about sorting objects into labelled boxes.'}]]*4
    events=model.generate_batch(prompts,62927,32,512)
    chosen=['model.embed_tokens.weight','model.layers.0.self_attn.q_proj.weight','model.layers.35.mlp.down_proj.weight','model.norm.weight']
    params=dict(model.model.named_parameters())
    before={n:params[n].detach().cpu().clone() for n in chosen}
    result=model.update([{'group_id':'synthetic','reward':float(i%2),'events':[e]} for i,e in enumerate(events)],1e-6,output/'update.json')
    changed={n:bool((before[n]!=params[n].detach().cpu()).any()) for n in chosen}
    masters=dict(zip(params,model.master))
    master_changed={n:bool((before[n].float()!=masters[n].detach()).any()) for n in chosen}
    report={'synthetic_only':True,'base_model':str(base),'no_lora':all('lora' not in n.lower() for n in params),
        'all_parameters_trainable':all(p.requires_grad for p in params.values()),'sampled_layers_changed':changed,'sampled_fp32_master_layers_changed':master_changed,
        'generation_lengths':[len(e['response_ids']) for e in events],'update':result,
        'cuda_peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30}
    save(output/'audit.json',report)
    if not all(master_changed.values()) or not any(changed.values()) or result['parameters_with_grad']!=result['total_parameters'] or not report['all_parameters_trainable'] or not report['no_lora']:
        raise RuntimeError('Full parameter smoke check failed')
    print(report,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);run(p.parse_args().output.resolve())
