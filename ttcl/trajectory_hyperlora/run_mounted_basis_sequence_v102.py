"""Frozen-hypernetwork online coefficient sequence on real Qwen rows."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM
from .direct_composition_pilot import DirectRelationHyperLoRA
from .basis_state_controller_v101 import BasisStateController
from .joint_domains_v93 import MODEL, ROOT, read, filehash

def h(xs):
    z = hashlib.sha256()
    for x in xs: z.update(x.detach().float().cpu().contiguous().numpy().tobytes())
    return z.hexdigest()

def main(device, output, rows_n):
    if output.exists(): raise FileExistsError(output)
    parent=ROOT/'results/trajectory_hyperlora/external_hyper_v100_20261010'
    ck=torch.load(parent/'trained.pt',map_location='cpu',weights_only=True)
    cache=torch.load(parent/'cache.pt',map_location='cpu',weights_only=True)
    model=AutoModelForCausalLM.from_pretrained(MODEL,local_files_only=True,
        torch_dtype=torch.bfloat16,attn_implementation='sdpa').to(device)
    agent=DirectRelationHyperLoRA(model,rank=4,layers=2,encoder_kind='contextual',task_conditioned=True).to(device)
    named=dict(agent.named_parameters())
    for n,v in ck['hyper'].items(): named[n].data.copy_(v.to(device))
    agent.requires_grad_(False); agent.eval()
    def fields(k): return {x:v.to(device) for x,v in cache['fields'][k].items()}
    bases=[]
    for row in cache['prepared'][:4]:
        with torch.no_grad():
            agent.set_source(fields(row['source']),target_fields=fields(row['query']))
            bases.append([a.b.detach().clone() for a in agent.adapters])
    controller=BasisStateController(bases,lr=.2,decay=.01)
    rows=[]
    for i,row in enumerate(cache['prepared'][:rows_n]):
        ids=torch.tensor([row['prefix']+row['answer']],device=device)
        labels=torch.tensor([[-100]*len(row['prefix'])+row['answer']],device=device)
        def loss_fn(bb):
            for a,b in zip(agent.adapters,bb,strict=True): a.b=b*1.
            return model(input_ids=ids,labels=labels,use_cache=False).loss
        pre=float(loss_fn(controller.matrices()).detach())
        info=controller.update_mounted(loss_fn)
        post=float(loss_fn(controller.matrices()).detach())
        rows.append({'row':i,'pre_loss':pre,'post_loss':post,'improved':post<pre,
                     'update':info,'state_coefficients':controller.state.coefficients.tolist(),
                     'state_hash':h([controller.state.coefficients,controller.state.gate_logit.reshape(1)])})
    result={'protocol':'frozen external hypernetwork; actual mounted-B coefficient updates on answer-confirmed rows; no benchmark score',
            'checkpoint_sha256':filehash(parent/'trained.pt'),'rows':rows,
            'improved_rows':sum(x['improved'] for x in rows),
            'mean_pre_loss':sum(x['pre_loss'] for x in rows)/len(rows),
            'mean_post_loss':sum(x['post_loss'] for x in rows)/len(rows)}
    output.parent.mkdir(parents=True,exist_ok=True); output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'improved_rows':result['improved_rows'],'mean_pre_loss':result['mean_pre_loss'],'mean_post_loss':result['mean_post_loss'],'output':str(output)}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--device',default='cuda:0');p.add_argument('--rows',type=int,default=24);p.add_argument('--output',type=Path,required=True);a=p.parse_args();main(a.device,a.output,a.rows)
