"""Measure whether an external hypernetwork responds to trajectory content."""
from __future__ import annotations
import argparse, json, random
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from ttcl.trajectory_hyperlora.direct_composition_pilot import DirectRelationHyperLoRA
from ttcl.trajectory_hyperlora.joint_domains_v93 import MODEL, digest, filehash

def main(args):
    if args.output.exists(): raise FileExistsError(args.output)
    device=torch.device(args.device)
    ext=torch.load(args.external/'trained.pt',map_location='cpu',weights_only=True)
    cache=torch.load(args.external/'cache.pt',map_location='cpu',weights_only=True)
    tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(MODEL,torch_dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True).to(device)
    agent=DirectRelationHyperLoRA(model,rank=4,layers=2,encoder_kind='contextual',task_conditioned=True).to(device)
    named=dict(agent.named_parameters())
    for n,v in ext['hyper'].items(): named[n].data.copy_(v.to(device))
    agent.requires_grad_(False); agent.eval()
    fields={k:{x:v.to(device) for x,v in val.items()} for k,val in cache['fields'].items()}
    prepared=cache['prepared']
    empty_key=digest('Causal trajectory prefix is empty.')
    if empty_key not in fields: raise ValueError('Missing encoded empty source')
    valid=[i for i,r in enumerate(prepared) if r['source'] != empty_key]
    rng=random.Random(153); rng.shuffle(valid); valid=valid[:args.limit]
    if len(valid)<2: raise ValueError('Need at least two nonempty sources')
    def score(r):
        ids=torch.tensor([r['prefix']+r['answer']],device=device)
        with torch.no_grad(): logits=model(input_ids=ids,use_cache=False).logits
        s=torch.log_softmax(logits[:,len(r['prefix'])-1:-1].float(),-1)
        y=torch.tensor(r['answer'],device=device).view(1,-1)
        return float(s.gather(-1,y.unsqueeze(-1)).mean())
    out=[]
    for pos,i in enumerate(valid):
        r=prepared[i]; j=valid[(pos+1)%len(valid)]
        assert prepared[j]['source'] != r['source']
        target=fields[r['query']]; vals={}
        saved={}
        for label,key in [('true',r['source']),('shuffled',prepared[j]['source']),('empty_source',empty_key)]:
            agent.set_source(fields[key],target_fields=target)
            with torch.no_grad():
                bs=[a.b.detach().float().clone() for a in agent.adapters]
            saved[label]=bs
            vals[label]={'score':score(r),'b_norm':float(torch.stack([b.pow(2).mean() for b in bs]).mean())}
        agent.set_source(None); vals['base']={'score':score(r),'b_norm':0.0}
        vals['true_shuffled_b_rmse']=float(torch.stack([(a-b).pow(2).mean() for a,b in zip(saved['true'],saved['shuffled'])]).mean().sqrt())
        out.append({'row_index':i,**vals})
    # Scores are mean token log likelihoods of recorded actions, not success rates.
    result={'protocol':'external_source_dependence_v155_corrected', 'checkpoint_sha256':filehash(args.external/'trained.pt'), 'cache_sha256':filehash(args.external/'cache.pt'),'external':str(args.external),'rows':out,
            'summary':{'n':len(out),'true_minus_base_logp':sum(x['true']['score']-x['base']['score'] for x in out)/len(out),
                       'shuffled_minus_base_logp':sum(x['shuffled']['score']-x['base']['score'] for x in out)/len(out),
                       'true_minus_empty_source_logp':sum(x['true']['score']-x['empty_source']['score'] for x in out)/len(out),
                       'true_minus_shuffled_logp':sum(x['true']['score']-x['shuffled']['score'] for x in out)/len(out),
                       'true_b_norm':sum(x['true']['b_norm'] for x in out)/len(out),
                       'shuffled_b_norm':sum(x['shuffled']['b_norm'] for x in out)/len(out)}}
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result['summary'],indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--external',type=Path,required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--device',default='cuda:0'); p.add_argument('--limit',type=int,default=24); main(p.parse_args())
