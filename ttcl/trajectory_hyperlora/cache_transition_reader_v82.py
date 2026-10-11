"""Frozen no-adapter field encoding with explicit token/content bindings."""
import argparse
import json
from pathlib import Path
import torch
from ttcl.trajectory_hyperlora.prepare_transition_reader_v82 import build,REVIEW,BOOT
from ttcl.trajectory_hyperlora.alfworld_online_from_empty import digest,file_hash
from ttcl.trajectory_hyperlora.alfworld_persistent_failure_delta_v36 import save
from ttcl.trajectory_hyperlora.alfworld_zero_shot_probe import load_agent

ROOT=Path('results/trajectory_hyperlora/alfworld_transition_reader_v82_20261010')
MODEL=Path('current_work/delta-Mem/model/Qwen3-4B-Instruct-2507')
FIELDS=('key_text','action_text','feedback_text')


def encode(agent,tok,text,device,limit=512):
    ids=tok(text,add_special_tokens=False).input_ids
    original=len(ids)
    if original>limit:ids=ids[:limit//2]+ids[-(limit-limit//2):]
    if not ids:raise ValueError('Empty field')
    agent.set_source(None)
    with torch.no_grad():
        tensor=torch.tensor([ids],device=device)
        output=agent.model(input_ids=tensor,attention_mask=torch.ones_like(tensor),output_hidden_states=True,use_cache=False)
        vector=output.hidden_states[-1][:,-1,:].detach().float().cpu()
    assert torch.isfinite(vector).all()
    return vector,{'input_ids_sha256':digest(ids),'original_tokens':original,'retained_tokens':len(ids)}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--device',default='cuda:3');args=p.parse_args()
    review,_=build();assert review==json.loads(REVIEW.read_text())
    if ROOT.exists():raise FileExistsError(ROOT)
    ROOT.mkdir(parents=True)
    manifest={'review_sha256':file_hash(REVIEW),'runner_sha256':file_hash(Path(__file__)),
              'loader_sha256':file_hash(Path(__file__).with_name('alfworld_zero_shot_probe.py')),
              'adapter_implementation_sha256':file_hash(Path(__file__).with_name('direct_composition_pilot.py')),
              'loader_checkpoint_sha256':file_hash(BOOT.parent/'hyper.pt'),
              'model_config_sha256':file_hash(MODEL/'config.json'),
              'model_weights':{str(p):file_hash(p) for p in sorted(MODEL.glob('*.safetensors'))},
              'encoding':review['encoding'],'scope':'Frozen field encoding. Cached future training fields are not available to a group unless its reviewed memory indices permit them.'}
    save(ROOT/'encoding_manifest.json',manifest)
    agent,tok=load_agent(MODEL,BOOT.parent/'hyper.pt',args.device,.6)
    agent.requires_grad_(False);agent.eval();agent.set_source(None)
    values={};bindings={};rows=[]
    result={'complete':False,'manifest_sha256':file_hash(ROOT/'encoding_manifest.json'),'failures':[]}
    try:
        for i,t in enumerate(review['transitions']):
            row={'input_content_sha256':t['input_content_sha256']}
            for field in FIELDS:
                key=digest(t[field]);row[field]=key
                if key not in values:
                    values[key],bindings[key]=encode(agent,tok,t[field],args.device,review['encoding']['max_field_tokens'])
            rows.append(row)
            result['summary']={'transitions':len(rows),'unique_fields':len(values)}
            save(ROOT/'cache_report.json',result)
            if (i+1)%25==0:print(json.dumps(result['summary']),flush=True)
        artifact=ROOT/'features.pt';torch.save({'values':values,'rows':rows,'token_bindings':bindings,'review_sha256':file_hash(REVIEW)},artifact)
        result.update(complete=True,features_sha256=file_hash(artifact));save(ROOT/'cache_report.json',result)
        print(json.dumps(result['summary']),flush=True)
    except Exception as exc:
        result['failures'].append(repr(exc));save(ROOT/'cache_report.json',result);raise

if __name__=='__main__':main()
