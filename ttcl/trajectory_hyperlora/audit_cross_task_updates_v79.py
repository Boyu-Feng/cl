"""Reload saved factors and independently recompute cross-task action losses."""
import argparse
import json
import math
from pathlib import Path
import torch
from ttcl.trajectory_hyperlora.evaluate_cross_task_updates_v79 import prepare, REVIEW, OUTPUT, TEACHERS, BOOT
from ttcl.trajectory_hyperlora.alfworld_online_from_empty import file_hash
from ttcl.trajectory_hyperlora.alfworld_zero_shot_probe import load_agent
from ttcl.trajectory_hyperlora.alfworld_online_accumulate_lora_v1 import factors_hash
from ttcl.trajectory_hyperlora.train_alf_next_task import target_loss
from ttcl.trajectory_hyperlora.alfworld_persistent_failure_delta_v36 import save


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--device',default='cuda:3');args=p.parse_args()
    output=OUTPUT.with_suffix('.audit.json')
    if output.exists():raise FileExistsError(output)
    review,targets=prepare()
    assert review==json.loads(REVIEW.read_text())
    result=json.loads(OUTPUT.read_text())
    assert result['complete'] and not result['failures']
    assert result['review_sha256']==file_hash(REVIEW)
    source_audit=json.loads((TEACHERS/'audit.json').read_text())
    source_review_path=Path('data/annotations/alfworld_support_updates_v76_20261010.json')
    assert source_audit['review_sha256']==file_hash(source_review_path)
    agent,tok=load_agent(Path('current_work/delta-Mem/model/Qwen3-4B-Instruct-2507'),BOOT.parent/'hyper.pt',args.device,.6)
    agent.requires_grad_(False);agent.eval();agent.set_source(None)
    boot=torch.load(BOOT,map_location=args.device,weights_only=True)
    with torch.no_grad():
        for layer,a in zip(agent.adapters,boot['A'],strict=True):layer.a.copy_(a.to(layer.a))
    audit_rows=[]
    for pair,row in zip(review['pairs'],result['rows'],strict=True):
        assert pair['source_index']<pair['query_index']<120
        assert row['source_index']==pair['source_index'] and row['query_index']==pair['query_index']
        assert row['binding']==pair['input_content_sha256']
        artifact=TEACHERS/f"teacher_{pair['source_index']:03d}.pt"
        assert file_hash(artifact)==pair['artifact_sha256']
        state=torch.load(artifact,map_location=args.device,weights_only=True)
        assert state['A_sha256']==factors_hash(boot['A'])
        assert state['bootstrap_sha256']==file_hash(BOOT)
        assert state['group_binding']==pair['source_binding']
        measured={}
        for arm in ('base','delta','random_delta'):
            values=boot['B'] if arm=='base' else [b+d for b,d in zip(boot['B'],state[arm],strict=True)]
            for layer,b in zip(agent.adapters,values,strict=True):layer.b=b
            losses=[]
            with torch.no_grad():
                for k in pair['query_targets']:
                    assert targets[k]['stream_index']==pair['query_index']
                    text=tok.apply_chat_template(targets[k]['actor_messages'],tokenize=False,add_generation_prompt=True)
                    prefix=tok(text,add_special_tokens=False).input_ids
                    losses.append(float(target_loss(agent,tok,targets[k],prefix,args.device)))
            assert len(losses)==len(row[arm])
            assert all(math.isfinite(x) and math.isclose(x,y,rel_tol=1e-4,abs_tol=1e-5) for x,y in zip(losses,row[arm],strict=True))
            measured[arm]=sum(losses)/len(losses)
        gain=measured['base']-measured['delta']; random_gain=measured['random_delta']-measured['delta']
        assert math.isclose(gain,row['gain'],rel_tol=1e-4,abs_tol=1e-5)
        assert math.isclose(random_gain,row['gain_over_random'],rel_tol=1e-4,abs_tol=1e-5)
        positive=gain>1e-6 and random_gain>1e-6
        assert positive==row['positive']
        audit_rows.append({'source_index':pair['source_index'],'query_index':pair['query_index'],'gain':gain,'gain_over_random':random_gain,'positive':positive})
        print(json.dumps(audit_rows[-1]),flush=True)
    assert result['summary']=={'pairs':len(audit_rows),'positive':sum(x['positive'] for x in audit_rows)}
    save(output,{'passed':True,'report_sha256':file_hash(OUTPUT),'review_sha256':file_hash(REVIEW),
        'audit_runner_sha256':file_hash(Path(__file__)),'summary':result['summary'],'rows':audit_rows,
        'scope':'Reconstructed cross-task bindings, audited saved candidate factors and independent neural loss recomputation; no optimization replay or environment reward claim'})


if __name__=='__main__':main()
