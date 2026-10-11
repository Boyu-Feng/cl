"""Audit causal bindings, recompute saved-feature reads, replay environment."""
import argparse
import json
import math
from pathlib import Path
import torch
from transformers import AutoTokenizer
from ttcl.trajectory_hyperlora.evaluate_transition_reader_v88 import bindings,REVIEW,OUTPUT,ROOT,MODEL,DATA,ARMS
from ttcl.trajectory_hyperlora.train_transition_reader_v88 import checked_cache
from ttcl.trajectory_hyperlora.transition_reader_v84 import TransitionReader
from ttcl.trajectory_hyperlora.transition_reader_v82 import effective_squared_norm
from ttcl.trajectory_hyperlora.alfworld_online_from_empty import digest,file_hash
from ttcl.trajectory_hyperlora.alfworld_online_accumulate_lora_v1 import factors_hash
from ttcl.trajectory_hyperlora.alfworld_persistent_failure_delta_v36 import save
from ttcl.trajectory_hyperlora.audit_my_online_update_v28_reward import replay


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--device',default='cuda:6');args=p.parse_args()
    torch.set_num_threads(2)
    output=OUTPUT.with_suffix('.audit.json')
    if output.exists():raise FileExistsError(output)
    review=bindings();assert review==json.loads(REVIEW.read_text())
    report=json.loads(OUTPUT.read_text());assert report['complete'] and not report['failures']
    assert report['review_sha256']==file_hash(REVIEW)
    assert report['query_features_sha256']==file_hash(ROOT/'eval_query_features.pt')
    queries=torch.load(ROOT/'eval_query_features.pt',map_location='cpu',weights_only=True)
    _,_,cache,_=checked_cache()
    state=torch.load(ROOT/'reader.pt',map_location=args.device,weights_only=True)
    assert state['top_k']==8 and state['temperature']==.05
    reader=TransitionReader(state['feature_dim'],state['factor_shapes'],state['width'],state['top_k'],state['temperature']).to(args.device)
    reader.load_state_dict(state['state']);reader.eval();reader.requires_grad_(False)
    shared=state['shared_B'];bases=state['shared_A'];memory=[]
    for field in ('key_text','action_text','feedback_text'):
        memory.append(torch.cat([cache['values'][cache['rows'][k][field]] for k in review['memory_indices']],dim=0).to(args.device))
    obs,act,fb=memory;tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
    reads_checked=0;episodes_checked=0;used=set()
    for row,target in zip(report['rows'],review['targets'],strict=True):
        assert (row['index'],row['game'],row['input_content_sha256'])==(target['index'],target['game'],target['input_content_sha256'])
        assert set(row['arms'])==set(ARMS)
        for arm in ARMS:
            ep=row['arms'][arm]
            assert ep['status']=='complete' and ep['steps']<=30
            assert digest(ep['initial_observation'])==target['initial_observation_sha256']
            assert len(ep['parameter_reads'])==(ep['steps'] if arm in ('real','shifted') else 0)
            for i,read in enumerate(ep['parameter_reads']):
                before=ep['initial_observation'] if i==0 else ep['trajectory'][i-1]['observation']
                text='Task initial observation:\n'+ep['initial_observation']+'\nObservation before action:\n'+before
                key=digest(text);assert read['query_text_sha256']==key
                used.add(key);q,token_binding=queries[key]
                ids=tok(text,add_special_tokens=False).input_ids;original=len(ids)
                if len(ids)>512:ids=ids[:256]+ids[-256:]
                expected={'input_ids_sha256':digest(ids),'original_tokens':original,'retained_tokens':len(ids)}
                assert token_binding==expected and all(read[k]==v for k,v in expected.items())
                with torch.no_grad():
                    values,info=reader.adapted_factors(q.to(args.device),obs,act.roll(1,0) if arm=='shifted' else act,fb.roll(1,0) if arm=='shifted' else fb,shared,bases)
                assert factors_hash(values)==read['B_sha256']
                assert torch.allclose(info['weights'].cpu()[0],torch.tensor(read['attention_weights']),rtol=1e-5,atol=1e-6)
                assert math.isclose(float(info['scale']),read['clip_scale'],rel_tol=1e-5,abs_tol=1e-6)
                delta=[v-b for v,b in zip(values,shared,strict=True)]
                relative=float(effective_squared_norm(delta,bases).sqrt()/effective_squared_norm(shared,bases).sqrt())
                assert relative<=.25001 and math.isclose(relative,read['effective_relative_norm'],rel_tol=1e-4,abs_tol=1e-6)
                reads_checked+=1
            replay(DATA/target['game'],ep);episodes_checked+=1
        print(json.dumps({'index':row['index'],'episodes_replayed':episodes_checked,'parameter_reads_checked':reads_checked}),flush=True)
    assert used==set(queries)
    summary={'n':len(report['rows']),**{a:sum(r['arms'][a]['reward'] for r in report['rows']) for a in ARMS}}
    assert summary==report['summary'] and episodes_checked==72
    save(output,{'passed':True,'report_sha256':file_hash(OUTPUT),'review_sha256':file_hash(REVIEW),
        'audit_runner_sha256':file_hash(Path(__file__)),'summary':summary,'episodes_replayed':episodes_checked,
        'parameter_reads_checked':reads_checked,'scope':'Causal memory and query-content bindings; exact reader B recomputation from saved frozen features; attention/norm checks; environment replay. Frozen encoder features and autoregressive actions are not independently regenerated; no optimizer replay.'})

if __name__=='__main__':main()
