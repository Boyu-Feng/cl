"""Measure effective parameter sensitivity on fixed visible v82 queries.

Read-only checkpoint diagnostic, not a task-score or new held-out evaluation.
"""
import hashlib
import json
import math
from pathlib import Path
import statistics
import torch
from ttcl.trajectory_hyperlora.transition_reader_v84 import TransitionReader
from ttcl.trajectory_hyperlora.transition_reader_v82 import effective_squared_norm

ROOT=Path('results/trajectory_hyperlora')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def mean(values):
    return statistics.mean(values)

def main():
    torch.set_num_threads(2)
    output=ROOT/'alfworld_transition_reader_v84_sensitivity_20261010.json'
    if output.exists():raise FileExistsError(output)
    parent=ROOT/'alfworld_transition_reader_v82_20261010'
    report_path=ROOT/'alfworld_transition_reader_v82_eval_20261010.json'
    report=json.loads(report_path.read_text())
    assert report['complete'] and not report['failures']
    query_path=parent/'eval_query_features.pt'
    assert report['query_features_sha256']==sha(query_path)
    queries=torch.load(query_path,map_location='cpu',weights_only=True)
    review_path=Path('data/annotations')/report_path.name
    assert report['review_sha256']==sha(review_path)
    review=json.loads(review_path.read_text())
    cache_path=parent/'features.pt'
    cache=torch.load(cache_path,map_location='cpu',weights_only=True)
    memory=[torch.cat([cache['values'][cache['rows'][k][field]] for k in review['memory_indices']],dim=0)
            for field in ('key_text','action_text','feedback_text')]
    obs,act,fb=memory
    ids=[r['arms']['real']['parameter_reads'][0]['query_text_sha256'] for r in report['rows']]
    results={}
    with torch.no_grad():
        for version in ('v84',):
            path=ROOT/f'alfworld_transition_reader_{version}_20261010/reader.pt'
            state=torch.load(path,map_location='cpu',weights_only=True)
            assert state['top_k']==8 and state['temperature']==.05
            reader=TransitionReader(state['feature_dim'],state['factor_shapes'],state['width'],state['top_k'],state['temperature'])
            reader.load_state_dict(state['state']);reader.eval()
            shared,bases=state['shared_B'],state['shared_A']
            reference=effective_squared_norm(shared,bases).sqrt().item()
            norm=lambda xs:effective_squared_norm(xs,bases).sqrt().item()
            sub=lambda xs,ys:[x-y for x,y in zip(xs,ys,strict=True)]
            rows=[];deltas=[]
            for index,key in enumerate(ids):
                q=queries[key][0]
                real,info=reader.adapted_factors(q,obs,act,fb,shared,bases)
                wrong,_=reader.adapted_factors(q,obs,act.roll(1,0),fb.roll(1,0),shared,bases)
                delta=sub(real,shared);wrong_delta=sub(wrong,shared)
                nr,nw=norm(delta),norm(wrong_delta);distance=norm(sub(real,wrong))
                weights=info['weights'][0].tolist()
                rows.append({'index':report['rows'][index]['index'],'query_sha256':key,
                    'normalized_attention_entropy':-sum(w*math.log(w) for w in weights if w>0)/math.log(len(weights)),
                    'max_attention':max(weights),'real_update_relative_to_bootstrap':nr/reference,
                    'mismatch_distance_relative_to_real_update':distance/max(nr,1e-20),
                    'real_wrong_update_cosine':(nr*nr+nw*nw-distance*distance)/max(2*nr*nw,1e-20)})
                deltas.append(delta)
            pair_distances=[norm(sub(deltas[i],deltas[j]))/max(norm(deltas[i]),1e-20)
                            for i in range(len(deltas)) for j in range(i)]
            results[version]={'checkpoint_sha256':sha(path),'rows':rows,
                'mean_entropy':mean(r['normalized_attention_entropy'] for r in rows),
                'mean_mismatch_relative_distance':mean(r['mismatch_distance_relative_to_real_update'] for r in rows),
                'mean_real_wrong_cosine':mean(r['real_wrong_update_cosine'] for r in rows),
                'mean_between_query_relative_distance':mean(pair_distances)}
    result={'report_sha256':sha(report_path),'review_sha256':sha(review_path),'query_features_sha256':sha(query_path),
        'cache_sha256':sha(cache_path),'runner_sha256':sha(Path(__file__)), 'results':results,
        'scope':'Fixed 24 visible initial-state queries from exposed v82 development tasks, CPU float32 diagnostic. Effective BA/r metric with frozen shared A. Does not regenerate frozen features, prove causality or score task success.'}
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({v:{k:x for k,x in d.items() if k not in ('rows','checkpoint_sha256')} for v,d in results.items()}))

if __name__=='__main__':main()
