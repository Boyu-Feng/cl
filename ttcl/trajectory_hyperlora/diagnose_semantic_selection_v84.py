"""Check fixed selector against existing chronological training action labels.

This is a training diagnostic, not an evaluation score or causal utility test.
"""
import json
from pathlib import Path
import statistics
import torch
from ttcl.trajectory_hyperlora.train_transition_reader_v82 import checked_cache
from ttcl.trajectory_hyperlora.prepare_attention_supervision_v83 import prepare,REVIEW
from ttcl.trajectory_hyperlora.transition_reader_v84 import TransitionReader
from ttcl.trajectory_hyperlora.alfworld_online_from_empty import file_hash


def main():
    torch.set_num_threads(2)
    out=Path('results/trajectory_hyperlora/alfworld_semantic_selection_v84_20261010.json')
    if out.exists():raise FileExistsError(out)
    review,targets,cache,_=checked_cache()
    labels=prepare();assert labels==json.loads(REVIEW.read_text())
    state_path=Path('results/trajectory_hyperlora/alfworld_transition_reader_v84_20261010/reader.pt')
    state=torch.load(state_path,map_location='cpu',weights_only=True)
    reader=TransitionReader(state['feature_dim'],state['factor_shapes'],state['width'],state['top_k'],state['temperature'])
    reader.load_state_dict(state['state']);reader.eval()
    field=lambda k,name:cache['values'][cache['rows'][k][name]]
    rows=[]
    with torch.no_grad():
        for g,l in zip(review['groups'],labels['groups'],strict=True):
            assert g['input_content_sha256']==l['parent_group_binding']
            bank=[torch.cat([field(k,f) for k in g['memory_indices']],dim=0) for f in ('key_text','action_text','feedback_text')]
            by_query={x['query_index']:x for x in l['annotations']}
            for k in g['update_order']:
                _,w=reader(field(k,'key_text'),*bank);w=w[0]
                pos=by_query[k]['positive_positions']
                rows.append({'stream_index':g['stream_index'],'query_index':k,'query_binding':targets[k]['input_content_sha256'],
                    'has_match':bool(pos),'memory_size':len(w),'selected_count':int((w>0).sum()),
                    'matching_mass':float(w[pos].sum()) if pos else 0.,'uniform_matching_mass':len(pos)/len(w),
                    'top1_matches':int(int(w.argmax()) in pos),
                    'any_selected_match':bool((w[pos]>0).any()) if pos else False})
    covered=[x for x in rows if x['has_match']]
    assert len(rows)==400 and len(covered)==labels['covered_steps']
    summary={'training_steps':len(rows),'covered_steps':len(covered),
        'mean_matching_mass_when_covered':statistics.mean(x['matching_mass'] for x in covered),
        'mean_uniform_matching_mass_when_covered':statistics.mean(x['uniform_matching_mass'] for x in covered),
        'top1_matches_when_covered':sum(x['top1_matches'] for x in covered),
        'any_selected_match_when_covered':sum(x['any_selected_match'] for x in covered),
        'selected_count_range':[min(x['selected_count'] for x in rows),max(x['selected_count'] for x in rows)]}
    result={'summary':summary,'rows':rows,'labels_sha256':file_hash(REVIEW),'reader_checkpoint_sha256':file_hash(state_path),
        'runner_sha256':file_hash(Path(__file__)),
        'scope':'Existing training labels only, weighted by actual update repetitions. Exact command identity is a proxy, not contextual action utility; selector uses frozen observation features and no target action. No new labels or test outcomes.'}
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(summary))

if __name__=='__main__':main()
