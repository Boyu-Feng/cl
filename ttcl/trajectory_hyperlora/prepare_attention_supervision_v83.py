"""Bind optional retrieval supervision derived only from existing train labels.

Exact action identity is a proxy for useful memory, not semantic equivalence.
No evaluation-task label is read. No wrong-memory loss is maximized.
"""
import argparse
import json
from pathlib import Path
from ttcl.trajectory_hyperlora.prepare_transition_reader_v82 import build,REVIEW as PARENT_REVIEW
from ttcl.trajectory_hyperlora.alfworld_online_from_empty import digest,file_hash
from ttcl.trajectory_hyperlora.alfworld_persistent_failure_delta_v36 import save

REVIEW=Path('data/annotations/alfworld_attention_supervision_v83_20261010.json')


def prepare():
    parent,targets=build();assert parent==json.loads(PARENT_REVIEW.read_text())
    groups=[];coverage=0
    for group in parent['groups']:
        annotations=[]
        for k in group['query_indices']:
            memory=group['memory_indices']
            positions=[j for j,m in enumerate(memory) if targets[m]['target_action']==targets[k]['target_action']]
            assert all(targets[m]['stream_index']<targets[k]['stream_index'] for m in memory)
            assert targets[k]['stream_index']==group['stream_index']<120
            content={'query_index':k,'query_binding':targets[k]['input_content_sha256'],
                'memory_bindings':group['memory_bindings'],'positive_positions':positions,
                'derivation':'Exact command equality with the already reviewed current training action; earlier completed own-success sources only',
                'no_match_behavior':'Zero auxiliary loss, ordinary actor loss still trained'}
            annotations.append({**content,'input_content_sha256':digest(content),'reviewed_target':True})
        by_query={a['query_index']:a for a in annotations}
        coverage+=sum(bool(by_query[k]['positive_positions']) for k in group['update_order'])
        item={'parent_group_binding':group['input_content_sha256'],'stream_index':group['stream_index'],
              'update_order':group['update_order'],'annotations':annotations}
        groups.append({**item,'input_content_sha256':digest(item)})
    return {'parent_review_sha256':file_hash(PARENT_REVIEW),'groups':groups,
        'runner_sha256':file_hash(Path(__file__)),'covered_steps':coverage,
        'budget':parent['budget'],'auxiliary_weight':.1,
        'objective':'Actor action CE plus 0.1 * negative log attention mass on earlier exact-action matches, only when present',
        'scope':'Prepared training-only candidate. No training or performance result. Same existing labels and step schedule; extra loss/computation for H. Exact action matches may still be contextually irrelevant.'}


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=('prepare','check'));args=p.parse_args();review=prepare()
    if args.command=='prepare':
        if REVIEW.exists():raise FileExistsError(REVIEW)
        save(REVIEW,review)
    else:assert review==json.loads(REVIEW.read_text())
    print(json.dumps({'groups':len(review['groups']),'covered_steps':review['covered_steps'],'budget':review['budget']}))

if __name__=='__main__':main()
