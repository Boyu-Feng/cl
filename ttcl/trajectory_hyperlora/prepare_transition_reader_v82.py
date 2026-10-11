"""Bind causal past-task transition memories for the v82 reader experiment."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from ttcl.trajectory_hyperlora.train_causal_prefix_v70 import checked
from ttcl.trajectory_hyperlora.alfworld_online_from_empty import digest,file_hash
from ttcl.trajectory_hyperlora.alfworld_persistent_failure_delta_v36 import save

PARENT=Path('results/trajectory_hyperlora/alfworld_cold_joint_memory_v63_hybrid_20261010.json')
PRIOR=Path('data/annotations/alfworld_causal_prefix_v70_train120_20261010.json')
BOOT=Path('results/trajectory_hyperlora/alfworld_centered_experience_v75b_20261010/bootstrap.pt')
REVIEW=Path('data/annotations/alfworld_transition_reader_v82_20261010.json')

def evenly(indices,limit):
    n=min(limit,len(indices))
    return [indices[k*len(indices)//n] for k in range(n)] if n else []


def build():
    prior=checked(SimpleNamespace(review=PRIOR));parent=json.loads(PARENT.read_text())
    audit=json.loads(PARENT.with_suffix('.audit.json').read_text())
    assert audit['passed'] and audit['report_sha256']==file_hash(PARENT)
    training=json.loads((BOOT.parent/'training.json').read_text())
    assert training['complete'] and training['bootstrap_sha256']==file_hash(BOOT)
    targets=prior['targets'];transitions=[];by_index={}
    for k,t in enumerate(targets):
        i=t['stream_index'];step=t['target_step'];ep=parent['rows'][i]['episode']
        assert i<120 and ep['reward']==1
        assert ep['trajectory'][step]['command']==t['target_action']
        before=ep['initial_observation'] if step==0 else ep['trajectory'][step-1]['observation']
        assert before==t['current_observation']
        item={'source_index':i,'source_step':step,'game':t['game'],'parent_target_binding':t['input_content_sha256'],
              'key_text':'Task initial observation:\n'+ep['initial_observation']+'\nObservation before action:\n'+before,
              'action_text':'Action taken:\n'+t['target_action'],
              'feedback_text':'Observed feedback after action:\n'+ep['trajectory'][step]['observation']}
        transitions.append({**item,'input_content_sha256':digest(item)})
        by_index.setdefault(i,[]).append(k)
    def memory(before):
        indices=[i for i in prior['own_success_indices'] if i<before and i in by_index][-8:]
        return [k for i in indices for k in evenly(by_index[i],8)]
    groups=[]
    for i in prior['own_success_indices'][8:]:
        ids=by_index.get(i,[])
        if not ids:continue
        bank=memory(i);assert bank and all(transitions[k]['source_index']<i for k in bank)
        item={'stream_index':i,'query_indices':ids,'memory_indices':bank,
              'memory_bindings':[transitions[k]['input_content_sha256'] for k in bank],
              'query_bindings':[targets[k]['input_content_sha256'] for k in ids],
              'update_order':[ids[s%len(ids)] for s in range(20)]}
        groups.append({**item,'input_content_sha256':digest(item)})
    dependencies=['prepare_transition_reader_v82.py','transition_reader_v82.py','train_causal_prefix_v70.py','prepare_causal_prefix_v70.py']
    return {'prior_review_sha256':file_hash(PRIOR),'parent_report_sha256':file_hash(PARENT),'parent_audit_sha256':file_hash(PARENT.with_suffix('.audit.json')),
        'bootstrap_sha256':file_hash(BOOT),'transitions':transitions,'groups':groups,'evaluation_memory_indices':memory(120),
        'dependencies':{n:file_hash(Path(__file__).with_name(n)) for n in dependencies},
        'budget':{'shared_bootstrap_steps':300,'steps_per_group':20,'steps_each_arm':sum(len(g['update_order']) for g in groups)},
        'encoding':{'max_field_tokens':512,'truncation':'first half plus last half','pooling':'last','frozen_base_no_adapter':True},
        'scope':'Offline chronological replay of own train-prefix successes; memory uses only earlier completed tasks and reviewed action targets. Same later action target schedule for H and direct LoRA. No unseen labels, no same-task retry, no claimed live-online collection.'},targets


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=('prepare','check'));a=p.parse_args()
    review,_=build()
    if a.command=='prepare':
        if REVIEW.exists():raise FileExistsError(REVIEW)
        save(REVIEW,review)
    else:assert json.loads(REVIEW.read_text())==review
    print(json.dumps({'transitions':len(review['transitions']),'groups':len(review['groups']),'budget':review['budget'],'eval_memory_transitions':len(review['evaluation_memory_indices'])}))

if __name__=='__main__':main()
