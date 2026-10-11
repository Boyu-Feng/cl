"""Bind cleaned train-prefix transitions to the unchanged v84 reader protocol."""
import argparse
import json
from pathlib import Path
from ttcl.trajectory_hyperlora.prepare_cleaned_targets_v88 import OUTPUT as ANNOTATIONS,AUDIT,CLEAN_ROOT
from ttcl.trajectory_hyperlora.prepare_transition_reader_v82 import BOOT,evenly
from ttcl.trajectory_hyperlora.alfworld_online_from_empty import digest,file_hash
from ttcl.trajectory_hyperlora.alfworld_persistent_failure_delta_v36 import save
REVIEW=Path('data/annotations/alfworld_transition_reader_v88_20261010.json')


def build():
    annotations=json.loads(ANNOTATIONS.read_text());audit=json.loads(AUDIT.read_text());clean=json.loads((CLEAN_ROOT/'report.json').read_text())
    assert audit['passed'] and audit['episodes_replayed']==28 and audit['new_annotation_sha256']==file_hash(ANNOTATIONS)
    assert audit['report_sha256']==annotations['cleaning_report_sha256']==file_hash(CLEAN_ROOT/'report.json')
    assert annotations['runner_sha256']==file_hash(Path(__file__).with_name('prepare_cleaned_targets_v88.py'))
    for n,h in annotations['dependencies'].items():assert file_hash(Path(__file__).with_name(n))==h
    targets=annotations['targets'];assert annotations['targets_sha256']==digest(targets)
    rows={r['index']:r for r in clean['rows']};transitions=[];by_index={}
    for k,t in enumerate(targets):
        content={a:b for a,b in t.items() if a not in ('input_content_sha256','reviewed_target')}
        assert t['reviewed_target'] and t['input_content_sha256']==digest(content)
        i=t['stream_index'];step=t['target_step'];ep=rows[i]['episode'];assert i<120 and ep['reward']==1 and digest(ep)==t['episode_sha256']
        assert ep['trajectory'][step]['command']==t['target_action']
        before=ep['initial_observation'] if step==0 else ep['trajectory'][step-1]['observation'];assert before==t['current_observation']
        assert digest(ep['trajectory'][:step])==t['visible_prefix_sha256']
        item={'source_index':i,'source_step':step,'game':t['game'],'parent_target_binding':t['input_content_sha256'],
            'key_text':'Task initial observation:\n'+ep['initial_observation']+'\nObservation before action:\n'+before,
            'action_text':'Action taken:\n'+t['target_action'],'feedback_text':'Observed feedback after action:\n'+ep['trajectory'][step]['observation']}
        transitions.append({**item,'input_content_sha256':digest(item)});by_index.setdefault(i,[]).append(k)
    wins=annotations['own_success_indices'];assert wins==sorted(rows)
    def memory(before):
        indices=[i for i in wins if i<before and i in by_index][-8:]
        return [k for i in indices for k in evenly(by_index[i],8)]
    groups=[]
    for i in wins[8:]:
        ids=by_index.get(i,[]);assert ids,'Do not silently drop shortened training groups'
        bank=memory(i);assert bank and all(transitions[k]['source_index']<i for k in bank)
        item={'stream_index':i,'query_indices':ids,'memory_indices':bank,
            'memory_bindings':[transitions[k]['input_content_sha256'] for k in bank],
            'query_bindings':[targets[k]['input_content_sha256'] for k in ids],'update_order':[ids[s%len(ids)] for s in range(20)]}
        groups.append({**item,'input_content_sha256':digest(item)})
    assert len(groups)==20
    return {'annotation_sha256':file_hash(ANNOTATIONS),'cleaning_audit_sha256':file_hash(AUDIT),'bootstrap_sha256':file_hash(BOOT),
        'transitions':transitions,'groups':groups,'evaluation_memory_indices':memory(120),
        'dependencies':{n:file_hash(Path(__file__).with_name(n)) for n in ['prepare_transition_reader_v88.py','prepare_cleaned_targets_v88.py','transition_reader_v84.py','transition_reader_v82.py']},
        'budget':{'shared_bootstrap_steps':300,'steps_per_group':20,'steps_each_arm':400},
        'encoding':{'max_field_tokens':512,'truncation':'first half plus last half','pooling':'last','frozen_base_no_adapter':True},
        'scope':'Cleaned own-success train-prefix chronological replay, same v84 reader architecture and old common bootstrap. New action labels/prompts and memory bound to cleaned content. H and direct LoRA share400-step schedule; extra cleaning environment feedback counted separately.'},targets

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=('prepare','check'));a=p.parse_args();review,_=build()
    if a.command=='prepare':
        if REVIEW.exists():raise FileExistsError(REVIEW)
        save(REVIEW,review)
    else:assert review==json.loads(REVIEW.read_text())
    print(json.dumps({'transitions':len(review['transitions']),'groups':len(review['groups']),'budget':review['budget']}))
