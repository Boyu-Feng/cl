"""Join audited auxiliary-H results with audited unchanged v82 controls."""
import hashlib
import json
from pathlib import Path
ROOT=Path('results/trajectory_hyperlora')
OUTPUT=ROOT/'alfworld_attention_supervision_v83_comparison_20261010.json'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()


def load(version,episodes):
    report_path=ROOT/f'alfworld_transition_reader_{version}_eval_20261010.json'
    review_path=Path('data/annotations')/report_path.name
    audit_path=report_path.with_suffix('.audit.json')
    r=json.loads(report_path.read_text());v=json.loads(review_path.read_text());a=json.loads(audit_path.read_text())
    assert r['complete'] and not r['failures'] and a['passed']
    assert a['report_sha256']==sha(report_path) and r['review_sha256']==a['review_sha256']==sha(review_path)
    assert len(r['rows'])==24 and a['episodes_replayed']==episodes and r['summary']==a['summary']
    return r,v,{'report':sha(report_path),'review':sha(review_path),'audit':sha(audit_path)}


def main():
    if OUTPUT.exists():raise FileExistsError(OUTPUT)
    new,nv,nh=load('v83',48);old,ov,oh=load('v82',120)
    assert nv['reference_review_sha256']==oh['review']
    for key in ('static_sha256','train_review_sha256','rollout_sha256','model_weights','memory_indices','memory_bindings','max_steps','max_new_tokens','history_turns','loop_guard_max'):
        assert nv[key]==ov[key],key
    pairs=list(zip(new['rows'],old['rows'],strict=True))
    for n,o,nt,ot in zip(new['rows'],old['rows'],nv['targets'],ov['targets'],strict=True):
        assert n['input_content_sha256']==nt['input_content_sha256'] and o['input_content_sha256']==ot['input_content_sha256']
        for key in ('index','game','game_sha256','initial_observation_sha256'):assert nt[key]==ot[key]
        assert n['arms']['real']['initial_observation']==o['arms']['real']['initial_observation']
    ntrain=json.loads((ROOT/'alfworld_transition_reader_v83_20261010/training.json').read_text())
    otrain=json.loads((ROOT/'alfworld_transition_reader_v82_20261010/training.json').read_text())
    assert ntrain['shared_A_sha256']==otrain['shared_A_sha256'] and ntrain['shared_B_sha256']==otrain['shared_B_sha256']
    assert ntrain['static_sha256']==otrain['static_sha256'] and ntrain['static_reused']
    seq=lambda e:[(s['command'],s['observation']) for s in e['trajectory']]
    comparisons={}
    for ref in ('old_real','new_shifted','empty','static','base'):
        selected=[(n,n['arms']['shifted'] if ref=='new_shifted' else o['arms']['real' if ref=='old_real' else ref]) for n,o in pairs]
        comparisons[ref]={'n':24,'auxiliary_H_wins':sum(n['arms']['real']['reward'] for n,r in selected),
            'reference_wins':sum(r['reward'] for n,r in selected),
            'gains':[n['index'] for n,r in selected if n['arms']['real']['reward']>r['reward']],
            'losses':[n['index'] for n,r in selected if n['arms']['real']['reward']<r['reward']],
            'different_sequences':[n['index'] for n,r in selected if seq(n['arms']['real'])!=seq(r)]}
    result={'bindings':{'v83':nh,'v82':oh},'comparisons':comparisons,
        'new_rollouts':48,'reused_control_rollouts':72,
        'scope':'Exposed24 frozen evaluation; verified common bootstrap and exact same static checkpoint. H and static have same target labels/optimizer budget but different parameter counts, encoding costs and H auxiliary objective. H initial numerical state was not saved in v82; no exact single-loss causal claim.'}
    OUTPUT.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(comparisons),flush=True)

if __name__=='__main__':main()
