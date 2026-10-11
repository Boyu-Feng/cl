"""Compare audited cleaned-data models with unchanged-architecture references."""
import json
from pathlib import Path
from ttcl.trajectory_hyperlora.alfworld_online_from_empty import file_hash
ROOT=Path('results/trajectory_hyperlora')
OUTPUT=ROOT/'alfworld_cleaned_reader_v88_comparison_20261010.json'


def load(version,episodes):
    p=ROOT/f'alfworld_transition_reader_{version}_eval_20261010.json';a=p.with_suffix('.audit.json');v=Path('data/annotations')/p.name
    d=json.loads(p.read_text());audit=json.loads(a.read_text());review=json.loads(v.read_text())
    assert d['complete'] and not d['failures'] and audit['passed'] and audit['episodes_replayed']==episodes
    assert audit['report_sha256']==file_hash(p) and d['review_sha256']==audit['review_sha256']==file_hash(v)
    assert d['summary']==audit['summary'] and len(d['rows'])==24
    return d,review,{'report_sha256':file_hash(p),'review_sha256':file_hash(v),'audit_sha256':file_hash(a)}


def main():
    if OUTPUT.exists():raise FileExistsError(OUTPUT)
    new,nv,nh=load('v88',72);old,ov,oh=load('v84',48);controls,cv,ch=load('v82',120)
    for ref in (ov,cv):
        for key in ('rollout_sha256','model_weights','max_steps','max_new_tokens','history_turns','loop_guard_max'):assert nv[key]==ref[key]
        for a,b in zip(nv['targets'],ref['targets'],strict=True):
            for key in ('index','game','game_sha256','initial_observation_sha256'):assert a[key]==b[key]
    paths={v:ROOT/f'alfworld_transition_reader_{v}_20261010' for v in ('v88','v84','v82')}
    training={v:json.loads((p/'training.json').read_text()) for v,p in paths.items()}
    manifest=json.loads((paths['v88']/'training_manifest.json').read_text())
    assert manifest['initial_reader_sha256']==file_hash(paths['v84']/'initial_reader.pt')
    for v in ('v84','v82'):
        for key in ('shared_A_sha256','shared_B_sha256'):assert training['v88'][key]==training[v][key]
    for arm in ('hyper','static'):
        groups=[g for g in training['v88']['groups'] if g['arm']==arm]
        assert sum(g['steps'] for g in groups)==400
    seq=lambda e:[(s['command'],s['observation']) for s in e['trajectory']]
    comparisons={}
    specs=[('real_vs_shifted','real',new,'shifted'),('real_vs_static','real',new,'static'),
           ('real_vs_original_reader','real',old,'real'),('static_vs_original_static','static',controls,'static'),
           ('real_vs_empty','real',controls,'empty'),('real_vs_base','real',controls,'base')]
    for name,arm,ref,refarm in specs:
        pairs=list(zip(new['rows'],ref['rows'],strict=True))
        comparisons[name]={'n':24,'new_wins':sum(n['arms'][arm]['reward'] for n,r in pairs),'reference_wins':sum(r['arms'][refarm]['reward'] for n,r in pairs),
            'gains':[n['index'] for n,r in pairs if n['arms'][arm]['reward']>r['arms'][refarm]['reward']],
            'losses':[n['index'] for n,r in pairs if n['arms'][arm]['reward']<r['arms'][refarm]['reward']],
            'different_sequences':[n['index'] for n,r in pairs if seq(n['arms'][arm])!=seq(r['arms'][refarm])]}
    cleanup=ROOT/'alfworld_success_loop_cleaning_v87_20261010/report.json';annotation=Path('data/annotations/alfworld_cleaned_targets_v88_20261010.json')
    cleaned=json.loads(cleanup.read_text());labels=json.loads(annotation.read_text())
    result={'bindings':{'new':nh,'original_reader':oh,'controls':ch,'cleaning_report':file_hash(cleanup),'new_labels':file_hash(annotation)},
        'summary':new['summary'],'comparisons':comparisons,'cleaning_environment_budget':cleaned['summary'],
        'annotation_environment_budget':{'episodes':labels['annotation_environment_replays'],'steps':labels['annotation_environment_steps']},
        'scope':'Exposed24 development comparison. Same H architecture/initial weights and common LoRA bootstrap; cleaned actor targets and source-memory content both change. Both new arms get the cleaning feedback; no equal-environment-budget claim against original uncleaned models. Gradient steps match, parameter counts and encoding compute do not.'}
    OUTPUT.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(comparisons),flush=True)

if __name__=='__main__':main()
