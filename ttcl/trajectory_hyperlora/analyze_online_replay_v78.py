"""Audited paired comparison of independent original and replay LoRA streams."""
import hashlib
import json
from pathlib import Path
ROOT=Path('results/trajectory_hyperlora')
OUTPUT=ROOT/'alfworld_online_replay_v78_comparison_20261010.json'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def load(version):
    p=ROOT/f'alfworld_cold_own_static_{version}_unrestricted_20261010.json'
    r=json.loads(p.read_text());a=p.with_suffix('.audit.json');audit=json.loads(a.read_text())
    assert r['complete'] and not r['failures'] and len(r['rows'])==240
    assert audit['passed'] and audit['episodes_replayed']==240 and audit['report_sha256']==sha(p)
    assert audit['summary']==r['summary']
    return r,{'report':sha(p),'audit':sha(a)}

def compare(rows):
    sequence=lambda ep:[(s['command'],s['observation']) for s in ep['trajectory']]
    return {'n':len(rows),'replay_wins':sum(n['episode']['reward'] for n,o in rows),
        'original_wins':sum(o['episode']['reward'] for n,o in rows),
        'gains':[n['index'] for n,o in rows if n['episode']['reward']>o['episode']['reward']],
        'losses':[n['index'] for n,o in rows if n['episode']['reward']<o['episode']['reward']],
        'identical_sequences':sum(sequence(n['episode'])==sequence(o['episode']) for n,o in rows),
        'replay_training_steps':sum(n['update']['steps'] for n,o in rows),
        'original_training_steps':sum(o['update']['steps'] for n,o in rows)}

def main():
    if OUTPUT.exists():raise FileExistsError(OUTPUT)
    new,nh=load('v78');old,oh=load('v65')
    for key in ('seed','limit','rank','layers','bootstrap_wins','bootstrap_steps','bootstrap_lr','online_steps_per_eligible_win','online_lr','max_steps','max_new_tokens','history_turns','loop_guard_max','constrain_actions','eligibility','source_report_sha256','model_config_sha256','initial_A_sha256','initial_parameters_sha256'):
        assert new[key]==old[key],key
    pairs=list(zip(new['rows'],old['rows'],strict=True))
    for n,o in pairs:
        assert (n['index'],n['game'])==(o['index'],o['game'])
        assert n['episode']['initial_observation']==o['episode']['initial_observation']
    boot_new=new['bootstrap']['index'];boot_old=old['bootstrap']['index']
    assert boot_new==boot_old and new['bootstrap']==old['bootstrap']
    assert all(n['episode']==o['episode'] for n,o in pairs[:boot_new+1])
    after_new=new['rows'][boot_new]['parameters_after_sha256'];after_old=old['rows'][boot_old]['parameters_after_sha256']
    result={'bindings':{'replay':nh,'original':oh},'overall':compare(pairs),
        'first120':compare(pairs[:120]),'last120':compare(pairs[120:]),
        'after_bootstrap':compare(pairs[boot_new+1:]),'bootstrap_index':boot_new,
        'bootstrap_parameters_identical':after_new==after_old,
        'own_success_counts':{'replay':new['summary']['own_successes'],'original':old['summary']['own_successes']},
        'scope':'Same scheduled tasks and per-success budget; policies collect independently evolving own trajectories. Total training steps may differ. Not a same-data replay causal effect or unseen generalization result.'}
    OUTPUT.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result['overall']),flush=True)

if __name__=='__main__':main()
