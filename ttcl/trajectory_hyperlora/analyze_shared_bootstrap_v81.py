"""Compare audited online policies from the identical saved bootstrap state."""
import json
from pathlib import Path
from ttcl.trajectory_hyperlora.analyze_online_replay_v78 import load,compare,sha

OUTPUT=Path('results/trajectory_hyperlora/alfworld_shared_bootstrap_v81_comparison_20261010.json')

def main():
    if OUTPUT.exists():raise FileExistsError(OUTPUT)
    new,nh=load('v78');control,ch=load('v81')
    binding=Path('data/annotations/alfworld_shared_bootstrap_v81_20261010.json')
    shared=json.loads(binding.read_text());assert control['shared_bootstrap_binding_sha256']==sha(binding)
    i=shared['bootstrap']['index'];assert new['bootstrap']==control['bootstrap']==shared['bootstrap']
    for report in (new,control):
        assert report['rows'][i]['parameters_after_sha256']==shared['parameters_sha256']
        assert report['initial_parameters_sha256']==shared['initial_parameters_sha256']
    for key in ('seed','rank','layers','online_steps_per_eligible_win','online_lr','max_steps','max_new_tokens','history_turns','loop_guard_max','constrain_actions','eligibility','model_config_sha256'):
        assert new[key]==control[key],key
    pairs=list(zip(new['rows'],control['rows'],strict=True))
    for n,c in pairs:
        assert (n['index'],n['game'])==(c['index'],c['game'])
        assert n['episode']['initial_observation']==c['episode']['initial_observation']
    assert all(n['episode']==c['episode'] for n,c in pairs[:i+1])
    result={'bindings':{'replay':nh,'latest_only':ch,'common_bootstrap':sha(binding)},
        'shared_bootstrap_parameters_identical':True,'overall':compare(pairs),
        'after_bootstrap':compare(pairs[i+1:]),'first120':compare(pairs[:120]),'last120':compare(pairs[120:]),
        'executed_training_steps':{'replay':sum(r['update'].get('executed_steps',r['update']['steps']) for r in new['rows']),
            'latest_only':sum(r['update'].get('executed_steps',r['update']['steps']) for r in control['rows'])},
        'shared_bootstrap_steps':300,
        'scope':'Exact common bootstrap; independent prequential own-trajectory policies afterwards. The reference labeled original in comparison fields is v81 latest-only. Different collected data and total updates remain possible.'}
    OUTPUT.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result['overall']),flush=True)

if __name__=='__main__':main()
