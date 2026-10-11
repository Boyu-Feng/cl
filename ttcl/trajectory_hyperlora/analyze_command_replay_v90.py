"""Compare audited command-balanced replay with exact-bootstrap references."""
import json
from pathlib import Path
from ttcl.trajectory_hyperlora.analyze_online_replay_v78 import load, compare, sha

OUTPUT = Path('results/trajectory_hyperlora/alfworld_command_replay_v90_comparison_20261010.json')


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    new, nh = load('v90')
    binding = Path('data/annotations/alfworld_shared_bootstrap_v81_20261010.json')
    shared = json.loads(binding.read_text())
    assert new['shared_bootstrap_binding_sha256'] == sha(binding)
    result = {'bindings': {'v90': nh, 'common_bootstrap': sha(binding)}, 'comparisons': {}}
    for version in ('v78', 'v81'):
        old, oh = load(version)
        i = shared['bootstrap']['index']
        for report in (new, old):
            assert report['bootstrap'] == shared['bootstrap']
            assert report['initial_parameters_sha256'] == shared['initial_parameters_sha256']
            assert report['rows'][i]['parameters_after_sha256'] == shared['parameters_sha256']
        for key in ('seed', 'limit', 'rank', 'layers', 'online_steps_per_eligible_win',
                    'online_lr', 'max_steps', 'max_new_tokens', 'history_turns',
                    'loop_guard_max', 'constrain_actions', 'eligibility',
                    'source_report_sha256', 'model_config_sha256'):
            assert new[key] == old[key], key
        pairs = list(zip(new['rows'], old['rows'], strict=True))
        for n, o in pairs:
            assert (n['index'], n['game']) == (o['index'], o['game'])
            assert n['episode']['initial_observation'] == o['episode']['initial_observation']
        assert all(n['episode'] == o['episode'] for n, o in pairs[:i+1])
        result['bindings'][version] = oh
        result['comparisons'][version] = {'overall': compare(pairs),
            'after_bootstrap': compare(pairs[i+1:]),
            'first120': compare(pairs[:120]), 'last120': compare(pairs[120:])}
    result['scope'] = ('All replay_wins fields mean v90; original_wins means named reference. '
        'Exact shared bootstrap. Independent online own-trajectory streams; same per-win '
        'budget, possibly different total updates and training data. No unseen confirmation '
        'or hypernetwork superiority claim.')
    OUTPUT.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result['comparisons']), flush=True)


if __name__ == '__main__':
    main()
