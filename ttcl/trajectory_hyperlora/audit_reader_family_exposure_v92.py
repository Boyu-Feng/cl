"""Describe task-family exposure in existing H training; no new model updates."""
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path('results/trajectory_hyperlora')
OUTPUT = ROOT/'alfworld_reader_family_exposure_v92_20261010.json'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
digest = lambda x: hashlib.sha256(json.dumps(x, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def main():
    assert not OUTPUT.exists()
    paths = {
        'prior': Path('data/annotations/alfworld_causal_prefix_v70_train120_20261010.json'),
        'bootstrap_manifest': ROOT/'alfworld_centered_experience_v75b_20261010/manifest.json',
        'bootstrap_training': ROOT/'alfworld_centered_experience_v75b_20261010/training.json',
        'cleaned': Path('data/annotations/alfworld_cleaned_targets_v88_20261010.json'),
        'reader_review': Path('data/annotations/alfworld_transition_reader_v88_20261010.json'),
        'reader_eval': ROOT/'alfworld_transition_reader_v88_eval_20261010.json',
        'reader_audit': ROOT/'alfworld_transition_reader_v88_eval_20261010.audit.json'}
    d = {k: json.loads(p.read_text()) for k,p in paths.items()}
    assert d['bootstrap_training']['complete'] and not d['bootstrap_training']['failures']
    assert d['bootstrap_training']['manifest_sha256'] == sha(paths['bootstrap_manifest'])
    assert d['bootstrap_manifest']['review_sha256'] == sha(paths['prior'])
    assert d['reader_review']['annotation_sha256'] == sha(paths['cleaned'])
    assert d['reader_audit']['passed'] and d['reader_audit']['report_sha256'] == sha(paths['reader_eval'])
    old, targets = d['prior']['targets'], d['cleaned']['targets']
    assert digest(old) == d['prior']['targets_sha256']
    assert digest(targets) == d['cleaned']['targets_sha256']
    family = lambda game: Path(game).parts[-3].split('-')[0]
    boot = d['bootstrap_manifest']['schedule'][0]
    assert boot['bootstrap'] and boot['steps'] == len(boot['target_indices']) == 300
    assert boot['target_order_sha256'] == digest(boot['target_indices'])
    bootstrap_counts = Counter(family(old[k]['game']) for k in boot['target_indices'])
    query_counts, source_counts = Counter(), Counter()
    cross_task_steps = 0
    transitions = d['reader_review']['transitions']
    for group in d['reader_review']['groups']:
        for k in group['update_order']:
            query = targets[k]
            assert query['stream_index'] == group['stream_index']
            query_counts[family(query['game'])] += 1
            for m in group['memory_indices']:
                source = transitions[m]
                assert source['source_index'] < query['stream_index']
                assert source['game'] != query['game']
                source_counts[family(source['game'])] += 1
            cross_task_steps += 1
    evaluation = Counter(family(r['game']) for r in d['reader_eval']['rows'])
    seen = set(bootstrap_counts) | set(query_counts) | set(source_counts)
    result = {'bindings': {k: sha(p) for k,p in paths.items()},
        'bootstrap_action_occurrences': dict(bootstrap_counts),
        'hyper_query_action_occurrences': dict(query_counts),
        'memory_transition_exposures_over_updates': dict(source_counts),
        'strictly_past_other_task_training_steps': cross_task_steps,
        'evaluation_task_families': dict(evaluation),
        'evaluation_families_absent_from_all_parameter_training_inputs': sorted(set(evaluation)-seen),
        'scope': 'Existing v88/v89 lineage and exposed24 evaluation only; family from public ALFWorld game path. Not a new generalization result.',
        'conclusion': 'Past-other-task training already exists. A family holdout must exclude that family from bootstrap, H query targets and source banks; changing only evaluation memory is insufficient. Dataset-general transfer remains unproved.'}
    OUTPUT.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k != 'bindings'}), flush=True)


if __name__ == '__main__':
    main()
