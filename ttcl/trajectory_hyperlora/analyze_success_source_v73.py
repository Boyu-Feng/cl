"""Compare the completed, audited source-policy intervention with frozen v68."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_bound(report_path, review_path, count):
    report = json.loads(report_path.read_text())
    review = json.loads(review_path.read_text())
    audit_path = report_path.with_suffix('.audit.json')
    audit = json.loads(audit_path.read_text())
    assert report['complete'] and not report['failures']
    assert len(report['rows']) == 134
    assert audit['passed'] and audit['replayed_episodes'] == count
    assert audit['report_sha256'] == sha(report_path)
    assert report['review_sha256'] == audit['review_sha256'] == sha(review_path)
    assert report['summary'] == audit['summary']
    for row, target in zip(report['rows'], review['targets'], strict=True):
        for field in ('index', 'game', 'family', 'input_content_sha256', 'source_records_sha256'):
            assert row[field] == target[field]
    return report, review, {'report': sha(report_path), 'review': sha(review_path), 'audit': sha(audit_path)}


def sequence(episode):
    return [(s['command'], s['observation']) for s in episode['trajectory']]


def compare(rows, arm):
    return {
        'n': len(rows),
        'success_source_wins': sum(new['hyper']['reward'] for new, old in rows),
        'reference_wins': sum(old[arm]['reward'] for new, old in rows),
        'gains': [new['index'] for new, old in rows if new['hyper']['reward'] > old[arm]['reward']],
        'losses': [new['index'] for new, old in rows if new['hyper']['reward'] < old[arm]['reward']],
        'identical_action_feedback_sequences': sum(sequence(new['hyper']) == sequence(old[arm]) for new, old in rows),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results', type=Path, default=Path('results/trajectory_hyperlora'))
    p.add_argument('--reviews', type=Path, default=Path('data/annotations'))
    p.add_argument('--output', type=Path, default=Path('results/trajectory_hyperlora/alfworld_success_source_v73_comparison_20261010.json'))
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    old_stem = 'alfworld_feedback_unseen_v68_20261010'
    new_stem = 'alfworld_success_source_v73_134_20261010'
    old, old_review, old_hashes = read_bound(args.results/(old_stem+'.json'), args.reviews/(old_stem+'.json'), 402)
    new, new_review, new_hashes = read_bound(args.results/(new_stem+'.json'), args.reviews/(new_stem+'.json'), 134)
    for field in ('hyper_sha256', 'static_sha256', 'training_review_sha256', 'training_audit_sha256',
                  'training_sha256', 'manifest_sha256', 'original_review_sha256', 'checkpoint_prefix',
                  'max_steps', 'max_new_tokens', 'history_turns', 'loop_guard_max', 'constrain_actions'):
        assert new_review[field] == old_review[field], field
    rows = list(zip(new['rows'], old['rows'], strict=True))
    for n, o in rows:
        assert (n['index'], n['game'], n['family']) == (o['index'], o['game'], o['family'])
        assert n['hyper']['initial_observation'] == o['hyper']['initial_observation']
    same = [(n, o) for n, o in rows if n['source_records_sha256'] == o['source_records_sha256']]
    changed = [(n, o) for n, o in rows if n['source_records_sha256'] != o['source_records_sha256']]
    result = {
        'bindings': {'v68': old_hashes, 'v73': new_hashes},
        'comparisons': {arm: compare(rows, arm) for arm in ('hyper', 'continual_lora', 'base')},
        'same_source_reproduction': compare(same, 'hyper'),
        'changed_source': compare(changed, 'hyper'),
        'families': {family: compare([(n,o) for n,o in rows if n['family'] == family], 'hyper')
                     for family in sorted({n['family'] for n,o in rows})},
        'scope': 'Audited exposed134; identical frozen trained H, source policy only. Source age/content/outcome co-vary. Not online parameter updates or a fresh confirmatory test.',
    }
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result['comparisons']), flush=True)


if __name__ == '__main__':
    main()
