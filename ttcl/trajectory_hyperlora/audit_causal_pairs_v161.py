"""Check v158 pairs against both complete public prefixes; create no labels."""
import argparse
import json
from collections import Counter
from pathlib import Path
from .prepare_paired_return_corpus_v158 import digest, filehash


def executable(action):
    return {k: v for k, v in action.items() if k not in ('thinking', 'thought')}


def prefix(trajectory, index, strip_rationale=False):
    return [{'query': s['query'],
             'action': executable(s['action']) if strip_rationale else s['action'],
             'feedback': s.get('public_feedback', '')}
            for s in trajectory[:index]]


def main(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    manifest = json.loads((args.corpus / 'manifest.json').read_text())
    data = json.loads((args.corpus / 'dataset.json').read_text())
    assert digest(data) == manifest['dataset_sha256']
    sources = {}
    for entry in manifest['source_files']:
        p = args.root / entry['path']
        assert filehash(p) == entry['sha256']
        sources[entry['sha256']] = json.loads(p.read_text())
    records = []
    for index, row in enumerate(data['rows']):
        a = sources[row['positive_source_sha256']]
        b = sources[row['negative_source_sha256']]
        ai = [i for i, s in enumerate(a) if str(s['query']) == row['query'] and
              json.dumps(s['action'], ensure_ascii=False, sort_keys=True) == row['answer']]
        bi = [i for i, s in enumerate(b) if str(s['query']) == row['query'] and
              json.dumps(s['action'], ensure_ascii=False, sort_keys=True) == row['negative']]
        assert len(ai) == len(bi) == 1, 'Ambiguous source binding'
        i, j = ai[0], bi[0]
        same = executable(a[i]['action']) == executable(b[j]['action'])
        strict = prefix(a, i) == prefix(b, j)
        public = prefix(a, i, True) == prefix(b, j, True)
        records.append({'row_index': index, 'initial_task_sha256': digest(a[0]['query']),
                        'positive_prefix_sha256': digest(prefix(a, i)),
                        'negative_prefix_sha256': digest(prefix(b, j)),
                        'same_executable_action': same,
                        'both_nonempty': i > 0 and j > 0,
                        'same_full_public_history': strict,
                        'same_history_ignoring_rationale': public,
                        'candidate': i > 0 and j > 0 and strict and not same})
    counts = Counter()
    for r in records:
        for key in ('same_executable_action', 'both_nonempty',
                    'same_full_public_history', 'same_history_ignoring_rationale', 'candidate'):
            counts[key] += int(r[key])
    result = {'dataset_sha256': manifest['dataset_sha256'], 'n': len(records),
              'counts': dict(counts), 'rows': records,
              'status': 'audit_only_not_reviewed_training_labels',
              'limitations': 'Matching public histories does not establish equal hidden state or causal action advantage. No task split can create missing supervision.'}
    with args.output.open('x') as f:
        json.dump(result, f, indent=2)
    print(json.dumps({'n': len(records), **counts}))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=Path('.'))
    p.add_argument('--corpus', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    main(p.parse_args())
