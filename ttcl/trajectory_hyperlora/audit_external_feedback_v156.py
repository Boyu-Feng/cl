"""Audit public feedback, without treating execution or completion as success."""
import argparse
import json
from collections import Counter
from pathlib import Path
from .prepare_external_clbench_corpus_v100 import digest, filehash


def main(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    manifest = json.loads(args.manifest.read_text())
    data = json.loads(args.dataset.read_text())
    assert digest(data) == manifest['dataset_sha256']
    counts, scheduled = Counter(), Counter(data['schedule'])
    weighted, records = Counter(), []
    index = 0
    for entry in manifest['source_files']:
        path = args.root / entry['path']
        assert filehash(path) == entry['sha256'], path
        for step in json.loads(path.read_text()):
            row = data['rows'][index]
            assert row['query'] == str(step['query'])
            assert row['answer'] == json.dumps(step['action'], ensure_ascii=False, sort_keys=True)
            feedback = str(step.get('public_feedback', ''))
            # Explicit environment verdict only. A query returning nothing
            # is not necessarily invalid, and completion is not correctness.
            if ': INCORRECT.' in feedback:
                status = 'explicit_incorrect'
            elif ': CORRECT' in feedback:
                status = 'explicit_correct'
            elif '(no results)' in feedback:
                status = 'no_results_unverified'
            else:
                status = 'unverified'
            counts[status] += 1
            weighted[status] += scheduled[index]
            records.append({'row_index': index, 'input_content_sha256': row['input_content_sha256'],
                            'trajectory_file': entry['path'], 'trajectory_sha256': entry['sha256'],
                            'step': step['step'], 'feedback_sha256': digest(feedback),
                            'feedback_status': status, 'training_occurrences': scheduled[index]})
            index += 1
    assert index == len(data['rows'])
    result = {'protocol': 'external_feedback_audit_v156',
              'dataset_sha256': manifest['dataset_sha256'], 'counts': dict(counts),
              'scheduled_counts': dict(weighted), 'rows': records,
              'limitations': 'Audit only; no generated utility labels. Execution and instance_complete do not imply correctness.'}
    with args.output.open('x') as f:
        json.dump(result, f, indent=2)
    print(json.dumps({k: result[k] for k in ('counts', 'scheduled_counts')}, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=Path('.'))
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    main(p.parse_args())
