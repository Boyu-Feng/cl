"""Audit existing corpus supervision without generating or reusing labels."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def audit(data):
    rows = data['rows']
    schedule = data['schedule']
    empty = 'Causal trajectory prefix is empty.'
    negatives = []
    for i, row in enumerate(rows):
        j = (i + 1) % len(rows)
        if rows[j]['answer'] == row['answer']:
            j = (i + 2) % len(rows)
        negatives.append(j)
    def stats(indices):
        indices = list(indices)
        return dict(
            count=len(indices),
            empty_source=sum(rows[i]['source'] == empty for i in indices),
            identical_positive_negative=sum(
                rows[i]['answer'] == rows[negatives[i]]['answer'] for i in indices),
            unique_sources=len({rows[i]['source'] for i in indices}),
            unique_queries=len({rows[i]['query'] for i in indices}),
        )
    return {'all_rows': stats(range(len(rows))), 'scheduled_updates': stats(schedule),
            'most_common_answers': [dict(answer_sha256=hashlib.sha256(a.encode()).hexdigest(), count=n)
                                    for a, n in Counter(r['answer'] for r in rows).most_common(10)],
            'limitations': ['String equality audit; no claim that unequal negatives are incorrect.',
                            'Existing corpus copied actions, not verified future utility labels.',
                            'Training loss alone does not establish use of trajectory content.']}


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    raw = args.dataset.read_bytes()
    result = audit(json.loads(raw))
    result['dataset_file_sha256'] = hashlib.sha256(raw).hexdigest()
    with args.output.open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print(json.dumps(result, indent=2))
