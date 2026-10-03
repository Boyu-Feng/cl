"""Count independent-input support for old content-bound credit signs."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

from .audit_credit_signal import audit as audit_dataset


def audit(dataset: dict, crossfit: dict) -> dict:
    source = audit_dataset(dataset, crossfit)
    groups = defaultdict(list)
    for example in dataset['examples']:
        binding = example['binding']
        if len(binding) != 7 or not all(isinstance(value, str)
                                        for value in binding):
            raise ValueError('Invalid source/content binding')
        groups[(binding[0], example['task'], binding[3])].append(example)
    rows = []
    for key, examples in sorted(groups.items()):
        inputs = {row['binding'][4] for row in examples}
        if len(inputs) != len(examples):
            raise ValueError('Duplicate task input under one experience text')
        if len(inputs) < 2:
            continue
        signs = []
        seed_conflicts = []
        for example in examples:
            deltas = example['raw_deltas']
            signs.append(1 if sum(deltas) > 0 else -1 if sum(deltas) < 0 else 0)
            seed_conflicts.append(any(value > 0 for value in deltas) and
                                  any(value < 0 for value in deltas))
        rows.append(dict(source=key[0], task=key[1], memory_text_sha256=key[2],
                         independent_inputs=len(inputs), input_signs=signs,
                         seed_conflicts=seed_conflicts,
                         same_nonzero_sign=(all(sign > 0 for sign in signs) or
                                            all(sign < 0 for sign in signs))))
    matching = [row for row in rows if row['same_nonzero_sign']]
    return dict(schema='credit_transfer_support_audit_v1',
                dataset_sha256=source['dataset_sha256'],
                examples=source['examples'],
                multiple_input_text_groups=len(rows),
                groups_with_any_nonzero_input=sum(any(sign != 0 for sign in
                                                      row['input_signs'])
                                                  for row in rows),
                groups_same_nonzero_sign=len(matching),
                same_sign_groups_with_seed_conflict=sum(any(
                    row['seed_conflicts']) for row in matching),
                rows=rows,
                caveat='Selected fixed-snapshot probes; no same-seed arm repeats or online transfer estimate')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--crossfit', type=Path, required=True)
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    result = audit(json.loads(args.dataset.read_text()),
                   json.loads(args.crossfit.read_text()))
    if args.report:
        args.report.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key:value for key,value in result.items()
                      if key != 'rows'}, indent=2))


if __name__ == '__main__':
    main()
