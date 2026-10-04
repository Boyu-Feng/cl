"""Test whether frozen actor-classified premise features improve Shapley credit.

Only reviewed group5–7 coalition labels are fitted. This is an internal
input/family cross-fit diagnostic; it does not change retrieval or Q.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from ttcl.icl_mem0_comparison.protocol import read, sha
from .annotate_credit_premises import PROMPT, _parse
from .credit_probe import memory_arms
from .fit_shapley_credit_v1 import (
    BASE_FEATURES, CONTENT_FEATURES, build as build_shapley,
)
from .probe_set_conditional_credit import _crossfit, _mse, _task
from .fit_content_credit_v2 import _family_crossfit


PREMISE_FEATURES = ('global_environment_block', 'actionable_precondition',
                    'environment_rewrite')


def fit(group5: Path, group6: Path, group7: Path,
        reviews5: Path, reviews6: Path, reviews7: Path,
        annotations: Path) -> dict:
    rows, lineage, families = build_shapley(
        group5, group6, group7, reviews5, reviews6, reviews7)
    design_path = annotations / 'design.json'
    design = read(design_path)
    if design['schema'] != 'alf_credit_premise_annotations_v1' or \
            design['expected'] != 54 or len(design['input_sha256']) != 18:
        raise ValueError('Unexpected premise annotation design')
    outputs = {'group5':group5, 'group6':group6, 'group7':group7}
    origins = {name:Path(read(output / 'design.json')['origin'])
               for name, output in outputs.items()}
    by_key = {}
    for path in annotations.glob('*.json'):
        if path == design_path:
            continue
        item = read(path)
        key = (item['input_sha256'], item['memory_text_sha256'])
        if key in by_key or item['design_sha256'] != sha(design_path) or \
                'parse_error' in item or set(item['features']) != set(PREMISE_FEATURES) or \
                any(type(item['features'][feature]) is not bool
                    for feature in PREMISE_FEATURES) or \
                _parse(item['raw_response']) != item['features'] or \
                path.stem != hashlib.sha256(('/'.join(key)).encode()).hexdigest():
            raise ValueError('Invalid or duplicate premise annotation')
        by_key[key] = item
    if len(by_key) != len(rows):
        raise ValueError('Missing or extra premise annotation')
    augmented = []
    for row in rows:
        item = by_key.pop((row['input_sha256'], row['memory_text_sha256']), None)
        if item is None or item['case'] != row['case'] or \
                item['snapshot_sha256'] != row['snapshot_sha256'] or \
                item['retrieval_sha256'] != row['retrieval_sha256'] or \
                item['memory_index'] != row['memory_index']:
            raise ValueError('Premise feature does not bind to credit row')
        origin = origins[row['origin']]
        spec, arms = memory_arms(origin, row['case'])
        task = _task(origin, row['case'])
        prompt = PROMPT.format(task=task,
                               experience=arms[f'only_{row["memory_index"]}'])
        if item['task_sha256'] != hashlib.sha256(task.encode()).hexdigest() or \
                item['prompt_sha256'] != hashlib.sha256(prompt.encode()).hexdigest() or \
                spec['memory_text_sha256'][row['memory_id']] != item['memory_text_sha256']:
            raise ValueError('Premise prompt does not match current source')
        augmented.append(dict(row, x=row['x'] +
                              [float(item['features'][name])
                               for name in PREMISE_FEATURES]))
    if by_key:
        raise ValueError('Unmatched premise annotations')
    base = tuple(range(len(BASE_FEATURES)))
    content = tuple(range(len(BASE_FEATURES) + len(CONTENT_FEATURES)))
    premise = tuple(range(len(rows[0]['x']), len(augmented[0]['x'])))
    variants = dict(base=base, base_premise=base + premise,
                    content=content, content_premise=content + premise)
    ridge = 32.0  # held constant from the preceding Shapley diagnostic
    candidates = [dict(name=name, input_loo_mse=_crossfit(augmented, cols, ridge),
                       family_loo_mse=_family_crossfit(augmented, cols, ridge))
                  for name, cols in variants.items()]
    return dict(schema='premise_shapley_credit_v2_diagnostic_v1',
                lineage=lineage, annotations_design_sha256=sha(design_path),
                annotator_script_sha256=design['annotator_script_sha256'],
                fit_script_sha256=sha(Path(__file__)),
                train_inputs=18, train_examples=len(rows), families=families,
                zero_mse=_mse(augmented, [0.] * len(augmented)),
                ridge=ridge, candidates=candidates,
                caveat='Actor-generated binary premise features on reviewed official-train inputs; internal cross-fit only, no group8 or online policy outcome')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('group5', 'group6', 'group7', 'reviews5', 'reviews6',
                 'reviews7', 'annotations', 'report'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    result = fit(*(getattr(args, name).resolve() for name in
                   ('group5', 'group6', 'group7', 'reviews5', 'reviews6',
                    'reviews7', 'annotations')))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
