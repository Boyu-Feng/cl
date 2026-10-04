"""Retrospective CPU diagnostic of pre-action scene-conditioned Shapley credit.

Adds public initial-room features to an older per-memory critic. Group8 and
enriched outcomes were already inspected when this feature hypothesis arose;
their scores are exploratory, never a prospective policy qualification.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random
import re
import statistics

from ttcl.icl_mem0_comparison.protocol import read, sha
from .credit_probe import memory_arms
from .evaluate_shapley_enrichment import _held_rows
from .fit_content_credit_v2 import _family_crossfit
from .fit_shapley_credit_v1 import (
    BASE_FEATURES, CONTENT_FEATURES, build as build_train,
)
from .probe_set_conditional_credit import _crossfit, _fit, _mse, _predict, _task, _tokens


SCENE_FEATURES = ('visible_locations', 'distinct_location_types',
                  'experience_visible_type_overlap',
                  'experience_goal_overlap', 'source_visible_type_overlap')
ROOM_ITEM = re.compile(r'\b(?:a|an) ([a-z]+) (\d+)\b')


def _scene_features(origin: Path, case: str, index: int) -> tuple[list[float], dict]:
    spec, arms = memory_arms(origin, case)
    episode = origin / 'runs' / case / 'attempt_1' / 'episode.json'
    initial = read(episode)['initial_observation']
    if 'Your task is to: ' not in initial:
        raise ValueError('Missing public goal')
    task = _task(origin, case)
    if task.casefold() not in initial.casefold():
        raise ValueError('Task and public initial observation differ')
    room = initial.split('Your task is to: ', 1)[0]
    locations = ROOM_ITEM.findall(room)
    location_types = {kind for kind, _ in locations}
    source, experience = arms[f'only_{index}'].split('\nExperience:', 1)
    source_tokens = _tokens(source.removeprefix('Task: '))
    experience_tokens = _tokens(experience)
    goal_tokens = _tokens(task)
    features = [len(locations) / 20., len(location_types) / 10.,
                len(location_types & experience_tokens) /
                max(1, len(location_types)),
                len(goal_tokens & experience_tokens) /
                max(1, len(goal_tokens)),
                len(location_types & source_tokens) /
                max(1, len(location_types))]
    return features, dict(initial_observation_sha256=hashlib.sha256(
        initial.encode()).hexdigest(), source_input_sha256=spec['source_input_sha256'])


def _augment(rows: list[dict], origins: dict[str, Path]) -> list[dict]:
    result = []
    for row in rows:
        origin = origins[row['origin']]
        features, binding = _scene_features(origin, row['case'], row['memory_index'])
        if binding['source_input_sha256'] != row['input_sha256']:
            raise ValueError('Scene input binding differs from Shapley label')
        result.append(dict(row, x=row['x'] + features,
                           scene_binding=binding['initial_observation_sha256']))
    return result


def run(groups: list[Path], reviews: list[Path], group8: Path,
        reviews8: Path, enriched: Path, enriched_reviews: Path,
        old_fit: Path) -> dict:
    old = read(old_fit)
    train, lineage, families = build_train(*groups, *reviews)
    if old['lineage'] != lineage or old['schema'] != 'shapley_credit_v1_fit_v1':
        raise ValueError('Old frozen fit lineage changed')
    blind = _held_rows(group8, reviews8, 'group8', families)
    selected = _held_rows(enriched, enriched_reviews, 'enriched', families)
    origin_paths = {}
    for name, output in (('group5', groups[0]), ('group6', groups[1]),
                         ('group7', groups[2]), ('group8', group8),
                         ('enriched', enriched)):
        origin_paths[name] = Path(read(output / 'design.json')['origin'])
    train, blind, selected = (_augment(rows, origin_paths)
                              for rows in (train, blind, selected))
    inputs = [row['input_sha256'] for rows in (train, blind, selected)
              for row in rows[::3]]
    if len(inputs) != len(set(inputs)):
        raise ValueError('Train and held public inputs overlap')
    start = len(train[0]['x']) - len(SCENE_FEATURES)
    base = tuple(range(len(BASE_FEATURES)))
    content = tuple(range(len(BASE_FEATURES) + len(CONTENT_FEATURES)))
    variants = dict(base=base, content=content,
                    base_scene=base + tuple(range(start, start + len(SCENE_FEATURES))),
                    content_scene=content + tuple(range(start, start + len(SCENE_FEATURES))))
    candidates = [dict(name=name, ridge=ridge, columns=list(columns),
                       input_loo_mse=_crossfit(train, columns, ridge),
                       family_loo_mse=_family_crossfit(train, columns, ridge))
                  for name, columns in variants.items() for ridge in (1., 8., 32.)]
    chosen = min(candidates, key=lambda item:(item['input_loo_mse'],
                                              item['family_loo_mse'],
                                              item['name'], item['ridge']))
    coefficients = _fit(train, tuple(chosen['columns']), chosen['ridge'])
    def score(rows: list[dict]) -> dict:
        prediction = _predict(rows, tuple(chosen['columns']), coefficients)
        frozen = [sum(row['x'][i] * value
                      for i, value in zip(old['selected_columns'],
                                          old['coefficients'])) for row in rows]
        per_input = defaultdict(list)
        for row, value in zip(rows, prediction):
            per_input[row['input_sha256']].append(
                row['label'] ** 2 - (row['label'] - value) ** 2)
        improvements = [statistics.fmean(values)
                        for values in per_input.values()]
        rng = random.Random(92721)
        bootstrap = sorted(statistics.fmean(rng.choice(improvements)
                                            for _ in improvements)
                           for _ in range(2000))
        return dict(inputs=len({row['input_sha256'] for row in rows}),
                    examples=len(rows),
                    zero_mse=_mse(rows, [0.] * len(rows)),
                    old_frozen_mse=_mse(rows, frozen),
                    scene_model_mse=_mse(rows, prediction),
                    mse_improvement_interval=[bootstrap[49], bootstrap[1949]],
                    predictions=[dict(case=row['case'],
                                      input_sha256=row['input_sha256'],
                                      scene_binding=row['scene_binding'],
                                      memory_index=row['memory_index'],
                                      memory_text_sha256=row['memory_text_sha256'],
                                      label=row['label'], prediction=float(value))
                                 for row, value in zip(rows, prediction)])
    return dict(schema='scene_conditioned_shapley_diagnostic_v1',
                script_sha256=sha(Path(__file__)), old_fit_sha256=sha(old_fit),
                source_design_sha256={kind:sha(output / 'design.json')
                                      for kind, output in zip(
                                          ('group5', 'group6', 'group7', 'group8', 'enriched'),
                                          groups + [group8, enriched])},
                scene_features=list(SCENE_FEATURES), train_inputs=18,
                train_examples=len(train), train_zero_mse=_mse(train, [0.] * len(train)),
                candidates=candidates, chosen=chosen,
                group8=score(blind), enriched=score(selected),
                eligible_for_online_ablation=False,
                caveat='Scene feature hypothesis was formed after inspecting both held groups; scores are retrospective, not prospective validation or task reward')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('group5', 'group6', 'group7', 'reviews5', 'reviews6',
                 'reviews7', 'group8', 'reviews8', 'enriched',
                 'reviews-enriched', 'old-fit', 'report'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    result = run([getattr(args, name).resolve() for name in
                  ('group5', 'group6', 'group7')],
                 [getattr(args, name).resolve() for name in
                  ('reviews5', 'reviews6', 'reviews7')],
                 args.group8.resolve(), args.reviews8.resolve(),
                 args.enriched.resolve(), args.reviews_enriched.resolve(),
                 args.old_fit.resolve())
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(chosen=result['chosen'],
                          group8={k:v for k,v in result['group8'].items()
                                  if k != 'predictions'},
                          enriched={k:v for k,v in result['enriched'].items()
                                    if k != 'predictions'})))


if __name__ == '__main__':
    main()
