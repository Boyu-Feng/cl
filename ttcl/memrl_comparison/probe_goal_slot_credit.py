"""Retrospective CPU test of ALFWorld goal-slot and initial-room credit features.

This checks whether source/current object, destination and operation bindings
add value beyond coarse overlap. Held outcomes were inspected before feature
design, so this is mechanism diagnosis, never prospective policy evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

from ttcl.icl_mem0_comparison.protocol import read, sha
from .credit_probe import memory_arms
from .evaluate_shapley_enrichment import _held_rows
from .fit_content_credit_v2 import _family_crossfit
from .fit_shapley_credit_v1 import BASE_FEATURES, build as build_train
from .probe_scene_conditioned_credit import _scene_features
from .probe_set_conditional_credit import _crossfit, _fit, _mse, _predict, _task, _tokens


GOAL_PUT = re.compile(r'^put (?:(a|some|two) )?(?:(clean|cool|hot) )?'
                      r'([a-z]+) (?:in|on) ([a-z]+)\.?$', re.I)
GOAL_PROCESS = re.compile(r'^(clean|cool|heat) some ([a-z]+) and put it in '
                          r'([a-z]+)\.?$', re.I)
GOAL_TWO = re.compile(r'^find two ([a-z]+) and put them in ([a-z]+)\.?$', re.I)
GOAL_LOOK = re.compile(r'^(?:examine|look at) (?:the )?([a-z]+) '
                         r'(?:with|under) (?:the )?([a-z]+)\.?$', re.I)
GOAL_FEATURES = ('same_object', 'same_destination', 'same_operation',
                 'object_in_experience', 'destination_in_experience',
                 'destination_visible_initially')


def _slots(task: str) -> tuple[str, str, str]:
    task = task.strip().casefold()
    match = GOAL_PUT.fullmatch(task)
    if match:
        quantity, state, obj, destination = match.groups()
        return obj, destination, state or ('two' if quantity == 'two' else 'place')
    match = GOAL_PROCESS.fullmatch(task)
    if match:
        state, obj, destination = match.groups()
        return obj, destination, {'heat':'hot'}.get(state, state)
    match = GOAL_TWO.fullmatch(task)
    if match:
        obj, destination = match.groups()
        return obj, destination, 'two'
    match = GOAL_LOOK.fullmatch(task)
    if match:
        obj, destination = match.groups()
        return obj, destination, 'look'
    raise ValueError(f'Unparsed public goal: {task}')


def _features(origin: Path, case: str, index: int) -> tuple[list[float], str]:
    spec, arms = memory_arms(origin, case)
    task = _task(origin, case)
    source, experience = arms[f'only_{index}'].split('\nExperience:', 1)
    current_obj, current_dst, current_op = _slots(task)
    source_obj, source_dst, source_op = _slots(source.removeprefix('Task: '))
    initial = read(origin / 'runs' / case / 'attempt_1' /
                   'episode.json')['initial_observation']
    if task.casefold() not in initial.casefold():
        raise ValueError('Current public goal differs from initial observation')
    room = initial.split('Your task is to: ', 1)[0].casefold()
    exp_tokens = _tokens(experience)
    values = [float(current_obj == source_obj),
              float(current_dst == source_dst),
              float(current_op == source_op),
              float(current_obj in exp_tokens),
              float(current_dst in exp_tokens),
              float(bool(re.search(r'\b' + re.escape(current_dst) +
                                   r' \d+\b', room)))]
    if spec['source_input_sha256'] != read(origin / 'runs' / case /
                                           'row.json')['input_sha256']:
        raise ValueError('Official current input changed')
    return values, hashlib.sha256(initial.encode()).hexdigest()


def _augment(rows: list[dict], origins: dict[str, Path]) -> list[dict]:
    augmented = []
    for row in rows:
        origin = origins[row['origin']]
        slots, initial_hash = _features(origin, row['case'], row['memory_index'])
        scene, scene_binding = _scene_features(origin, row['case'],
                                               row['memory_index'])
        if (scene_binding['source_input_sha256'] != row['input_sha256'] or
                scene_binding['initial_observation_sha256'] != initial_hash):
            raise ValueError('Goal slots and scene do not bind to same input')
        augmented.append(dict(row, x=row['x'] + slots + scene,
                              initial_observation_sha256=initial_hash))
    return augmented


def run(groups: list[Path], reviews: list[Path], group8: Path,
        reviews8: Path, enriched: Path, enriched_reviews: Path) -> dict:
    train, lineage, families = build_train(*groups, *reviews)
    blind = _held_rows(group8, reviews8, 'group8', families)
    enriched_rows = _held_rows(enriched, enriched_reviews, 'enriched', families)
    outputs = dict(zip(('group5', 'group6', 'group7', 'group8', 'enriched'),
                       groups + [group8, enriched]))
    origins = {name:Path(read(output / 'design.json')['origin'])
               for name, output in outputs.items()}
    train, blind, enriched_rows = (_augment(rows, origins)
                                   for rows in (train, blind, enriched_rows))
    inputs = [row['input_sha256'] for rows in (train, blind, enriched_rows)
              for row in rows[::3]]
    if len(inputs) != len(set(inputs)):
        raise ValueError('Overlapping train and held public game content')
    start = len(train[0]['x']) - len(GOAL_FEATURES) - 5
    base = tuple(range(len(BASE_FEATURES)))
    slots = tuple(range(start, start + len(GOAL_FEATURES)))
    scene = tuple(range(start + len(GOAL_FEATURES), len(train[0]['x'])))
    variants = dict(base=base, base_slots=base + slots,
                    base_slots_scene=base + slots + scene)
    ridge = 32.0
    candidates = [dict(name=name, columns=list(columns),
                       input_loo_mse=_crossfit(train, columns, ridge),
                       family_loo_mse=_family_crossfit(train, columns, ridge))
                  for name, columns in variants.items()]
    chosen = min(candidates, key=lambda row:(row['input_loo_mse'],
                                             row['family_loo_mse'], row['name']))
    coefficients = _fit(train, tuple(chosen['columns']), ridge)
    def score(rows: list[dict]) -> dict:
        predicted = _predict(rows, tuple(chosen['columns']), coefficients)
        return dict(inputs=len({row['input_sha256'] for row in rows}),
                    examples=len(rows), zero_mse=_mse(rows, [0.] * len(rows)),
                    model_mse=_mse(rows, predicted),
                    predictions=[dict(input_sha256=row['input_sha256'],
                                      initial_observation_sha256=row['initial_observation_sha256'],
                                      memory_text_sha256=row['memory_text_sha256'],
                                      memory_index=row['memory_index'],
                                      label=row['label'], prediction=float(value))
                                 for row, value in zip(rows, predicted)])
    return dict(schema='goal_slot_credit_diagnostic_v1',
                script_sha256=sha(Path(__file__)),
                scene_script_sha256=sha(Path(__file__).with_name(
                    'probe_scene_conditioned_credit.py')),
                lineage=lineage,
                source_design_sha256={name:sha(output / 'design.json')
                                      for name, output in outputs.items()},
                train_inputs=18, train_examples=len(train),
                train_zero_mse=_mse(train, [0.] * len(train)),
                features=list(GOAL_FEATURES) + ['scene_' + name for name in
                                                ('visible_locations',
                                                 'distinct_location_types',
                                                 'experience_visible_type_overlap',
                                                 'experience_goal_overlap',
                                                 'source_visible_type_overlap')],
                ridge=ridge, candidates=candidates, chosen=chosen,
                group8=score(blind), enriched=score(enriched_rows),
                eligible_for_online_ablation=False,
                caveat='Goal-slot hypothesis formed after observing group8/enriched outcomes; held scores are retrospective, not prospective policy evidence')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('group5', 'group6', 'group7', 'reviews5', 'reviews6',
                 'reviews7', 'group8', 'reviews8', 'enriched',
                 'reviews-enriched', 'report'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    result = run([getattr(args, name).resolve() for name in
                  ('group5', 'group6', 'group7')],
                 [getattr(args, name).resolve() for name in
                  ('reviews5', 'reviews6', 'reviews7')],
                 args.group8.resolve(), args.reviews8.resolve(),
                 args.enriched.resolve(), args.reviews_enriched.resolve())
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(chosen=result['chosen'],
                          group8={k:v for k,v in result['group8'].items()
                                  if k != 'predictions'},
                          enriched={k:v for k,v in result['enriched'].items()
                                    if k != 'predictions'})))


if __name__ == '__main__':
    main()
