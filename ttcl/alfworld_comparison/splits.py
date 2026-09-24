"""Audit ALFWorld task provenance and construct a training-disjoint test manifest.

This module performs no model inference or environment execution. Historical
selection manifests are treated conservatively: reserved training/probe tasks
count as training, and reserved evaluation tasks count as previously evaluated.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import time

from ttcl.experience_evolution.core import FAMILIES, workspace_root

HISTORICAL_PROJECTS = (
    'experience_evolution', 'experience_v2', 'experience_repair',
    'experience_design', 'experience_feedback',
)
REQUIRED_PROJECTS = set(HISTORICAL_PROJECTS[:4])
OFFICIAL_SPLITS = ('train', 'valid_seen', 'valid_unseen')
EVALUATION_KEYS = {'evaluation', 'evaluation_alf'}
TRAINING_KEYS = {'training', 'screening', 'calibration', 'probes'}
MANIFEST_PATTERNS = (
    '*/plan.json', '*/training_plan.json', '*/training/curriculum.json',
    '*/histories.json', '*/dataset.json',
    '*/failure_mix/training_plan.json', '*/failure_mix/training/curriculum.json',
)


def _sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def _json_sha(value):
    return _sha_bytes(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                separators=(',', ':')).encode())


def _normalize_game(value):
    if not isinstance(value, str) or not value.endswith('/game.tw-pddl'):
        return None
    parts = Path(value).parts
    if '..' in parts:
        raise ValueError(f'Unsafe ALFWorld task reference: {value}')
    if 'json_2.1.1' in parts:
        parts = parts[parts.index('json_2.1.1'):]
    elif parts and parts[0] in OFFICIAL_SPLITS:
        parts = ('json_2.1.1', *parts)
    else:
        raise ValueError(f'Unrecognized ALFWorld task reference: {value}')
    if len(parts) != 5 or parts[1] not in OFFICIAL_SPLITS:
        raise ValueError(f'Unexpected ALFWorld task layout: {value}')
    return '/'.join(parts)


def _games(value):
    if isinstance(value, str):
        game = _normalize_game(value)
        if game:
            yield game
    elif isinstance(value, list):
        for item in value:
            yield from _games(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _games(item)


def _normalize_pddl(text):
    text = re.sub(r';[^\n]*', '', text).lower()
    text = re.sub(r'\(\s*problem\s+[^\s()]+', '(problem instance', text)
    return ' '.join(re.findall(r'\(|\)|[^\s()]+', text))


def _pddl_fingerprints(record):
    problem = record.get('pddl_problem')
    domain = record.get('pddl_domain')
    if not isinstance(problem, str) or not isinstance(domain, str):
        return None, None
    normalized = _normalize_pddl(problem)
    full_hash = _json_sha([_normalize_pddl(domain), normalized])
    start = normalized.find('( :goal ')
    if start < 0:
        raise ValueError('PDDL problem has no recognizable goal section')
    depth = 0
    for position in range(start, len(normalized)):
        if normalized[position] == '(':
            depth += 1
        elif normalized[position] == ')':
            depth -= 1
            if depth == 0:
                return full_hash, _sha_bytes(normalized[start:position + 1].encode())
    raise ValueError('Unbalanced PDDL goal section')


def _inventory(data_root):
    inventory = {}
    for split in OFFICIAL_SPLITS:
        files = sorted((data_root / 'json_2.1.1' / split).glob('*/*/game.tw-pddl'))
        if not files:
            raise FileNotFoundError(f'Official ALFWorld split is missing or empty: {split}')
        for path in files:
            content = path.read_bytes()
            record = json.loads(content)
            family = path.parent.parent.name.split('-')[0]
            if family not in FAMILIES:
                raise ValueError(f'Unrecognized task family: {path}')
            name = path.relative_to(data_root).as_posix()
            problem_sha256, goal_sha256 = _pddl_fingerprints(record)
            inventory[name] = {
                'path': name, 'family': family, 'split': split,
                'sha256': _sha_bytes(content), 'solvable': record.get('solvable') is True,
                'normalized_problem_sha256': problem_sha256, 'goal_sha256': goal_sha256,
            }
    return inventory


def _historical_usage(results_root, require_history):
    training, evaluation = defaultdict(set), defaultdict(set)
    sources, covered = [], set()
    for project in HISTORICAL_PROJECTS:
        directory = results_root / project
        manifests = sorted({p for pattern in MANIFEST_PATTERNS for p in directory.glob(pattern)})
        for path in manifests:
            content = path.read_bytes()
            value = json.loads(content)
            source = path.relative_to(results_root).as_posix()
            referenced = set(_games(value))
            if not referenced:
                continue
            covered.add(project)
            sources.append({'path': source, 'sha256': _sha_bytes(content),
                            'distinct_referenced_tasks': len(referenced)})
            if path.name in {'histories.json', 'dataset.json'}:
                # These are writer training inputs, including any source history
                # and future utility-scoring probes, never evaluation examples.
                for game in referenced:
                    training[game].add(source)
                continue
            if not isinstance(value, dict):
                raise ValueError(f'Expected a mapping in {source}')
            accounted = set()
            for key, section in value.items():
                games = set(_games(section))
                if not games:
                    continue
                if key in EVALUATION_KEYS:
                    destination = evaluation
                elif key in TRAINING_KEYS:
                    destination = training
                else:
                    raise ValueError(
                        f'Unclassified ALFWorld references in {source}#{key}; '
                        'audit their role before building a test split.'
                    )
                for game in games:
                    destination[game].add(f'{source}#{key}')
                accounted.update(games)
            if accounted != referenced:
                raise AssertionError(f'Unaccounted task references in {source}')
    if require_history and not REQUIRED_PROJECTS.issubset(covered):
        raise FileNotFoundError(
            'Historical experiment manifests required for a complete contamination audit: '
            + ', '.join(sorted(REQUIRED_PROJECTS - covered))
        )
    return training, evaluation, sources, covered


def build_manifest(data_root=None, results_root=None, test_split='valid_unseen',
                   *, require_history=True):
    """Return a JSON-serializable maximal official test split plus provenance.

    All official training tasks are denied, not just tasks found in historical
    runs. Historical training/calibration/screening/probe references and exact
    content duplicates are also denied even if they occur in an evaluation split.
    `never_evaluated` additionally excludes prior evaluation paths/content.
    """
    if test_split not in {'valid_unseen', 'valid_seen'}:
        raise ValueError('Comparison tasks must come from valid_unseen or valid_seen, never train')
    root = workspace_root()
    data_root = Path(data_root or root / 'ttcl/data/alfworld_delta').resolve()
    results_root = Path(results_root or root / 'ttcl/results').resolve()
    inventory = _inventory(data_root)
    training, evaluation, sources, covered = _historical_usage(results_root, require_history)
    missing = sorted((set(training) | set(evaluation)) - set(inventory))
    if missing:
        raise FileNotFoundError(f'Historical task files absent from official data: {missing}')
    official_training = {g for g, item in inventory.items() if item['split'] == 'train'}
    denied = official_training | set(training)
    training_by_hash, past_by_hash = defaultdict(set), defaultdict(set)
    training_by_problem, past_by_problem = defaultdict(set), defaultdict(set)
    training_goal_hashes = set()
    for game in denied:
        training_by_hash[inventory[game]['sha256']].add(game)
        if inventory[game]['normalized_problem_sha256']:
            training_by_problem[inventory[game]['normalized_problem_sha256']].add(game)
            training_goal_hashes.add(inventory[game]['goal_sha256'])
    for game in evaluation:
        past_by_hash[inventory[game]['sha256']].add(game)
        if inventory[game]['normalized_problem_sha256']:
            past_by_problem[inventory[game]['normalized_problem_sha256']].add(game)
    tasks, excluded = [], []
    for game, item in sorted(inventory.items()):
        if item['split'] != test_split:
            continue
        reasons = []
        if not item['solvable']:
            reasons.append({'kind': 'official_unsolvable'})
        if game in denied:
            reasons.append({'kind': 'training_path', 'sources': sorted(training[game])})
        collisions = sorted(training_by_hash.get(item['sha256'], ()))
        if collisions:
            reasons.append({'kind': 'training_content_sha256', 'matching_tasks': collisions,
                            'sources': sorted({source for match in collisions
                                               for source in training.get(match, ())})})
        problem_matches = sorted(training_by_problem.get(item['normalized_problem_sha256'], ()))
        if problem_matches:
            reasons.append({'kind': 'training_normalized_pddl_problem',
                            'matching_tasks': problem_matches})
        if reasons:
            excluded.append({**item, 'reasons': reasons})
            continue
        prior_matches = sorted(set(past_by_hash.get(item['sha256'], ()))
                               | set(past_by_problem.get(item['normalized_problem_sha256'], ())))
        past_sources = sorted({source for match in prior_matches for source in evaluation[match]})
        past_sources = sorted(set(past_sources) | evaluation.get(game, set()))
        tasks.append({**item, 'past_evaluation_sources': past_sources,
                      'past_evaluation_content_matches': prior_matches,
                      'goal_template_seen_in_training': item['goal_sha256'] in training_goal_hashes,
                      'never_evaluated': not past_sources})
    if not tasks:
        raise ValueError('No test tasks remain after training-contamination exclusions')
    chosen = {item['path'] for item in tasks}
    chosen_hashes = {item['sha256'] for item in tasks}
    fresh = [item for item in tasks if item['never_evaluated']]
    intersections = {
        'selected_vs_training_paths': sorted(chosen & denied),
        'selected_vs_training_sha256': sorted(chosen_hashes & set(training_by_hash)),
        'selected_vs_training_normalized_problem_sha256': sorted(
            {item['normalized_problem_sha256'] for item in tasks if item['normalized_problem_sha256']}
            & set(training_by_problem)),
        'fresh_vs_past_evaluation_normalized_problem_sha256': sorted(
            {item['normalized_problem_sha256'] for item in fresh if item['normalized_problem_sha256']}
            & set(past_by_problem)),
        'fresh_vs_past_evaluation_paths': sorted({item['path'] for item in fresh} & set(evaluation)),
        'fresh_vs_past_evaluation_sha256': sorted({item['sha256'] for item in fresh} & set(past_by_hash)),
    }
    if any(intersections.values()):
        raise AssertionError(f'Training-disjoint split invariant failed: {intersections}')
    manifest = {
        'schema_version': 1, 'created_at': time.time(), 'data_root': str(data_root),
        'results_root': str(results_root), 'test_split': test_split,
        'policy': 'Maximal solvable official evaluation split; exclude every official train task and every historical training/screening/calibration/probe path, byte-identical task or identical normalized full PDDL world-and-goal problem. Prior evaluation tasks remain in the main comparison and are explicitly flagged.',
        'official_splits': {
            split: {'tasks': sum(v['split'] == split for v in inventory.values()),
                    'solvable': sum(v['split'] == split and v['solvable'] for v in inventory.values()),
                    'by_family': dict(sorted(Counter(v['family'] for v in inventory.values()
                                                    if v['split'] == split).items()))}
            for split in OFFICIAL_SPLITS
        },
        'historical_manifests': sources,
        'historical_audit_complete': REQUIRED_PROJECTS.issubset(covered),
        'training_audit': {
            'official_training_tasks': len(official_training),
            'official_training_content_manifest_sha256': _json_sha(
                [(g, inventory[g]['sha256']) for g in sorted(official_training)]),
            'historical_training_tasks': len(training),
            'historical_training_by_split': dict(Counter(inventory[g]['split'] for g in training)),
            'historical_training_tasks_detail': [
                {'path': game, 'sha256': inventory[game]['sha256'], 'sources': sorted(origins)}
                for game, origins in sorted(training.items())],
            'historical_training_evaluation_overlap': sorted(set(training) & set(evaluation)),
        },
        'semantic_audit': {
            'normalization': 'Normalize PDDL case and whitespace, remove comments, replace only the problem instance name; hash domain plus full problem (objects, initial world state and goal). Object IDs and world state are retained. This is a duplicate check, not a proof of semantic equivalence.',
            'goal_policy': 'Hash the goal section separately for descriptive template overlap only. Shared goals across different worlds are expected and are NOT excluded as training contamination.',
            'official_tasks_with_pddl': sum(item['normalized_problem_sha256'] is not None for item in inventory.values()),
            'selected_tasks_with_training_goal_template': sum(item['goal_template_seen_in_training'] for item in tasks),
            'selected_tasks_with_pddl': sum(item['normalized_problem_sha256'] is not None for item in tasks),
        },
        'past_evaluation_audit': {
            'tasks': len(evaluation),
            'by_split': dict(Counter(inventory[g]['split'] for g in evaluation)),
            'interpretation': 'Conservative union of prior evaluation and reserved evaluation manifests; not a claim that every reserved task completed.'
        },
        'tasks': tasks, 'excluded': excluded,
        'counts': {'selected': len(tasks), 'excluded': len(excluded),
                   'never_evaluated': len(fresh), 'previously_evaluated_or_reserved': len(tasks) - len(fresh),
                   'by_family': dict(sorted(Counter(t['family'] for t in tasks).items())),
                   'never_evaluated_by_family': {f: sum(t['family'] == f for t in fresh) for f in FAMILIES}},
        'intersections': intersections, 'checks_passed': True,
    }
    manifest['task_selection_sha256'] = _json_sha(
        [{k: item[k] for k in ('path', 'family', 'sha256', 'never_evaluated')} for item in tasks])
    return manifest


def write_manifest(path, **kwargs):
    path = Path(path)
    manifest = build_manifest(**kwargs)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Do not overwrite the predeclared input of an existing comparison run.
    with path.open('x') as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--results-root', type=Path)
    parser.add_argument('--test-split', choices=['valid_unseen', 'valid_seen'], default='valid_unseen')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    manifest = write_manifest(args.output, data_root=args.data_root,
                              results_root=args.results_root, test_split=args.test_split)
    print(json.dumps(manifest['counts'], ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
