"""Compile reviewed, full-trajectory experience targets for a future writer.

Targets are local claims with exact action/feedback support, not labels copied
from a historical memory ID or a terminal reward broadcast across a trace.
The whole game input hash is the unit for train/validation splitting.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

from ttcl.experience_evolution.core import digest
from ttcl.icl_mem0_comparison.protocol import read, save, sha
from .feedback_skill_actor import PREP, effect


MOVE = re.compile(r'move (\w+ \d+) to (\w+ \d+)', re.I)
GO = re.compile(r'go to (\w+ \d+)', re.I)


def _row_hash(row: dict) -> str:
    return digest(json.dumps(row, sort_keys=True, ensure_ascii=False))


def _move_success(row: dict, obj: str, destination: str) -> bool:
    return f'you move the {obj} to the {destination}.' in row['observation'].lower()


def claims_from_trajectory(trajectory: list[dict]) -> list[dict]:
    """Induce only two directly checkable types of conditional experience."""
    claims = []
    seen = set()
    for index, row in enumerate(trajectory):
        action = row['action'].lower().strip()
        match = PREP.fullmatch(action)
        if match and effect(row['action'], row['observation']):
            verb, _, tool = match.groups()
            tool_type = tool.split()[0]
            key = ('verified_preparation_tool', verb, tool_type)
            if key not in seen:
                claims.append(dict(kind='verified_preparation_tool', operation=verb,
                    tool_type=tool_type,
                    when='holding a task-relevant object that still needs this operation',
                    advice=f'Navigate to a {tool_type} and use it for {verb}; confirm the effect in feedback.',
                    evidence_indices=[index], evidence_row_sha256=[_row_hash(row)]))
                seen.add(key)
        failed_move = MOVE.fullmatch(action)
        if not failed_move or 'Nothing happens.' not in row['observation']:
            continue
        obj, destination = failed_move.groups()
        go_index = next((j for j in range(index + 1, min(index + 12, len(trajectory)))
            if trajectory[j]['action'].lower().strip() == f'go to {destination}'
            and f'you arrive at {destination}.' in trajectory[j]['observation'].lower()), None)
        if go_index is None:
            continue
        success_index = next((j for j in range(go_index + 1, min(index + 13, len(trajectory)))
            if trajectory[j]['action'].lower().strip() == action
            and _move_success(trajectory[j], obj, destination)), None)
        if success_index is None:
            continue
        key = ('move_retry_after_navigation',)
        if key not in seen:
            indices = [index, go_index, success_index]
            claims.append(dict(kind='move_retry_after_navigation',
                when='a move-to-destination action returned Nothing happens',
                advice='Navigate to that destination, then retry the same move if the object is still held.',
                evidence_indices=indices,
                evidence_row_sha256=[_row_hash(trajectory[j]) for j in indices]))
            seen.add(key)
    return claims


def validate_claims(trajectory: list[dict], claims: list[dict]) -> None:
    for claim in claims:
        indices = claim['evidence_indices']
        hashes = claim['evidence_row_sha256']
        if (not indices or len(indices) != len(hashes) or
                any(not isinstance(i, int) or not 0 <= i < len(trajectory)
                    or _row_hash(trajectory[i]) != h for i, h in zip(indices, hashes))):
            raise ValueError('Claim source row binding changed')
        if claim['kind'] == 'verified_preparation_tool':
            if len(indices) != 1:
                raise ValueError('Preparation claim needs one exact effect')
            event = effect(trajectory[indices[0]]['action'], trajectory[indices[0]]['observation'])
            if (not event or event[0] != claim['operation'] or
                    event[2].split()[0] != claim['tool_type']):
                raise ValueError('Preparation claim is unsupported')
        elif claim['kind'] == 'move_retry_after_navigation':
            if len(indices) != 3 or indices != sorted(indices):
                raise ValueError('Recovery claim needs an ordered failure, navigation, retry')
            fail, go, success = (trajectory[i] for i in indices)
            match = MOVE.fullmatch(fail['action'].lower().strip())
            if not match or 'Nothing happens.' not in fail['observation']:
                raise ValueError('Recovery failure is unsupported')
            obj, destination = match.groups()
            if (go['action'].lower().strip() != f'go to {destination}'
                    or f'you arrive at {destination}.' not in go['observation'].lower()
                    or success['action'].lower().strip() != fail['action'].lower().strip()
                    or not _move_success(success, obj, destination)):
                raise ValueError('Recovery sequence is unsupported')
        else:
            raise ValueError('Unknown experience claim')


def writer_input(episode: dict) -> dict:
    """Only the completed public source trace; never a future evaluation task."""
    return dict(initial_observation=episode['initial_observation'],
        trajectory=[dict(action=row['action'], observation=row['observation'])
                    for row in episode['trajectory']],
        outcome=episode['reward'])


def compile_corpus(roots: list[Path], data_root: Path) -> dict:
    samples = []
    for root in roots:
        for path in sorted(root.glob('**/episode.json')):
            episode = read(path)
            game = episode.get('game', '')
            if not game.startswith('json_2.1.1/train/'):
                continue
            game_path = (data_root / game).resolve()
            if not game_path.is_file() or not game_path.is_relative_to(data_root.resolve()):
                raise ValueError(f'Source is not an official train game: {game}')
            claims = claims_from_trajectory(episode['trajectory'])
            if not claims:
                continue
            validate_claims(episode['trajectory'], claims)
            sample_input = writer_input(episode)
            samples.append(dict(game=game, input_sha256=sha(game_path),
                source_episode=str(path.resolve()), source_episode_sha256=sha(path),
                writer_input_sha256=digest(json.dumps(sample_input, sort_keys=True, ensure_ascii=False)),
                claims=claims, reviewed=True))
    samples.sort(key=lambda row: (row['input_sha256'], row['source_episode']))
    return dict(schema='alf_trajectory_experience_teacher_v1',
        source_roots=[str(root.resolve()) for root in roots],
        split='official json_2.1.1/train only; group all episodes of one game content hash together',
        annotation_rule='Each claim checked against exact positive effect or ordered failure-navigation-success rows, bound to input and source episode SHA-256',
        samples=samples)


def audit_corpus(corpus: dict, data_root: Path) -> dict:
    if corpus.get('schema') != 'alf_trajectory_experience_teacher_v1':
        raise ValueError('Unknown corpus schema')
    games = set()
    kinds = {}
    for sample in corpus['samples']:
        path = Path(sample['source_episode'])
        if (not sample.get('reviewed') or
                not any(path.is_relative_to(Path(root)) for root in corpus['source_roots']) or
                sha(path) != sample['source_episode_sha256'] or
                not sample['game'].startswith('json_2.1.1/train/') or
                sha(data_root / sample['game']) != sample['input_sha256']):
            raise ValueError('Source, split, or input-content binding changed')
        episode = read(path)
        if episode['game'] != sample['game'] or digest(json.dumps(
                writer_input(episode), sort_keys=True, ensure_ascii=False)) != sample['writer_input_sha256']:
            raise ValueError('Writer input binding changed')
        expected = claims_from_trajectory(episode['trajectory'])
        validate_claims(episode['trajectory'], sample['claims'])
        if sample['claims'] != expected:
            raise ValueError('Teacher target changed')
        games.add(sample['input_sha256'])
        for claim in sample['claims']:
            kinds[claim['kind']] = kinds.get(claim['kind'], 0) + 1
    return dict(schema='alf_trajectory_experience_teacher_audit_v1',
        samples=len(corpus['samples']), distinct_train_inputs=len(games), claim_counts=kinds,
        corpus_sha256=digest(json.dumps(corpus, sort_keys=True, ensure_ascii=False)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-roots', type=Path, nargs='+', required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    data_root = args.data_root.resolve()
    corpus = compile_corpus([root.resolve() for root in args.source_roots], data_root)
    audit = audit_corpus(corpus, data_root)
    save(args.output, corpus)
    save(args.output.with_name(args.output.stem + '_audit.json'), audit)
    print(json.dumps(audit), flush=True)


if __name__ == '__main__':
    main()
