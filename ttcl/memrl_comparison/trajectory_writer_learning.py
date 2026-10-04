"""Evidence-checked continual writer targets and cumulative claim memory.

The writer sees its previous compact memory and one completed trajectory.
It proposes add/reinforce operations with source step indices. The validator
accepts only claims entailed by exact environment action/feedback evidence.
The export is a supervised warm start; future official paired chain returns,
not local success alone, must govern downstream RL writer credit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random

from ttcl.experience_evolution.core import digest
from ttcl.icl_mem0_comparison.protocol import read, save, sha
from .trajectory_experience_teacher import (audit_corpus, claims_from_trajectory,
                                             writer_input)


SYSTEM = ("Maintain reusable cross-task experience from the completed trajectory. "
          "Return JSON only: {\"operations\":[...]}. Each operation is add or reinforce, "
          "and must cite zero-based source trajectory step indices. Use only exact "
          "action/feedback evidence. Preserve prior claims unless new evidence supports an update. "
          "Do not copy source object instance IDs into a reusable claim. A final task reward "
          "does not prove that every step was good.")


def claim_key(claim: dict) -> str:
    if claim['kind'] == 'verified_preparation_tool':
        return f"{claim['kind']}:{claim['operation']}:{claim['tool_type']}"
    if claim['kind'] == 'move_retry_after_navigation':
        return claim['kind']
    raise ValueError('Unknown claim kind')


class ClaimMemory:
    def __init__(self):
        self.entries: dict[str, dict] = {}

    def public_state(self) -> list[dict]:
        return [dict(key=key, kind=value['claim']['kind'],
                     operation=value['claim'].get('operation'),
                     tool_type=value['claim'].get('tool_type'),
                     when=value['claim']['when'], advice=value['claim']['advice'],
                     supporting_games=len(value['sources']))
                for key, value in sorted(self.entries.items())]

    def apply(self, operations: list[dict], episode: dict, input_sha: str) -> dict:
        expected = {claim_key(claim): claim for claim in claims_from_trajectory(episode['trajectory'])}
        if not isinstance(operations, list):
            raise ValueError('Writer operations must be a list')
        applied, rejected = [], []
        for index, operation in enumerate(operations):
            try:
                if not isinstance(operation, dict):
                    raise ValueError('Operation must be an object')
                key = operation['key']
                kind = operation['op']
                claim = expected.get(key)
                if (kind not in {'add', 'reinforce'} or claim is None or
                        operation.get('evidence_indices') != claim['evidence_indices'] or
                        operation.get('evidence_row_sha256') != claim['evidence_row_sha256']):
                    raise ValueError('Operation lacks exact new trajectory evidence')
                if kind != ('reinforce' if key in self.entries else 'add'):
                    raise ValueError('Operation conflicts with previous memory state')
                if key not in self.entries:
                    self.entries[key] = dict(claim=claim, sources=set())
                self.entries[key]['sources'].add(input_sha)
                applied.append(dict(key=key, op=kind, source_input_sha256=input_sha))
            except (KeyError, ValueError, TypeError) as exc:
                rejected.append(dict(index=index, reason=str(exc)))
        return dict(applied=applied, rejected=rejected, after=self.public_state())

    def experience_text(self, max_claims: int = 6) -> str:
        ranked = sorted(self.entries.items(), key=lambda item: (-len(item[1]['sources']), item[0]))
        return '\n'.join(f"[{key}; {len(value['sources'])} independent source games] "
                         f"When {value['claim']['when']}, {value['claim']['advice']}"
                         for key, value in ranked[:max_claims])


def teacher_operations(memory: ClaimMemory, episode: dict) -> list[dict]:
    return [dict(op='reinforce' if claim_key(claim) in memory.entries else 'add',
                 key=claim_key(claim), evidence_indices=claim['evidence_indices'],
                 evidence_row_sha256=claim['evidence_row_sha256'])
            for claim in claims_from_trajectory(episode['trajectory'])]


def writer_messages(memory: ClaimMemory, episode: dict) -> list[dict]:
    public = writer_input(episode)
    user = dict(previous_experience=memory.public_state(),
        completed_interaction=dict(initial_observation=public['initial_observation'],
            trajectory=[dict(index=i, **row) for i, row in enumerate(public['trajectory'])],
            outcome=public['outcome']))
    return [dict(role='system', content=SYSTEM),
            dict(role='user', content=json.dumps(user, ensure_ascii=False, separators=(',', ':')))]


def _choose_one_per_game(corpus: dict, excluded: set[str]) -> list[dict]:
    by_game = {}
    for sample in corpus['samples']:
        if sample['input_sha256'] not in excluded:
            by_game.setdefault(sample['input_sha256'], []).append(sample)
    chosen = []
    for rows in by_game.values():
        rows.sort(key=lambda row: (-len(row['claims']), row['source_episode_sha256']))
        chosen.append(rows[0])
    return chosen


def export(corpus: dict, data_root: Path, pilot_design: dict) -> dict:
    audit_corpus(corpus, data_root)
    excluded = {row['input_sha256'] for row in pilot_design['cases']}
    chosen = _choose_one_per_game(corpus, excluded)
    groups = {'train': [], 'validation': []}
    for sample in chosen:
        bucket = int(sample['input_sha256'][:8], 16) % 5
        groups['validation' if bucket == 0 else 'train'].append(sample)
    if min(len(groups['train']), len(groups['validation'])) < 3:
        raise ValueError('Need source-content-disjoint train and validation groups')
    output = {'train': [], 'validation': []}
    for split, samples in groups.items():
        for order_seed in (0, 1, 2) if split == 'train' else (0,):
            ordered = sorted(samples, key=lambda row: hashlib.sha256(
                f'{order_seed}:{row["input_sha256"]}'.encode()).hexdigest())
            memory = ClaimMemory()
            for position, sample in enumerate(ordered):
                episode_path = Path(sample['source_episode'])
                if (sha(episode_path) != sample['source_episode_sha256'] or
                        sha(data_root / sample['game']) != sample['input_sha256']):
                    raise ValueError('Source or game input changed')
                episode = read(episode_path)
                before = memory.public_state()
                messages = writer_messages(memory, episode)
                operations = teacher_operations(memory, episode)
                result = memory.apply(operations, episode, sample['input_sha256'])
                if result['rejected'] or len(result['applied']) != len(operations):
                    raise ValueError('Teacher update failed independent validator')
                output[split].append(dict(schema='alf_continual_writer_sft_v1',
                    split=split, order_seed=order_seed, position=position,
                    game=sample['game'], input_sha256=sample['input_sha256'],
                    source_episode=sample['source_episode'],
                    source_episode_sha256=sample['source_episode_sha256'],
                    previous_memory_sha256=digest(json.dumps(before, sort_keys=True, ensure_ascii=False)),
                    messages=messages,
                    completion=json.dumps({'operations': operations}, ensure_ascii=False, separators=(',', ':')),
                    after_memory_sha256=digest(json.dumps(result['after'], sort_keys=True, ensure_ascii=False)),
                    teacher_claims=sample['claims'], reviewed=True))
    train_inputs = {row['input_sha256'] for row in output['train']}
    validation_inputs = {row['input_sha256'] for row in output['validation']}
    if train_inputs & validation_inputs or excluded & (train_inputs | validation_inputs):
        raise ValueError('Training, validation, or frozen pilot input leakage')
    return dict(schema='alf_continual_writer_sft_export_v1', rows=output,
        distinct_inputs={split: len({row['input_sha256'] for row in rows})
                         for split, rows in output.items()},
        frozen_pilot_excluded_inputs=sorted(excluded),
        note='Warm-start supervised targets only; no writer weights updated or downstream reward claim')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--pilot-design', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    corpus_path, pilot_path = args.corpus.resolve(), args.pilot_design.resolve()
    result = export(read(corpus_path), args.data_root.resolve(), read(pilot_path))
    result.update(corpus_sha256=sha(corpus_path), pilot_design_sha256=sha(pilot_path),
                  exporter_sha256=sha(Path(__file__)))
    save(args.output, result)
    print(json.dumps(dict(rows={split: len(rows) for split, rows in result['rows'].items()},
                          distinct_inputs=result['distinct_inputs'])), flush=True)


if __name__ == '__main__':
    main()
