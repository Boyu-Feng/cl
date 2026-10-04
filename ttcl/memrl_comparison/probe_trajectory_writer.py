"""Diagnostic: can the frozen writer extract evidence-bound trajectory claims?"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import requests

from ttcl.experience_evolution.core import digest
from ttcl.icl_mem0_comparison.protocol import read, save, sha
from .trajectory_experience_teacher import claims_from_trajectory, writer_input


SYSTEM = ("Extract reusable experience from the completed trajectory. Return only a JSON object "
          "with a 'claims' list. Allowed claim types: "
          "(1) {'kind':'verified_preparation_tool','operation':'clean|cool|heat',"
          "'tool_type':'tool type','evidence_indices':[index]} only if that exact action received "
          "explicit success feedback for the same object and tool; "
          "(2) {'kind':'move_retry_after_navigation','evidence_indices':[failed_move_index,"
          "successful_go_to_destination_index,successful_retry_index]} only if the same move failed "
          "with 'Nothing happens', navigation to its destination succeeded, then that move succeeded. "
          "Use zero-based indices from the supplied trajectory. Do not infer a claim from final task reward, "
          "invent missing evidence, or copy object instance IDs into a general rule. Return [] if uncertain.")


def prompt_for(episode: dict) -> list[dict]:
    public = writer_input(episode)
    compact = dict(initial_observation=public['initial_observation'],
        trajectory=[dict(index=i, **row) for i, row in enumerate(public['trajectory'])])
    return [dict(role='system', content=SYSTEM),
            dict(role='user', content=json.dumps(compact, ensure_ascii=False, separators=(',', ':')))]


def parse(raw: str, expected: list[dict]) -> dict:
    try:
        decoder = json.JSONDecoder()
        start = raw.index('{')
        value, _ = decoder.raw_decode(raw[start:])
        rows = value['claims']
        if not isinstance(rows, list):
            raise ValueError('Claims must be a list')
    except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        return dict(valid=[], invalid=[dict(reason=str(exc))], exact_recall=0)
    expected_keys = {(row['kind'], tuple(row['evidence_indices']),
                      row.get('operation'), row.get('tool_type')) for row in expected}
    valid, invalid = [], []
    for row in rows:
        try:
            if not isinstance(row, dict) or row.get('kind') not in {
                    'verified_preparation_tool', 'move_retry_after_navigation'}:
                raise ValueError('Unknown kind')
            indices = row.get('evidence_indices')
            if not isinstance(indices, list) or not all(isinstance(i, int) for i in indices):
                raise ValueError('Bad evidence indices')
            key = (row['kind'], tuple(indices), row.get('operation'), row.get('tool_type'))
            if key not in expected_keys:
                raise ValueError('Unsupported claim or wrong evidence')
            if key not in valid:
                valid.append(key)
        except ValueError as exc:
            invalid.append(dict(claim=row, reason=str(exc)))
    return dict(valid=[list(key) for key in valid], invalid=invalid,
                exact_recall=len(valid) / len(expected_keys) if expected_keys else 1.0)


def select(corpus: dict) -> list[dict]:
    by_game = {}
    for sample in corpus['samples']:
        by_game.setdefault(sample['input_sha256'], []).append(sample)
    recovery, preparation = [], []
    for game, rows in by_game.items():
        rows.sort(key=lambda row: hashlib.sha256(row['source_episode'].encode()).hexdigest())
        candidate = next((row for row in rows if any(c['kind'] == 'move_retry_after_navigation'
                                                    for c in row['claims'])), rows[0])
        (recovery if any(c['kind'] == 'move_retry_after_navigation' for c in candidate['claims'])
         else preparation).append(candidate)
    recovery.sort(key=lambda row: row['input_sha256'])
    preparation.sort(key=lambda row: row['input_sha256'])
    return recovery[:3] + preparation[:3]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--url', default='http://127.0.0.1:18559')
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    corpus_path, output = args.corpus.resolve(), args.output.resolve()
    corpus = read(corpus_path)
    cases = select(corpus)
    if len(cases) != 6 or len({case['input_sha256'] for case in cases}) != 6:
        raise ValueError('Need six distinct source games')
    design = dict(schema='alf_trajectory_writer_pilot_v1',
        selection='Three source games with verified move recovery and three with preparation effect only; SHA-ranked within strata before model output; diagnostic, not unbiased prevalence',
        cases=[dict(game=row['game'], input_sha256=row['input_sha256'],
            source_episode=row['source_episode'], source_episode_sha256=row['source_episode_sha256'],
            expected_claims=row['claims']) for row in cases],
        corpus_path=str(corpus_path), corpus_sha256=sha(corpus_path),
        runner_sha256=sha(Path(__file__)), url=args.url,
        budget='One frozen 4B writer call per six selected official train trajectories; temperature 0, max_tokens 512; no training or test data')
    output.mkdir(parents=True, exist_ok=True)
    frozen = output / 'design.json'
    if frozen.exists():
        if read(frozen) != design:
            raise ValueError('Frozen pilot design changed')
    else:
        save(frozen, design)
    print(json.dumps({'design_sha256': sha(frozen), 'cases': len(cases)}), flush=True)
    if args.prepare_only:
        return
    for index, case in enumerate(design['cases']):
        path = output / f'{index:02d}.json'
        if path.exists():
            continue
        episode_path = Path(case['source_episode'])
        if sha(episode_path) != case['source_episode_sha256']:
            raise ValueError('Source trajectory changed')
        episode = read(episode_path)
        if episode['game'] != case['game'] or claims_from_trajectory(episode['trajectory']) != case['expected_claims']:
            raise ValueError('Expected evidence changed')
        messages = prompt_for(episode)
        started = time.monotonic()
        response = requests.post(args.url + '/v1/chat/completions', json=dict(
            model='frozen-actor', messages=messages, temperature=0., top_p=1.,
            max_tokens=512, seed=95000 + index), timeout=240)
        response.raise_for_status()
        result = response.json()
        raw = result['choices'][0]['message']['content']
        judgment = parse(raw, case['expected_claims'])
        row = dict(index=index, game=case['game'], input_sha256=case['input_sha256'],
            source_episode_sha256=case['source_episode_sha256'],
            prompt_sha256=digest(json.dumps(messages, ensure_ascii=False)),
            raw_response=raw, raw_response_sha256=digest(raw),
            usage=result.get('usage', {}), seconds=time.monotonic() - started,
            expected_claims=case['expected_claims'], judgment=judgment)
        save(path, row)
        print(json.dumps(dict(index=index, expected=len(case['expected_claims']),
            valid=len(judgment['valid']), invalid=len(judgment['invalid']),
            recall=judgment['exact_recall'])), flush=True)


if __name__ == '__main__':
    main()
