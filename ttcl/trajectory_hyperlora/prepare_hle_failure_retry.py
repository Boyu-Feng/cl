"""Prepare disjoint, answer-isolated HLE text-MCQ retry pilot inputs.

Reads an authorized local copy; does not bypass the official dataset gate.
This prepares inputs only and does not claim any HLE evaluation result.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def prepare_rows(rows, source_count=64, evaluation_count=64):
    if source_count < 1 or evaluation_count < 1:
        raise ValueError('Both partitions must be nonempty')
    eligible = []
    seen = set()
    for row in rows:
        key = str(row['id'])
        if key in seen:
            raise ValueError('Duplicate HLE id')
        seen.add(key)
        kind = str(row.get('answer_type', '')).lower().replace('_', '').replace('-', '')
        if row.get('image') or kind not in ('multiplechoice', 'mcq'):
            continue
        if not isinstance(row['question'], str) or not isinstance(row['answer'], str):
            raise ValueError('Expected text question and answer')
        if not row['question'].strip() or not row['answer'].strip():
            raise ValueError('Empty question or answer')
        eligible.append(row)
    eligible.sort(key=lambda x: digest(['hle_failure_retry_v1', str(x['id'])]))
    total = source_count + evaluation_count
    if len(eligible) < total:
        raise ValueError(f'Need {total} eligible text MCQs, found {len(eligible)}')
    actor, grading = [], []
    for i, row in enumerate(eligible[:total]):
        item = {'id': str(row['id']), 'question': row['question'],
                'partition': 'source_collection' if i < source_count else 'evaluation'}
        binding = digest(item)
        actor.append({**item, 'input_content_sha256': binding})
        grading.append({'id': str(row['id']), 'answer': row['answer'], 'input_content_sha256': binding})
    return actor, grading, len(eligible)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True, type=Path, help='Authorized HLE JSON list, JSONL, or parquet file')
    p.add_argument('--output-dir', required=True, type=Path, help='New private/ignored data directory')
    p.add_argument('--source-count', type=int, default=64)
    p.add_argument('--evaluation-count', type=int, default=64)
    args = p.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if args.input.suffix == '.parquet':
        import pyarrow.parquet as pq
        rows = pq.read_table(args.input).to_pylist()
    elif args.input.suffix == '.jsonl':
        rows = [json.loads(line) for line in args.input.read_text().splitlines() if line.strip()]
    else:
        rows = json.loads(args.input.read_text())
    if not isinstance(rows, list):
        raise ValueError('Expected a list of question objects')
    actor, grading, eligible = prepare_rows(rows, args.source_count, args.evaluation_count)
    manifest = {'stage': 'inputs_prepared_not_evaluated', 'scope': 'text-only MCQ pilot, not full HLE score',
        'input_sha256': hashlib.sha256(args.input.read_bytes()).hexdigest(),
        'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'eligible_text_mcqs': eligible, 'source_count': args.source_count, 'evaluation_count': args.evaluation_count,
        'actor_inputs_sha256': digest(actor), 'grading_keys_sha256': digest(grading),
        'source_policy': 'Generate own responses on disjoint source questions; only verified own successes can become source trajectories; no reference-answer text',
        'retry_policy': 'Score first attempts; at most one retry per arm on the identical first-failure set; correctness bit only; no correct answers or judge rationale in actor or residual',
        'planned_controls': ['unchanged_retry', 'failure_residual', 'matched_norm_random_residual'],
        'limitation': 'ALFWorld-trained hypernetwork transfer requires explicit reporting; source collection consumes additional inference budget; no source success means no valid success-minus-failure arm'}
    args.output_dir.mkdir(parents=True)
    for name, value in [('actor_inputs.json', actor), ('grading_keys.json', grading), ('manifest.json', manifest)]:
        (args.output_dir/name).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({'prepared_source_questions': args.source_count, 'prepared_evaluation_questions': args.evaluation_count,
                      'eligible_text_mcqs': eligible, 'evaluation_started': False}))


if __name__ == '__main__':
    main()
