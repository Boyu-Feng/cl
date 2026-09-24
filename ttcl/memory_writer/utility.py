"""Delayed frozen-reader feedback and held-out utility evaluation, outside CLBench training."""
import argparse
import json
from pathlib import Path

from ttcl.memory_writer.core import apply_update, messages, parse_update, state_score
from ttcl.memory_writer.evaluate import WriterBackend, aggregate, append, update
from ttcl.memory_writer.train import read_rows, save_json


def probe_prompt(memory, probe):
    return [{'role': 'user', 'content': (
        f"Using only the memory, what is the stored {probe['attribute']} of {probe['entity']} "
        f"in {probe['scope']}? Return ONLY JSON {{\"value\": [strings], \"status\": \"observed or hypothesis\"}}. "
        'If unavailable or withdrawn return {"value": [], "status": "unknown"}.\nMemory:\n'
        + json.dumps(memory, ensure_ascii=False, sort_keys=True))}]


def probe_correct(raw, expected):
    try:
        raw = raw.strip()
        if raw.startswith('```') and raw.endswith('```'):
            raw = '\n'.join(raw.splitlines()[1:-1])
        answer = json.loads(raw)
        return (isinstance(answer, dict) and isinstance(answer.get('value'), list)
                and all(isinstance(v, str) for v in answer['value'])
                and sorted(answer['value']) == sorted(expected['value'])
                and answer.get('status') == expected['status'])
    except (ValueError, TypeError):
        return False


def probe_memory(model, memory, row, variant):
    answers = []
    for i, probe in enumerate(row['probes']):
        # Identical question and decoding seed across all candidate memory variants.
        identity = row['id'] + f'/future-question-{i}'
        completion = model.call(probe_prompt(memory, probe), identity, reader=True, max_new_tokens=128)
        record = dict(row_id=row['id'], variant=variant, question=i,
                      correct=completion['finish_reason'] == 'stop' and probe_correct(completion['raw_response'], probe['expected']),
                      raw_response=completion['raw_response'], expected=probe['expected'],
                      reader_adapter_enabled=False)
        append(model.out / 'probe_answers.jsonl', record)
        answers.append(record)
    return sum(r['correct'] for r in answers) / len(answers)


def select_pair(candidates):
    valid = [c for c in candidates if c['error'] is None]
    # Chosen memory also has to be supported by this independent training world's
    # known state. This filter is not an online benchmark reward or a model judge.
    supported = [c for c in valid if c['state']['precision'] == 1.0]
    if not supported or not valid:
        return None
    chosen = max(supported, key=lambda c: (c['utility'], -len(c['response'])))
    rejected = min(valid, key=lambda c: (c['utility'], len(c['response'])))
    if chosen['utility'] - rejected['utility'] < 0.5 or chosen['response'] == rejected['response']:
        return None
    return chosen, rejected


def collect(args):
    args.output.mkdir(parents=True, exist_ok=False)
    save_json(args.output / 'config.json', vars(args))
    model = WriterBackend(args)
    pairs, total, sampled = [], [], 0
    for index, row in enumerate(read_rows(args.data / 'preference_train.jsonl')):
        known = {e['id'] for e in row['events']} | {e for r in row['memory_before'] for e in r['evidence']}
        candidates = []
        for c in range(4):
            if c == 0:
                raw, finish = '{"operations": []}', 'stop'
            else:
                completion = model.call(messages(row['memory_before'], row['events'], row['schema']),
                                        row['id'] + f'/candidate-{c}',
                                        writer_temperature=(0.0, 0.0, 0.7, 1.0)[c])
                raw, finish = completion['raw_response'], completion['finish_reason']
                sampled += 1
            error, after, response = None, row['memory_before'], raw
            try:
                if finish != 'stop':
                    raise ValueError('Writer token budget exhausted')
                proposal = parse_update(raw)
                after = apply_update(row['memory_before'], proposal, known)
                response = json.dumps(proposal, sort_keys=True)
            except (ValueError, KeyError, TypeError) as exc:
                error = str(exc)
            # Do not spend reader calls repeatedly on an identical memory state.
            signature = json.dumps(after, sort_keys=True)
            previous = next((x for x in candidates if x['memory_signature'] == signature), None)
            utility = previous['utility'] if previous else probe_memory(model, after, row, f'candidate-{c}')
            candidates.append(dict(candidate=c, response=response, error=error, utility=utility,
                                   state=state_score(after, row['expected_after']), memory_signature=signature,
                                   memory_after=after))
        pair = select_pair(candidates)
        if pair:
            chosen, rejected = pair
            entry = {k: row[k] for k in ('id', 'memory_before', 'events', 'schema', 'family')}
            entry.update(chosen=chosen['response'], rejected=rejected['response'],
                         chosen_utility=chosen['utility'], rejected_utility=rejected['utility'])
            pairs.append(entry)
            append(args.output / 'pairs.jsonl', entry)
        append(args.output / 'candidates.jsonl', dict(id=row['id'], candidates=candidates, selected=bool(pair)))
        total.extend(candidates)
        save_json(args.output / 'progress.json', dict(phase='candidate_feedback', completed=index + 1,
                                                     pairs=len(pairs), model_calls=model.call_count))
        print('prefix', index + 1, 'pairs', len(pairs), 'calls', model.call_count, flush=True)
    metrics = dict(prefixes=index + 1, pairs=len(pairs), model_calls=model.call_count,
                   sampled_writer_candidates=sampled, candidate_count=len(total),
                   invalid_candidates=sum(c['error'] is not None for c in total),
                   reader_calls=model.call_count - sampled, reader_adapter_enabled=False,
                   clbench_used=False, minimum_required_pairs=8,
                   scoring='same future question, same frozen reader, same seed; chosen state support filter')
    save_json(args.output / 'metrics.json', metrics)
    if len(pairs) < 8:
        raise ValueError(f'Only {len(pairs)} informative pairs: not enough for preference training; no artificial ties generated')
    save_json(args.output / 'progress.json', dict(phase='complete', **metrics))


def evaluate(args):
    args.output.mkdir(parents=True, exist_ok=False)
    save_json(args.output / 'config.json', vars(args))
    model = WriterBackend(args)
    summary, records = {}, []
    for row in read_rows(args.data / 'test_utility.jsonl'):
        after, _, error = update(model, row['memory_before'], row['events'], row['schema'], row['id'])
        record = dict(id=row['id'], error=error, **state_score(after, row['expected_after']), memory_after=after)
        records.append(record)
        append(args.output / 'single_updates.jsonl', record)
    summary['single_update'] = aggregate(records)
    save_json(args.output / 'partial_metrics.json', summary)
    stream_records, utilities, ablations, oracles = [], [], [], []
    for sequence in read_rows(args.data / 'test_utility_streams.jsonl'):
        memory = []
        for row in sequence['rows']:
            memory, _, error = update(model, memory, row['events'], row['schema'], 'stream/' + row['id'])
            record = dict(id=row['id'], error=error, **state_score(memory, row['expected_after']), memory_after=memory)
            append(args.output / 'stream_updates.jsonl', record)
            stream_records.append(record)
        last = dict(sequence['rows'][-1])
        last['probes'] = list(last['probes']) + [
            dict(scope=r['scope'], entity=r['entity'], attribute=r['attribute'],
                 expected={'value': r['value'], 'status': r['status']})
            for r in last['expected_after'] if r['attribute'] in ('count', 'validation_command')]
        utilities.append(probe_memory(model, memory, last, 'predicted_memory'))
        ablations.append(probe_memory(model, [], last, 'memory_removed'))
        oracles.append(probe_memory(model, last['expected_after'], last, 'oracle_memory_diagnostic'))
        save_json(args.output / 'progress.json', dict(phase='utility_eval', streams=len(utilities),
                                                     model_calls=model.call_count))
    summary.update(stream=aggregate(stream_records), reader_accuracy=sum(utilities) / len(utilities),
                   memory_removed_accuracy=sum(ablations) / len(ablations),
                   oracle_memory_accuracy=sum(oracles) / len(oracles), streams=len(utilities),
                   questions=len(utilities) * 4, model_calls=model.call_count, reader_adapter_enabled=False)
    save_json(args.output / 'metrics.json', summary)
    save_json(args.output / 'progress.json', dict(phase='complete', **summary))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['collect', 'evaluate'])
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--adapter', type=Path, required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.device, args.dtype, args.context_limit = 'cuda:0', 'bfloat16', 16384
    args.max_new_tokens, args.temperature, args.top_p, args.top_k = 768, 0.7, 0.9, 0
    try:
        (collect if args.mode == 'collect' else evaluate)(args)
    except Exception as exc:
        args.output.mkdir(parents=True, exist_ok=True)
        save_json(args.output / 'failure.json', {'error_type': type(exc).__name__, 'error': str(exc)})
        raise


if __name__ == '__main__':
    main()
