"""Matched writer evaluations with an adapter-disabled, frozen answer model."""
import argparse
from collections import defaultdict
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import sys

from ttcl.common.local_qwen import LocalQwen
from ttcl.memory_writer.core import apply_update, messages, parse_update, state_score
from ttcl.memory_writer.train import read_rows, save_json


def append(path, row):
    with Path(path).open('a') as handle:
        handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')


def seed_for(identity, reader=False):
    raw = json.dumps([42, identity, 0 if reader else 'memory_writer'], ensure_ascii=False)
    return int.from_bytes(hashlib.sha256(raw.encode()).digest()[:8], 'big') % (2**63)


class WriterBackend(LocalQwen):
    def __init__(self, args):
        super().__init__(args)
        self.adapter = args.adapter
        self.out = args.output
        self.call_count = 0
        if self.adapter:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, self.adapter, is_trainable=False).eval()
            self.model.requires_grad_(False)

    def call(self, prompt, identity, reader=False, max_new_tokens=768, writer_temperature=0.0):
        self.call_count += 1
        seed = seed_for(identity, reader)
        meta = {'call_id': self.call_count, 'identity': identity, 'purpose': 'reader' if reader else 'writer',
                'seed': seed, 'adapter_enabled': bool(self.adapter) and not reader,
                'temperature': 0.7 if reader else writer_temperature}
        append(self.out / 'requests.jsonl', {**meta, 'messages': prompt})
        context = self.model.disable_adapter() if reader and self.adapter else nullcontext()
        with context:
            result = super().generate(prompt, seed, max_new_tokens=max_new_tokens,
                                      temperature=0.7 if reader else writer_temperature)
        append(self.out / 'generations.jsonl', {**meta, **result})
        return result


def update(model, memory, events, schema, identity):
    result = model.call(messages(memory, events, schema), identity)
    try:
        if result['finish_reason'] != 'stop':
            raise ValueError('Writer output exhausted token budget')
        proposal = parse_update(result['raw_response'])
        known = {e['id'] for e in events} | {e for r in memory for e in r['evidence']}
        after = apply_update(memory, proposal, known)
        error = None
    except (ValueError, KeyError, TypeError) as exc:
        after, error = memory, str(exc)
    return after, result, error


def aggregate(rows):
    n = len(rows)
    return {'count': n, 'valid_rate': sum(r['error'] is None for r in rows) / max(n, 1),
            'exact_state': sum(r['exact_state'] for r in rows) / max(n, 1),
            'mean_f1': sum(r['f1'] for r in rows) / max(n, 1)}


def run_evaluation(args):
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    save_json(out / 'config.json', vars(args))
    save_json(out / 'progress.json', {'phase': 'loading'})
    model = WriterBackend(args)
    summary = {}
    for dataset in ('test_sgd', 'test_synthetic'):
        records, groups = [], defaultdict(list)
        for row in read_rows(args.data / (dataset + '.jsonl'))[:args.examples]:
            after, completion, error = update(model, row['memory_before'], row['events'], row['schema'], row['id'])
            record = dict(id=row['id'], family=row['family'], error=error,
                          expected_change=bool(row['target']['operations']),
                          **state_score(after, row['expected_after']), memory_after=after)
            append(out / (dataset + '.jsonl'), record)
            records.append(record)
            groups[row['family']].append(record)
            save_json(out / 'progress.json', {'phase': dataset, **aggregate(records), 'model_calls': model.call_count})
            if len(records) % 8 == 0:
                print(dataset, len(records), aggregate(records), flush=True)
        summary[dataset + '_gold_history'] = {
            **aggregate(records), 'changed_only': aggregate([r for r in records if r['expected_change']]),
            'noop_only': aggregate([r for r in records if not r['expected_change']]),
            'by_family': {k: aggregate(v) for k, v in groups.items()}}
        save_json(out / 'partial_metrics.json', summary)
    stream_records, probes = [], []
    for sequence in read_rows(args.data / 'test_streams.jsonl'):
        memory = []
        for row in sequence['rows']:
            before = memory
            memory, completion, error = update(model, memory, row['events'], row['schema'], 'rollout/' + row['id'])
            record = dict(stream=sequence['id'], id=row['id'], error=error,
                          **state_score(memory, row['expected_after']), memory_before=before, memory_after=memory)
            append(out / 'stream_rollouts.jsonl', record)
            stream_records.append(record)
            save_json(out / 'progress.json', {'phase': 'stream_rollout', **aggregate(stream_records),
                                            'model_calls': model.call_count})
        expected = sequence['rows'][-1]['expected_after']
        if expected:
            # Ask about an early retained key, not simply the latest event.
            probe = expected[min(1, len(expected) - 1)]
            query = (f"Using only the supplied memory, return the stored values and status for scope={probe['scope']}, "
                     f"entity={probe['entity']}, attribute={probe['attribute']}. "
                     'Return JSON {"value": [strings], "status": "observed or hypothesis"}. '
                     'If unavailable, return {"value": [], "status": "unknown"}.\nMemory:\n' + json.dumps(memory, sort_keys=True))
            completion = model.call([{'role': 'user', 'content': query}], sequence['id'] + '/probe', reader=True)
            try:
                raw = completion['raw_response'].strip()
                if raw.startswith('```'):
                    raw = '\n'.join(raw.splitlines()[1:-1])
                answer = json.loads(raw)
                correct = sorted(answer['value']) == sorted(probe['value']) and answer['status'] == probe['status']
            except (ValueError, KeyError, TypeError):
                correct = False
            result = dict(stream=sequence['id'], correct=correct, raw_response=completion['raw_response'],
                          expected=probe, reader_adapter_enabled=False)
            append(out / 'reader_probes.jsonl', result)
            probes.append(result)
        print('stream', sequence['id'], aggregate(stream_records), flush=True)
    summary['stream_predicted_history'] = aggregate(stream_records)
    summary['reader_probes'] = {'count': len(probes), 'accuracy': sum(r['correct'] for r in probes) / max(1, len(probes))}
    save_json(out / 'partial_metrics.json', summary)
    if args.bsm_scans:
        summary['bsm_independent'] = run_bsm(args, model, use_memory=False)
        summary['bsm_structured_memory'] = run_bsm(args, model, use_memory=True)
        summary['bsm_delta'] = summary['bsm_structured_memory']['mean_score'] - summary['bsm_independent']['mean_score']
    summary.update(model_calls=model.call_count, reader_adapter_enabled=False, parameter_updates_during_eval=0,
                   note='Custom strict memory-state metrics, not official SGD benchmark scores. BSM uses official task score.')
    save_json(out / 'metrics.json', summary)
    save_json(out / 'progress.json', {'phase': 'complete', 'model_calls': model.call_count})


def run_bsm(args, model, use_memory):
    bench = args.repo_root / 'current_work/continual-learning-bench'
    sys.path.insert(0, str(bench))
    from src.interface import Response
    from src.tasks.blind_spectrum_monitoring.task import BlindSpectrumMonitoringTask
    from ttcl.common.bsm import parse_report
    from ttcl.llm_memory.memory import answer_messages
    import math

    mode = 'bsm_structured' if use_memory else 'bsm_independent'
    task = BlindSpectrumMonitoringTask(dataset_path=str(bench / 'data/blind_spectrum_monitoring/mixed_grid_lifecycle.jsonl'),
                                      num_instances=args.bsm_scans, seed=42, repeat_instructions=True)
    query = task.reset()
    memory, records = [], []
    for scan in range(1, args.bsm_scans + 1):
        identity = query.instance_id or query.prompt
        before = memory
        prompt = answer_messages(query.prompt, query.response_schema.model_json_schema(), json.dumps(memory, sort_keys=True) if memory else '')
        completion = model.call(prompt, identity, reader=True, max_new_tokens=1536)
        try:
            report = parse_report(completion['raw_response'], query.response_schema)
            if any(not math.isfinite(v) for r in report.transmitters for v in (r.center_freq, r.bandwidth, r.estimated_power)):
                raise ValueError('Nonfinite report')
            response, error = Response(action=report), None
        except ValueError as exc:
            response = Response(action=query.response_schema(transmitters=[]), metadata={'latency_timeout': True})
            error = str(exc)
        step = task.step(response)
        row = dict(scan=scan, reward=float(step.instance_outcome.reward), error=error,
                   raw_response=completion['raw_response'], memory_before=before, reader_adapter_enabled=False)
        records.append(row)
        append(args.output / (mode + '.jsonl'), row)
        if use_memory:
            events = [dict(id=f'scan-{scan}', role='completed_interaction',
                           text=json.dumps({'task': query.prompt, 'response': completion['raw_response'],
                                            'feedback': step.observation.content}))]
            memory, write_result, write_error = update(model, memory, events, None, f'bsm-memory-{scan}')
            append(args.output / 'bsm_memory.jsonl', dict(scan=scan, error=write_error, memory_after=memory,
                                                        raw_response=write_result['raw_response']))
        save_json(args.output / 'progress.json', {'phase': mode, 'completed': scan,
                                                'mean_score': sum(r['reward'] for r in records) / scan})
        print(mode, scan, row['reward'], flush=True)
        if step.done:
            break
        query = step.next_query
    return {'completed': len(records), 'mean_score': sum(r['reward'] for r in records) / len(records),
            'invalid_reports': sum(r['error'] is not None for r in records), 'reward_calls': len(records),
            'score': task.evaluate().score, 'scalar_reward_in_memory': False}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--adapter', type=Path)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--repo-root', type=Path, required=True)
    p.add_argument('--examples', type=int, default=64)
    p.add_argument('--bsm-scans', type=int, default=12)
    args = p.parse_args()
    args.device, args.dtype, args.context_limit = 'cuda:0', 'bfloat16', 16384
    args.max_new_tokens, args.temperature, args.top_p, args.top_k = 768, 0.7, 0.9, 0
    try:
        run_evaluation(args)
    except Exception as exc:
        args.output.mkdir(parents=True, exist_ok=True)
        save_json(args.output / 'failure.json', {'error_type': type(exc).__name__, 'error': str(exc)})
        raise


if __name__ == '__main__':
    main()
