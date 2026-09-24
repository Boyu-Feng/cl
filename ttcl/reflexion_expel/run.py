"""Pinned, resumable CLBench adaptation of Reflexion and offline ExpeL."""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import time
import traceback
from types import SimpleNamespace

from ttcl.experience_v2.common import BENCH, Client, PYTHON, read, save, seed
from .upstream import Upstream, Retriever

ARMS = ['none', 'retry_none', 'reflexion', 'expel']


def public_episode(episode, row):
    fields = ['public_task_brief', 'initial_public_query', 'response_schemas',
              'steps', 'completed', 'reward', 'reward_scope', 'format_failures',
              'local_execution_error']
    result = {key: episode[key] for key in fields if key in episode}
    result['official_success'] = row.get('success')
    return result


def trajectory_text(episode, row):
    return json.dumps(public_episode(episode, row), ensure_ascii=False)


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def llm_call(client, messages, path, random_seed):
    if path.exists():
        result = read(path)
        assert result['messages'] == messages, f'Resume prompt changed: {path}'
        return result
    result = client.complete(messages, random_seed=random_seed, tokens=1024,
                             temperature=0.0, top_p=1.0)
    if not result['raw_response']:
        raise RuntimeError(f'Empty writer response at {path}')
    result['messages'] = messages
    save(path, result)
    return result


def actor_episode(client, args, index, directory, context, repeat, attempt):
    from ttcl.structured_memory.online_bank import run_episode
    client.repeat = repeat if attempt == 0 else seed(repeat, 'retry', attempt)
    request = {'task': args.task, 'index': index, 'context': context,
               'repeat': repeat, 'attempt': attempt, 'actor_repeat': client.repeat}
    if (directory / 'row.json').exists():
        assert read(directory / 'request.json') == request
        return read(directory / 'row.json'), read(directory / 'trajectory.json')
    if directory.exists():
        directory.rename(directory.with_name(directory.name + f'.interrupted_{time.time_ns()}'))
    row, episode = run_episode(args, client, index, directory, context)
    if row['status'] != 'complete' and not any(x in row.get('error', '') for x in
            ['no schema-valid JSON action', 'Safety cap exceeded']):
        raise RuntimeError(f'Infrastructure failure: {row.get("error")}')
    save(directory / 'request.json', request)
    save(directory / 'row.json', row)
    return row, episode


def run_trials(client, upstream, args, index, directory, repeat,
               reflection=True, first=None, attempts=3):
    reflections, rows, episodes = [], [], []
    for attempt in range(attempts):
        dest = directory / f'trial_{attempt:02}'
        context = ('Plans from previous failed attempts on this SAME task:\n' +
            '\n\n'.join(reflections[-3:])) if reflection and reflections else ''
        if attempt == 0 and first is not None:
            source = Path(first)
            row, episode = read(source / 'row.json'), read(source / 'trajectory.json')
            dest.mkdir(parents=True, exist_ok=True)
            save(dest / 'row.json', dict(row, reused_from=str(source)))
            save(dest / 'trajectory.json', episode)
        else:
            row, episode = actor_episode(client, args, index, dest, context, repeat, attempt)
        rows.append(row)
        episodes.append(episode)
        if row.get('success') is True:
            break
        if attempt < attempts - 1 and reflection:
            messages = [{'role': 'user', 'content': upstream.reflection(
                trajectory_text(episode, row), reflections)}]
            update = llm_call(client, messages, dest / 'reflection.json',
                              seed(923, args.task, repeat, index, attempt, 'reflection'))
            reflections.append(update['raw_response'])
    # The policy returns the first success or the last attempt, never maximum reward.
    final = dict(rows[-1], attempts=len(rows), first_reward=rows[0]['reward'],
        first_success=rows[0].get('success'), canonical_index=index,
        total_actor_calls=sum(x['actor_calls'] for x in rows),
        total_actor_input_tokens=sum(x['actor_input_tokens'] for x in rows),
        total_actor_output_tokens=sum(x['actor_output_tokens'] for x in rows))
    save(directory / 'result.json', final)
    return rows, episodes


def extract_rules(client, upstream, gathered, out, task, repeat):
    rules, successes = [], []
    jobs = []
    for item in gathered:
        successful = [(r, ep) for r, ep in zip(item['rows'], item['episodes'])
                      if r.get('success') is True]
        failed = [(r, ep) for r, ep in zip(item['rows'], item['episodes'])
                  if r.get('success') is not True]
        if not successful:
            continue
        row, episode = successful[0]
        text = trajectory_text(episode, row)
        successes.append({'index': item['index'], 'query': episode['initial_public_query'],
                          'trajectory': text})
        for failed_row, failed_episode in failed:
            jobs.append({'kind': 'compare', 'indices': [item['index']],
                'success': text, 'failure': trajectory_text(failed_episode, failed_row),
                'task': episode['initial_public_query']})
    # Split only between complete trajectories to preserve all evidence.
    chunks, group, group_tokens = [], [], 0
    for example in successes:
        tokens = len(client.tokenizer.encode(example['trajectory'], add_special_tokens=False))
        if group and (len(group) == 8 or group_tokens + tokens > 48000):
            chunks.append(group)
            group, group_tokens = [], 0
        group.append(example)
        group_tokens += tokens
    if group:
        chunks.append(group)
    for group in chunks:
        jobs.append({'kind': 'all_success', 'indices': [s['index'] for s in group],
                     'success': '\n\n'.join(s['trajectory'] for s in group),
                     'failure': None, 'task': ''})
    for i, job in enumerate(jobs):
        messages = [{'role': 'system', 'content':
            'You are an advanced reasoning agent that can add, edit or remove rules '
            'based on completed CLBench task trajectories and their outcome feedback. '
            'Use only the supplied past interactions. Higher task reward is better. '
            'Do not infer a failure cause from the scalar alone.'},
            {'role': 'user', 'content': upstream.critique(rules,
                job['success'], job['failure'], job['task'])}]
        update = llm_call(client, messages, out / f'critique_{i:03}.json',
                          seed(923, task, repeat, 'critique', i))
        before = copy.deepcopy(rules)
        rules, operations, rejected = upstream.update(rules, update['raw_response'])
        save(out / f'operations_{i:03}.json', dict(kind=job['kind'],
            training_indices=job['indices'], before=before, after=rules,
            parsed_operations=operations, rejected_invalid_indices=rejected))
    state = {'rules': rules, 'successful_examples': successes,
             'training_indices': [x['index'] for x in gathered],
             'num_critique_calls': len(jobs), 'test_feedback_used': False}
    save(out / 'state.json', state)
    return state


def retrieve_context(state, query, retriever, tokenizer):
    if not state['rules'] and not state['successful_examples']:
        return '', {'selected': [], 'skipped': [], 'context_sha256': digest('')}
    rules = '\n'.join(f'{i}. {r[0]}' for i, r in enumerate(state['rules'], 1))
    context = 'Insights extracted from separate experience-gathering tasks:\n' + rules
    selected, skipped = [], []
    for index, score in retriever.rank(query, state['successful_examples']):
        document = state['successful_examples'][index]
        if document['query'] == query:
            skipped.append({'index': document['index'], 'reason': 'identical_query'})
            continue
        text = '\n\nSuccessful past task example:\n' + document['trajectory']
        # Whole-trajectory retrieval only; disclose skips rather than truncating evidence.
        if len(tokenizer.encode(context + text, add_special_tokens=False)) > 12000:
            skipped.append({'index': document['index'], 'reason': 'retrieval_token_budget'})
            continue
        context += text
        selected.append({'training_index': document['index'], 'cosine': score})
        if len(selected) == 2:
            break
    return context, {'selected': selected, 'skipped': skipped,
                     'context_sha256': digest(context)}


def worker(root, task, repeat):
    from ttcl.experience_v2.evaluate import configure_tasks
    from ttcl.structured_memory import run_benchmark as base
    configure_tasks()
    os.chdir(BENCH)
    plan = read(root / 'plan.json')
    settings = plan['tasks'][task]
    client = Client(plan['url'], repeat=repeat)
    upstream = Upstream(root / 'upstream')
    args = SimpleNamespace(task=task, seed=42, num_instances=settings['total'],
        memory_chars=20000, max_turns_per_instance=64, action_retries=2,
        normalize_action=True, allow_initial_experience=True)
    dest = root / 'runs' / task / str(repeat)
    progress = root / 'workers' / f'{task}_{repeat}.json'
    def update(**values):
        save(progress, dict(task=task, repeat=repeat, updated_at=time.time(), **values))
    # Persist the initial public queries before any policy is evaluated. This
    # establishes canonical split identity without looking at held-out outcomes.
    queries = {}
    for index in settings['train'] + settings['test']:
        env = base.make_task(task, 42, independent=True)
        try:
            q = env.reset_baseline_instance(index)
            queries[index] = q.prompt
        finally:
            connection = getattr(env, '_conn', None)
            if connection is not None:
                connection.close()
    assert not set(settings['train']) & set(settings['test'])
    # A prompt can contain canonical ordinals; exact duplicates are still reported.
    duplicates = [(a, b) for a in settings['train'] for b in settings['test']
                  if queries[a] == queries[b]]
    save(dest / 'split_audit.json', {'train': settings['train'], 'test': settings['test'],
        'disjoint_indices': True, 'identical_initial_queries': duplicates,
        'query_hashes': {str(i): digest(q) for i, q in queries.items()}})
    if duplicates:
        raise ValueError(f'Train/test duplicate public queries: {duplicates}')
    gathered = []
    for position, index in enumerate(settings['train']):
        update(phase='gathering', completed=position, expected=len(settings['train']), index=index)
        rows, episodes = run_trials(client, upstream, args, index,
            dest / 'gathering' / f'episode_{index+1:03}', repeat)
        gathered.append({'index': index, 'rows': rows, 'episodes': episodes})
    update(phase='extracting_rules', completed=len(gathered), expected=len(gathered))
    state = extract_rules(client, upstream, gathered, dest / 'insights', task, repeat)
    retriever = Retriever(plan['embedding_model'])
    for position, index in enumerate(settings['test']):
        episode_dir = dest / 'evaluation' / f'episode_{index+1:03}'
        update(phase='evaluation', completed=position, expected=len(settings['test']),
               index=index, arm='none')
        first = episode_dir / 'none' / 'trial_00'
        row, _ = actor_episode(client, args, index, first, '', repeat, 0)
        save(episode_dir / 'none' / 'result.json', dict(row, attempts=1,
            first_reward=row['reward'], total_actor_calls=row['actor_calls'],
            total_actor_input_tokens=row['actor_input_tokens'],
            total_actor_output_tokens=row['actor_output_tokens']))
        for arm in ['retry_none', 'reflexion']:
            update(phase='evaluation', completed=position, expected=len(settings['test']),
                   index=index, arm=arm)
            run_trials(client, upstream, args, index, episode_dir / arm, repeat,
                       reflection=arm == 'reflexion', first=first)
        update(phase='evaluation', completed=position, expected=len(settings['test']),
               index=index, arm='expel')
        context, audit = retrieve_context(state, queries[index], retriever, client.tokenizer)
        assert all(x['training_index'] in settings['train'] for x in audit['selected'])
        save(episode_dir / 'expel' / 'retrieval.json', audit)
        row, _ = actor_episode(client, args, index, episode_dir / 'expel' / 'trial_00',
                               context, repeat, 0)
        save(episode_dir / 'expel' / 'result.json', dict(row, attempts=1,
            first_reward=row['reward'], total_actor_calls=row['actor_calls'],
            total_actor_input_tokens=row['actor_input_tokens'],
            total_actor_output_tokens=row['actor_output_tokens']))
        print(json.dumps({'task': task, 'repeat': repeat, 'evaluated': position + 1,
                          'expected': len(settings['test'])}), flush=True)
    update(phase='complete', completed=len(settings['test']), expected=len(settings['test']))


def report(root):
    plan = read(root / 'plan.json')
    summary = {'updated_at': datetime.now(timezone.utc).isoformat(), 'tasks': {}}
    lines = ['# Reflexion / ExpeL CLBench', '', f'Updated: {summary["updated_at"]}', '',
        'Local Qwen3-4B ports of pinned official mechanisms; not a reproduction of paper scores.',
        'ExpeL learns rules/retrieval examples from a separate 20% prefix. The other 80% is held out.',
        'Reflexion and retry_none stop at first official success or after 3 attempts; final, not best, reward.',
        'Terminal reward/success are visible after an attempt. ExpeL rules stay frozen during evaluation.',
        'Scores below use only cases completed by all four methods; means across tasks are not combined.',
        'Model-format/turn-cap failures are unscored and counted explicitly, not silently zeroed.', '']
    for task, settings in plan['tasks'].items():
        groups = {arm: {} for arm in ARMS}
        for repeat in plan['repeats']:
            for index in settings['test']:
                for arm in ARMS:
                    p = root/'runs'/task/str(repeat)/'evaluation'/f'episode_{index+1:03}'/arm/'result.json'
                    if p.exists():
                        groups[arm][(repeat, index)] = read(p)
        common = set.intersection(*[{k for k, r in g.items()
            if r.get('reward') is not None and r['status'] == 'complete'} for g in groups.values()])
        for key in common:
            if len({g[key]['instance_id'] for g in groups.values()}) != 1:
                raise ValueError(f'Pair identity mismatch: {task} {key}')
        expected = len(settings['test']) * len(plan['repeats'])
        lines += [f'## {task}', '', f'Common completed pairs: {len(common)}/{expected}', '',
            '| Method | Recorded | Unscored | Mean reward | Delta vs none | Mean attempts |',
            '|---|---:|---:|---:|---:|---:|']
        info = {'common_pairs': len(common), 'expected': expected, 'arms': {}, 'experience_banks': {}}
        for rep in plan['repeats']:
            p = root/'runs'/task/str(rep)/'insights/state.json'
            if p.exists():
                bank = read(p)
                info['experience_banks'][str(rep)] = {
                    'source_tasks': len(bank['training_indices']),
                    'successful_tasks': len(bank['successful_examples']),
                    'rules': len(bank['rules']), 'critique_calls': bank['num_critique_calls']}
        for arm, rows in groups.items():
            vals = [rows[k]['reward'] for k in sorted(common)]
            delta = [rows[k]['reward'] - groups['none'][k]['reward'] for k in sorted(common)]
            stat = {'recorded': len(rows), 'unscored': sum(r['reward'] is None for r in rows.values()),
                'mean_reward': statistics.mean(vals) if vals else None,
                'delta_vs_none': statistics.mean(delta) if delta else None,
                'mean_attempts': statistics.mean(rows[k]['attempts'] for k in common) if common else None,
                'logical_actor_calls': sum(r['total_actor_calls'] for r in rows.values()),
                'logical_actor_input_tokens': sum(r['total_actor_input_tokens'] for r in rows.values()),
                'logical_actor_output_tokens': sum(r['total_actor_output_tokens'] for r in rows.values()),
                'by_repeat': {str(rep): statistics.mean([rows[k]['reward'] for k in common if k[0] == rep])
                    for rep in plan['repeats'] if any(k[0] == rep for k in common)}}
            info['arms'][arm] = stat
            def fmt(v):
                return '—' if v is None else f'{v:.4f}'
            lines.append(f'| {arm} | {len(rows)}/{expected} | {stat["unscored"]} | '
                f'{fmt(stat["mean_reward"])} | {fmt(stat["delta_vs_none"])} | {fmt(stat["mean_attempts"])} |')
        info['reflexion_minus_retry_none'] = statistics.mean(
            groups['reflexion'][k]['reward'] - groups['retry_none'][k]['reward']
            for k in common) if common else None
        summary['tasks'][task] = info
        lines += ['', f'Reflexion minus retry_none: {info["reflexion_minus_retry_none"]}', '']
        for rep, bank in info['experience_banks'].items():
            lines.append(f'ExpeL source seed {rep}: {bank["successful_tasks"]}/{bank["source_tasks"]} '
                         f'successful tasks, {bank["rules"]} extracted rules.')
        lines.append('')
    # Costs are physical across all stages, alongside logical reused first attempts above.
    costs = {'actor_calls': 0, 'actor_input_tokens': 0, 'actor_output_tokens': 0,
             'writer_calls': 0, 'writer_input_tokens': 0, 'writer_output_tokens': 0,
             'writer_length_stops': 0, 'empty_rule_updates': 0}
    for p in (root/'runs').glob('**/trial_*/row.json'):
        r = read(p)
        if r.get('reused_from'):
            continue
        for field in ['actor_calls', 'actor_input_tokens', 'actor_output_tokens']:
            costs[field] += r[field]
    for pattern in ['**/reflection.json', '**/critique_*.json']:
        for p in (root/'runs').glob(pattern):
            r = read(p)
            costs['writer_calls'] += 1
            costs['writer_input_tokens'] += r['input_tokens']
            costs['writer_output_tokens'] += r['output_tokens']
            costs['writer_length_stops'] += r['finish_reason'] == 'length'
    for p in (root/'runs').glob('**/operations_*.json'):
        costs['empty_rule_updates'] += not read(p)['parsed_operations']
    summary['costs'] = costs
    summary['workers'] = [read(p) for p in sorted((root/'workers').glob('*.json'))]
    lines += ['## Progress', '']
    for w in summary['workers']:
        lines.append(f'- {w["task"]}, seed {w["repeat"]}: {w["phase"]}, '
                     f'{w.get("completed", 0)}/{w.get("expected", "?")}')
    lines += ['', '## Costs (including experience gathering)', '', json.dumps(costs, indent=2), '',
        'Sales prediction and codebase adaptation remain blocked by Docker permissions.',
        'See PROTOCOL.md, provenance.json, per-trial trajectories and insights/operations_*.json.']
    save(root/'summary.json', summary)
    tmp = root/'REPORT.md.tmp'
    tmp.write_text('\n'.join(lines) + '\n')
    tmp.replace(root/'REPORT.md')


def supervise(root):
    plan = read(root/'plan.json')
    jobs = [(task, rep) for task in plan['tasks'] for rep in plan['repeats']]
    active, finished, failed = {}, [], []
    (root/'logs').mkdir(exist_ok=True)
    while jobs or active:
        while jobs and len(active) < plan['concurrency']:
            task, repeat = jobs.pop(0)
            name = f'{task}_{repeat}'
            log = (root/'logs'/f'{name}.log').open('a')
            command = [str(PYTHON), '-m', 'ttcl.reflexion_expel.run', 'worker',
                       '--root', str(root), '--task', task, '--repeat', str(repeat)]
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            log.close()
            active[name] = child
        for name, child in list(active.items()):
            if child.poll() is not None:
                (finished if child.returncode == 0 else failed).append(name)
                del active[name]
        save(root/'status.json', {'phase': 'running' if jobs or active else
             ('complete' if not failed else 'finished_with_failures'),
             'supervisor_pid': os.getpid(), 'active': {n: p.pid for n, p in active.items()},
             'queued': len(jobs), 'finished': finished, 'failed': failed, 'updated_at': time.time()})
        report(root)
        if jobs or active:
            time.sleep(15)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['worker', 'supervise', 'report'])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--task')
    parser.add_argument('--repeat', type=int, default=303)
    args = parser.parse_args()
    args.root = args.root.resolve()
    try:
        if args.command == 'worker':
            worker(args.root, args.task, args.repeat)
        elif args.command == 'supervise':
            supervise(args.root)
        else:
            report(args.root)
    except Exception:
        save(args.root/'failures'/f'{args.command}_{args.task}_{args.repeat}.json',
             {'traceback': traceback.format_exc(), 'time': time.time()})
        raise


if __name__ == '__main__':
    main()
