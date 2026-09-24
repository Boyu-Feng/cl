from __future__ import annotations

import argparse
from concurrent.futures import CancelledError, ThreadPoolExecutor, as_completed
from collections import defaultdict
import fcntl
import json
import os
from pathlib import Path
import random
import shutil
import signal
import subprocess
import threading
import time
import traceback

from ttcl.experience_evolution.core import FAMILIES, read, save, seed, digest
from .environment import Actor, make_env, concurrent_preflight
from ttcl.experience_v2.common import WORKSPACE, OLD, MODEL, PYTHON, Client, environment, sha_file, start_server, stop_server
from ttcl.reflexion_expel.upstream import Retriever
from .report import report

ROOT = WORKSPACE/'ttcl/results/alfworld_comparison/20260924_parserfix'
ARMS = ['retry_none', 'untrained', 'delta', 'reflexion', 'expel']
REPORT_LOCK = threading.Lock()

PROTOCOL = '''# Matched ALFWorld Delta / Reflexion / ExpeL evaluation

Primary test set: the complete official valid_unseen split, subject to the frozen
path and full-file content-hash exclusion audit. All previous training, calibration,
screening and training-probe tasks are excluded. Previously evaluated test tasks
remain allowed but are explicitly identified; the never-evaluated subset is also
reported. It is not claimed that all valid_unseen tasks are newly untouched.
No test outcome selects data, checkpoints, prompts, hyperparameters or candidates.

All five methods use frozen Qwen3-4B-Instruct-2507, identical command-only prompts
and admissible commands, no shared fixed actor few-shot demonstrations, temperature
.7, top_p 1, 64 output tokens, 50 actions per attempt, at most three fresh environment
resets per task. ExpeL retrieved training examples are part of its bounded memory
context.
Stop at the first official success; otherwise report failure after the last attempt.
Task/attempt/action seeds are method independent. Memory has a shared 2048-token
maximum; whole entries are retained or explicitly skipped, never silent truncation.
Complete per-attempt trajectories and generation costs are recorded. Writer output
limit is 768 tokens and greedy decoding. Context limit is 65536 tokens for everyone.

Three sampling seeds, separately shuffled fixed same-family chains. Methods see
the same ordered tasks. Each method keeps its own trajectory and memory; seeds and
families never share online state. The primary score includes the initial empty
memory task. Also report scores excluding each family chain's first task.

retry_none: no experience; up to three attempts. Its first attempt is the single-
attempt no-experience control. Identical prompts may reuse an executed episode,
with provenance and logical versus physical computation distinguished.
untrained: the original writer prompt with the unadapted base model.
delta: the fixed original Delta LoRA, used only by the writer. The actor never
loads this adapter for its own calls. Delta/untrained update after completed
attempts, including before a retry, retaining only their own bounded document.
The original Delta prompt describes transfer to a different task; applying its
unchanged update to same-task retries is an explicitly declared retry adaptation.
reflexion: official ALFWorld reflection prompt on failed attempts; reflections are
visible only within that same task, then reset. The actor remains the common actor.
expel: official ExpeL rule extraction/update primitives, success/failure contrasts
and successful-trajectory induction on the existing original Delta TRAIN rollouts;
task-family-filtered all-mpnet-base-v2 retrieval. Rules and retrieval examples freeze
BEFORE any test execution. No test trajectory updates ExpeL's offline bank.

ExpeL may use the 240 already executed development episodes on the 144 original
Delta training tasks, including empty-baseline branches. These are common training
task exposure, not identical writer inputs or equal offline compute. Development
cost and provenance are reported separately. No additional hidden expert action or
test demonstration is introduced. These are controlled local method adaptations,
not reproductions of original-paper actor prompts or headline scores.

Report first-attempt success and success within the common three-attempt budget,
seed/family results, never-previously-evaluated subset, invalid actions and token/
environment costs. First-attempt scores are from this online three-attempt protocol:
previous tasks may already have used retries; they are not a separate one-attempt
experiment. Final paired bootstrap intervals resample task IDs with seeds grouped.
Completed environment timeouts are failures. Infrastructure errors remain errors,
stop the affected run, and are never silently converted into zero success.
No online parameter training or best-of-N score selection is performed.

Infrastructure revision: all TextWorld create/load/reset/step/close calls use a
process-local reentrant lock because its TatSu parsers share mutable global state.
Only CPU environment operations are serialized; model HTTP calls remain concurrent.
Before model startup, six TRAIN tasks are checked against serial reference states
with the production three-chain concurrency. This check cannot read test outcomes.
When recovery_from is declared, the prior failed directory is preserved and its
bank reused byte-for-byte after hash verification; no test condition or budget is
changed, and no completed formal episode from a parser-corrupted run is reused.
'''


def prepare(root, recovery_from=None):
    from .splits import build_manifest
    root = Path(root)
    if (root/'plan.json').exists():
        raise FileExistsError('Refusing to overwrite a frozen experiment')
    root.mkdir(parents=True, exist_ok=True)
    data = WORKSPACE/'ttcl/data/alfworld_delta'
    manifest = build_manifest(data, WORKSPACE/'ttcl/results')
    save(root/'split_manifest.json', manifest)
    tasks = manifest['tasks']
    repeats = [92601, 92602, 92603]
    sequences = []
    for repeat in repeats:
        for family in FAMILIES:
            chosen = [t for t in tasks if t['family'] == family]
            random.Random(seed(926, repeat, family, 'order')).shuffle(chosen)
            sequences.append({'repeat': repeat, 'family': family, 'tasks': chosen})
    plan = {'created_at': time.time(), 'data_root': str(data), 'model': str(MODEL),
            'actor_url': 'http://127.0.0.1:18267', 'port': 18267, 'server_gpu': 3,
            'context': 65536, 'actor_temperature': .7, 'actor_max_tokens': 64,
            'max_steps': 50, 'max_attempts': 3, 'memory_tokens': 2048,
            'writer_max_tokens': 768, 'writer_temperature': 0.0,
            'arms': ARMS, 'eval_seeds': repeats, 'sequences': sequences,
            'test_tasks': len(tasks), 'expected_cells': len(tasks)*len(repeats)*len(ARMS),
            'maximum_logical_actor_episodes': len(tasks)*len(repeats)*len(ARMS)*3,
            'chain_workers': 3, 'development_source': str(OLD),
            'embedding_model': str(WORKSPACE/'ttcl/models/all-mpnet-base-v2'),
            'checkpoint_selection': 'original Delta fixed final', 'actor_frozen': True}
    if recovery_from is not None:
        recovery_from = Path(recovery_from).resolve()
        previous = read(recovery_from/'plan.json')
        changed = [key for key, value in previous.items()
                   if key not in {'created_at', 'recovery_from', 'bank_reuse'} and plan.get(key) != value]
        if changed:
            raise ValueError(f'Recovery changes declared experimental conditions: {changed}')
        if read(recovery_from/'status.json')['phase'] != 'failed':
            raise ValueError('Recovery expects a preserved failed run')
        if list((recovery_from/'evaluation').rglob('episode.json')):
            raise ValueError('This recovery route requires zero completed formal episodes')
        plan['recovery_from'] = str(recovery_from)
        plan['bank_reuse'] = True
    save(root/'plan.json', plan)
    (root/'PROTOCOL.md').write_text(PROTOCOL)
    shutil.copytree(OLD/'training/delta/adapter', root/'adapters/delta')
    target = root/'source/ttcl'
    target.mkdir(parents=True, exist_ok=True)
    (target/'__init__.py').write_text('')
    for name in ['alfworld_comparison', 'experience_evolution', 'experience_v2', 'reflexion_expel']:
        shutil.copytree(WORKSPACE/'ttcl'/name, target/name,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    upstream_files = {
        'ExpeL/agent/expel.py': 'expel/expel.py',
        'ExpeL/prompts/templates/human.py': 'expel/human.py',
        'ExpeL/prompts/alfworld.py': 'expel/alfworld.py',
        'reflexion/alfworld_runs/generate_reflections.py': 'reflexion/generate_reflections.py',
        'reflexion/alfworld_runs/reflexion_few_shot_examples.txt': 'reflexion/reflexion_few_shot_examples.txt'}
    for source, dest in upstream_files.items():
        output = root/'upstream'/dest
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(WORKSPACE/'current_work'/source, output)
    if recovery_from is not None:
        from .recovery import restore_bank
        provenance = restore_bank(recovery_from, root/'expel_bank', root/'upstream')
        provenance.update(reason='TextWorld shared parser concurrency failure before any formal episode completed',
                          original_failure_status_sha256=sha_file(recovery_from/'status.json'),
                          experimental_conditions_unchanged=True,
                          restarted_formal_episodes_from_zero=True)
        save(root/'recovery.json', provenance)
        save(root/'bank_hashes.json', {str(p):sha_file(p) for p in (root/'expel_bank').rglob('*.json')})
    paths = [p for d in ['source', 'adapters', 'upstream'] for p in (root/d).rglob('*') if p.is_file()]
    paths += [root/n for n in ['plan.json', 'split_manifest.json', 'PROTOCOL.md']]
    if recovery_from is not None:
        paths += [root/'recovery.json']
    paths += [data/t['path'] for t in tasks]
    paths += [Path(manifest['results_root'])/item['path'] for item in manifest['historical_manifests']]
    save(root/'input_hashes.json', {str(p): sha_file(p) for p in paths})
    save(root/'status.json', {'phase': 'prepared', 'expected_cells': plan['expected_cells']})
    return plan


def verify(root):
    bad = [p for p, h in read(root/'input_hashes.json').items() if sha_file(p) != h]
    if bad:
        raise ValueError(f'Frozen input changed: {bad}')
    if (root/'bank_hashes.json').exists():
        bad = [p for p, h in read(root/'bank_hashes.json').items() if sha_file(p) != h]
        if bad:
            raise ValueError(f'Frozen ExpeL bank changed: {bad}')


def check_episode(ep, job):
    from .methods import public_episode
    for key in ['game', 'seed', 'memory']:
        if ep[key] != job[key]:
            raise ValueError(f'Resume input mismatch: {key}')
    if ep.get('status') != 'complete' or ep.get('actor_adapter_enabled') is not False:
        raise ValueError('Invalid completed actor record')
    if ep.get('memory_sha256') != digest(job['memory']):
        raise ValueError('Cached memory fingerprint does not match its input')
    public_episode(ep)
    if len(ep['generations']) != ep['steps']:
        raise ValueError('Cached generation count does not match actor steps')
    if any(not isinstance(step.get('valid_command'), bool) for step in ep['trajectory']):
        raise ValueError('Missing cached command validation')


def run_jobs(actor, jobs):
    if not jobs:
        return []
    paths = [Path(job['output']).resolve() for job in jobs]
    if len(set(paths)) != len(paths):
        raise ValueError('Actor jobs must use distinct output directories')
    sources, pending, aliases = {}, [], []
    for job in jobs:
        path = Path(job['output'])/'episode.json'
        key = (job['game'], job['seed'], job['memory'])
        if path.exists():
            check_episode(read(path), job)
            sources.setdefault(key, job)
            continue
        if path.parent.exists():
            path.parent.rename(path.parent.with_name(path.parent.name+f'.interrupted_{time.time_ns()}'))
        if key in sources:
            aliases.append((job, sources[key]))
        else:
            sources[key] = job
            pending.append(job)
    if pending:
        actor.run_many(pending)
    for job, source in aliases:
        source_path = Path(source['output'])/'episode.json'
        ep = dict(read(source_path), reused_from=str(source_path))
        save(Path(job['output'])/'episode.json', ep)
    results = [read(Path(j['output'])/'episode.json') for j in jobs]
    states = defaultdict(set)
    for ep, job in zip(results, jobs):
        check_episode(ep, job)
        states[job['game']].add((ep['initial_observation'], ep['initial_commands_sha256']))
    if any(len(value) > 1 for value in states.values()):
        raise ValueError('Unmatched initial environment states')
    return results


def check_cancelled(stop_event):
    if stop_event is not None and stop_event.is_set():
        raise CancelledError('Another evaluation chain failed or execution was interrupted')


class CancellableActor(Actor):
    def __init__(self, plan, stop_event):
        super().__init__(plan)
        self.stop_event = stop_event

    def generate(self, messages, random_seed):
        check_cancelled(self.stop_event)
        return super().generate(messages, random_seed)


def tokens(client, text):
    return len(client.tokenizer.encode(text, add_special_tokens=False))


def check_memory(client, text, budget):
    if tokens(client, text) > budget:
        raise ValueError(f'Memory exceeds common {budget}-token budget; no truncation')


def chain(root, sequence, bank, retriever, stop_event=None):
    from .methods import delta_update, reflexion_update, pack_reflections
    from .bank import query_context
    check_cancelled(stop_event)
    plan = read(root/'plan.json')
    repeat, family = sequence['repeat'], sequence['family']
    client = Client(plan['actor_url'], context=plan['context'])
    actor = CancellableActor(plan, stop_event)
    memories = {'delta': '', 'untrained': ''}
    try:
        for position, task in enumerate(sequence['tasks']):
            check_cancelled(stop_event)
            game = task['path']
            dest = root/'evaluation'/family/str(repeat)/f'task_{position:03}'
            save(root/'workers'/f'{family}_{repeat}.json',
                 {'phase': 'running', 'stage': 'initializing_environment',
                  'completed_tasks': position, 'active_game': game,
                  'expected_tasks': len(sequence['tasks']), 'updated_at': time.time()})
            env = make_env(Path(plan['data_root'])/game)
            try:
                query = str(env.reset()['feedback'])
            finally:
                env.close()
            expel_context, retrieval = query_context(bank, query, family, retriever,
                                                      client.tokenizer, budget=plan['memory_tokens'])
            save(dest/'retrieval.json', retrieval)
            context = {'retry_none': '', 'reflexion': '', 'expel': expel_context, **memories}
            reflections, episodes, updates = [], {a: [] for a in ARMS}, {a: [] for a in ARMS}
            for attempt in range(plan['max_attempts']):
                check_cancelled(stop_event)
                active = [a for a in ARMS if not episodes[a] or not episodes[a][-1]['reward']]
                if not active:
                    break
                save(root/'workers'/f'{family}_{repeat}.json',
                     {'phase': 'running', 'stage': 'actor_attempt', 'attempt': attempt+1,
                      'active_methods': active, 'completed_tasks': position, 'active_game': game,
                      'expected_tasks': len(sequence['tasks']), 'updated_at': time.time()})
                jobs = []
                for arm in active:
                    check_memory(client, context[arm], plan['memory_tokens'])
                    jobs.append({'game': game, 'seed': seed(repeat, game, attempt, 'actor'),
                                 'memory': context[arm], 'output': str(dest/arm/f'attempt_{attempt:02}')})
                results = run_jobs(actor, jobs)
                for arm, ep in zip(active, results):
                    check_cancelled(stop_event)
                    if ep['initial_observation'] != query:
                        raise ValueError('Actor reset differs from the retrieval task query')
                    episodes[arm].append(ep)
                    if arm in memories:
                        update = delta_update(client, memories[arm], ep, dest/arm/f'update_{attempt:02}.json',
                                              seed(repeat, game, attempt, 'writer'),
                                              model='delta' if arm == 'delta' else 'frozen-actor')
                        memories[arm] = context[arm] = update['raw_response']
                        check_memory(client, memories[arm], plan['memory_tokens'])
                        updates[arm].append(update)
                    elif arm == 'reflexion' and not ep['reward'] and attempt+1 < plan['max_attempts']:
                        update = reflexion_update(client, ep, reflections, dest/arm/f'reflection_{attempt:02}.json',
                                                  seed(repeat, game, attempt, 'writer'), root/'upstream')
                        reflections.append(update['raw_response'])
                        context[arm], audit = pack_reflections(reflections, client.tokenizer, plan['memory_tokens'])
                        save(dest/arm/f'packing_{attempt:02}.json', audit)
                        updates[arm].append(update)
            for arm in ARMS:
                eps = episodes[arm]
                calls = [g for ep in eps for g in ep['generations']]
                physical_calls = [g for ep in eps if 'reused_from' not in ep for g in ep['generations']]
                row = {'status': 'complete', 'arm': arm, 'game': game, 'family': family,
                       'repeat': repeat, 'position': position, 'never_evaluated': task['never_evaluated'],
                       'first_success': int(eps[0]['reward']), 'success': int(eps[-1]['reward']),
                       'attempts': len(eps), 'attempt_successes': [int(ep['reward']) for ep in eps],
                       'actor_calls': sum(ep['steps'] for ep in eps),
                       'physical_actor_calls': sum(ep['steps'] for ep in eps if 'reused_from' not in ep),
                       'actor_input_tokens': sum(g['usage'].get('prompt_tokens', 0) for g in calls),
                       'actor_output_tokens': sum(g['usage'].get('completion_tokens', 0) for g in calls),
                       'physical_actor_input_tokens': sum(g['usage'].get('prompt_tokens', 0) for g in physical_calls),
                       'physical_actor_output_tokens': sum(g['usage'].get('completion_tokens', 0) for g in physical_calls),
                       'actor_seconds': sum(g['seconds'] for g in calls),
                       'physical_actor_seconds': sum(g['seconds'] for g in physical_calls),
                       'writer_calls': len(updates[arm]),
                       'writer_seconds': sum(g['seconds'] for g in updates[arm]),
                       'writer_input_tokens': sum(g['input_tokens'] for g in updates[arm]),
                       'writer_output_tokens': sum(g['output_tokens'] for g in updates[arm]),
                       'invalid_commands': sum(not t['valid_command'] for ep in eps for t in ep['trajectory'])}
                save(dest/arm/'result.json', row)
            save(root/'workers'/f'{family}_{repeat}.json', {'phase': 'running', 'completed_tasks': position+1,
                                                         'expected_tasks': len(sequence['tasks']), 'updated_at': time.time()})
            with REPORT_LOCK:
                report(root)
        save(root/'workers'/f'{family}_{repeat}.json', {'phase': 'complete', 'completed_tasks': len(sequence['tasks'])})
    except BaseException as exc:
        save(root/'workers'/f'{family}_{repeat}.json', {
            'phase': 'cancelled' if isinstance(exc, CancelledError) else 'failed',
            'updated_at': time.time(), 'traceback': traceback.format_exc()})
        raise
    finally:
        actor.pool.shutdown(wait=True, cancel_futures=True)
        client.session.close()


def evaluate(root):
    plan = read(root/'plan.json')
    report(root)
    bank = read(root/'expel_bank/state.json')
    retriever = Retriever(plan['embedding_model'])
    # Cache every possible embedding before sharing the retriever between workers.
    # There is then no concurrent mutation of its tokenizer/model/cache state.
    for document in bank['successful_examples']:
        retriever.encode(document['query'])
    tasks = {task['path'] for sequence in plan['sequences'] for task in sequence['tasks']}
    for game in sorted(tasks):
        env = make_env(Path(plan['data_root'])/game)
        try:
            retriever.encode(str(env.reset()['feedback']))
        finally:
            env.close()
    stop_event = threading.Event()
    pool = ThreadPoolExecutor(max_workers=plan['chain_workers'])
    futures = []
    try:
        futures = [pool.submit(chain, root, seq, bank, retriever, stop_event)
                   for seq in plan['sequences']]
        for future in as_completed(futures):
            future.result()
    except BaseException:
        stop_event.set()
        for future in futures:
            future.cancel()
        raise
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
    summary = report(root, final=True)
    if summary['completed_cells'] != plan['expected_cells']:
        raise RuntimeError('Evaluation finished without the complete planned grid')



def preflight(root, bank):
    """Exercise real actor and writer integration on one audited training task."""
    root = Path(root)
    plan = read(root/'plan.json')
    old_plan = read(Path(plan['development_source'])/'plan.json')
    training_games = {game for group in old_plan['training'] for game in group['games']}
    # Two-object tasks cannot finish within three commands, ensuring that the
    # preflight also exercises failed-attempt reflection and retry generation.
    examples = bank['successful_examples']
    preferred = [doc for doc in examples if doc['family'] == 'pick_two_obj_and_place']
    if preferred:
        chosen = preferred[0]
        game, family = chosen['game'], chosen['family']
    else:
        group = next((item for item in old_plan['training']
                      if item['family'] == 'pick_two_obj_and_place'), old_plan['training'][0])
        game, family = group['games'][0], group['family']
    if game not in training_games or '/train/' not in game:
        raise ValueError('Preflight must use a declared training task')
    if any(task['path'] == game for seq in plan['sequences'] for task in seq['tasks']):
        raise ValueError('Preflight training task overlaps the evaluation manifest')
    smoke = root/'preflight'
    smoke.mkdir(parents=True, exist_ok=True)
    sequence = {'repeat': 92699, 'family': family,
                'tasks': [{'path': game, 'family': family, 'never_evaluated': False}]}
    smoke_plan = dict(plan, max_steps=3, max_attempts=2, sequences=[sequence],
                      eval_seeds=[92699], test_tasks=1, expected_cells=len(ARMS),
                      maximum_logical_actor_episodes=2*len(ARMS),
                      purpose='training-only integration preflight; excluded from test results')
    smoke_plan_path = smoke/'plan.json'
    if smoke_plan_path.exists() and read(smoke_plan_path) != smoke_plan:
        raise ValueError('Preflight plan changed on resume')
    save(smoke_plan_path, smoke_plan)
    upstream = smoke/'upstream'
    if upstream.exists() or upstream.is_symlink():
        if upstream.resolve() != (root/'upstream').resolve():
            raise ValueError('Preflight upstream prompt source changed')
    else:
        upstream.symlink_to((root/'upstream').resolve(), target_is_directory=True)
    retriever = Retriever(plan['embedding_model'])
    chain(smoke, sequence, bank, retriever)
    summary = report(smoke, final=True)
    destination = smoke/'evaluation'/family/'92699'/'task_000'
    delta = read(destination/'delta/update_00.json')
    reflection_path = destination/'reflexion/reflection_00.json'
    reflection = read(reflection_path) if reflection_path.exists() else None
    if delta['served_model'] != 'delta' or (reflection is not None and reflection['served_model'] != 'frozen-actor'):
        raise ValueError('Preflight writer adapter routing failed')
    if summary['completed_cells'] != len(ARMS):
        raise ValueError('Preflight did not complete all five method cells')
    result = {'passed': True, 'training_game': game, 'sampling_seed': 92699,
              'steps_per_attempt': 3, 'maximum_attempts': 2, 'completed_cells': len(ARMS),
              'delta_writer_adapter_checked': True, 'failed_reflection_checked': reflection is not None,
              'reflection_note': 'Executed on failed train attempt' if reflection is not None else
                  'Training preflight succeeded before a retry; no reflection was needed',
              'actor_frozen_checked': True, 'test_data_used': False,
              'bank_id': bank.get('bank_id'), 'finished_at': time.time()}
    save(root/'preflight.json', result)
    return result


def supervise(root):
    from .bank import build_bank
    root = Path(root)
    lock = (root/'supervisor.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    verify(root)
    plan, server = read(root/'plan.json'), None
    try:
        save(root/'status.json', {'phase': 'environment_preflight', 'supervisor_pid': os.getpid(), 'updated_at': time.time()})
        old_plan = read(Path(plan['development_source'])/'plan.json')
        smoke_games = [next(seq['games'][0] for seq in old_plan['training'] if seq['family'] == family)
                       for family in FAMILIES]
        environment_check = concurrent_preflight(plan['data_root'], smoke_games,
                                                workers=plan['chain_workers'], repeats=2, steps=2)
        save(root/'environment_preflight.json', environment_check)
        save(root/'status.json', {'phase': 'starting_server', 'supervisor_pid': os.getpid(), 'updated_at': time.time()})
        server = start_server(root, plan['server_gpu'], plan['port'], {'delta': root/'adapters/delta'}, context=plan['context'])
        save(root/'status.json', {'phase': 'building_expel_bank', 'supervisor_pid': os.getpid(), 'updated_at': time.time()})
        client = Client(plan['actor_url'], context=plan['context'])
        try:
            if plan.get('bank_reuse'):
                from .recovery import verify_reused_bank
                bank = verify_reused_bank(root/'expel_bank')
            else:
                bank = build_bank(client, Path(plan['development_source']), root/'expel_bank', root/'upstream')
        finally:
            client.session.close()
        if not (root/'bank_hashes.json').exists():
            paths = [p for p in (root/'expel_bank').rglob('*.json') if p.is_file()]
            save(root/'bank_hashes.json', {str(p): sha_file(p) for p in paths})
        verify(root)
        save(root/'status.json', {'phase': 'preflight', 'supervisor_pid': os.getpid(), 'updated_at': time.time()})
        preflight(root, bank)
        verify(root)
        save(root/'status.json', {'phase': 'evaluation', 'supervisor_pid': os.getpid(), 'updated_at': time.time()})
        evaluate(root)
        verify(root)
        save(root/'status.json', {'phase': 'complete', 'finished_at': time.time(), 'expected_cells': plan['expected_cells']})
    except BaseException:
        save(root/'status.json', {'phase': 'failed', 'updated_at': time.time(), 'traceback': traceback.format_exc()})
        raise
    finally:
        stop_server(server)


def launch(root):
    verify(root)
    command = [str(PYTHON), '-m', 'ttcl.alfworld_comparison.run', 'supervise', '--root', str(root)]
    with (root/'supervisor.log').open('a') as log:
        process = subprocess.Popen(command, cwd=root/'source', env=environment(root),
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    save(root/'supervisor_pid.json', {'pid': process.pid, 'command': command})
    print(json.dumps({'pid': process.pid, 'root': str(root)}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['prepare', 'launch', 'supervise', 'report', 'verify'])
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--recover-from', type=Path)
    args = parser.parse_args()
    if args.command == 'prepare':
        print(json.dumps(prepare(args.root, args.recover_from), default=str)[:800])
    elif args.command == 'launch':
        launch(args.root)
    elif args.command == 'supervise':
        def stop(signum, frame):
            raise KeyboardInterrupt(f'Signal {signum}')
        signal.signal(signal.SIGTERM, stop)
        supervise(args.root)
    elif args.command == 'verify':
        verify(args.root)
    else:
        print(json.dumps(report(args.root), indent=2))


if __name__ == '__main__':
    main()
