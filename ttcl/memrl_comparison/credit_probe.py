"""Paired, frozen-memory leave-one-out diagnostic for a completed MemRL run.

This is an exploratory probe of the *retrieved text*, not a replay of online
learning. Every arm uses the same task, memory snapshot, actor seed and budget.
The original run directory is read-only; no arm writes or updates memory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from ttcl.icl_mem0_comparison.protocol import Client, read, save, sha
from ttcl.experience_evolution.core import seed


def memory_arms(origin: Path, case: str):
    episode = origin / 'runs' / case
    benchmark, task, repeat, arm, episode_name = Path(case).parts
    if arm != 'memrl':
        raise ValueError('Case must point to a MemRL episode')
    index = int(episode_name.split('_')[1]) - 1
    retrieval = read(episode / ('retrieval_1.json' if benchmark == 'alfworld' else 'retrieval.json'))
    ids = retrieval['ids']
    if len(ids) < 2:
        raise ValueError(f'{case}: fewer than two injected memories')
    previous = episode.parent / f'episode_{index:03d}' / 'memory_after.json'
    snapshot = read(previous)
    entries = {mid: 'Task: ' + snapshot['items'][mid]['metadata']['task_description'] +
               '\nExperience: ' + snapshot['items'][mid]['metadata']['public_abstract'] for mid in ids}
    def context(kept):
        return '\n\n'.join(entries[mid] for mid in kept)
    full = context(ids)
    if full != retrieval['context'] or hashlib.sha256(full.encode()).hexdigest() != retrieval['context_sha256']:
        raise ValueError(f'{case}: reconstruction does not match frozen retrieval')
    arms = {'full': full, 'none': ''}
    arms.update({f'only_{i}': context([mid]) for i, mid in enumerate(ids)})
    arms.update({f'drop_{i}': context([m for m in ids if m != mid]) for i, mid in enumerate(ids)})
    base = episode.parent.parent / 'none' / episode_name / 'row.json'
    return dict(benchmark=benchmark, task=task, repeat=int(repeat), index=index,
                ids=ids, case=case, original_memrl=read(episode / 'row.json'),
                original_none=read(base), original_update=read(episode / ('update_1.json' if benchmark == 'alfworld' else 'update.json')),
                snapshot_sha256=sha(previous), retrieval_sha256=sha(episode / ('retrieval_1.json' if benchmark == 'alfworld' else 'retrieval.json'))), arms


def run_alf(plan, client, spec, contexts, output):
    from ttcl.alfworld_comparison.environment import Actor
    from ttcl.memrl_comparison.memory import digest

    class LocalActor(Actor):
        def generate(self, messages, random_seed):
            result = client.complete(messages, random_seed, tokens=plan['alf']['actor_max_tokens'],
                                     temperature=plan['alf']['actor_temperature'], top_p=1.)
            return dict(text=result['raw_response'], finish_reason=result['finish_reason'],
                        usage={'prompt_tokens':result['input_tokens'], 'completion_tokens':result['output_tokens']},
                        seed=random_seed, prompt_sha256=digest(messages), seconds=result['seconds'],
                        rendered_prompt_sha256=result['rendered_prompt_sha256'])

    game = spec['original_memrl']['game']
    if sha(Path(plan['alf']['data_root']) / game) != spec['original_memrl']['input_sha256']:
        raise ValueError('Frozen game content changed')
    actor = LocalActor(plan['alf'])
    try:
        for name, context in contexts.items():
            target = output / name
            if (target / 'episode.json').exists():
                continue
            target.mkdir(parents=True, exist_ok=True)
            actor.run_many([dict(game=game, memory=context,
                seed=seed(spec['repeat'], game, 0, 'actor'), output=target)])
    finally:
        actor.pool.shutdown(wait=True)
    return {name:dict(reward=read(output / name / 'episode.json')['reward'],
                      steps=read(output / name / 'episode.json')['steps'],
                      first_action=read(output / name / 'episode.json')['trajectory'][0]['action'])
            for name in contexts}


def run_cl(plan, client, spec, contexts, output):
    from ttcl.icl_mem0_comparison.worker import System, make_task, base
    from ttcl.icl_mem0_comparison.protocol import normalize_prompt

    results = {}
    os.chdir(os.environ['TTCL_BENCH'])
    for name, context in contexts.items():
        target = output / name
        if (target / 'result.json').exists():
            results[name] = read(target / 'result.json')
            continue
        target.mkdir(parents=True, exist_ok=True)
        task = make_task(spec['task'], plan['task_seed'])
        try:
            query = task.reset_baseline_instance(spec['index'])
            brief = task.get_agent_brief()
            brief = base.format_task_agent_brief(brief) if brief else ''
            system = System(plan, client, 'none', [], None, target, brief, spec['index'], plan['tasks'][spec['task']])
            # Keep the original system identity; output directories distinguish arms.
            system.arm = 'memrl'; system.mode = 'memrl'
            prompt = normalize_prompt(query.prompt, brief, spec['index'], plan['tasks'][spec['task']])
            if hashlib.sha256(query.prompt.encode()).hexdigest() != spec['original_memrl']['initial_query_sha256']:
                raise ValueError('CLBench input changed')
            if context:
                system.messages[0]['content'] += '\n\nPast experience:\n' + context
            recorder = base.Recorder(target, system, 1)
            result = base.run_task(task, system, trace_recorder=recorder,
                                   show_progress=False, reset_system=False, initial_query=query)
            if len(result.instance_outcomes) != 1:
                raise ValueError('Expected one outcome')
            outcome = result.instance_outcomes[0]
            row = dict(reward=float(outcome.reward), success=outcome.success,
                       actor_calls=system.calls, query_sha256=hashlib.sha256(prompt.encode()).hexdigest())
            save(target / 'result.json', row)
            results[name] = row
        finally:
            conn = getattr(task, '_conn', None)
            if conn is not None:conn.close()
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--origin', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--url', required=True)
    parser.add_argument('--validate-only', action='store_true')
    parser.add_argument('--paired-drop-index', type=int,
                        help='Replay only full context and one leave-one-out arm')
    parser.add_argument('--cl-sampling-repeats', type=int, nargs='+',
                        help='Probe fixed CL memory with several actor sampling seeds; run four unique arms per seed')
    parser.add_argument('--alf-actor-repeats', type=int, nargs='+',
                        help='Probe fixed ALFWorld memory with several paired actor seeds')
    parser.add_argument('cases', nargs='+', help='Paths relative to origin/runs')
    args = parser.parse_args()
    origin = args.origin.resolve(); output = args.output.resolve()
    plan = read(origin / 'plan.json')
    plan['url'] = args.url; plan['alf']['actor_url'] = args.url
    output.mkdir(parents=True, exist_ok=True)
    manifest = dict(origin=str(origin), origin_plan_sha256=sha(origin / 'plan.json'),
                    cases=args.cases, url=args.url,
                    note='Exploratory fixed-memory text ablation; no Q-value update or online policy replay')
    if args.cl_sampling_repeats:
        manifest['cl_sampling_repeats'] = args.cl_sampling_repeats
    if args.alf_actor_repeats:
        if args.cl_sampling_repeats:
            raise ValueError('Choose one actor-repeat mode')
        manifest['alf_actor_repeats'] = args.alf_actor_repeats
    if args.paired_drop_index is not None:
        if args.cl_sampling_repeats:
            raise ValueError('Use either --paired-drop-index or --cl-sampling-repeats')
        manifest['paired_drop_index'] = args.paired_drop_index
    if (output / 'design.json').exists() and read(output / 'design.json') != manifest:
        raise ValueError('Existing diagnostic design differs')
    save(output / 'design.json', manifest)
    for case in args.cases:
        spec, arms = memory_arms(origin, case)
        if args.paired_drop_index is not None:
            index = args.paired_drop_index
            if index < 0 or index >= len(spec['ids']):
                raise ValueError(f'Invalid drop index {index} for {case}')
            arms = {'full': arms['full'], f'drop_{index}': arms[f'drop_{index}']}
        folder = output / spec['benchmark'] / spec['task'] / str(spec['repeat']) / f"episode_{spec['index']+1:03d}"
        folder.mkdir(parents=True, exist_ok=True)
        save(folder / 'source.json', {k:v for k,v in spec.items() if k not in {'original_memrl','original_none','original_update'}})
        if args.validate_only:
            print('validated', case, 'ids', spec['ids'], flush=True)
            continue
        if args.alf_actor_repeats:
            if spec['benchmark'] != 'alfworld':
                raise ValueError('ALF actor-repeat probe requires an ALFWorld case')
            for actor_repeat in args.alf_actor_repeats:
                client = Client(plan, actor_repeat)
                replay_spec = dict(spec, repeat=actor_repeat)
                target = folder / f'actor_repeat_{actor_repeat}'
                result = run_alf(plan, client, replay_spec, arms, target)
                save(target / 'summary.json', dict(ids=spec['ids'],
                     actor_repeat=actor_repeat, replay=result))
                print(json.dumps(dict(case=case, actor_repeat=actor_repeat,
                                      replay=result), ensure_ascii=False), flush=True)
            continue
        if args.cl_sampling_repeats:
            if spec['benchmark'] != 'clbench' or len(spec['ids']) != 2:
                raise ValueError('Sampling-repeat probe requires a two-memory CLBench case')
            unique = {name:arms[name] for name in ('full','none','drop_0','drop_1')}
            for actor_repeat in args.cl_sampling_repeats:
                client = Client(plan, actor_repeat)
                result = run_cl(plan, client, spec, unique, folder / f'actor_repeat_{actor_repeat}')
                save(folder / f'actor_repeat_{actor_repeat}' / 'summary.json',
                     dict(ids=spec['ids'], actor_repeat=actor_repeat, replay=result))
                print(json.dumps(dict(case=case, actor_repeat=actor_repeat, replay=result),
                                 ensure_ascii=False), flush=True)
            continue
        client = Client(plan, spec['repeat'])
        result = (run_alf(plan, client, spec, arms, folder) if spec['benchmark'] == 'alfworld' else
                  run_cl(plan, client, spec, arms, folder))
        save(folder / 'summary.json', dict(ids=spec['ids'], original_full=spec['original_memrl']['first_attempt'] if spec['benchmark']=='alfworld' else spec['original_memrl']['reward'],
             original_none=spec['original_none']['first_attempt'] if spec['benchmark']=='alfworld' else spec['original_none']['reward'],
             replay=result, original_q_updates=spec['original_update']['q_updates']))
        print(json.dumps(dict(case=case, replay=result), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
