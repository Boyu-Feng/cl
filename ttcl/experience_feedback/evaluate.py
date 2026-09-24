from __future__ import annotations
import os
import json
import time
from types import SimpleNamespace
from ttcl.experience_v2.common import BENCH, Client, read, save, seed

def configure_tasks():
    from ttcl.structured_memory import run_benchmark as base
    original = base.make_task

    def make_task(name, random_seed, independent=False):
        if name == 'blind_spectrum_monitoring':
            from src.tasks.blind_spectrum_monitoring.task import BlindSpectrumMonitoringTask
            return BlindSpectrumMonitoringTask(seed=random_seed, schedule='default',
                                               response_timeout_seconds=0)
        return original(name, random_seed, independent)

    base.make_task = make_task
    # Independent mode never reads/updates this object; it is required by reset().
    base.MEMORIES['blind_spectrum_monitoring'] = base.DatabaseMemory


def clbench(root, suite, task, repeat):
    from ttcl.structured_memory.online_bank import run_episode
    from .protocol import public_messages
    configure_tasks()
    os.chdir(BENCH)
    plan = read(root / 'plan.json')
    settings = plan['suites'][suite]
    arms = settings['arms']
    count = settings['tasks'][task]
    model = Client(settings['url'], repeat=repeat)
    args = SimpleNamespace(task=task, seed=42, num_instances=count, memory_chars=20000,
        max_turns_per_instance=64, action_retries=2, normalize_action=True,
        allow_initial_experience=True)
    progress = root / suite / 'workers' / f'{task}_{repeat}.json'
    memories = {arm: '' for arm in arms}
    for index in range(count):
        cache = {}
        for arm in arms:
            dest = root / suite / 'clbench' / task / str(repeat) / arm / f'episode_{index+1:03}'
            context = memories[arm]
            save(progress, dict(phase='actor', task=task, repeat=repeat, index=index, arm=arm,
                                updated_at=time.time()))
            if (dest / 'row.json').exists():
                row, episode = read(dest/'row.json'), read(dest/'trajectory.json')
                assert read(dest/'memory_before.json')['text'] == context
            elif context in cache:
                old_row, episode, source = cache[context]
                row = dict(old_row, reused_from=str(source))
                save(dest/'trajectory.json', episode)
            else:
                if dest.exists():
                    dest.rename(dest.with_name(dest.name + f'.interrupted_{time.time_ns()}'))
                row, episode = run_episode(args, model, index, dest, context)
                if row['status'] != 'complete' and not any(s in row.get('error','') for s in
                        ['no schema-valid JSON action', 'Safety cap exceeded']):
                    raise RuntimeError(row.get('error', 'Unknown task failure'))
            row.update(task=task, arm=arm, repeat=repeat, actor_adapter_enabled=False,
                       feedback_protocol='completed_episode_reward_v1')
            save(dest/'memory_before.json', {'text':context})
            save(dest/'row.json', row)
            cache[context] = (row, episode, dest)
            print(json.dumps({k:row.get(k) for k in ['task','repeat','episode','arm','reward','status']}), flush=True)
            if arm == 'none':
                continue
            messages = public_messages(context, episode)
            if (dest/'writer.json').exists():
                update = read(dest/'writer.json')
                assert update['messages'] == messages
            else:
                update = model.complete(messages, model=arms[arm],
                    random_seed=seed(923,task,repeat,index,'writer'), tokens=768)
                if not update['raw_response']:
                    raise ValueError('Empty writer output')
                update['messages'] = messages
                save(dest/'writer.json', update)
            memories[arm] = update['raw_response']
    save(progress, dict(phase='complete', task=task, repeat=repeat, cells=count*len(arms)))

