from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import statistics
import time
import traceback
from types import SimpleNamespace

from .common import BENCH, Client, WRITER_SYSTEM, read, save, seed


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
    from .common import public_messages
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
            row.update(task=task, arm=arm, repeat=repeat, actor_adapter_enabled=False)
            save(dest/'memory_before.json', {'text':context})
            save(dest/'row.json', row)
            cache[context] = (row, episode, dest)
            print(json.dumps({k:row.get(k) for k in ['task','repeat','episode','arm','reward','status']}), flush=True)
            if arm == 'none' or index == count-1:
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


def session_messages(memory, session):
    return [{'role':'system', 'content':WRITER_SYSTEM}, {'role':'user', 'content':json.dumps({
        'previous_experience':memory,
        'completed_interaction':{'observation':session,
                                'outcome':'A conversation session has finished; no task reward is available.'}})}]


def locomo(root, suite, sample_index):
    from .locomo_protocol import (build_locomo_history_messages, OFFICIAL_SYSTEM_PROMPT,
        prepare_locomo_question, build_official_question_prompt, canonicalize_locomo_prediction,
        score_locomo_prediction)
    plan=read(root/'plan.json'); settings=plan['suites'][suite]
    sample=read(root/'locomo10.json')[sample_index]
    dest=root/suite/'locomo'/str(sample_index)
    progress=root/suite/'workers'/f'locomo_{sample_index}.json'
    model=Client(settings['url'])
    sessions=build_locomo_history_messages(sample)[1:]
    arms=dict(settings['arms'])
    if not settings.get('inherit_suite'):
        arms['full_history']='frozen-actor'
    memories={}
    for arm, name in arms.items():
        if arm in {'none','full_history'}:
            continue
        memory=''
        for j,session in enumerate(sessions):
            save(progress,dict(phase='memory', sample=sample_index, arm=arm, session=j,
                               total_sessions=len(sessions), updated_at=time.time()))
            path=dest/arm/f'session_{j:03}.json'
            messages=session_messages(memory,session['content'])
            if path.exists():
                result=read(path)
                assert result['messages']==messages
            else:
                result=model.complete(messages, model=name,
                    random_seed=seed(923,sample['sample_id'],j,'writer'), tokens=768)
                if not result['raw_response']:
                    raise ValueError('Empty writer output')
                result['messages']=messages
                save(path,result)
            memory=result['raw_response']
        memories[arm]=memory
    full_context='\n\n'.join(s['content'] for s in sessions)
    def answer(item):
        i,question,arm=item
        path=dest/arm/'qa'/f'{i:04}.json'
        if path.exists():
            return
        spec=prepare_locomo_question(question,sample_id=sample['sample_id'],question_index=i,seed=923)
        context=full_context if arm=='full_history' else memories.get(arm,'')
        messages=[{'role':'system','content':OFFICIAL_SYSTEM_PROMPT},
                  {'role':'user','content':context+'\n\n'+build_official_question_prompt(spec)}]
        result=model.complete(messages,random_seed=seed(923,sample['sample_id'],i,'answer'),
                              tokens=50,temperature=0)
        prediction=canonicalize_locomo_prediction(result['raw_response'],spec)
        result.update(sample_id=sample['sample_id'],question_index=i,category=question['category'],
            arm=arm, prediction=prediction,score=score_locomo_prediction(question,prediction),
            question=question['question'],answer=question.get('answer'),
            context_source='full_dialogue' if arm=='full_history' else 'rolling_memory_only',
            actor_adapter_enabled=False)
        save(path,result)
    chosen=read(root/'locomo_selection.json')[str(sample_index)]
    work=[(i,sample['qa'][i],arm) for i in chosen for arm in arms]
    save(progress,dict(phase='qa',sample=sample_index,total=len(work)))
    with ThreadPoolExecutor(max_workers=4) as pool:
        for n,_ in enumerate(pool.map(answer,work),1):
            if n%50==0:
                save(progress,dict(phase='qa',sample=sample_index,completed=n,total=len(work)))
                print(f'LoCoMo {sample_index}: {n}/{len(work)}',flush=True)
    save(progress,dict(phase='complete',sample=sample_index,completed=len(work)))


def report(root,suite):
    settings=read(root/'plan.json')['suites'][suite]
    inherited=settings.get('inherit_suite')
    base_settings=read(root/'plan.json')['suites'][inherited] if inherited else settings
    report_arms={**base_settings['arms'],**settings['arms']}
    sources=[root/suite]+([root/inherited] if inherited else [])
    result={'clbench':{},'locomo':{}}
    lines=[f'# {suite}', '', 'Frozen actors; all means are descriptive. Tasks retain separate metrics.',
           'CLBench writer sees only public observations, not hidden evaluator scores.',
           'LoCoMo: fixed 300 questions across ten dialogues; question-blind session memory; greedy QA; official category scoring.',
           'LoCoMo full_history is a separate larger-context reference. This is a textual-memory adaptation, not the official HF decoding recipe.', '']
    for task,count in settings['tasks'].items():
        rows=[read(p) for source in sources for p in (source/'clbench'/task).glob('*/*/episode_*/row.json')
              if '.interrupted_' not in str(p)]
        expected=count*len(settings['repeats'])*len(report_arms)
        groups={a:{(r['repeat'],r['canonical_index']):r for r in rows
                   if r['arm']==a and r['status']=='complete'} for a in report_arms}
        common=set.intersection(*(set(v) for v in groups.values()))
        task_result={'recorded':len(rows),'expected':expected,'paired':len(common),'arms':{}}
        lines += [f'## {task}: {len(rows)}/{expected}', '', '| Arm | Mean reward (common pairs) | Wins/losses/ties vs untrained |', '|---|---:|---:|']
        for arm,g in groups.items():
            values=[g[k]['reward'] for k in common]
            diffs=[g[k]['reward']-groups['untrained'][k]['reward'] for k in common]
            v={'mean':statistics.mean(values) if values else None,
               'wins':sum(d>1e-12 for d in diffs),'losses':sum(d < -1e-12 for d in diffs),
               'ties':sum(abs(d)<=1e-12 for d in diffs),
               'by_repeat':{str(s):statistics.mean([r['reward'] for (rs,_),r in g.items() if rs==s])
                            for s in settings['repeats'] if any(rs==s for rs,_ in g)},
               'failed':sum(r['arm']==arm and r['status']!='complete' for r in rows)}
            task_result['arms'][arm]=v
            lines.append(f"| {arm} | {v['mean']} | {v['wins']}/{v['losses']}/{v['ties']} |")
        result['clbench'][task]=task_result
    qa=[read(p) for source in sources for p in (source/'locomo').glob('*/*/qa/*.json')]
    lines += ['', '## LoCoMo', '', '| Arm | Questions | Categories 1–4 mean F1 | Category 5 accuracy |', '|---|---:|---:|---:|']
    for arm in [*report_arms,'full_history']:
        own=[r for r in qa if r['arm']==arm]
        cats={str(c):[r['score'] for r in own if r['category']==c] for c in range(1,6)}
        expected_qa=read(root/'plan.json')['locomo_questions']
        v={'recorded':len(own),'expected':expected_qa,
           'category':{c:{'n':len(xs),'mean':statistics.mean(xs) if xs else None} for c,xs in cats.items()}}
        main=[r['score'] for r in own if r['category']!=5]
        v['categories_1_4_f1']=statistics.mean(main) if main else None
        result['locomo'][arm]=v
        lines.append(f"| {arm} | {len(own)}/{expected_qa} | {v['categories_1_4_f1']} | {v['category']['5']['mean']} |")
    save(root/suite/'comparison.json',result)
    (root/suite/'REPORT.md').write_text('\n'.join(lines)+'\n')
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('command',choices=['clbench','locomo','report'])
    p.add_argument('--root',type=Path,required=True);p.add_argument('--suite',required=True)
    p.add_argument('--task');p.add_argument('--repeat',type=int);p.add_argument('--sample',type=int)
    a=p.parse_args()
    try:
        if a.command=='clbench':clbench(a.root,a.suite,a.task,a.repeat)
        elif a.command=='locomo':locomo(a.root,a.suite,a.sample)
        else:report(a.root,a.suite)
    except Exception:
        save(a.root/a.suite/'failures'/f'{a.command}_{a.task}_{a.repeat}_{a.sample}.json',
             {'traceback':traceback.format_exc()})
        raise


if __name__=='__main__':main()
