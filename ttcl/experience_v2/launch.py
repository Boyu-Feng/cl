from __future__ import annotations

import argparse
from collections import deque
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import time
import traceback

from .common import (BENCH, MODEL, OLD, PYTHON, WORKSPACE, ROOT, environment, freeze,
                     read, save, sha_file, start_server, stop_server, select_locomo_questions)
from ttcl.experience_evolution.core import FAMILIES


def prepare(root):
    root.mkdir(parents=True,exist_ok=False)
    for arm in ['delta','absolute']:
        shutil.copytree(OLD/f'training/{arm}/adapter',root/f'adapters/{arm}')
    shutil.copy2(WORKSPACE/'current_work/delta-Mem/data/locomo10.json',root/'locomo10.json')
    locomo_selection=select_locomo_questions(read(root/'locomo10.json'))
    save(root/'locomo_selection.json',locomo_selection)
    tasks={'blind_spectrum_monitoring':90,'exploitable_poker':120,
           'database_exploration':20,'cohort_studies':20}
    train_arms={'continued_k1':{'seed':923,'paired_samples':1},
                'improved_k2_s923':{'seed':923,'paired_samples':2},
                'improved_k2_s924':{'seed':924,'paired_samples':2}}
    plan={'created_at':time.time(),'original':str(OLD),'suites':{
        'frozen_transfer':{'tasks':tasks,'repeats':[303,404], 'url':'http://127.0.0.1:18217',
            'arms':{'none':'frozen-actor','untrained':'frozen-actor','delta':'delta','absolute':'absolute'}},
        'improved_transfer':{'tasks':tasks,'repeats':[303,404], 'url':'http://127.0.0.1:18217',
            'arms':{a:a for a in train_arms},'inherit_suite':'frozen_transfer'}},
        'docker_blocked':{'sales_prediction':'Docker socket permission denied; sudo requires password',
                          'codebase_adaptation':'Docker socket permission denied; sudo requires password'},
        'locomo_source':'https://github.com/snap-research/locomo/blob/main/data/locomo10.json',
        'locomo_sha256':sha_file(root/'locomo10.json'),'writer_max_tokens':768,
        'locomo_questions':sum(map(len,locomo_selection.values())),
        'writer_prompt':'unchanged original WRITER_SYSTEM, 200-word instruction',
        'score_visibility':'CLBench public observations only; hidden scalar excluded, including cohort and BSM',
        'candidate_selection':'All predeclared final checkpoints reported. No test-selected winner.'}
    save(root/'plan.json',plan)
    prior=read(OLD/'plan.json');data=Path(prior['data_root']);rng=random.Random(923)
    used={g for seq in prior['training']+prior['evaluation'] for g in seq['games']}
    used.update(x['game'] for x in prior['calibration'])
    screening=[];evaluation=[]
    for family in FAMILIES:
        for split in ['train','valid_unseen']:
            pool=[]
            for game in sorted((data/'json_2.1.1'/split).glob(f'{family}-*/*/game.tw-pddl')):
                name=game.relative_to(data).as_posix()
                if name in used or 'movable' in name or 'Sliced' in name or not read(game).get('solvable',False):continue
                pool.append(name)
            rng.shuffle(pool)
            if split=='train':
                if len(pool)<24:raise ValueError('Insufficient screening pool')
                screening += [{'family':family,'game':g} for g in pool[:24]]
            else:
                if len(pool)<9:raise ValueError('Insufficient untouched evaluation pool')
                for j in range(3):
                    evaluation.append({'id':f'fresh:{family}:{j}','family':family,'games':pool[j*3:j*3+3]})
    training=dict(prior, model=str(MODEL), actor_url='http://127.0.0.1:18218',
        screening=screening,evaluation=evaluation,training_arms=train_arms,
        writer_max_tokens=768,writer_context_limit=16384,actor_context_limit=32768,
        sequence_length=4,train_seed=923,eval_seeds=[707,808],learning_rate=5e-6,
        initialization=str(root/'adapters/delta'),training=[],calibration=[],
        checkpoint_selection='fixed final; report all arms; no held-out score tuning')
    save(root/'training_plan.json',training)
    save(root/'data_hashes.json',{g:sha_file(data/g) for g in
         [s['game'] for s in screening]+[g for seq in evaluation for g in seq['games']]})
    (root/'PROTOCOL.md').write_text('''# Frozen transfer and improved reward training

Experiment 1: reuse the saved Delta and Absolute LoRA from September 22. Frozen base actor.
CLBench full canonical sequences: BSM 90, Poker 120, Database 20, Cohort 20; actor repeats 303/404.
Docker-dependent Sales/Codebase are blocked by host permissions, not silently replaced.
Only public environment observations reach the writer. Hidden evaluator scalars are excluded.
This intentionally differs from the previous reward-visible Cohort pilot.
Writer prompt is unchanged, output cap raised equally to 768 to reduce truncation; no retry selecting a better memory.
All per-task means, failures, costs, per-repeat results and writer truncations must be reported.

LoCoMo: fixed 300 QA items, 30 per each of 10 conversations, proportional category sampling (seed 923).
Memory constructed chronologically from all sessions of each conversation,
without QA, evidence annotations, generated observations, or annotated summaries. No QA feedback is retained.
Conditions: no history, rolling untrained summary, rolling Delta, rolling Absolute, full conversation.
Text/captions and timestamps only; the final rolling document is the only history for memory arms.
The original procedural-experience prompt is deliberately unchanged: this tests transfer as-is.
Official category F1 routines reused; categories 1–4 reported separately from adversarial category 5.
Category 5 uses official randomized choices. Greedy 50-token QA is a declared decoding adaptation.
Full history uses chronological sessions without truncation and is a larger-context reference, not a matched-memory control.

Experiment 2: continue the original Delta LoRA. Only ALFWorld official training split supplies gradients.
Screen 24 fresh training instances per family with two no-memory rollouts. Prefer successful source
tasks and intermediate-difficulty targets, but reserve one quarter of chains for uniformly sampled targets.
48 same-family four-task chains; no task repeats within a chain; training tasks may recur between chains.
One generated memory per state; paired rewards averaged over K=2 trials; fixed replica 0 continues the chain.
Reward remains success(next task, new memory) minus success(same task, empty memory).
No group centering, no best-of-K trajectory selection. Two PPO-style clipped epochs per batch,
KL to frozen base 0.01, LR 5e-6, 144 writer actions and 24 optimizer steps per arm.
Controls: matched curriculum/continuation with K=1, seed 923; K=2 with seeds 923 and 924.
The K=1 control has fewer environment calls; report both action budget and rollout cost, not equal-cost superiority.
Fresh ALFWorld test: 54 instances from valid_unseen, excluding the old pilot evaluation,
18 three-task sequences, two sampling repeats, fixed final checkpoints. No result-driven checkpoint selection.
Evaluate all three new checkpoints on the same CLBench and LoCoMo protocols; reuse exact frozen control results.
No benchmark QA or hidden CLBench data enter parameter training. Do not guarantee improvement in advance.
''')
    freeze(root)
    save(root/'status.json',{'phase':'prepared'})


def process(root,module,args,logname,gpu=None):
    env=environment(root)
    if gpu is not None:env['CUDA_VISIBLE_DEVICES']=str(gpu)
    path=root/logname;path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a') as log:
        return subprocess.Popen([str(PYTHON),'-m',module,*map(str,args)],cwd=root/'source',
                                env=env,stdout=log,stderr=subprocess.STDOUT)


def run_checked(root,module,args,logname,gpu=None):
    p=process(root,module,args,logname,gpu)
    code=p.wait()
    if code:raise RuntimeError(f'{module} {args} failed ({code}); see {logname}')


def training_pipeline(root):
    server=None
    try:
        server=start_server(root,1,18218,{},context=32768)
        save(root/'training_pipeline_status.json',{'phase':'screening'})
        run_checked(root,'ttcl.experience_v2.train',['screen','--root',root],'logs/screen.log')
        arms=list(read(root/'training_plan.json')['training_arms'])
        for arm in arms:
            save(root/'training_pipeline_status.json',{'phase':'training','arm':arm})
            run_checked(root,'ttcl.experience_v2.train',['train','--root',root,'--arm',arm],f'logs/train_{arm}.log',3)
        # Transfer can begin as soon as all adapters are saved.
        save(root/'adapters_ready.json',{'arms':arms,'time':time.time()})
        for arm in ['none','untrained','delta','absolute',*arms]:
            save(root/'training_pipeline_status.json',{'phase':'held_out_evaluation','arm':arm})
            run_checked(root,'ttcl.experience_v2.train',['evaluate','--root',root,'--arm',arm],f'logs/alf_eval_{arm}.log',3)
        save(root/'training_pipeline_status.json',{'phase':'complete'})
    except Exception:
        save(root/'training_pipeline_status.json',{'phase':'failed','traceback':traceback.format_exc()})
        raise
    finally:stop_server(server)


def transfer_suite(root,suite):
    from .evaluate import report
    settings=read(root/'plan.json')['suites'][suite]
    adapters={name:(root/'adapters'/name if name in {'delta','absolute'} else root/'training'/name/'adapter')
              for name in set(settings['arms'].values()) if name!='frozen-actor'}
    server=None;active=[]
    try:
        server=start_server(root,0,18217,adapters)
        jobs=[]
        for task in settings['tasks']:
            for repeat in settings['repeats']:
                jobs.append((f'{task}_{repeat}',['clbench','--task',task,'--repeat',repeat]))
        # Interleave two streams of CLBench with dialogue memory construction.
        locomo_jobs=[(f'locomo_{i}',['locomo','--sample',i]) for i in range(10)]
        queue=deque()
        for i in range(max(len(jobs),len(locomo_jobs))):
            if i<len(jobs):queue.append(jobs[i])
            if i<len(locomo_jobs):queue.append(locomo_jobs[i])
        finished=[];failed=[];last_report=0
        while queue or active:
            while queue and len(active)<4:
                name,args=queue.popleft()
                worker=root/suite/'workers'/f'{name}.json'
                if worker.exists() and read(worker).get('phase')=='complete':
                    finished.append(name);continue
                p=process(root,'ttcl.experience_v2.evaluate',[*args,'--root',root,'--suite',suite],f'{suite}/logs/{name}.log')
                active.append((name,p))
            for name,p in list(active):
                code=p.poll()
                if code is None:continue
                (failed if code else finished).append(name)
                active.remove((name,p))
            if time.monotonic()-last_report>30:
                report(root,suite);last_report=time.monotonic()
            save(root/f'{suite}_status.json',dict(phase='running',finished=finished,failed=failed,
                 active=[{'name':n,'pid':p.pid} for n,p in active],queued=len(queue),updated_at=time.time()))
            time.sleep(5)
        report(root,suite)
        save(root/f'{suite}_status.json',dict(phase='failed' if failed else 'complete',finished=finished,failed=failed))
        if failed:raise RuntimeError(f'Failed transfer jobs: {failed}')
    finally:
        for _,p in active:
            if p.poll() is None:p.terminate()
        stop_server(server)


def supervise(root):
    train_process=process(root,'ttcl.experience_v2.launch',['training','--root',root],'logs/training_pipeline.log')
    save(root/'status.json',dict(phase='running',supervisor_pid=os.getpid(),training_pid=train_process.pid))
    errors=[]
    try:transfer_suite(root,'frozen_transfer')
    except Exception:errors.append(traceback.format_exc())
    while not (root/'adapters_ready.json').exists() and train_process.poll() is None:
        save(root/'status.json',dict(phase='waiting_for_adapters',training_pid=train_process.pid,errors=errors))
        time.sleep(10)
    if (root/'adapters_ready.json').exists():
        try:transfer_suite(root,'improved_transfer')
        except Exception:errors.append(traceback.format_exc())
    if train_process.wait():errors.append('Training pipeline failed; see training_pipeline_status.json')
    hashes=read(root/'input_hashes.json')
    changed=[p for p,h in hashes.items() if sha_file(p)!=h]
    save(root/'integrity.json',dict(passed=not changed,changed=changed))
    if changed:errors.append('Frozen source/checkpoint hash changed')
    save(root/'status.json',dict(phase='failed' if errors else 'complete',errors=errors,
         blocked=read(root/'plan.json')['docker_blocked'],finished_at=time.time()))


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','run','training','transfer'])
    p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--suite',default='frozen_transfer')
    a=p.parse_args()
    if a.command=='prepare':prepare(a.root)
    elif a.command=='run':supervise(a.root)
    elif a.command=='training':training_pipeline(a.root)
    else:transfer_suite(a.root,a.suite)


if __name__=='__main__':main()
