from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import fcntl
import json
import math
import os
from pathlib import Path
import random
import shutil
import signal
import statistics
import subprocess
import time
import traceback

from ttcl.experience_evolution.core import FAMILIES, read, save, seed, writer_messages, append
from ttcl.experience_v2.common import (
    MODEL, OLD, PYTHON, WORKSPACE, Client, environment, sha_file, start_server, stop_server,
)
from .annotations import corrected, annotation_path, load_annotations

V2 = WORKSPACE / 'ttcl/results/experience_v2/20260923'
DEFAULT_ROOT = WORKSPACE / 'ttcl/results/experience_repair/20260924'
DIAGNOSTIC_ARMS = ['empty', 'keep', 'generated', 'corrected']
SFT_ARMS = ['unfiltered_sft', 'corrected_sft']


def select_histories():
    sequences = read(V2 / 'training/curriculum.json')['training']
    pools = defaultdict(list)
    for path in sorted((V2 / 'training/continued_k1').glob('batch_*/seq_*/update_*/writer.json')):
        position = int(path.parent.name.split('_')[1])
        if position < 2:
            continue
        index = int(path.parents[1].name.split('_')[1])
        batch = int(path.parents[2].name.split('_')[1])
        messages = read(path)['messages']
        value = json.loads(messages[1]['content'])
        seq = sequences[batch * 4 + index]
        pools[int(value['completed_interaction']['reward'])].append({
            'source': str(path.resolve()), 'game': seq['games'][position - 1],
            'history_games': seq['games'][:position], 'family': seq['family'],
            'messages': messages, 'previous': value['previous_experience'],
            'episode': value['completed_interaction'],
        })
    rng = random.Random(924)
    selected, used = [], set()
    for reward in [1, 0]:
        groups = defaultdict(list)
        for item in pools[reward]:
            groups[item['family']].append(item)
        for values in groups.values():
            rng.shuffle(values)
        count = 0
        while count < 16:
            before = count
            for family in sorted(groups):
                while groups[family] and groups[family][-1]['game'] in used:
                    groups[family].pop()
                if groups[family] and count < 16:
                    item = groups[family].pop()
                    used.add(item['game'])
                    selected.append(item)
                    count += 1
            if count == before:
                raise ValueError('Insufficient distinct histories for balanced selection')
    for index, item in enumerate(selected):
        item['id'] = f'h{index:02}'
    return selected


def prepare(root):
    load_annotations()  # Fail clearly before creating a run if reviewed data is absent.
    root.mkdir(parents=True, exist_ok=False)
    histories = select_histories()
    # corrected() binds every reviewed annotation to its exact public input.
    # New rollouts must be reviewed again before any probe target is selected.
    for item in histories:
        target, operation, steps = corrected(item)
        item.update(corrected=target, operation=operation,
                    evidence=[{'step': i, **item['episode']['trajectory'][i - 1]} for i in steps],
                    annotation_author='Codex assistant; evidence-reviewed supervision, not autonomous student output',
                    annotation_scope='previous memory and completed public trajectory only; no future task or reward')
        assert item['messages'] == writer_messages(item['previous'], item['episode'])
    save(root / 'histories.json', histories)
    # Fix annotations before sampling future tasks. No outcome-based target selection.
    previous = read(V2 / 'training_plan.json')
    data = Path(previous['data_root'])
    used_train = set(g for row in histories for g in row['history_games'])
    for seq in read(OLD / 'plan.json')['training'] + read(V2 / 'training/curriculum.json')['training']:
        used_train.update(seq['games'])
    used_eval = set()
    for plan in [read(OLD / 'plan.json'), previous]:
        used_eval.update(g for seq in plan['evaluation'] for g in seq['games'])
    rng = random.Random(92401)
    train_pools, eval_pools = {}, {}
    for family in FAMILIES:
        for split, excluded, dest in [('train', used_train, train_pools), ('valid_unseen', used_eval, eval_pools)]:
            values = []
            for path in sorted((data / 'json_2.1.1' / split).glob(f'{family}-*/*/game.tw-pddl')):
                game = path.relative_to(data).as_posix()
                if game not in excluded and 'movable' not in game and 'Sliced' not in game and read(path).get('solvable', False):
                    values.append(game)
            rng.shuffle(values)
            dest[family] = values
    probes = {}
    for row in histories:
        probes[row['id']] = [train_pools[row['family']].pop() for _ in range(2)]
    evaluation = []
    quotas = {'pick_and_place_simple': 3, 'look_at_obj_in_light': 1,
              'pick_clean_then_place_in_recep': 3, 'pick_cool_then_place_in_recep': 2,
              'pick_heat_then_place_in_recep': 2, 'pick_two_obj_and_place': 1}
    for family, count in quotas.items():
        for i in range(count):
            evaluation.append({'id': f'{family}_{i}', 'family': family,
                               'games': [eval_pools[family].pop() for _ in range(2)]})
    plan = {k: previous[k] for k in ['data_root', 'actor_temperature', 'actor_max_tokens', 'max_steps']}
    plan.update(model=str(MODEL), actor_url='http://127.0.0.1:18247', port=18247,
                server_gpu=0, training_gpu=1, context=32768, probes=probes,
                evaluation=evaluation, probe_repeats=[92411,92412], eval_repeats=[92421,92422],
                writer_max_tokens=768, writer_temperature=1.0, diagnostic_arms=DIAGNOSTIC_ARMS,
                sft_arms=SFT_ARMS, evaluation_arms=['none','delta',*SFT_ARMS],
                training={'seed':924, 'epochs':4, 'accumulation':8, 'learning_rate':5e-6, 'max_length':16384},
                expected={'histories':32, 'diagnostic_episodes':512, 'sft_examples_per_arm':32,
                          'sft_optimizer_steps_per_arm':16, 'fresh_source_episodes':24, 'fresh_target_episodes':96},
                keep_labels=sum(x['operation']=='keep' for x in histories),
                created_at=time.time(), primary_score='official success; paired differences by history and family',
                selection='fixed final checkpoints; no test-selected candidate or checkpoint')
    save(root / 'plan.json', plan)
    shutil.copytree(V2 / 'adapters/delta', root / 'adapters/delta')
    target = root / 'source/ttcl'
    target.mkdir(parents=True)
    (target / '__init__.py').write_text('')
    for name in ['experience_repair','experience_evolution','experience_v2','common','llm_memory','structured_memory']:
        shutil.copytree(WORKSPACE / 'ttcl' / name, target / name,
                        ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    all_probes = [g for games in probes.values() for g in games]
    all_eval = [g for seq in evaluation for g in seq['games']]
    assert len(set(all_probes)) == 64 and len(set(all_eval)) == 24
    assert not set(all_probes) & used_train and not set(all_eval) & used_eval
    assert all('/train/' in g for g in all_probes)
    assert all('/valid_unseen/' in g for g in all_eval)
    save(root / 'split_audit.json', {'passed':True, 'source_rewards':dict(Counter(x['episode']['reward'] for x in histories)),
         'histories':32, 'distinct_source_games':len({x['game'] for x in histories}),
         'new_train_probes':64, 'new_valid_unseen_games':24,
         'evaluation_excludes':'all evaluation games in original Delta pilot and experience_v2',
         'no_clbench_training':True, 'annotation_uses_future':False})
    protocol = '''# Evidence-grounded memory repair pilot

Authorized scope: fixed-history diagnostic, controlled small SFT, and fresh ALFWorld evaluation.
32 completed training histories (16 success, 16 failure), all with nonempty previous memory,
selected from continued_k1 rollouts, round-robin by family with unique source games.
These are reused training histories, not untouched histories. Only train-split data supplies gradients.

Four diagnostic arms: empty, keep previous, newly generated old Delta, Codex-authored correction.
Corrections are evidence-reviewed supervision, NOT autonomous student or independently human-verified
outputs. The author read only old memory and completed public trajectories before probe selection.
Each correction cites exact steps; unexecuted recovery procedures are hypotheses. Two no-op targets
preserve old text. Corrections are frozen before scoring and never revised based on probe rewards.
32 histories x 4 arms x 2 distinct same-family training probes x 2 sampling repeats = 512 episodes.
No branch trajectory is used to alter another branch's input. Same task resets and seeds across arms.

SFT: corrected_sft vs unfiltered_sft, same 32 histories, same old Delta initialization,
same original writer input prompt, 4 epochs, batch accumulation 8, LR 5e-6, 16 steps per arm.
unfiltered_sft uses the one predeclared original Delta generation per history, without utility filtering.
corrected_sft uses the frozen authored targets, also without outcome filtering. This is supervision
quality/style plus data content as a package, not a pure reward-baseline ablation. Both are run and
reported regardless of diagnostic reward; no best-of-N candidate selection. Actor weights frozen.
Assistant targets alone receive SFT loss; no context truncation. Base fingerprints checked.

Fresh evaluation: 12 two-task sequences (24 valid_unseen games), excluding original pilot and v2
evaluation games; 2 sampling repeats; none, original Delta, unfiltered_sft, corrected_sft.
The first task is identical empty-memory data shared across four arms with explicit provenance;
each writer generates one memory from that first task and the second task is scored once.
24 source episodes + 96 target episodes, or 192 logical two-task arm episodes.
No training or checkpoint selection uses these outcomes. Family counts are unequal; report
per-family and macro-family as well as pooled scores. This tests one update, not long-chain retention.

Infrastructure failures are not converted to reward zero. Completed ALFWorld timeouts are official
failures. Report paired results, sample counts, seed/family breakdowns, tokens, steps, lengths,
invalid commands and uncertainty. Small diagnostic sample, no guaranteed improvement.
This pilot does not launch the previously prepared RL objectives or a new CLBench test sweep.
'''
    (root / 'PROTOCOL.md').write_text(protocol)
    paths = [p for d in [root/'source',root/'adapters'] for p in d.rglob('*') if p.is_file()]
    paths += [root/n for n in ['plan.json','histories.json','PROTOCOL.md','split_audit.json']]
    paths += [Path(x['source']) for x in histories]
    paths += [annotation_path()]
    paths += [data/g for g in set(all_probes+all_eval+[g for x in histories for g in x['history_games']])]
    save(root / 'input_hashes.json', {str(p):sha_file(p) for p in paths})
    save(root / 'status.json', {'phase':'prepared','expected':plan['expected']})


def verify(root):
    changed = [p for p,h in read(root/'input_hashes.json').items() if sha_file(p)!=h]
    if changed:
        raise ValueError(f'Frozen inputs changed: {changed}')


def validate_episode(ep, job):
    for k in ['game','seed','memory']:
        if ep[k] != job[k]:
            raise ValueError(f'Cached episode disagrees on {k}')
    if ep.get('status') != 'complete' or ep.get('actor_adapter_enabled') is not False:
        raise ValueError('Invalid completed frozen-actor record')


def run_jobs(actor, jobs):
    missing = []
    for job in jobs:
        path = Path(job['output'])
        if (path/'episode.json').exists():
            validate_episode(read(path/'episode.json'),job)
        else:
            if path.exists():
                path.rename(path.with_name(path.name+f'.interrupted_{time.time_ns()}'))
            missing.append(job)
    if missing:
        actor.run_many(missing)
    results = [read(Path(job['output'])/'episode.json') for job in jobs]
    for ep,job in zip(results,jobs):
        validate_episode(ep,job)
    return results


def matched(episodes):
    for key in ['game','seed','initial_observation','initial_commands_sha256']:
        if len({ep[key] for ep in episodes}) != 1:
            raise ValueError(f'Mismatched branches: {key}')


def generate(root, client, messages, model, directory, random_seed):
    path = directory/'writer.json'
    if path.exists():
        result = read(path)
        if result['messages'] != messages or result['served_model'] != model or result['actual_generation_seed'] != random_seed:
            raise ValueError('Writer resume input mismatch')
    else:
        result = client.complete(messages, model=model, random_seed=random_seed, tokens=768, temperature=1.0)
        if not result['raw_response']:
            raise ValueError('Empty writer result')
        result['messages'] = messages
        save(path,result)
    return result['raw_response']


def make_training_data(root):
    plan=read(root/'plan.json')
    client=Client(plan['actor_url'],context=plan['context'])
    histories=read(root/'histories.json')
    rows={a:[] for a in SFT_ARMS}
    for i,row in enumerate(histories):
        generated=generate(root,client,row['messages'],'delta',root/'candidates'/row['id'],seed(924,'writer',row['id']))
        for arm,target in [('unfiltered_sft',generated),('corrected_sft',row['corrected'])]:
            rows[arm].append({'id':row['id'],'messages':row['messages'],'target':target,
                              'source_reward':row['episode']['reward'],'family':row['family']})
        save(root/'generation_status.json',{'phase':'running','completed':i+1,'expected':32})
    for arm,values in rows.items():
        save(root/'training_data'/f'{arm}.json',values)
    save(root/'training_data/hashes.json',{a:sha_file(root/'training_data'/f'{a}.json') for a in SFT_ARMS})
    save(root/'generation_status.json',{'phase':'complete','completed':32,'expected':32})


def diagnostic(root, actor):
    plan=read(root/'plan.json'); rows=[]
    for history in read(root/'histories.json'):
        memories={'empty':'','keep':history['previous'],'corrected':history['corrected'],
                  'generated':read(root/'candidates'/history['id']/'writer.json')['raw_response']}
        jobs=[]; metadata=[]
        for target_index,game in enumerate(plan['probes'][history['id']]):
            for repeat in plan['probe_repeats']:
                for arm in DIAGNOSTIC_ARMS:
                    jobs.append({'game':game,'memory':memories[arm],'seed':seed(repeat,history['id'],target_index),
                                 'output':str(root/'diagnostic'/history['id']/str(target_index)/str(repeat)/arm)})
                    metadata.append({'history':history['id'],'family':history['family'],'source_reward':history['episode']['reward'],
                                     'target':target_index,'repeat':repeat,'arm':arm})
        episodes=run_jobs(actor,jobs)
        for i in range(0,len(episodes),4):matched(episodes[i:i+4])
        for ep,meta in zip(episodes,metadata):
            rows.append({**meta,**episode_metrics(ep)})
        save(root/'diagnostic_rows.json',rows)
        save(root/'diagnostic_status.json',{'phase':'running','completed':len(rows),'expected':512})
        report(root)
        print(json.dumps({'diagnostic':len(rows),'expected':512}),flush=True)
    save(root/'diagnostic_status.json',{'phase':'complete','completed':len(rows),'expected':512})


def episode_metrics(ep):
    return {'game':ep['game'],'reward':ep['reward'],'steps':ep['steps'],
            'invalid_commands':sum(not s['valid_command'] for s in ep['trajectory']),
            'actor_calls':len(ep['generations']),
            'input_tokens':sum(g.get('usage',{}).get('prompt_tokens',0) for g in ep['generations']),
            'output_tokens':sum(g.get('usage',{}).get('completion_tokens',0) for g in ep['generations'])}


def encode_training(tokenizer,row,limit):
    prefix=tokenizer.apply_chat_template(row['messages'],tokenize=True,add_generation_prompt=True)
    full=tokenizer.apply_chat_template(row['messages']+[{'role':'assistant','content':row['target']}],tokenize=True)
    if full[:len(prefix)]!=prefix:raise ValueError('Chat-template boundary mismatch')
    if len(full)>limit:raise ValueError(f'Overlength history {row["id"]}: {len(full)}; no truncation')
    return {'ids':full,'target_length':len(full)-len(prefix)}


def train(root,arm):
    import torch
    from transformers import AutoTokenizer,AutoModelForCausalLM
    from peft import PeftModel
    from ttcl.experience_evolution.writer import fingerprint
    verify(root)
    plan=read(root/'plan.json'); config=plan['training']; out=root/'training'/arm
    if (out/'status.json').exists() and read(out/'status.json')['phase']=='complete':return
    if out.exists():raise RuntimeError('Partial SFT requires explicit recovery; refusing silent restart')
    out.mkdir(parents=True)
    save(out/'status.json',{'phase':'loading'})
    hashes=read(root/'training_data/hashes.json')
    assert all(sha_file(root/'training_data'/f'{a}.json')==hashes[a] for a in SFT_ARMS)
    tokenizer=AutoTokenizer.from_pretrained(str(MODEL),local_files_only=True)
    # Validate both arms before either starts, so filtering cannot change histories.
    for a in SFT_ARMS:
        values=read(root/'training_data'/f'{a}.json')
        assert len(values)==32
        for row in values:encode_training(tokenizer,row,config['max_length'])
    data=[encode_training(tokenizer,r,config['max_length']) for r in read(root/'training_data'/f'{arm}.json')]
    torch.manual_seed(config['seed'])
    base=AutoModelForCausalLM.from_pretrained(str(MODEL),local_files_only=True,torch_dtype=torch.bfloat16,
                                            attn_implementation='sdpa').to('cuda:0')
    model=PeftModel.from_pretrained(base,str(root/'adapters/delta'),is_trainable=True)
    model.config.use_cache=False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    model.enable_input_require_grads()
    parameters=[p for p in model.parameters() if p.requires_grad]
    assert parameters and all('lora_' in n for n,p in model.named_parameters() if p.requires_grad)
    before=fingerprint(model);initial_adapter=fingerprint(model,True)
    optimizer=torch.optim.AdamW(parameters,lr=config['learning_rate'],weight_decay=0.0)
    rng=random.Random(config['seed']);step=0;model.train()
    for epoch in range(config['epochs']):
        order=list(range(len(data)));rng.shuffle(order)
        for offset in range(0,len(order),config['accumulation']):
            batch=order[offset:offset+config['accumulation']];optimizer.zero_grad(set_to_none=True);losses=[]
            for index in batch:
                item=data[index]; ids=torch.tensor([item['ids'][:-1]],device='cuda:0')
                targets=torch.tensor([item['ids'][-item['target_length']:]],device='cuda:0')
                logits=model(input_ids=ids,attention_mask=torch.ones_like(ids),
                             logits_to_keep=item['target_length'],use_cache=False).logits
                loss=torch.nn.functional.cross_entropy(logits.float().reshape(-1,logits.shape[-1]),targets.reshape(-1))
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite SFT loss')
                losses.append(float(loss.detach()));(loss/len(batch)).backward()
                del logits,loss,ids,targets
            norm=float(torch.nn.utils.clip_grad_norm_(parameters,1.0,error_if_nonfinite=True))
            if not math.isfinite(norm):raise FloatingPointError('Nonfinite gradient')
            optimizer.step();step+=1
            state={'phase':'training','epoch':epoch+1,'step':step,'expected_steps':16,
                   'mean_loss':statistics.mean(losses),'gradient_norm':norm,'updated_at':time.time()}
            append(out/'training.jsonl',state);save(out/'status.json',state);print(json.dumps(state),flush=True)
    after=fingerprint(model)
    if before!=after:raise AssertionError('Actor base weights changed')
    model.save_pretrained(out/'adapter');tokenizer.save_pretrained(out/'adapter')
    save(out/'audit.json',{'base_unchanged':True,'base_fingerprint':before,'initial_adapter':initial_adapter,
                         'final_adapter':fingerprint(model,True),'only_lora_trainable':True,
                         'data_sha256':hashes[arm],'assistant_only_loss':True,'examples':len(data)})
    save(out/'status.json',{'phase':'complete','steps':step,'examples':len(data)})


def training_pipeline(root):
    child=None
    try:
        for arm in SFT_ARMS:
            save(root/'training_status.json',{'phase':'running','arm':arm,'updated_at':time.time()})
            child=worker(root,'train',['--arm',arm],f'train_{arm}.log',gpu=read(root/'plan.json')['training_gpu'])
            if child.wait():raise RuntimeError(f'SFT failed: {arm}')
        save(root/'training_status.json',{'phase':'complete','arms':SFT_ARMS})
    finally:
        if child is not None and child.poll() is None:
            stop_server(child)


def evaluate(root,actor):
    plan=read(root/'plan.json');client=Client(plan['actor_url'],context=plan['context']);rows=[]
    for sequence in plan['evaluation']:
        for repeat in plan['eval_repeats']:
            directory=root/'evaluation'/sequence['id']/str(repeat)
            source=run_jobs(actor,[{'game':sequence['games'][0],'memory':'','seed':seed(repeat,sequence['id'],0),
                                   'output':str(directory/'source')}])[0]
            memories={'none':''}
            for arm in plan['evaluation_arms'][1:]:
                memories[arm]=generate(root,client,writer_messages('',source),arm,directory/arm,
                                       seed(repeat,sequence['id'],'writer'))
            jobs=[{'game':sequence['games'][1],'memory':memories[arm],'seed':seed(repeat,sequence['id'],1),
                   'output':str(directory/arm/'target')} for arm in plan['evaluation_arms']]
            episodes=run_jobs(actor,jobs);matched(episodes)
            for arm,ep in zip(plan['evaluation_arms'],episodes):
                rows.append({'sequence':sequence['id'],'family':sequence['family'],'repeat':repeat,'arm':arm,
                             'shared_source':str(directory/'source/episode.json'),**episode_metrics(ep)})
            save(root/'evaluation_rows.json',rows)
            save(root/'evaluation_status.json',{'phase':'running','completed':len(rows),'expected':96})
            report(root)
    save(root/'evaluation_status.json',{'phase':'complete','completed':len(rows),'expected':96})


def report(root):
    result={};lines=['# Experience repair pilot','',
        'Evidence-reviewed Codex supervision; fixed actor; corrected SFT versus self-generated SFT control.',
        'Partial means are descriptive only. No test-selected checkpoints. See PROTOCOL.md for scope.','']
    for filename,key,baseline in [('diagnostic_rows.json','diagnostic','empty'),('evaluation_rows.json','fresh_evaluation','none')]:
        if not (root/filename).exists():continue
        rows=read(root/filename); groups=defaultdict(list)
        for row in rows:groups[row['arm']].append(row)
        ids=lambda r: (r.get('history',r.get('sequence')),r.get('target',0),r['repeat'])
        maps={arm:{ids(r):r for r in rs} for arm,rs in groups.items()}
        common=set.intersection(*(set(v) for v in maps.values()))
        result[key]={'recorded':len(rows),'common_per_arm':len(common),'arms':{}}
        lines += [f'## {key}: {len(rows)} records; {len(common)} shared samples per arm','',
                  '| arm | mean reward | delta baseline | mean steps | family macro reward |','|---|---:|---:|---:|---:|']
        for arm,rs in groups.items():
            vals=[maps[arm][k] for k in common];families=sorted({r['family'] for r in vals})
            by_family={f:statistics.mean(r['reward'] for r in vals if r['family']==f) for f in families}
            v={'n':len(vals),'mean_reward':statistics.mean(r['reward'] for r in vals),
               'delta_baseline':statistics.mean(maps[arm][k]['reward']-maps[baseline][k]['reward'] for k in common),
               'mean_steps':statistics.mean(r['steps'] for r in vals),'by_family':by_family,
               'family_macro_reward':statistics.mean(by_family.values()),
               'by_seed':{str(s):statistics.mean(r['reward'] for r in vals if r['repeat']==s) for s in sorted({r['repeat'] for r in vals})},
               'actor_calls':sum(r['actor_calls'] for r in vals),'input_tokens':sum(r['input_tokens'] for r in vals),
               'output_tokens':sum(r['output_tokens'] for r in vals),'invalid_commands':sum(r['invalid_commands'] for r in vals)}
            for other in maps:
                ds=[maps[arm][k]['reward']-maps[other][k]['reward'] for k in common]
                v['vs_'+other]={'mean':statistics.mean(ds),'wins':sum(d>0 for d in ds),'losses':sum(d<0 for d in ds),'ties':sum(d==0 for d in ds)}
            result[key]['arms'][arm]=v
            lines.append(f'| {arm} | {v["mean_reward"]:.4f} | {v["delta_baseline"]:+.4f} | {v["mean_steps"]:.2f} | {v["family_macro_reward"]:.4f} |')
        lines.append('')
    save(root/'summary.json',result)
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')


def worker(root,command,args,logname,gpu=None):
    env=environment(root)
    if gpu is not None:env['CUDA_VISIBLE_DEVICES']=str(gpu)
    path=root/'logs'/logname;path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a') as log:
        p=subprocess.Popen([str(PYTHON),'-m','ttcl.experience_repair.run',command,'--root',str(root),*args],
                            env=env,cwd=root/'source',stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    save(root/'processes'/f'{command}.json',{'pid':p.pid,'command':command,'args':args,'gpu':gpu})
    return p


def supervise(root):
    lock=(root/'supervisor.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    verify(root);plan=read(root/'plan.json');server=None;trainer=None
    def stop(signum,frame):raise KeyboardInterrupt(f'Signal {signum}')
    signal.signal(signal.SIGTERM,stop)
    from ttcl.experience_evolution.environment import Actor
    actor=Actor(plan)
    try:
        save(root/'status.json',{'phase':'starting_server','supervisor_pid':os.getpid(),'updated_at':time.time()})
        server=start_server(root,plan['server_gpu'],plan['port'],{'delta':root/'adapters/delta'},context=plan['context'])
        save(root/'status.json',{'phase':'generating_training_pairs','supervisor_pid':os.getpid(),'updated_at':time.time()})
        make_training_data(root)
        trainer=worker(root,'training',[],'training_pipeline.log')
        save(root/'status.json',{'phase':'diagnostic_and_sft','supervisor_pid':os.getpid(),'training_pid':trainer.pid,'updated_at':time.time()})
        diagnostic(root,actor)
        save(root/'status.json',{'phase':'waiting_for_sft','supervisor_pid':os.getpid(),'training_pid':trainer.pid,'updated_at':time.time()})
        if trainer.wait():raise RuntimeError('Training pipeline failed; see logs')
        stop_server(server);server=None
        adapters={'delta':root/'adapters/delta',**{a:root/'training'/a/'adapter' for a in SFT_ARMS}}
        server=start_server(root,plan['server_gpu'],plan['port'],adapters,context=plan['context'])
        save(root/'status.json',{'phase':'fresh_evaluation','supervisor_pid':os.getpid(),'updated_at':time.time()})
        evaluate(root,actor);verify(root);report(root)
        save(root/'status.json',{'phase':'complete','finished_at':time.time(),'expected':plan['expected']})
    except BaseException as exc:
        save(root/'status.json',{'phase':'stopped' if isinstance(exc,KeyboardInterrupt) else 'failed',
                                'traceback':traceback.format_exc(),'updated_at':time.time()})
        raise
    finally:
        if trainer is not None and trainer.poll() is None:stop_server(trainer)
        stop_server(server)
        actor.pool.shutdown(wait=False,cancel_futures=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','supervise','train','training','report'])
    p.add_argument('--root',type=Path,default=DEFAULT_ROOT);p.add_argument('--arm',choices=SFT_ARMS)
    a=p.parse_args()
    def stopping(signum,frame):raise KeyboardInterrupt(f'Signal {signum}')
    signal.signal(signal.SIGTERM,stopping)
    try:
        if a.command=='prepare':prepare(a.root)
        elif a.command=='supervise':supervise(a.root)
        elif a.command=='train':train(a.root,a.arm)
        elif a.command=='training':training_pipeline(a.root)
        else:report(a.root)
    except BaseException as exc:
        if a.command in {'train','training'}:
            path=a.root/'training'/a.arm/'status.json' if a.command=='train' else a.root/'training_status.json'
            save(path,{'phase':'stopped' if isinstance(exc,KeyboardInterrupt) else 'failed',
                       'traceback':traceback.format_exc(),'updated_at':time.time()})
        raise


if __name__=='__main__':main()
