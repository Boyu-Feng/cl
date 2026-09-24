from __future__ import annotations

from collections import Counter
from pathlib import Path
import json
import random
import shutil
import time

from ttcl.experience_evolution.core import read,save,writer_messages
from ttcl.experience_v2.common import MODEL, OLD, WORKSPACE, sha_file
from .environments import world, task, scripted_episode

PILOT = WORKSPACE / "ttcl/results/experience_repair/20260924"
V2 = WORKSPACE / "ttcl/results/experience_v2/20260923"
ROOT = WORKSPACE / "ttcl/results/experience_design/20260924"


def brief(w):
    if w['domain']=='sql':
        return f"For environment {w['id']} only, table {w['table']} has identifier {w['key']}, category {w['category']}, and integer amount {w['amount']}."
    return f"For service {w['id']} only, call {w['read']} with argument {w['key_arg']}, then {w['convert']} with argument {w['value_arg']} equal to the fetched raw value, then submit the converted result."


def synthetic_rows(domain,counts):
    rows=[];counter=0
    for category,n in counts.items():
        for j in range(n):
            w=world(domain,counter,'train');prev=world(domain,1000+counter,'train')
            spec=task(w,counter%9);mode=category
            if category=='keep' and j%2:mode='ambiguous'
            ep=scripted_episode(spec,mode)
            local=brief(w)
            if category=='scope':
                old=brief(prev)
                target=(old+' '+local+' Revision: A different environment identifier selects a different interface; do not overwrite the older mapping or apply it globally. Procedure: match the current environment before using a stored mapping, and inspect the current interface when no mapping matches. Uncertainty: no claim is made about unseen environments.')
                targets=[task(w,(counter+1)%9),task(prev,(counter+2)%9)]
            elif category=='keep':
                old=local+' Use the current observation and exact identifiers. A repeated observation does not require a new rule. Failure without a specific contradiction does not justify inventing a cause.'
                target=old;targets=[task(w,(counter+1)%9),task(w,(counter+2)%9)]
            elif category=='correction':
                old=('All databases use products with a price column; this mapping can be reused without inspection.' if domain=='sql'
                     else 'All services use fetch with argument id; a failed call means the item is absent.')
                detail=('The old products/price query failed, while inspecting the current schema exposed a different table and columns. Do not replace a computed query result with an invented answer.' if domain=='sql'
                        else 'The assumed fetch/id interface was rejected. The service description gave different names and a fetch-before-convert requirement. Submission must use the converted value, not the raw value.')
                target=f'Observed: {detail} Revision: withdraw the universal interface assumption. {local} Procedure: verify the environment and use its observed interface. Keep error causes uncertain unless the returned error identifies them.'
                targets=[task(w,(counter+1)%9),task(w,(counter+2)%9)]
            else:
                old='Use observed tool results to answer the current task; do not assume an interface before observing it.'
                detail=('Schema inspection, aggregation and submission succeeded.' if domain=='sql' else 'Discovering the interface, fetching the requested item, converting and submitting succeeded.')
                target=f'Observed: {detail} {local} Procedure: retain the observed interface and reuse it only when the environment matches. Specific answer values are task-specific; recompute them for new requests. Uncertainty: this success does not establish a universal schema or tool naming convention.'
                targets=[task(w,(counter+1)%9),task(w,(counter+2)%9)]
            assert len(target.split())<=200
            rows.append({'id':f'{domain}_{counter:03}', 'domain':domain,'category':category,
                         'previous':old,'episode':ep,'messages':writer_messages(old,ep),'target':target,
                         'probes':targets,'provenance':'scripted source actions executed in deterministic local environment; program-authored evidence-grounded supervision',
                         'source_world':w['id'],'previous_world':prev['id'] if category=='scope' else w['id'],
                         'evidence':ep['trajectory'],'operation':'keep' if category=='keep' else 'revise'})
            counter+=1
    return rows


def alf_rows():
    hs=read(PILOT/'histories.json');by_id={h['id']:h for h in hs}
    groups={
        'correction':[f'h{i:02}' for i in range(16,32)]+['h00','h02'],
        'success':['h01','h03','h04','h05','h07','h08','h09','h10','h12','h13','h14','h15'],
        'keep':['h00','h01','h02','h03','h04','h05','h06','h10','h11'],
        'scope':['h01','h03','h04','h07','h09','h10','h13','h14','h15'],
    }
    rows=[]
    for category,ids in groups.items():
        for index,hid in enumerate(ids):
            h=by_id[hid];old=h['previous'];target=h['corrected']
            if category=='keep':old=target
            elif category=='scope':
                old=h['corrected']+' This previous procedure and its object locations apply unchanged to every future household task.'
                target=h['corrected']+' Do not extend this observation into a universal location rule.'
            if len(target.split())>200:raise ValueError('ALF target over length')
            rows.append({'id':f'alf_{category}_{index:02}','domain':'alfworld','category':category,'family':h['family'],
                         'previous':old,'episode':h['episode'],'messages':writer_messages(old,h['episode']),
                         'target':target,'source_history':hid,'source_game':h['game'],'history_games':h['history_games'],
                         'evidence':h['evidence'],'operation':'keep' if category=='keep' else 'revise',
                         'provenance':'frozen Codex-reviewed ALF source; keep and scope include declared controlled previous-memory variants'})
    return rows


def prepare(root):
    root.mkdir(parents=True,exist_ok=False)
    rows=alf_rows()+synthetic_rows('sql',{'correction':18,'success':12,'keep':9,'scope':9})+synthetic_rows('tools',{'correction':12,'success':8,'keep':6,'scope':6})
    assert len(rows)==128
    assert Counter(r['category'] for r in rows)=={'correction':48,'success':32,'keep':24,'scope':24}
    assert all(r['messages']==writer_messages(r['previous'],r['episode']) for r in rows)
    # ALF teacher histories are frozen before choosing any new target instances.
    prior=read(V2/'training_plan.json');data=Path(prior['data_root']);used=set()
    for p in [read(OLD/'plan.json'),read(V2/'training/curriculum.json')]:
        used.update(g for s in p['training'] for g in s['games'])
    pilot_plan=read(PILOT/'plan.json')
    used.update(g for gs in pilot_plan['probes'].values() for g in gs)
    used.update(g for r in rows if r['domain']=='alfworld' for g in r['history_games'])
    rng=random.Random(92431);pools={}
    for family in {r['family'] for r in rows if r['domain']=='alfworld'}:
        pool=[]
        for p in sorted((data/'json_2.1.1/train').glob(f'{family}-*/*/game.tw-pddl')):
            g=p.relative_to(data).as_posix()
            if g not in used and 'movable' not in g and 'Sliced' not in g and read(p).get('solvable',False):pool.append(g)
        rng.shuffle(pool);pools[family]=pool
    for row in rows:
        if row['domain']=='alfworld':
            row['probes']=[{'domain':'alfworld','id':pools[row['family']].pop()} for _ in range(2)]
    save(root/'dataset.json',rows)
    # Fresh synthetic worlds plus the pilot's predeclared ALF test partition, never training.
    chains=[]
    for domain in ['sql','tools']:
        for i in range(4):
            a=world(domain,i*2,'test');b=world(domain,i*2+1,'test')
            sequence=[task(a,j) for j in range(4)]+[task(b,j) for j in range(3)]+[task(a,j) for j in range(4,7)]
            chains.append({'id':f'{domain}_{i}','domain':domain,'tasks':sequence,'phases':['A']*4+['B']*3+['A_return']*3})
    plan={k:prior[k] for k in ['data_root','actor_temperature','actor_max_tokens','max_steps']}
    plan.update(model=str(MODEL),actor_url='http://127.0.0.1:18257',port=18257,server_gpu=1,training_gpu=3,
                context=32768,writer_max_tokens=768,writer_context_limit=16384,rank=8,train_seed=92431,
                learning_rate=5e-6,kl_beta=.01,probe_repeats=[92511,92512],eval_repeats=[92521,92522],
                rl_arms={'rl_absolute':'absolute','rl_empty':'empty','rl_previous':'previous'},
                sft_arms=['raw_sft','balanced_sft'],evaluation_arms=['none','delta','raw_sft','balanced_sft','rl_absolute','rl_empty','rl_previous'],
                training={'seed':92431,'epochs':4,'accumulation':8,'learning_rate':5e-6,'max_length':16384},
                candidates_per_history=2,histories_per_batch=4,rl_batch_count=32,
                evaluation_alf=pilot_plan['evaluation'],evaluation_chains=chains,
                expected={'training_histories':128,'sft_steps_per_arm':64,'rl_writer_actions_per_arm':256,
                          'rl_optimizer_steps_per_arm':64,'rl_actor_episodes_per_arm':2048,
                          'rl_actor_episodes_total':6144,'evaluation_alf_sources':24,'evaluation_alf_targets':168,
                          'evaluation_synthetic_tasks':1120},created_at=time.time())
    save(root/'plan.json',plan)
    save(root/'training_plan.json',plan)
    shutil.copytree(V2/'adapters/delta',root/'adapters/delta')
    target=root/'source/ttcl';target.mkdir(parents=True);(target/'__init__.py').write_text('')
    for name in ['experience_design','experience_repair','experience_evolution','experience_v2','common','llm_memory','structured_memory']:
        shutil.copytree(WORKSPACE/'ttcl'/name,target/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    audit={'passed':True,'category_counts':dict(Counter(r['category'] for r in rows)),
           'domain_counts':dict(Counter(r['domain'] for r in rows)),
           'alf_distinct_source_histories':len({r['source_history'] for r in rows if r['domain']=='alfworld'}),
           'alf_controlled_previous_memory_variants':18,'synthetic_sources':'80 executed scripted episodes; not student-collected trajectories',
           'keep_targets_unchanged':all(r['target']==r['previous'] for r in rows if r['category']=='keep'),
           'test_data_in_writer_training_inputs':False,'test_partition':'ALF shared pilot reservation; SQL/tools disjoint test world identifiers',
           'annotations_fixed_before_probe_scoring':True,'distinct_alf_probe_games':len({p['id'] for r in rows if r['domain']=='alfworld' for p in r['probes']})}
    assert audit['distinct_alf_probe_games']==96 and audit['keep_targets_unchanged']
    save(root/'data_audit.json',audit)
    protocol='''# Data composition and reward-baseline experiment

128 supervised inputs: 48 correction, 32 successful procedure extraction, 24 keep, 24 scope revision.
Domains: 48 ALFWorld, 48 independent SQLite, 32 deterministic local API tasks.
ALF uses 32 distinct completed training histories plus controlled prior-memory variants (not 48
independent source trajectories). SQL/API sources are executed scripted demonstrations, not claims
of autonomous student discovery. Every label has raw public evidence and authorship provenance.
Both SFT arms see exactly the same histories and original writer input prompt. raw_sft learns one
unfiltered original Delta generation; balanced_sft learns fixed evidence-grounded targets. Both
start from original Delta, LR 5e-6, 4 epochs, accumulation 8, 64 updates. This isolates the target
supervision package (content/style), not the individual effects of each domain or data category.
No target is silently filtered using future rewards. Null infrastructure results are not zeroed.

Three RL arms start from the SAME balanced_sft final checkpoint. Four fixed histories per batch,
two independently sampled candidates per history, two predeclared future training tasks, and two
actor sampling repeats. Every arm executes every branch: candidate0, candidate1, previous, empty.
Reward: absolute=mean(R_new); empty=mean(R_new-R_empty); previous=mean(R_new-R_previous).
No hidden length/cost rewards or group centering. All custom rewards are exact task success in [0,1].
Cost, invalid actions and factual consistency are separate diagnostics, not conflated with utility.
The baseline choice changes finite-sample credit/variance, not necessarily the ideal policy optimum.
Candidate outcomes never enter writer prompts; candidate text is fixed before probe execution.
Two PPO-style epochs per batch, same action and environment budgets, no best-of-N test selection.
KL reference is the frozen shared balanced_sft INITIALIZATION, not the unadapted base model.
Zero-advantage samples retain the KL term; all candidates enter updates. The executed histories are
fixed across arms, but candidates/outcomes vary after model updates. Report that distinction.
Each RL arm: 256 writer actions, 64 optimizer updates, 2048 actor episodes. Total 6144 actor episodes.
Checkpoint selection: fixed final. Intermediate checkpoints saved for recovery only.

Evaluation: none, Delta, raw_sft, balanced_sft, rl_absolute, rl_empty, rl_previous.
ALF uses the previous pilot's reserved 12 two-task sequences, shared test partition for comparison;
not a claim of a second untouched ALF test set. No previous pilot outcome is used to change labels.
24 shared empty-memory source episodes; 168 target episodes. Two sampling repeats.
SQL/API: four chains per domain, each A(4 tasks)->B(3 tasks)->A return(3 tasks), two sampling repeats;
1120 task episodes. Independent test databases/service names and data. The current completed test
trajectory may update online memory for its next task, but never supplies parameter gradients.
One generation per update; no test-time oracle or candidate reward selection. Score the first task
separately, and report post-first and A-return results, domains and seeds separately, plus macro-domain.
These synthetic environments are mechanism tests, not CLBench benchmark scores. Historical CLBench
scores are not reused as if they belonged to these test tasks. No cross-domain transfer claim is made
for SQL/API after training on these domains. Longer ALF retention remains outside this pilot.

The original 32-history repair pipeline stays untouched. This run uses GPUs 1 and 3 and its own port.
All code, data, annotations, adapters and splits are frozen and hashed before launch.
'''
    (root/'PROTOCOL.md').write_text(protocol)
    paths=[p for d in [root/'source',root/'adapters'] for p in d.rglob('*') if p.is_file()]
    paths += [root/n for n in ['plan.json','training_plan.json','dataset.json','data_audit.json','PROTOCOL.md']]
    paths += [PILOT/'histories.json',PILOT/'plan.json']
    games={p['id'] for r in rows if r['domain']=='alfworld' for p in r['probes']}
    games.update(g for r in rows if r['domain']=='alfworld' for g in r['history_games'])
    games.update(g for s in plan['evaluation_alf'] for g in s['games'])
    paths += [data/g for g in games]
    save(root/'input_hashes.json',{str(p):sha_file(p) for p in paths})
    save(root/'status.json',{'phase':'prepared','expected':plan['expected']})
