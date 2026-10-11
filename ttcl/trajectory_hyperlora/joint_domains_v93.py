"""Audited joint-domain pilot: matched data/order for generated and direct LoRA."""
from __future__ import annotations
import argparse, hashlib, json, random, time
from pathlib import Path
from collections import Counter
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from ttcl.trajectory_hyperlora.direct_composition_pilot import DirectRelationHyperLoRA
from ttcl.trajectory_hyperlora.contextual_alf_source import contextual_text_fields, task_context_text, source_text
from ttcl.trajectory_hyperlora.train_alf_next_task import compact_messages
from ttcl.trajectory_hyperlora.train_xland_raw_hyperlora import checked_query
from ttcl.trajectory_hyperlora.train_xland_official_history_hyperlora import question
from ttcl.trajectory_hyperlora.score_hle_mcq_pilot import parse_letter

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'results/trajectory_hyperlora/joint_domains_v93_20261010'
MODEL = ROOT/'current_work/delta-Mem/model/Qwen3-4B-Instruct-2507'

def digest(x):
    return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
def filehash(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):
    return json.loads(Path(p).read_text())
def write(p,d):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(d,ensure_ascii=False,indent=2));tmp.replace(p)
def ordered(xs,key):
    return sorted(xs,key=lambda x:digest(['v93',key(x)]))
def select_source(query,bank,exclude):
    # Selection depends only on public text, never labels or evaluation outcomes.
    words=set(query.lower().split())
    eligible=[x for x in bank if x['id']!=exclude]
    return max(eligible,key=lambda x:(len(words & set(x['query'].lower().split()))/max(1,len(words | set(x['query'].lower().split()))),digest(x['id'])))
def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'dataset.json').exists():
        raise RuntimeError('Refusing to overwrite frozen prepared dataset')
    paths={
      'alf_targets':ROOT/'data/annotations/alfworld_cleaned_targets_v88_20261010.json',
      'alf_episodes':ROOT/'results/trajectory_hyperlora/alfworld_success_loop_cleaning_v87_20261010/report.json',
      'alf_audit':ROOT/'results/trajectory_hyperlora/alfworld_success_loop_cleaning_v87_20261010/annotation_audit.json',
      'xland':ROOT/'data/annotations/xland_trivial_official_reviewed_128_v1_20261006.json',
      'hle_inputs':ROOT/'data/hle/failure_retry_pilot_20261010/actor_inputs.json',
      'hle_keys':ROOT/'data/hle/failure_retry_pilot_20261010/grading_keys.json',
      'hle_responses':ROOT/'results/trajectory_hyperlora/hle_base_responses_20261010.json'}
    audit=read(paths['alf_audit']);assert audit['passed']
    assert audit['new_annotation_sha256']==filehash(paths['alf_targets'])
    assert audit['report_sha256']==filehash(paths['alf_episodes'])
    rows=[];banks={};evals={}
    def add(domain,ident,messages,answer,query,source,parent):
        row=dict(domain=domain,id=ident,messages=messages,answer=answer,query=query,
                 source_id=source['id'],source=source['text'],parent=parent)
        row['input_content_sha256']=digest(row);row['reviewed_target']=True;rows.append(row)
    banks['alfworld']=[]
    for r in read(paths['alf_episodes'])['rows']:
        e=r['episode'];assert e['reward']==1
        obs=e['initial_observation'];records=[]
        for t in e['trajectory']:
            records.append(dict(observation=obs,action=t['command'],feedback=t['observation']));obs=t['observation']
        banks['alfworld'].append(dict(id=r['game'],query=e['initial_observation'],text=source_text(records)))
    counts=Counter()
    for r in read(paths['alf_targets'])['targets']:
        assert r['reviewed_target']
        if counts[r['game']]>=8:continue
        counts[r['game']]+=1
        src=select_source(r['initial_observation'],banks['alfworld'],r['game'])
        add('alfworld',f"{r['game']}:{r['target_step']}",compact_messages(r['actor_messages'],2),r['target_action'],
            task_context_text(r['initial_observation'],r['current_observation']),src,
            dict(original_input_sha256=r['input_content_sha256'],game=r['game'],target_step=r['target_step']))
    assert len(counts)==28
    x=read(paths['xland']);xt=ordered(x['split']['train'],lambda z:z['task_id'])[:24]
    assert not ({z['ruleset_id'] for z in xt}&{z['ruleset_id'] for z in x['split']['test']})
    for task in xt:
        for i,q in enumerate(task['queries'][:8]):
            content,label=checked_query(q)
            assert task['source_history_id']!=q['target_history_id']
            assert label in range(6)
            src=dict(id=f"xland:{task['task_id']}:{task['source_history_id']}",text=json.dumps(content['source_episodes'],ensure_ascii=False,separators=(',',':')))
            query=question(content,6)
            add('xland',f"{task['task_id']}:{i}",[dict(role='user',content=query)],str(label),query,src,
                dict(original_input_sha256=q['input_sha256'],task_id=task['task_id'],ruleset_id=task['ruleset_id'],review_basis=q['review_basis']))
    evals['xland']=ordered(x['split']['test'],lambda z:z['task_id'])[:8]
    inputs=read(paths['hle_inputs']);keys={r['id']:r for r in read(paths['hle_keys'])}
    resp={r['id']:r for r in read(paths['hle_responses'])['rows']}
    train=[r for r in inputs if r['partition']=='source_collection'];assert len(train)==64
    banks['hle']=[]
    for r in train:
        k=keys[r['id']]; rr=resp[r['id']]
        assert r['input_content_sha256']==k['input_content_sha256']==rr['input_content_sha256']
        letter,_=parse_letter(rr['response'])
        if letter==k['answer'].strip():
            banks['hle'].append(dict(id=r['id'],query=r['question'],text='Completed prior question:\n'+r['question']+'\nResponse:\n'+rr['response']))
    assert len(banks['hle'])==8
    for r in train:
        src=select_source(r['question'],banks['hle'],r['id'])
        add('hle',r['id'],[dict(role='system',content='Solve the question. Return only one line: Answer: X, where X is the option letter.'),dict(role='user',content=r['question'])],
            'Answer: '+keys[r['id']]['answer'].strip(),r['question'],src,dict(original_input_sha256=r['input_content_sha256'],supervision='official training-subset answer'))
    evals['hle']=ordered([r for r in inputs if r['partition']=='evaluation'],lambda z:z['id'])[:24]
    assert not ({r['id'] for r in train}&{r['id'] for r in evals['hle']})
    games=sorted((ROOT/'ttcl/data/alfworld_delta/json_2.1.1/valid_unseen').glob('*/*/game.tw-pddl'))
    if not games:games=sorted((ROOT/'data/ttcl/alfworld_delta/json_2.1.1/valid_unseen').glob('*/*/game.tw-pddl'))
    families={p.parts[-3].split('-')[0] for p in games}
    assert len(families)==6,(len(games),families)
    evals['alfworld']=[str(p) for fam in sorted(families) for p in ordered([p for p in games if p.parts[-3].split('-')[0]==fam],str)[:2]]
    schedule=[];rng=random.Random(93)
    pools={d:[i for i,r in enumerate(rows) if r['domain']==d] for d in ['alfworld','xland','hle']}
    for cycle in range(200):
        for d in pools:schedule.append(rng.choice(pools[d]))
    dataset=dict(rows=rows,banks=banks,evaluation=evals,schedule=schedule)
    manifest=dict(protocol='joint_domains_v93',training_counts=dict(Counter(r['domain'] for r in rows)),training_tasks=dict(alfworld=28,xland=24,hle=64),
        updates_per_arm=600,domain_updates=dict(alfworld=200,xland=200,hle=200),rank=4,layers=2,lr=1e-4,seed=93,
        actor_max_tokens=4096,encoder_max_tokens=2048,encoder_clipping='first 1024 plus last 1024 token IDs; recorded exact IDs',
        init='fresh contextual task-conditioned H; shared initial random A; zero B for both',
        evaluation_counts={d:len(v) for d,v in evals.items()},evaluation_updates=0,
        provenance=dict(alfworld='audited cleaned own successful trajectories',xland='official histories and expert action labels',hle='64 supervised training-subset answers; source bank of 8 own option-correct responses, reasoning unverified'),
        limitations=['same update budget, different parameter counts and feature-encoding cost','existing evaluation pools previously exposed; exploratory not pristine confirmation','new HLE answer-only prompt; old 8/64 not a paired control'],
        dependencies={str(p.relative_to(ROOT)):filehash(p) for p in paths.values()},dataset_sha256=digest(dataset),schedule_sha256=digest(schedule))
    write(OUT/'dataset.json',dataset);write(OUT/'manifest.json',manifest)
    print(json.dumps(manifest,ensure_ascii=False),flush=True)

def run(device):
    manifest=read(OUT/'manifest.json');data=read(OUT/'dataset.json');assert digest(data)==manifest['dataset_sha256']
    for p,h in manifest['dependencies'].items():assert filehash(ROOT/p)==h,p
    if (OUT/'complete.json').exists():raise RuntimeError('Already complete')
    torch.manual_seed(93);torch.cuda.manual_seed_all(93)
    tok=AutoTokenizer.from_pretrained(MODEL)
    model=AutoModelForCausalLM.from_pretrained(MODEL,torch_dtype=torch.bfloat16,attn_implementation='sdpa').to(device)
    agent=DirectRelationHyperLoRA(model,rank=4,layers=2,encoder_kind='contextual',task_conditioned=True).to(device)
    agent.eval() # identical deterministic dropout policy in both arms; gradients still enabled
    hparams={n:p for n,p in agent.named_parameters() if p.requires_grad}
    ha=[a.a for a in agent.adapters]
    sa=[torch.nn.Parameter(p.detach().clone()) for p in ha]
    sb=[torch.nn.Parameter(torch.zeros(1,a.base.out_features,a.rank,device=device)) for a in agent.adapters]
    assert all(torch.equal(a,b) for a,b in zip(ha,sa))
    initial={n:p.detach().cpu().clone() for n,p in hparams.items()}
    torch.save(dict(hyper=initial,static_a=[p.detach().cpu() for p in sa],static_b=[p.detach().cpu() for p in sb]),OUT/'initial.pt')
    cache={};prepared=[];token_bindings=[]
    write(OUT/'status.json',dict(stage='encoding',rows=0,total=len(data['rows'])))
    def encode(text):
        key=digest(text)
        if key not in cache:
            ids=tok(text,add_special_tokens=False).input_ids
            used=ids if len(ids)<=2048 else ids[:1024]+ids[-1024:]
            with torch.no_grad():
                agent.set_source(None)
                output=model(input_ids=torch.tensor([used],device=device),output_hidden_states=True,use_cache=False)
                hidden=output.hidden_states[-1]
                cache[key]=dict(contextual=hidden[:,-1].float().detach().cpu(),pair_contextual=hidden.float().mean(1).detach().cpu())
            token_bindings.append(dict(text_sha256=key,original_tokens=len(ids),used_token_ids=used))
        return key
    for i,r in enumerate(data['rows']):
        ids=tok.apply_chat_template(r['messages'],tokenize=True,add_generation_prompt=True)
        assert len(ids)<=4096,(r['id'],len(ids))
        answer=tok(r['answer']+tok.eos_token,add_special_tokens=False).input_ids
        prepared.append(dict(prefix=ids,answer=answer,source=encode(r['source']),query=encode(r['query']),binding=r['input_content_sha256']))
        if (i+1)%10==0:
            print(f'encoding {i+1}/{len(data["rows"])} unique={len(cache)}',flush=True)
            write(OUT/'status.json',dict(stage='encoding',rows=i+1,total=len(data['rows'])))
    torch.save(dict(fields=cache,prepared=prepared),OUT/'cache.pt')
    write(OUT/'token_bindings.json',token_bindings)
    opt_h=torch.optim.AdamW(list(hparams.values()),lr=1e-4,weight_decay=0)
    opt_s=torch.optim.AdamW(sa+sb,lr=1e-4,weight_decay=0)
    history=[]
    def save(step):
        torch.save(dict(step=step,manifest_sha256=filehash(OUT/'manifest.json'),hyper={n:p.detach().cpu() for n,p in hparams.items()},static_a=[p.detach().cpu() for p in sa],static_b=[p.detach().cpu() for p in sb],optimizer_h=opt_h.state_dict(),optimizer_s=opt_s.state_dict()),OUT/f'checkpoint_{step:04d}.pt')
        write(OUT/'training.json',history)
    started=time.time()
    for step,idx in enumerate(data['schedule'],1):
        r=prepared[idx];losses={}
        ids=torch.tensor([r['prefix']+r['answer']],device=device)
        labels=torch.tensor([[-100]*len(r['prefix'])+r['answer']],device=device)
        for arm in ('hyper','lora'):
            if arm=='hyper':
                for a,p in zip(agent.adapters,ha):a.a=p
                agent.set_source({k:v.to(device) for k,v in cache[r['source']].items()},target_fields={k:v.to(device) for k,v in cache[r['query']].items()})
                opt=opt_h;params=list(hparams.values())
            else:
                for a,p,b in zip(agent.adapters,sa,sb):a.a=p;a.b=b*1.0
                opt=opt_s;params=sa+sb
            opt.zero_grad(set_to_none=True)
            loss=model(input_ids=ids,labels=labels,use_cache=False).loss
            assert torch.isfinite(loss), (step,arm)
            loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(params,1.0,error_if_nonfinite=True)
            opt.step();losses[arm]=dict(loss=float(loss.detach()),grad_norm=float(norm))
        history.append(dict(step=step,row_index=idx,domain=data['rows'][idx]['domain'],input_content_sha256=r['binding'],**losses))
        if step==1 or step%10==0:
            print(json.dumps(history[-1]),flush=True)
            write(OUT/'status.json',dict(stage='training',updates_per_arm=step,total_per_arm=600,elapsed_seconds=time.time()-started))
        if step%100==0:save(step)
    write(OUT/'complete.json',dict(training_complete=True,updates_per_arm=600,domain_updates=dict(Counter(x['domain'] for x in history)),hyper_parameters=sum(p.numel() for p in hparams.values()),lora_parameters=sum(p.numel() for p in sa+sb),checkpoint_sha256=filehash(OUT/'checkpoint_0600.pt'),evaluation_complete=False))
    write(OUT/'status.json',dict(stage='training_complete',updates_per_arm=600,evaluation_complete=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','train']);p.add_argument('--device',default='cuda:2');a=p.parse_args()
    try:
        prepare() if a.stage=='prepare' else run(a.device)
    except Exception as e:
        write(OUT/'failure.json',dict(stage=a.stage,type=type(e).__name__,message=str(e)));raise
