"""Frozen five-arm evaluation of the preselected joint-domain pilot."""
from __future__ import annotations
import argparse,json,time
from pathlib import Path
import torch
from ttcl.trajectory_hyperlora.joint_domains_v93 import (ROOT,OUT,MODEL,read,write,digest,filehash,select_source,question,parse_letter,task_context_text,DirectRelationHyperLoRA,AutoTokenizer,AutoModelForCausalLM)
from ttcl.trajectory_hyperlora.alfworld_zero_shot_probe import run_episode
from ttcl.trajectory_hyperlora.audit_my_online_update_v28_reward import replay
from ttcl.trajectory_hyperlora.evaluate_xland_official_live import Worker,state
ARMS=('base','lora','hyper','empty','shifted')

def main(device):
    data=read(OUT/'dataset.json');manifest=read(OUT/'manifest.json');complete=read(OUT/'complete.json')
    assert digest(data)==manifest['dataset_sha256']
    assert complete['training_complete'] and complete['updates_per_arm']==600
    checkpoint=OUT/'checkpoint_0600.pt';assert filehash(checkpoint)==complete['checkpoint_sha256']
    for f,h in read(OUT/'code_freeze.json').items():assert filehash(ROOT/'ttcl/trajectory_hyperlora'/f)==h
    target=OUT/'evaluation.json'
    if target.exists():raise RuntimeError('Refusing to overwrite evaluation; preserve failures')
    review=dict(arms=ARMS,dataset_sha256=digest(data),checkpoint_sha256=complete['checkpoint_sha256'],training_updates=0,
        counts={d:len(x) for d,x in data['evaluation'].items()},
        alfworld=dict(max_steps=30,max_new_tokens=64,history_turns=2,legal_actions=True,loop_guard=2,query='initial plus current public feedback'),
        xland=dict(max_steps=200,seed='930000+ruleset_id*1009',action_rule='greedy among six legal actions, excluding tried actions in identical observation until exhausted',epsilon=0,scoring='official return and any positive reward; not expert-action accuracy'),
        hle=dict(max_new_tokens=256,prompt='Answer-only, identical to joint training',parser='original strict Answer: single letter',source='training-only bank, lexical selector'),
        empty='Zero both source feature vectors, retain actual query features and learned H biases/A; inference ablation, not separately trained',
        shifted='Next source in fixed same-domain bank; XLand next selected test ruleset history; no target outcomes used',
        audit='ALF feedback/action/reward replay; XLand seed/action/observation/reward replay; HLE deterministic parsing; no independent regeneration of neural outputs',
        dependencies={f:filehash(ROOT/'ttcl/trajectory_hyperlora'/f) for f in ['evaluate_joint_domains_v93.py','alfworld_zero_shot_probe.py','evaluate_xland_official_live.py','xland_official_live_worker.py','audit_my_online_update_v28_reward.py']},
        game_hashes={p:filehash(p) for p in data['evaluation']['alfworld']},benchmark_sha256=filehash(ROOT/'data/xland_minigrid/trivial_1m'))
    write(OUT/'evaluation_review.json',review)
    report=dict(complete=False,review_sha256=filehash(OUT/'evaluation_review.json'),rows=[],failures=[],summary={})
    write(target,report)
    tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(MODEL,torch_dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True).to(device)
    agent=DirectRelationHyperLoRA(model,rank=4,layers=2,encoder_kind='contextual',task_conditioned=True).to(device)
    saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
    params=dict(agent.named_parameters())
    with torch.no_grad():
        for n,v in saved['hyper'].items():params[n].copy_(v.to(device))
    ha=[a.a.detach().clone() for a in agent.adapters]
    sa=[x.to(device) for x in saved['static_a']];sb=[x.to(device) for x in saved['static_b']]
    for p in agent.parameters():p.requires_grad_(False)
    agent.eval();agent.task_context_scope='initial'
    cache={};bindings={}
    def encode(text):
        key=digest(text)
        if key not in cache:
            ids=tok(text,add_special_tokens=False).input_ids;used=ids if len(ids)<=2048 else ids[:1024]+ids[-1024:]
            agent.set_source(None)
            with torch.no_grad():
                h=model(input_ids=torch.tensor([used],device=device),output_hidden_states=True,use_cache=False).hidden_states[-1]
                cache[key]=dict(contextual=h[:,-1].float().detach(),pair_contextual=h.float().mean(1).detach())
            bindings[key]=dict(original_tokens=len(ids),used_token_ids=used)
        return cache[key]
    def mount(arm,source,query,trace):
        with torch.no_grad():
            for a,p in zip(agent.adapters,sa if arm=='lora' else ha):a.a.copy_(p)
            if arm=='base':agent.set_source(None)
            elif arm=='lora':
                for a,b in zip(agent.adapters,sb):a.b=b
            else:
                sf=encode(source['text']);qf=encode(query)
                if arm=='empty':sf={k:torch.zeros_like(v) for k,v in sf.items()}
                agent.set_source(sf,target_fields=qf)
            trace.append(dict(arm=arm,source_id=source['id'],source_sha256=digest(source['text']),query_sha256=digest(query),b_norms=[float(a.b.norm()) if a.b is not None else 0 for a in agent.adapters]))
    def shifted(source,bank):return bank[(next(i for i,x in enumerate(bank) if x['id']==source['id'])+1)%len(bank)]
    def record(domain,ident,arm,result,trace):
        report['rows'].append(dict(domain=domain,id=ident,arm=arm,result=result,parameter_reads=trace))
        report['summary']={d:{a:dict(completed=sum(r['domain']==d and r['arm']==a for r in report['rows']),success=sum(r['domain']==d and r['arm']==a and r['result']['success'] for r in report['rows'])) for a in ARMS} for d in data['evaluation']}
        write(target,report);write(OUT/'evaluation_token_bindings.json',bindings)
        write(OUT/'evaluation_status.json',dict(stage='evaluating',domain=domain,completed_runs=len(report['rows']),total_runs=220,summary=report['summary']))
        print(json.dumps(dict(domain=domain,id=ident,arm=arm,success=result['success'],completed=len(report['rows']))),flush=True)
    # Labels are used only after response generation, never by source selection or mount.
    keys={x['id']:x for x in read(ROOT/'data/hle/failure_retry_pilot_20261010/grading_keys.json')}
    worker=None
    try:
        for r in data['evaluation']['hle']:
            source=select_source(r['question'],data['banks']['hle'],r['id'])
            messages=[dict(role='system',content='Solve the question. Return only one line: Answer: X, where X is the option letter.'),dict(role='user',content=r['question'])]
            ids=tok.apply_chat_template(messages,tokenize=True,add_generation_prompt=True);assert len(ids)<=4096
            for arm in ARMS:
                trace=[];src=shifted(source,data['banks']['hle']) if arm=='shifted' else source
                mount(arm,src,r['question'],trace)
                inp=torch.tensor([ids],device=device)
                with torch.no_grad():output=model.generate(input_ids=inp,attention_mask=torch.ones_like(inp),do_sample=False,max_new_tokens=256,pad_token_id=tok.eos_token_id)
                response=tok.decode(output[0,len(ids):],skip_special_tokens=True);letter,status=parse_letter(response)
                assert keys[r['id']]['input_content_sha256']==r['input_content_sha256']
                record('hle',r['id'],arm,dict(response=response,parsed=letter,parse_status=status,success=letter==keys[r['id']]['answer'].strip(),prompt_sha256=digest(ids),generated_tokens=len(output[0])-len(ids)),trace)
        for game in data['evaluation']['alfworld']:
            for arm in ARMS:
                trace=[]
                def provider(initial,current,available):
                    src=select_source(initial,data['banks']['alfworld'],game)
                    if arm=='shifted':src=shifted(src,data['banks']['alfworld'])
                    mount(arm,src,task_context_text(initial,current),trace)
                    return ''
                episode=run_episode(agent,tok,Path(game),{},adapter=False,device=device,max_steps=30,max_new_tokens=64,constrain_actions=True,actor_history_turns=2,loop_guard_max=2,evidence_provider=provider)
                if episode['status']!='complete':raise RuntimeError(episode)
                replay(Path(game),episode)
                record('alfworld',game,arm,dict(success=episode['reward']==1,episode=episode,environment_replay_passed=True),trace)
        tasks=data['evaluation']['xland'];bank=[]
        for t in tasks:
            c=t['queries'][0]['model_input']
            bank.append(dict(id=str(t['task_id']),text=json.dumps(c['source_episodes'],ensure_ascii=False,separators=(',',':'))))
        worker=Worker(ROOT/'ttcl/.runtime/xland_env/bin/python',ROOT/'data/xland_minigrid','trivial-1m','XLand-MiniGrid-R1-9x9',ROOT/'data/xland_minigrid/trivial_1m')
        choice=[tok(str(i),add_special_tokens=False).input_ids for i in range(6)];assert all(len(x)==1 for x in choice)
        choice=[x[0] for x in choice]
        for ti,t in enumerate(tasks):
            seed=930000+int(t['ruleset_id'])*1009
            for arm in ARMS:
                current=worker.send(dict(command='reset',ruleset_id=t['ruleset_id'],seed=seed));assert current['num_actions']==6
                initial=current;trace=[];trajectory=[];seen={};src=bank[(ti+1)%len(bank)] if arm=='shifted' else bank[ti]
                for step in range(min(200,current['max_steps'])):
                    query=question(dict(target_initial_state=state(current['observation'])),6)
                    ids=tok.apply_chat_template([dict(role='user',content=query)],tokenize=True,add_generation_prompt=True)
                    mount(arm,src,query,trace)
                    with torch.no_grad():logits=model(input_ids=torch.tensor([ids],device=device),use_cache=False).logits[0,-1,choice].float()
                    key=digest(current['observation']);used=seen.setdefault(key,set());allowed=[a for a in range(6) if a not in used] or list(range(6))
                    action=max(allowed,key=lambda a:float(logits[a]));used.add(action)
                    after=worker.send(dict(command='step',action=action))
                    trajectory.append(dict(observation=current['observation'],action=action,next_observation=after['observation'],reward=after['reward'],done=after['done'],prompt_sha256=digest(ids),logits=logits.tolist()))
                    current={**after,'max_steps':initial['max_steps']}
                    if after['done']:break
                replayed=worker.send(dict(command='reset',ruleset_id=t['ruleset_id'],seed=seed));assert replayed==initial
                for s in trajectory:
                    assert replayed['observation']==s['observation']
                    replayed=worker.send(dict(command='step',action=s['action']))
                    assert replayed['observation']==s['next_observation'] and replayed['reward']==s['reward'] and replayed['done']==s['done']
                record('xland',str(t['task_id']),arm,dict(success=any(s['reward']>0 for s in trajectory),official_return=sum(s['reward'] for s in trajectory),seed=seed,ruleset_id=t['ruleset_id'],trajectory=trajectory,environment_replay_passed=True),trace)
        assert len(report['rows'])==220
        report['complete']=True;write(target,report)
        write(OUT/'evaluation_status.json',dict(stage='complete',completed_runs=220,summary=report['summary']))
    except Exception as e:
        report['failures'].append(dict(type=type(e).__name__,message=str(e)));write(target,report)
        write(OUT/'evaluation_status.json',dict(stage='failed',completed_runs=len(report['rows']),error=str(e)));raise
    finally:
        if worker is not None:worker.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--device',default='cuda:6');args=p.parse_args();main(args.device)
