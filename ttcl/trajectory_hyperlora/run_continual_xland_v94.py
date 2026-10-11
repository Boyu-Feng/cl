"""Self-trajectory -> persistent B pilot, followed by matched and on-policy tests.

Offline expert labels only train the update rule and the direct-LoRA prior.
Test adaptation sees only completed own trajectories and official rewards.
"""
from __future__ import annotations
import argparse,hashlib,json,random,time
from pathlib import Path
import torch
from torch.nn import functional as F
from transformers import AutoTokenizer,AutoModelForCausalLM
from ttcl.trajectory_hyperlora.joint_domains_v93 import ROOT,MODEL,read,write,digest,filehash,ordered
from ttcl.trajectory_hyperlora.direct_composition_pilot import DirectRelationHyperLoRA
from ttcl.trajectory_hyperlora.train_xland_raw_hyperlora import checked_query
from ttcl.trajectory_hyperlora.train_xland_official_history_hyperlora import question
from ttcl.trajectory_hyperlora.evaluate_xland_official_live import Worker,state
from ttcl.trajectory_hyperlora.continual_update_v94 import UpdateRule,effective_norm,bound,trajectory_text,positive_targets
OUT=ROOT/'results/trajectory_hyperlora/continual_xland_v94_20261010'
ANNOT=ROOT/'data/annotations/xland_trivial_official_reviewed_128_v1_20261006.json'
ARMS=('base','online_zero','online_prior','hyper','reset','shifted')
MILESTONES=(0,1,3,7)

def tensor_hash(xs):
    h=hashlib.sha256()
    for x in xs:h.update(x.detach().float().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'plan.json').exists():raise RuntimeError('Refusing plan overwrite')
    d=read(ANNOT);train=ordered(d['split']['train'],lambda t:t['task_id'])[:8]
    test=ordered(d['split']['test'],lambda t:t['task_id'])[:8]
    assert not ({t['ruleset_id'] for t in train}&{t['ruleset_id'] for t in test})
    for t in train:
        for q in t['queries']:
            checked_query(q);assert q['target_history_id']!=t['source_history_id']
    # Evaluation plans deliberately omit official expert-action queries.
    test=[{k:t[k] for k in ('task_id','ruleset_id')} for t in test]
    code=['run_continual_xland_v94.py','continual_update_v94.py','direct_composition_pilot.py','evaluate_xland_official_live.py','xland_official_live_worker.py','train_xland_official_history_hyperlora.py','train_xland_raw_hyperlora.py','joint_domains_v93.py']
    plan=dict(train=train,test=test,rank=4,layers=2,seed=94,train_collection_episodes=4,test_shared_episodes=7,max_steps=100,
      collection_epsilon=0.15,evaluation_epsilon=0.0,action_guard='exclude already tried actions in identical observation, fallback all six',
      meta_updates=96,unroll_length=4,supervised_losses_per_update=5,loss='mean four future expert action CEs + .25 replay CE on first query',
      offline_lr=1e-4,online_lr=1e-3,online_steps_per_episode=8,online_replay='last four own episodes; reward-confirmed transition and up to three preceding actions; no labels for failure-only data',
      cumulative_delta_limit=0.05,total_limit=0.15,limits_relative_to='frozen dense down_proj Frobenius norm of effective BA/rank',
      milestone_attempt_numbers=[1,2,4,8],matched_arms=ARMS,onpolicy_arms=['hyper','online_zero','online_prior'],
      own_training_sources=True,offline_supervision='reviewed official expert queries in disjoint histories of the same training ruleset',
      limitations=['small historically exposed test rulesets; exploratory','H uses offline expert supervision; online_prior has identical offline labels and fixed A, online_zero is a cold-start reference','online LoRA uses a declared reward-confirmed imitation surrogate; not all possible test-time gradient algorithms','B reset between independent rulesets; first study within-ruleset accumulation across layouts, not lifelong cross-ruleset retention','failure-retry teacher not included in this first stage','shared-source comparison replays a precollected frozen-base stream; onpolicy evaluation is separate'],
      annotation_sha256=filehash(ANNOT),benchmark_sha256=filehash(ROOT/'data/xland_minigrid/trivial_1m'),
      code_freeze={f:filehash(ROOT/'ttcl/trajectory_hyperlora'/f) for f in code})
    write(OUT/'plan.json',plan);print('prepared',len(train),len(test),flush=True)

class Runner:
    def __init__(self,device):
        self.plan=read(OUT/'plan.json');self.device=device
        for f,h in self.plan['code_freeze'].items():assert filehash(ROOT/'ttcl/trajectory_hyperlora'/f)==h
        assert filehash(ANNOT)==self.plan['annotation_sha256']
        assert filehash(ROOT/'data/xland_minigrid/trivial_1m')==self.plan['benchmark_sha256']
        torch.manual_seed(94);torch.cuda.manual_seed_all(94)
        self.tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
        base=AutoModelForCausalLM.from_pretrained(MODEL,torch_dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True).to(device)
        self.agent=DirectRelationHyperLoRA(base,rank=4,layers=2,encoder_kind='contextual').to(device)
        self.agent.eval()
        for p in self.agent.parameters():p.requires_grad_(False)
        self.aa=[a.a.detach().clone() for a in self.agent.adapters]
        self.norms=[float(a.base.weight.float().norm()) for a in self.agent.adapters]
        self.shapes=[(a.base.out_features,4) for a in self.agent.adapters]
        self.rule=UpdateRule(base.config.hidden_size,self.shapes).to(device)
        self.cache={};self.bindings={}
        self.choice=[self.tok(str(i),add_special_tokens=False).input_ids for i in range(6)]
        assert all(len(t)==1 for t in self.choice);self.choice=[t[0] for t in self.choice]
        self.worker=Worker(ROOT/'ttcl/.runtime/xland_env/bin/python',ROOT/'data/xland_minigrid','trivial-1m','XLand-MiniGrid-R1-9x9',ROOT/'data/xland_minigrid/trivial_1m')
        torch.save(dict(a=[a.cpu() for a in self.aa],norms=self.norms,initial_rule=self.rule.state_dict()),OUT/'initial.pt')
    def zero(self):return [torch.zeros(1,*s,device=self.device) for s in self.shapes]
    def mount(self,bb):
        for a,b in zip(self.agent.adapters,bb,strict=True):a.b=b
    def ids(self,observation):
        q=question(dict(target_initial_state=state(observation)),6)
        ids=self.tok.apply_chat_template([dict(role='user',content=q)],tokenize=True,add_generation_prompt=True)
        assert len(ids)<=1024
        return torch.tensor([ids],device=self.device)
    def logits(self,observation,bb):
        self.mount(bb);return self.agent.model(input_ids=self.ids(observation),use_cache=False).logits[0,-1,self.choice].float()
    def fields(self,ep):
        key=ep['content_sha256']
        assert digest({k:v for k,v in ep.items() if k!='content_sha256'})==key
        if key not in self.cache:
            text=trajectory_text(ep);ids=self.tok(text,add_special_tokens=False).input_ids
            used=ids if len(ids)<=2048 else ids[:1024]+ids[-1024:]
            self.agent.set_source(None)
            with torch.no_grad():
                h=self.agent.model(input_ids=torch.tensor([used],device=self.device),output_hidden_states=True,use_cache=False).hidden_states[-1]
                self.cache[key]=dict(last=h[:,-1].float().detach(),mean=h.float().mean(1).detach())
            self.bindings[key]=dict(text_sha256=digest(text),original_tokens=len(ids),used_token_ids=used)
            write(OUT/'source_bindings.json',self.bindings)
        return self.cache[key]
    def rollout(self,task,seed,bb,epsilon=0.0):
        started=time.time();rng=random.Random(seed)
        current=self.worker.send(dict(command='reset',ruleset_id=task['ruleset_id'],seed=seed));initial=current
        assert current['num_actions']==6
        seen={};traj=[]
        for step in range(min(self.plan['max_steps'],initial['max_steps'])):
            with torch.no_grad():logits=self.logits(current['observation'],bb)
            key=digest(current['observation']);used=seen.setdefault(key,set())
            allowed=[a for a in range(6) if a not in used] or list(range(6))
            a=rng.choice(allowed) if rng.random()<epsilon else max(allowed,key=lambda a:float(logits[a]))
            used.add(a);after=self.worker.send(dict(command='step',action=a))
            traj.append(dict(observation=current['observation'],action=a,next_observation=after['observation'],reward=after['reward'],done=after['done']))
            current=after
            if after['done']:break
        # Fresh reset replay verifies exact public feedback and official scoring.
        replay=self.worker.send(dict(command='reset',ruleset_id=task['ruleset_id'],seed=seed));assert replay==initial
        for s in traj:
            assert replay['observation']==s['observation']
            replay=self.worker.send(dict(command='step',action=s['action']))
            assert replay['observation']==s['next_observation'] and replay['reward']==s['reward'] and replay['done']==s['done']
        ep=dict(task_id=task['task_id'],ruleset_id=task['ruleset_id'],seed=seed,trajectory=traj,
                initial=initial,return_=sum(s['reward'] for s in traj),positive=any(s['reward']>0 for s in traj),b_before_sha256=tensor_hash(bb),environment_replay_passed=True)
        ep['return']=ep.pop('return_');ep['content_sha256']=digest(ep)
        return ep,time.time()-started
    def status(self,stage,**kw):
        write(OUT/'status.json',dict(stage=stage,**kw));print(json.dumps(dict(stage=stage,**kw)),flush=True)
    def collect(self):
        output=dict(train=[],test=[],complete=False)
        for split,num,offset in [('train',4,940000),('test',7,1940000)]:
            for ti,task in enumerate(self.plan[split]):
                eps=[]
                for j in range(num):
                    seed=offset+int(task['ruleset_id'])*1009+j
                    ep,elapsed=self.rollout(task,seed,self.zero(),self.plan['collection_epsilon'])
                    eps.append(ep)
                    write(OUT/f'collection_{split}_{ti}_{j}.json',ep)
                    self.status('collecting',split=split,task=ti+1,tasks=8,episode=j+1,episodes=num,positive=ep['positive'])
                output[split].append(eps);write(OUT/'collection.json',output)
        output['complete']=True;write(OUT/'collection.json',output)
        return output
    def train(self,collection):
        targets=[]
        for ti,t in enumerate(self.plan['train']):
            qs=[]
            for q in t['queries']:
                content,label=checked_query(q)
                row=dict(observation=content['target_initial_state']['observation'],action=label,parent_sha256=q['input_sha256'],source_episode_hashes=[e['content_sha256'] for e in collection['train'][ti]])
                row['input_content_sha256']=digest(row);row['reviewed_target']=True;qs.append(row)
            targets.append(qs)
        write(OUT/'training_targets.json',dict(rows=targets,scope='New bindings join replay-verified own sources to unchanged content-verified official query labels; no new guessed failure action labels'))
        for group in collection['train']:
            for ep in group:self.fields(ep)
        opt=torch.optim.AdamW(self.rule.parameters(),lr=1e-4,weight_decay=0)
        prior=[torch.nn.Parameter(b) for b in self.zero()]
        optprior=torch.optim.AdamW(prior,lr=1e-4,weight_decay=0)
        rng=random.Random(94);history=[]
        for it in range(96):
            ti=it%8;episodes=collection['train'][ti];qs=rng.sample(targets[ti],4)
            opt.zero_grad(set_to_none=True);bb=self.zero();losses=[];gates=[]
            for ep,q in zip(episodes,qs,strict=True):
                bb,gate=self.rule(self.fields(ep),bb,self.aa,self.norms)
                losses.append(F.cross_entropy(self.logits(q['observation'],bb).unsqueeze(0),torch.tensor([q['action']],device=self.device)))
                gates.append(float(gate.detach()))
            replayloss=F.cross_entropy(self.logits(qs[0]['observation'],bb).unsqueeze(0),torch.tensor([qs[0]['action']],device=self.device))
            # Small relative weight penalty regularizes the accumulated effective update.
            penalty=sum(((b.squeeze(0).T@b.squeeze(0))*(a@a.T)).sum()/16/(n*n) for b,a,n in zip(bb,self.aa,self.norms))
            total=torch.stack(losses).mean()+.25*replayloss+1e-3*penalty
            assert torch.isfinite(total);total.backward();torch.nn.utils.clip_grad_norm_(self.rule.parameters(),1,error_if_nonfinite=True);opt.step()
            optprior.zero_grad(set_to_none=True);pl=[]
            for q in qs+[qs[0]]:
                pl.append(F.cross_entropy(self.logits(q['observation'],prior).unsqueeze(0),torch.tensor([q['action']],device=self.device)))
            pt=torch.stack(pl).sum()*.25
            assert torch.isfinite(pt);pt.backward();torch.nn.utils.clip_grad_norm_(prior,1,error_if_nonfinite=True);optprior.step()
            with torch.no_grad():
                for b,a,n in zip(prior,self.aa,self.norms):b.copy_(bound(b,a,.15*n))
            history.append(dict(update=it+1,task=ti,query_hashes=[q['input_content_sha256'] for q in qs],source_hashes=[e['content_sha256'] for e in episodes],hyper_loss=float(total.detach()),prior_loss=float(pt.detach()),gates=gates))
            write(OUT/'training.json',history)
            if (it+1)%8==0:self.status('training',updates=it+1,total=96)
        self.rule.eval()
        for p in self.rule.parameters():p.requires_grad_(False)
        prior=[b.detach().clone() for b in prior]
        torch.save(dict(rule=self.rule.state_dict(),a=[a.cpu() for a in self.aa],prior=[b.cpu() for b in prior],plan_sha256=filehash(OUT/'plan.json'),training_targets_sha256=filehash(OUT/'training_targets.json'),training_sha256=filehash(OUT/'training.json')),OUT/'trained.pt')
        return prior
    def online(self,bb,episodes,optstate=None):
        start=time.time();rows=positive_targets(episodes[-4:]);before=tensor_hash(bb)
        if not rows:return bb,optstate,dict(updates=0,reason='no reward-confirmed targets',before=before,after=before,seconds=time.time()-start)
        params=[torch.nn.Parameter(b.detach().clone()) for b in bb]
        opt=torch.optim.AdamW(params,lr=1e-3,weight_decay=0)
        if optstate is not None:opt.load_state_dict(optstate)
        rng=random.Random(9400+len(episodes));chosen=[]
        for _ in range(8):
            q=rng.choice(rows);opt.zero_grad(set_to_none=True)
            loss=F.cross_entropy(self.logits(q['observation'],params).unsqueeze(0),torch.tensor([q['action']],device=self.device))
            assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(params,1,error_if_nonfinite=True);opt.step()
            with torch.no_grad():
                for b,a,n in zip(params,self.aa,self.norms):b.copy_(bound(b,a,.15*n))
            chosen.append(dict(**q,input_content_sha256=digest(q),reviewed_target=True,review_basis='own environment-replayed rewarding transition or preceding action, not guaranteed optimal'))
        result=[b.detach() for b in params]
        return result,opt.state_dict(),dict(updates=8,targets=chosen,before=before,after=tensor_hash(result),seconds=time.time()-start)
    def hyper_update(self,bb,ep):
        start=time.time();before=tensor_hash(bb)
        with torch.no_grad():out,gate=self.rule(self.fields(ep),bb,self.aa,self.norms)
        result=[b.detach() for b in out]
        return result,dict(updates=1,source_hash=ep['content_sha256'],before=before,after=tensor_hash(result),gate=float(gate),seconds=time.time()-start)
    def evaluate(self,collection,prior):
        report=dict(complete=False,shared=[],onpolicy=[],updates=[],failures=[],plan_sha256=filehash(OUT/'plan.json'),trained_sha256=filehash(OUT/'trained.pt'))
        def save(stage):
            write(OUT/'evaluation.json',report)
            self.status(stage,shared_completed=len(report['shared']),shared_total=192,onpolicy_completed=len(report['onpolicy']),onpolicy_total=192)
        for ti,t in enumerate(self.plan['test']):
            sources=collection['test'][ti]
            bs={a:[b.clone() for b in prior] if a=='online_prior' else self.zero() for a in ARMS};os={a:None for a in ARMS}
            for k in range(8):
                if k in MILESTONES:
                    for arm in ARMS:
                        ep,elapsed=self.rollout(t,2940000+int(t['ruleset_id'])*1009+k,bs[arm])
                        report['shared'].append(dict(task_id=t['task_id'],arm=arm,observed_episodes=k,attempt_number=k+1,episode=ep,seconds=elapsed))
                        save('shared_evaluation')
                if k==7:break
                for arm in ARMS:
                    if arm=='base':continue
                    if arm.startswith('online'):
                        bs[arm],os[arm],u=self.online(bs[arm],sources[:k+1],os[arm])
                    else:
                        source=collection['test'][(ti+1)%8][k] if arm=='shifted' else sources[k]
                        bs[arm],u=self.hyper_update(self.zero() if arm=='reset' else bs[arm],source)
                    report['updates'].append(dict(stage='shared',task_id=t['task_id'],arm=arm,after_episode=k+1,**u))
                save('shared_evaluation')
        # Genuine on-policy attempts: each arm updates only from its own past.
        for ti,t in enumerate(self.plan['test']):
            for arm in ('hyper','online_zero','online_prior'):
                bb=[b.clone() for b in prior] if arm=='online_prior' else self.zero();episodes=[];os=None
                for k in range(8):
                    ep,elapsed=self.rollout(t,3940000+int(t['ruleset_id'])*1009+k,bb)
                    episodes.append(ep)
                    report['onpolicy'].append(dict(task_id=t['task_id'],arm=arm,attempt_number=k+1,episode=ep,seconds=elapsed))
                    if k<7:
                        if arm=='hyper':bb,u=self.hyper_update(bb,ep)
                        else:bb,os,u=self.online(bb,episodes,os)
                        report['updates'].append(dict(stage='onpolicy',task_id=t['task_id'],arm=arm,after_episode=k+1,**u))
                    save('onpolicy_evaluation')
        assert len(report['shared'])==192 and len(report['onpolicy'])==192
        assert tensor_hash([a.a for a in self.agent.adapters])==tensor_hash(self.aa)
        report['complete']=True;write(OUT/'evaluation.json',report)
        summary={}
        for stage in ('shared','onpolicy'):
            summary[stage]={}
            for arm in (ARMS if stage=='shared' else ('hyper','online_zero','online_prior')):
                summary[stage][arm]={str(k):dict(n=sum(r['arm']==arm and r['attempt_number']==k for r in report[stage]),positive=sum(r['arm']==arm and r['attempt_number']==k and r['episode']['positive'] for r in report[stage])) for k in (1,2,4,8)}
        write(OUT/'summary.json',summary);self.status('complete',summary=summary)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','run']);p.add_argument('--device',default='cuda:2');args=p.parse_args()
    runner=None
    try:
        if args.stage=='prepare':prepare()
        else:
            if (OUT/'initial.pt').exists():raise RuntimeError('Existing run; do not overwrite artifacts')
            runner=Runner(args.device);collection=runner.collect();prior=runner.train(collection);runner.evaluate(collection,prior)
    except Exception as e:
        write(OUT/'failure.json',dict(type=type(e).__name__,message=str(e)));raise
    finally:
        if runner is not None:runner.worker.close()
