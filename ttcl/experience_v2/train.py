from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
import statistics
import time
import traceback

from .common import OLD, read, save, seed
from ttcl.experience_evolution.core import FAMILIES, append, writer_messages
from ttcl.experience_evolution.run import check_pair


def choose_sequences(candidates, scores, random_seed=923):
    rng=random.Random(random_seed)
    sequences=[]
    selection=[]
    for family in FAMILIES:
        items=[x for x in candidates if x['family']==family]
        means={x['game']:statistics.mean(scores[x['game']]) for x in items}
        # Sources include successful trajectories; targets prioritize intermediate difficulty.
        sources=sorted(items,key=lambda x:(-means[x['game']],seed(random_seed,x['game'])))
        targets=sorted(items,key=lambda x:(abs(means[x['game']]-.5),seed(random_seed,x['game'])))
        good=[x for x in sources if means[x['game']]>0]
        target_pool=targets[:max(8,len(targets)//2)]
        for j in range(8):
            first=rng.choice(good or sources)
            pool=[x for x in (items if j%4==3 else target_pool) if x['game']!=first['game']]
            chosen=[first]+rng.sample(pool,3)
            sequences.append({'id':f'curriculum:{family}:{j}', 'family':family,
                              'games':[x['game'] for x in chosen]})
        selection.append({'family':family,'screened':len(items),'source_successes':len(good),
                          'target_success_rates':[means[x['game']] for x in target_pool]})
    rng.shuffle(sequences)
    return sequences,selection


def actor_job(game,memory,random_seed,path):
    return {'game':game,'memory':memory,'seed':random_seed,'output':str(path)}


def screen(root):
    from ttcl.experience_evolution.environment import Actor
    plan=read(root/'training_plan.json'); actor=Actor(plan)
    scores=defaultdict(list)
    jobs=[]
    for i,item in enumerate(plan['screening']):
        for k in range(2):
            jobs.append(actor_job(item['game'],'',seed(923,'screen',i,k),
                                 root/'training/screening'/f'{i:03}_{k}'))
    for start in range(0,len(jobs),16):
        batch=jobs[start:start+16]
        missing=[j for j in batch if not (Path(j['output'])/'episode.json').exists()]
        if missing:actor.run_many(missing)
        for j in batch:
            ep=read(Path(j['output'])/'episode.json')
            scores[j['game']].append(ep['reward'])
        save(root/'training/screen_status.json',dict(phase='screening',completed=min(start+16,len(jobs)),total=len(jobs)))
        print(f'SCREEN {min(start+16,len(jobs))}/{len(jobs)}',flush=True)
    sequences,selection=choose_sequences(plan['screening'],scores)
    save(root/'training/curriculum.json',{'training':sequences,'selection':selection,'scores':scores})
    save(root/'training/screen_status.json',dict(phase='complete',completed=len(jobs)))


def train(root,arm):
    from ttcl.experience_evolution.environment import Actor
    from ttcl.experience_evolution.writer import Writer
    plan=read(root/'training_plan.json')
    config=plan['training_arms'][arm]
    plan['train_seed']=config['seed']
    sequences=read(root/'training/curriculum.json')['training']
    out=root/'training'/arm
    out.mkdir(parents=True,exist_ok=True)
    if (out/'status.json').exists() and read(out/'status.json')['phase']=='complete':return
    if (out/'training.jsonl').exists():
        raise RuntimeError('Partial training needs explicit checkpoint recovery; refusing silent restart')
    writer=Writer(plan,adapter=root/'adapters/delta',train=True)
    actor=Actor(plan)
    totals=[]
    k_samples=config['paired_samples']
    for batch_i,start in enumerate(range(0,len(sequences),4)):
        batch=sequences[start:start+4]; directory=out/f'batch_{batch_i:03}'
        states=['']*len(batch)
        save(out/'status.json',dict(phase='rollout',batch=batch_i,total_batches=len(sequences)//4))
        current=actor.run_many([actor_job(seq['games'][0],'',seed(config['seed'],seq['id'],0),
                                         directory/f'seq_{i}/source') for i,seq in enumerate(batch)])
        samples=[]
        for position in range(1,4):
            messages=[writer_messages(m,ep) for m,ep in zip(states,current)]
            generated=writer.generate(messages,seed(config['seed'],'writer',batch_i,position),
                [directory/f'seq_{i}/update_{position}' for i in range(len(batch))])
            jobs=[]
            for i,(seq,sample) in enumerate(zip(batch,generated)):
                for k in range(k_samples):
                    rs=seed(config['seed'],seq['id'],position,k)
                    for branch,memory in [('with',sample['text']),('without','')]:
                        jobs.append(actor_job(seq['games'][position],memory,rs,
                            directory/f'seq_{i}/task_{position}/rep_{k}/{branch}'))
            outcomes=actor.run_many(jobs); current=[]
            for i,sample in enumerate(generated):
                pairs=[outcomes[2*(i*k_samples+k):2*(i*k_samples+k)+2] for k in range(k_samples)]
                for yes,no in pairs:check_pair(yes,no)
                deltas=[yes['reward']-no['reward'] for yes,no in pairs]
                advantage=statistics.mean(deltas)
                sample.update(advantage=advantage,delta=advantage)
                label={'advantage':advantage,'paired_deltas':deltas,
                       'with_rewards':[a['reward'] for a,b in pairs],
                       'without_rewards':[b['reward'] for a,b in pairs],
                       'game':batch[i]['games'][position],'position':position,
                       'continuation_replica':0,'writer_finish_reason':sample['finish_reason']}
                save(directory/f'seq_{i}/update_{position}/label.json',label)
                samples.append(sample)
                states[i]=sample['text']
                # Replica 0 is predeclared; never select the best outcome for continuation.
                current.append(pairs[0][0])
        metric=writer.update(samples)
        metric.update(batch=batch_i,examples=len(samples),paired_samples=k_samples,time=time.time())
        totals.append(metric);append(out/'training.jsonl',metric)
        print(json.dumps(metric),flush=True)
        if (batch_i+1)%4==0:
            writer.save(out/f'checkpoint_{batch_i+1:03}')
    audit=writer.save(out)
    save(out/'status.json',dict(phase='complete',batches=len(totals),writer_actions=sum(x['examples'] for x in totals),
         positive=sum(x['positive'] for x in totals),negative=sum(x['negative'] for x in totals),
         zero=sum(x['zero'] for x in totals),audit=audit))


def evaluate(root,arm):
    from ttcl.experience_evolution.environment import Actor
    from ttcl.experience_evolution.writer import Writer
    plan=read(root/'training_plan.json');out=root/'alfworld_evaluation'/arm
    if (out/'status.json').exists() and read(out/'status.json')['phase']=='complete':return
    if (out/'scores.jsonl').exists():
        raise RuntimeError('Partial ALFWorld evaluation; explicit recovery required')
    adapters={'delta':root/'adapters/delta','absolute':root/'adapters/absolute'}
    adapter=adapters.get(arm,root/'training'/arm/'adapter')
    writer=None if arm=='none' else Writer(plan,adapter=None if arm=='untrained' else adapter,train=False)
    actor=Actor(plan);rows=[]
    for repeat in plan['eval_seeds']:
        for start in range(0,len(plan['evaluation']),4):
            sequences=plan['evaluation'][start:start+4];states=['']*len(sequences)
            current=None
            for position in range(3):
                if position and writer:
                    generated=writer.generate([writer_messages(m,ep) for m,ep in zip(states,current)],
                        seed(repeat,'eval_writer',start,position),
                        [out/str(repeat)/seq['id'].replace(':','_')/f'update_{position}' for seq in sequences])
                    states=[s['text'] for s in generated]
                current=actor.run_many([actor_job(seq['games'][position],states[i],
                    seed(repeat,seq['id'],position),out/str(repeat)/seq['id'].replace(':','_')/f'task_{position}')
                    for i,seq in enumerate(sequences)])
                for seq,ep in zip(sequences,current):
                    row=dict(arm=arm,repeat=repeat,sequence=seq['id'],position=position,
                             reward=ep['reward'],game=ep['game'],steps=ep['steps'])
                    rows.append(row);append(out/'scores.jsonl',row)
                save(out/'status.json',dict(phase='evaluating',completed=len(rows),total=len(plan['evaluation'])*6))
    later=[r['reward'] for r in rows if r['position']>0]
    save(out/'status.json',dict(phase='complete',completed=len(rows),after_first_success=statistics.mean(later)))


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['screen','train','evaluate'])
    p.add_argument('--root',type=Path,required=True);p.add_argument('--arm')
    args=p.parse_args()
    try:
        if args.command=='screen':screen(args.root)
        elif args.command=='train':train(args.root,args.arm)
        else:evaluate(args.root,args.arm)
    except Exception:
        save(args.root/'failures'/f'{args.command}_{args.arm}.json',{'traceback':traceback.format_exc()})
        raise


if __name__=='__main__':main()
