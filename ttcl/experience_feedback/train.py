from __future__ import annotations
import json
from pathlib import Path
import time
from ttcl.experience_evolution.core import read, save, seed, append, writer_messages
from ttcl.experience_evolution.run import check_pair
from .protocol import training_label

def actor_job(game,memory,random_seed,path):
    return {'game':game,'memory':memory,'seed':random_seed,'output':str(path)}

def train(root,arm):
    from ttcl.experience_evolution.environment import Actor
    from ttcl.experience_evolution.writer import Writer
    plan=read(root/'training_plan.json')
    config=plan['training_arms'][arm]
    plan['train_seed']=config['seed']
    sequences=read(root/plan['curriculum'])['training']
    out=root/'training'/arm
    out.mkdir(parents=True,exist_ok=True)
    if (out/'status.json').exists() and read(out/'status.json')['phase']=='complete':return
    if (out/'training.jsonl').exists():
        raise RuntimeError('Partial training needs explicit checkpoint recovery; refusing silent restart')
    writer=Writer(plan,adapter=Path(plan['initialization']),train=True)
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
                advantage,delta=training_label(pairs,config['objective'])
                sample.update(advantage=advantage,delta=delta)
                label={'objective':config['objective'],'advantage':advantage,'delta':delta,'paired_deltas':deltas,
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
        metric.update(batch=batch_i,examples=len(samples),paired_samples=k_samples,objective=config['objective'],time=time.time())
        totals.append(metric);append(out/'training.jsonl',metric)
        print(json.dumps(metric),flush=True)
        if (batch_i+1)%4==0:
            writer.save(out/f'checkpoint_{batch_i+1:03}')
    audit=writer.save(out)
    save(out/'status.json',dict(phase='complete',batches=len(totals),writer_actions=sum(x['examples'] for x in totals),
         positive=sum(x['positive'] for x in totals),negative=sum(x['negative'] for x in totals),
         zero=sum(x['zero'] for x in totals),audit=audit))

