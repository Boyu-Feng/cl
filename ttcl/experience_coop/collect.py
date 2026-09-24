"""Eight on-policy writer actions and independent actor actions per public history."""
from __future__ import annotations

from collections import defaultdict
import math
from pathlib import Path
import statistics

from ttcl.experience_evolution.core import read, save, seed
from ttcl.experience_v2.common import sha_file
from ttcl.experience_lab.collect import messages_for
from .environments import Environments, writer
from .protocol import binding, paired_advantages, validate_sample


def source_shard(root, shard):
    root=Path(root); plan=read(root/'plan.json'); env=Environments(plan, training=True)
    try:
        for h in plan['histories']:
            if h.get('family',h['domain']) != shard: continue
            dest=root/'sources'/h['id']
            if (dest/'input.json').exists(): continue
            warmup=env.run(h['warmup'],'',h['warmup_seed'],dest/'warmup')
            generated=writer(env.client,messages_for('',h['warmup'],warmup),'original_delta',
                dest/'previous.json',seed(92710,h['id'],'previous'),plan)
            previous=generated['raw_response'] if generated['usable'] else ''
            source=env.run(h['source'],previous,h['source_seed'],dest/'source')
            messages=messages_for(previous,h['source'],source)
            save(dest/'input.json',{'history':h, 'previous':previous, 'messages':messages,
                'input_binding':binding(messages), 'source_reward':source['reward'],
                'public_source_sha256':sha_file(dest/'source/completed.json'),
                'future_probe_input_excluded':True})
    finally: env.close()


def freeze_sources(root):
    root=Path(root); plan=read(root/'plan.json'); values=defaultdict(list); hashes={}
    for h in plan['histories']:
        directory=root/'sources'/h['id']; p=directory/'input.json'; row=read(p)
        if binding(row['messages'])!=row['input_binding']:
            raise ValueError('Shared source input binding failed')
        if sha_file(directory/'source/completed.json')!=row['public_source_sha256']:
            raise ValueError('Public source changed')
        reward=row['source_reward']
        if reward is not None:
            if not math.isfinite(reward): raise ValueError('Nonfinite official reward')
            values[h['domain']].append(reward)
        for name in ['input.json','previous.json','source/completed.json','warmup/completed.json']:
            p=directory/name; hashes[str(p)]=sha_file(p)
    domains={h['domain'] for h in plan['histories']}
    if set(values)!=domains: raise ValueError('No finite training-source calibration in a domain')
    scales={d:max(1., math.sqrt(statistics.mean(v*v for v in values[d]))) for d in domains}
    save(root/'source_audit.json',{'hashes':hashes,'reward_scales':scales,
        'scale_counts':{d:len(v) for d,v in values.items()},
        'histories':len(plan['histories']), 'only_training_sources':True,
        'supervision':'Automatic on-policy samples bound to fresh public inputs; rewards verified independently'})
    return scales


def collect_history(root, arm, block_id, history_id):
    root=Path(root); plan=read(root/'plan.json')
    block=next(b for b in plan['blocks'] if b['id']==block_id)
    if history_id not in block['histories']: raise ValueError('History not in declared block')
    h=next(h for h in plan['histories'] if h['id']==history_id)
    directory=root/'training'/arm/block_id; dest=directory/'histories'/history_id
    source_path=root/'sources'/history_id/'input.json'
    source=read(source_path); audit=read(root/'source_audit.json')
    if sha_file(source_path)!=audit['hashes'][str(source_path)]:
        raise ValueError('Frozen source changed')
    env=Environments(plan,'reader_current' if arm=='dual' else 'frozen-actor',
                     training=True,capture=arm=='dual')
    try:
        candidates={}; texts={'empty':'','keep':source['previous']}
        for i in range(8):
            name=f'candidate_{i}'
            result=writer(env.client,source['messages'],'writer_current',dest/'candidates'/f'{name}.json',
                          seed(92710,history_id,'candidate',i),plan,sample=True)
            validate_sample(result['sample'],source['input_binding'])
            candidates[name]=result
            texts[name]=result['raw_response'] if result['usable'] else source['previous']
        # Always preserve all eight draws (including duplicates or unusable output).
        # Context exclusion applies to the whole history, not selected winners.
        if any(len(v['sample']['input_ids'])>plan['training']['max_length'] for v in candidates.values()):
            save(dest/'labels.json',{'history':history_id,'domain':h['domain'],'excluded':'writer_context_budget',
                 'writer_rows':[],'reader_rows':[], 'input_binding':source['input_binding']})
            return
        rewards={k:[] for k in texts}; observations={k:[] for k in texts}; cached={}
        identities={}
        for repeat in plan['probe_seeds']:
            for pi,spec in enumerate(h['probes']):
                actor_seed=seed(repeat,history_id,pi)
                for branch,text in texts.items():
                    output=dest/'probes'/str(repeat)/str(pi)/branch
                    key=binding([spec,text,actor_seed])
                    if key in cached:
                        result,path=cached[key]
                        save(output/'reuse.json',{'physical_result':str(path),'request_sha256':key})
                    else:
                        result=env.run(spec,text,actor_seed,output); path=output/'completed.json'
                        cached[key]=(result,path)
                    identity=[result['task_identity'],result['initial_query_hash']]
                    if (repeat,pi) in identities and identities[repeat,pi]!=identity:
                        raise ValueError('Counterfactual probe initial states differ')
                    identities[repeat,pi]=identity
                    rewards[branch].append(result['reward'])
                    observations[branch].append((result,path))
        if any(v is None for values in rewards.values() for v in values):
            save(dest/'labels.json',{'history':history_id,'domain':h['domain'],'excluded':'missing_official_score',
                'rewards':rewards,'writer_rows':[],'reader_rows':[], 'input_binding':source['input_binding']})
            return
        advantages=paired_advantages(rewards,audit['reward_scales'][h['domain']])
        writer_rows=[]; reader_rows=[]; exclusions=[]
        for branch,result in candidates.items():
            advantage=advantages[branch]['writer']
            if not result['usable']: advantage-=plan['training']['unusable_writer_penalty']
            writer_rows.append({'path':str(dest/'candidates'/f'{branch}.json'),'sample_key':'sample',
                'advantage':advantage,'domain':h['domain'],'history':history_id,
                'input_binding':source['input_binding']})
            if arm!='dual': continue
            for pi,(observed,path) in enumerate(observations[branch]):
                eligible=[]
                for si,sample in enumerate(observed['sampled_actions']):
                    validate_sample(sample)
                    n=len(sample['input_ids'])-sample['prompt_length']
                    if len(sample['input_ids'])>plan['training']['max_length'] or n>plan['training']['reader_max_response']:
                        exclusions.append({'branch':branch,'probe':pi,'sample':si,'reason':'reader_length_budget'})
                        continue
                    eligible.append({'path':str(path),'sample_key':'sampled_actions','sample_index':si,
                        'advantage':advantages[branch]['reader'][pi], 'domain':h['domain'],
                        'history':history_id,'episode':f'{history_id}/{branch}/{pi}'})
                # Weights keep each episode equal, including a sampled long turn
                # that was excluded; no replacement of missing observations.
                for row in eligible:
                    row['within_episode_weight']=1/max(1,len(observed['sampled_actions']))
                reader_rows.extend(eligible)
        save(dest/'labels.json',{'history':history_id,'domain':h['domain'],
            'input_binding':source['input_binding'],'writer_rows':writer_rows,'reader_rows':reader_rows,
            'rewards':rewards,'advantages':advantages,'reader_exclusions':exclusions,
            'sampled_candidates':8,'distinct_candidate_texts':len({v['raw_response'] for v in candidates.values()}),
            'usable_candidates':sum(v['usable'] for v in candidates.values()),
            'logical_probe_episodes':sum(map(len,rewards.values())),'physical_probe_episodes':len(cached),
            'automatic_reward_review':'complete finite matched task/seed grid; original public input binding verified'})
    finally: env.close()


def freeze_block(root, arm, block_id):
    root=Path(root); plan=read(root/'plan.json')
    block=next(b for b in plan['blocks'] if b['id']==block_id)
    directory=root/'training'/arm/block_id; datasets={'writer':[],'reader':[]}; hashes={}; excluded=[]
    for hid in block['histories']:
        p=directory/'histories'/hid/'labels.json'; labels=read(p); hashes[str(p)]=sha_file(p)
        source=read(root/'sources'/hid/'input.json')
        if labels['input_binding']!=source['input_binding']: raise ValueError('Reward label input mismatch')
        if labels.get('excluded'): excluded.append({'history':hid,'reason':labels['excluded']})
        checked = None if labels.get('excluded') else paired_advantages(
            labels['rewards'],read(root/'source_audit.json')['reward_scales'][labels['domain']])
        for role in datasets:
            for row in labels[role+'_rows']:
                p=Path(row['path']); value=read(p)
                if role=='writer':
                    expected=checked[p.stem]['writer']
                    if not value['usable']: expected-=plan['training']['unusable_writer_penalty']
                    validate_sample(value['sample'],source['input_binding'])
                else:
                    _,branch,probe=row['episode'].rsplit('/',2)
                    expected=checked[branch]['reader'][int(probe)]
                    validate_sample(value['sampled_actions'][row['sample_index']])
                if abs(row['advantage']-expected)>1e-12:
                    raise ValueError('Automatic reward label failed independent recomputation')
                hashes[str(p)]=sha_file(p); datasets[role].append(row)
    for role,rows in datasets.items():
        save(directory/f'{role}_dataset.json',rows)
        hashes[str(directory/f'{role}_dataset.json')]=sha_file(directory/f'{role}_dataset.json')
    save(directory/'dataset_audit.json',{'hashes':hashes,'excluded_histories':excluded,
        'examples':{k:len(v) for k,v in datasets.items()},'automatic_review_complete':True})
    return {k:len(v) for k,v in datasets.items()}
