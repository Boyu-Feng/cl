"""Verify chronological schedules, frozen A, paired budgets and all probes."""
from __future__ import annotations

import json
from pathlib import Path
import random
import torch
from transformers import AutoTokenizer

from ttcl.trajectory_hyperlora.train_alfworld_feedback_stream_v60 import parser, checked
from ttcl.trajectory_hyperlora.alfworld_online_from_empty import digest, file_hash
from ttcl.trajectory_hyperlora.alfworld_online_shared_trajectory_v53 import alignment_pair
from ttcl.trajectory_hyperlora.audit_my_online_update_v28_reward import replay
from ttcl.trajectory_hyperlora.analyze_alfworld_online_shared_trajectory_v53 import paired
from ttcl.trajectory_hyperlora.alfworld_persistent_failure_delta_v36 import save


def main():
    args=parser().parse_args(); review=checked(args); config=review['config']; stream=review['stream']
    completion=json.loads((args.output_dir/'completed.json').read_text())
    assert completion['training_sha256']==file_hash(args.output_dir/'training.json')
    manifest=json.loads((args.output_dir/'manifest.json').read_text())
    assert completion['manifest_sha256']==file_hash(args.output_dir/'manifest.json')
    assert manifest['review_sha256']==file_hash(args.review)
    assert manifest['trainer_sha256']==file_hash(Path(__file__).with_name('train_alfworld_feedback_stream_v60.py'))
    for name,value in manifest['dependencies'].items():
        assert file_hash(Path(__file__).with_name(name))==value
    training=json.loads((args.output_dir/'training.json').read_text())
    assert len(training['rows'])==120
    rng=random.Random(config['seed']); wins=0; booted=False; lp=None; ln=None; expected_steps=0
    frozen_a=None
    for i,(row,result) in enumerate(zip(stream,training['rows'],strict=True)):
        assert result['index']==i and result['input_content_sha256']==row['input_content_sha256']
        eligible=row['source_index'] is not None
        if row['positive_actions'] and eligible: lp=i
        if row['negative_targets'] and eligible: ln=i
        event=False
        if not booted:
            wins+=bool(row['positive_actions'])
            if wins==8:
                booted=True; assert training['bootstrap']['index']==i
                assert training['bootstrap']['hyper']['steps']==training['bootstrap']['static']['steps']==300
                frozen_a=result['hyper_A_sha256']
        elif eligible and (row['positive_actions'] or row['negative_targets']) and lp is not None:
            event=True; expected=[]
            for _ in range(20):
                positive=stream[lp]['positive_actions'][rng.randrange(len(stream[lp]['positive_actions']))]
                negative=stream[ln]['negative_targets'][rng.randrange(len(stream[ln]['negative_targets']))] if ln is not None else None
                expected.append({'positive_binding':digest([stream[lp]['input_content_sha256'],positive]),
                    'positive_index':lp,'positive_source':stream[lp]['source_index'],
                    'negative_binding':negative['input_content_sha256'] if negative else None,
                    'negative_index':ln,'negative_source':stream[ln]['source_index'] if ln is not None else None})
            assert result['schedule']==expected
            assert result['hyper_steps']==result['static_steps']==20
            for arm in ('hyper_fit','static_fit'):
                fit=result[arm]
                assert fit['optimizer_steps']==fit['positive_target_passes']==20
                assert fit['negative_target_passes']==(20 if ln is not None else 0)
                if ln is not None:
                    assert fit['negative_weight']==.1
                    assert [(x['positive_binding'],x['negative_binding'],x['update_index']) for x in fit['rows']]==[(x['positive_binding'],x['negative_binding'],i) for x in expected]
            expected_steps+=20
        if not event: assert result['hyper_steps']==result['static_steps']==0
        if booted: assert result['hyper_A_sha256']==frozen_a
    assert [(x['label'],x['prefix_length']) for x in training['snapshots']]==[
        ('bootstrap',training['bootstrap']['index']+1),('40',40),('80',80),('120',120)]
    bootstrap=torch.load(args.output_dir/'hyper_bootstrap.pt',map_location='cpu',weights_only=True)
    a_names=[n for n in bootstrap['trainable_state'] if n.endswith('.a')]
    assert len(a_names)==2
    for snapshot in training['snapshots']:
        for arm in ('hyper','static'):
            path=args.output_dir/f"{arm}_{snapshot['label']}.pt"
            assert snapshot[arm+'_sha256']==file_hash(path)
            checkpoint=torch.load(path,map_location='cpu',weights_only=True)
            assert checkpoint['review_sha256']==file_hash(args.review)
            assert checkpoint['prefix_length']==snapshot['prefix_length']
            if arm=='hyper':
                assert all(torch.equal(bootstrap['trainable_state'][n],checkpoint['trainable_state'][n]) for n in a_names)
    evaluation=json.loads((args.output_dir/'evaluation.json').read_text())
    assert evaluation['training_sha256']==completion['training_sha256']
    assert evaluation['review_sha256']==file_hash(args.review)
    assert evaluation['runner_sha256']==file_hash(Path(__file__).with_name('evaluate_alfworld_feedback_stream_v60.py'))
    assert not evaluation['failures'] and len(evaluation['rows'])==72
    assert [(r['prefix_length'],r['probe_index']) for r in evaluation['rows']]==[(n,p['index']) for n in (40,80,120) for p in review['probes']]
    tok=AutoTokenizer.from_pretrained(args.model,local_files_only=True)
    count=0; cache={}
    for row in evaluation['rows']:
        n=row['prefix_length']; probe=review['probes'][row['probe_index']-120]
        source=next(x for x in reversed(stream[:n]) if x['family']==probe['family'])
        assert row['game']==probe['game'] and row['probe_binding']==probe['input_content_sha256']
        assert row['source_index']==source['index'] and row['source_binding']==source['input_content_sha256']
        if source['index'] not in cache:
            normal,shifted=alignment_pair(source['source_records'],tok)
            cache[source['index']]=(digest(normal),digest(shifted))
        assert (row['coherent_records_sha256'],row['shifted_records_sha256'])==cache[source['index']]
        for arm in config['arms']:
            ep=row[arm]
            assert ep['status']=='complete' and ep['steps']<=30
            assert digest(ep['initial_observation'])==probe['initial_observation_sha256']
            replay(args.data_root/probe['game'],ep); count+=1
        if count%32==0: print(json.dumps({'replayed_probe_episodes':count}),flush=True)
    summaries=[]
    for n in (40,80,120):
        rows=[r for r in evaluation['rows'] if r['prefix_length']==n]
        summaries.append({'prefix_length':n,'n':24,'comparisons':[paired(rows,'hyper_online',a) for a in config['arms'] if a!='hyper_online']})
    assert summaries==evaluation['summaries']
    primary=summaries[-1]['comparisons']; running=0.
    for i,item in enumerate(sorted(primary,key=lambda x:x['mcnemar_exact_two_sided_p'])):
        running=max(running,min(1.,(3-i)*item['mcnemar_exact_two_sided_p']))
        item['holm_adjusted_p']=running
    save(args.output_dir/'audit.json',{'passed':True,'evaluation_sha256':file_hash(args.output_dir/'evaluation.json'),
        'training_sha256':completion['training_sha256'],'auditor_sha256':file_hash(Path(__file__)),
        'replayed_episodes':count,'matched_online_steps':expected_steps,'primary_prefix120':primary,
        'limitation':'24 exposed development tasks, one seed; earlier checkpoints share probes and are not independent tests'})
    print(json.dumps({'passed':True,'replayed_episodes':count,'matched_online_steps':expected_steps}),flush=True)


if __name__=='__main__':
    main()
