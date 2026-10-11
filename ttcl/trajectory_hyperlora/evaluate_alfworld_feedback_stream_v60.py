"""Fixed development probes for preregistered accumulation checkpoints."""
from __future__ import annotations

import json
from pathlib import Path
import torch

from ttcl.trajectory_hyperlora.train_alfworld_feedback_stream_v60 import parser, checked
from ttcl.trajectory_hyperlora.alfworld_online_from_empty import digest, file_hash
from ttcl.trajectory_hyperlora.alfworld_zero_shot_probe import load_agent, run_episode
from ttcl.trajectory_hyperlora.alfworld_online_own_trajectory_v52b import bounded_source_fields
from ttcl.trajectory_hyperlora.alfworld_online_shared_trajectory_v53 import alignment_pair
from ttcl.trajectory_hyperlora.alfworld_persistent_failure_delta_v36 import save
from ttcl.trajectory_hyperlora.analyze_alfworld_online_shared_trajectory_v53 import paired


def main():
    args=parser().parse_args()
    review=checked(args); config=review['config']
    output=args.output_dir/'evaluation.json'
    if output.exists(): raise FileExistsError(output)
    completed=json.loads((args.output_dir/'completed.json').read_text())
    assert completed['training_sha256']==file_hash(args.output_dir/'training.json')
    assert completed['manifest_sha256']==file_hash(args.output_dir/'manifest.json')
    manifest=json.loads((args.output_dir/'manifest.json').read_text())
    assert manifest['review_sha256']==file_hash(args.review)
    assert manifest['trainer_sha256']==file_hash(Path(__file__).with_name('train_alfworld_feedback_stream_v60.py'))
    for name,value in manifest['dependencies'].items():
        assert file_hash(Path(__file__).with_name(name))==value
    training=json.loads((args.output_dir/'training.json').read_text())
    for snapshot in training['snapshots']:
        for arm in ('hyper','static'):
            assert snapshot[arm+'_sha256']==file_hash(args.output_dir/f"{arm}_{snapshot['label']}.pt")
    bootstrap=args.output_dir/'hyper_bootstrap.pt'
    hyper,ht=load_agent(args.model,bootstrap,args.hyper_device,.8)
    frozen,ft=load_agent(args.model,bootstrap,args.hyper_device,.8)
    static,st=load_agent(args.model,bootstrap,args.static_device,.75)
    result={'protocol':'Fixed exposed TRAIN probes, no probe trajectory enters training or memory; prefix120 endpoint, earlier checkpoints descriptive',
        'review_sha256':file_hash(args.review),'training_sha256':completed['training_sha256'],
        'runner_sha256':file_hash(Path(__file__)),'rows':[],'summaries':[],'failures':[]}
    def episode(agent,tok,game,fields,device,factors=None):
        return run_episode(agent,tok,game,fields,adapter=factors is None,fixed_adapter=factors,
            device=device,max_steps=30,max_new_tokens=64,constrain_actions=True,
            actor_history_turns=2,loop_guard_max=2)
    for length in config['checkpoint_prefix_lengths']:
        state=torch.load(args.output_dir/f'hyper_{length}.pt',map_location='cpu',weights_only=True)
        names={name:p for name,p in hyper.named_parameters() if p.requires_grad}
        assert set(names)==set(state['trainable_state'])
        with torch.no_grad():
            for name,p in names.items(): p.copy_(state['trainable_state'][name].to(p))
        sstate=torch.load(args.output_dir/f'static_{length}.pt',map_location='cpu',weights_only=True)
        with torch.no_grad():
            for layer,a in zip(static.adapters,sstate['A'],strict=True): layer.a.copy_(a.to(layer.a))
        factors=[b.to(args.static_device) for b in sstate['factors']]
        latest={row['family']:row for row in review['stream'][:length]}
        cache={}
        for probe in review['probes']:
            source=latest[probe['family']]
            assert source['index']<length<=probe['index']
            if source['index'] not in cache:
                normal,shifted=alignment_pair(source['source_records'],ht)
                cache[source['index']]={
                    'hyper':bounded_source_fields(hyper,ht,normal,args.hyper_device),
                    'frozen':bounded_source_fields(frozen,ft,normal,args.hyper_device),
                    'shifted':bounded_source_fields(hyper,ht,shifted,args.hyper_device),
                    'normal_sha256':digest(normal),'shifted_sha256':digest(shifted)}
            fields=cache[source['index']]; game=args.data_root/probe['game']
            row={'prefix_length':length,'probe_index':probe['index'],'game':probe['game'],
                'probe_binding':probe['input_content_sha256'],'source_index':source['index'],
                'source_binding':source['input_content_sha256'],
                'coherent_records_sha256':fields['normal_sha256'],'shifted_records_sha256':fields['shifted_sha256']}
            row['hyper_online']=episode(hyper,ht,game,fields['hyper'],args.hyper_device)
            row['hyper_frozen_bootstrap']=episode(frozen,ft,game,fields['frozen'],args.hyper_device)
            row['online_static_lora']=episode(static,st,game,{},args.static_device,factors)
            row['hyper_shifted_source_actions']=episode(hyper,ht,game,fields['shifted'],args.hyper_device)
            for arm in config['arms']:
                ep=row[arm]
                if ep['status']!='complete' or digest(ep['initial_observation'])!=probe['initial_observation_sha256']:
                    result['failures'].append({'prefix_length':length,'probe_index':probe['index'],'arm':arm,'episode':ep})
                    save(output,result); raise RuntimeError('Incomplete or changed probe')
            result['rows'].append(row); save(output,result)
            current=[r for r in result['rows'] if r['prefix_length']==length]
            print(json.dumps({'prefix_length':length,'probes':len(current),
                **{arm:sum(r[arm]['reward'] for r in current) for arm in config['arms']}}),flush=True)
        current=[r for r in result['rows'] if r['prefix_length']==length]
        result['summaries'].append({'prefix_length':length,'n':len(current),
            'comparisons':[paired(current,'hyper_online',arm) for arm in config['arms'] if arm!='hyper_online']})
        save(output,result)
    assert len(result['rows'])==72 and not result['failures']
    print(json.dumps({'evaluation_complete':True,'environment_replay_audit_pending':True}),flush=True)


if __name__=='__main__':
    main()
