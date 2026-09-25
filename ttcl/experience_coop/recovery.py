"""Restart from verified initial rollouts, never from a partially changed policy."""
from __future__ import annotations

import copy
from pathlib import Path
import shutil
import time

from ttcl.experience_evolution.core import read, save
from ttcl.experience_v2.common import WORKSPACE, sha_file
from .collect import freeze_sources, freeze_block
from .protocol import correction_settings


def verify_files(hashes):
    for p,h in hashes.items():
        if sha_file(p)!=h: raise ValueError('Recovery input changed: '+p)


def prepare_recovery(root, origin, gpu=3, port=18287):
    root=Path(root).resolve();origin=Path(origin).resolve()
    if root.exists() or root.is_relative_to(origin):
        raise FileExistsError('Recovery requires a separate new directory')
    if read(origin/'status.json')['phase']!='failed':
        raise ValueError('Only an explicitly failed run can supply this recovery')
    original=read(origin/'plan.json')
    verify_files(read(origin/'input_hashes.json'))
    verify_files(read(origin/'reader_initial_hashes.json'))
    verify_files(read(origin/'source_audit.json')['hashes'])
    initial=original['blocks'][0]['id']
    inherited_hashes={}
    for arm in original['arms']:
        directory=origin/'training'/arm/initial
        inputs=read(directory/'block_input.json')
        if (inputs['writer']!=original['initial_adapter'] or inputs['reader']!=original['initial_reader']
                or inputs['writer_optimizer'] is not None or inputs['reader_optimizer'] is not None):
            raise ValueError('Reused rollout must have been generated before any policy update')
        for mapping in [inputs['hashes'],read(directory/'dataset_audit.json')['hashes']]:
            verify_files(mapping);inherited_hashes.update(mapping)
    root.mkdir(parents=True)
    plan=copy.deepcopy(original)
    plan.update(created_at=time.time(),gpu=gpu,port=port,actor_url=f'http://127.0.0.1:{port}',
        predecessor=None,initial_adapter=str(root/'adapters/original_delta'),
        initial_reader=str(root/'adapters/reader_initial'),reused_initial_rollouts=initial)
    plan['training'].update(correction_settings('decoupled_token_is'))
    save(root/'plan.json',plan)
    shutil.copytree(origin/'adapters',root/'adapters')
    shutil.copytree(origin/'source',root/'source',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.rmtree(root/'source/ttcl/experience_coop')
    shutil.copytree(WORKSPACE/'ttcl/experience_coop',root/'source/ttcl/experience_coop',
                    ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for name in ['split_audit.json','split_supplier.json']:
        shutil.copy2(origin/name,root/name)
    shutil.copytree(origin/'sources',root/'sources')
    # These are byte copies of the exact original public histories, not new
    # trajectories with old labels. Revalidate their content-bound rewards.
    freeze_sources(root)
    copied_hashes={}
    for arm in plan['arms']:
        source=origin/'training'/arm/initial/'histories'
        dest=root/'training'/arm/initial/'histories'
        shutil.copytree(source,dest)
        for p in source.rglob('*'):
            if p.is_file():
                q=dest/p.relative_to(source)
                if sha_file(p)!=sha_file(q): raise ValueError('Rollout byte copy differs')
                copied_hashes[str(q)]=sha_file(q)
        freeze_block(root,arm,initial)
    save(root/'recovery.json',{'origin':str(origin),'origin_plan_sha256':sha_file(origin/'plan.json'),
        'reused_source_histories':len(plan['histories']),'reused_block':initial,
        'reused_initial_writer_samples_per_arm':80,'reused_logical_probe_episodes_per_arm':360,
        'original_policy_updates_reused':False,'all_optimizers_restart_from_initial_policies':True,
        'absolute_raw_sample_references_preserved':True,
        'origin_must_be_retained':True,'original_failure_retained':True,
        'algorithm_change':'Decoupled clipped PPO with bounded token importance correction; strict original mode retained',
        'method_reference':'https://github.com/verl-project/verl/blob/main/docs/algo/rollout_corr.md'})
    paths=[p for d in ['source','adapters','sources'] for p in (root/d).rglob('*') if p.is_file()]
    paths += [root/n for n in ['plan.json','split_audit.json','split_supplier.json','source_audit.json','recovery.json']]
    external={p:h for p,h in read(origin/'input_hashes.json').items() if not Path(p).is_relative_to(origin)}
    manifest={**external,**inherited_hashes,**copied_hashes,**{str(p):sha_file(p) for p in paths}}
    # Keep the source run's protocol, binding records and failure permanently auditable.
    for name in ['plan.json','input_hashes.json','status.json','source_audit.json']:
        p=origin/name;manifest[str(p)]=sha_file(p)
    for arm in plan['arms']:
        for name in ['writer_dataset.json','reader_dataset.json','dataset_audit.json']:
            p=root/'training'/arm/initial/name;manifest[str(p)]=sha_file(p)
    save(root/'input_hashes.json',manifest)
    save(root/'reader_initial_hashes.json',{str(p):sha_file(p) for p in Path(plan['initial_reader']).rglob('*') if p.is_file()})
    save(root/'status.json',{'phase':'prepared','recovery_origin':str(origin),'histories':len(plan['histories']),
        'writer_candidates_per_arm':len(plan['histories'])*8,'blocks_per_arm':len(plan['blocks'])})
    return read(root/'status.json')
