"""Freeze shared train-only histories and a new pair of trainable adapters."""
from __future__ import annotations

from collections import defaultdict
import copy
from pathlib import Path
import shutil
import time

from ttcl.experience_evolution.core import read, save, seed
from ttcl.experience_lab.prepare import prepare as prepare_splits
from ttcl.experience_v2.common import WORKSPACE, BENCH, sha_file
from .protocol import evaluation_routes, correction_settings

DEFAULT_ROOT = WORKSPACE/'results/experience_coop/20260924_ppo8'
DEFAULT_SPLITS = WORKSPACE/'results/experience_lab/20260924_r01/plan.json'


def prepare(root, split_plan=DEFAULT_SPLITS, gpu=3, port=18287, predecessor=None,
            *, rollout_correction="strict", fresh_lineage=False):
    correction = correction_settings(rollout_correction)
    root = Path(root).resolve()
    if root.exists():
        raise FileExistsError('New experiment requires a new directory')
    split_plan = Path(split_plan).resolve()
    if fresh_lineage and split_plan.exists():
        raise ValueError('Fresh lineage requires a new split plan path; do not reuse an old split')
    if not split_plan.exists():
        # Fresh lineage still requires a completed, auditable initial Delta run.
        prepare_splits(root/'split_source', gpu=gpu, port=port, rounds=2,
                       fresh_lineage=fresh_lineage)
        split_plan = root/'split_source/plan.json'
    parent = read(split_plan)
    if len(parent['rounds']) != 2:
        raise ValueError('This protocol expects two predeclared 80-history splits')
    original_audit = read(split_plan.parent/'split_audit.json')
    if not original_audit['full_scene_hash_disjoint']:
        raise ValueError('Audited scene-disjoint split required')
    # Check the supplier manifest before copying roles, not after taking scores.
    for p,h in read(split_plan.parent/'input_hashes.json').items():
        if sha_file(p) != h:
            raise ValueError('Split supplier frozen input changed: '+p)
    histories = [copy.deepcopy(h) for r in parent['rounds'] for h in r['histories']]
    blocks = []
    for ri,rp in enumerate(parent['rounds']):
        groups = defaultdict(list)
        for h in rp['histories']:
            groups[h.get('family', h['domain'])].append(h['id'])
        if len(groups) != 10 or {len(v) for v in groups.values()} != {8}:
            raise ValueError('Expected six ALF families and four CLBench domains, eight histories each')
        for j in range(8):
            blocks.append({'id': f'block_{len(blocks):03}', 'round': ri+1,
                           'histories': [groups[k][j] for k in sorted(groups)]})
    plan = {k: copy.deepcopy(parent[k]) for k in ['model','data_root','context','actor_max_tokens',
        'max_steps','writer_tokens','memory_tokens','environment_seed','development','final_sequences',
        'development_seeds','final_seeds','clbench_splits','already_exposed_test']}
    model_config = read(Path(plan['model'])/'config.json')
    layers = model_config['num_hidden_layers']
    plan.update(created_at=time.time(), gpu=gpu, port=port, actor_url=f'http://127.0.0.1:{port}',
        predecessor=str(Path(predecessor).resolve()) if predecessor else None,
        initial_adapter=str(root/'adapters/original_delta'),
        initial_reader=str(root/'adapters/reader_initial'), histories=histories, blocks=blocks,
        arms=['writer_only','dual'], rollout_per_history=8, probe_seeds=[927101,927102],
        evaluation_routes=evaluation_routes(),
        reader_lora={'r':8, 'lora_alpha':16, 'lora_dropout':0., 'bias':'none',
            'task_type':'CAUSAL_LM', 'target_modules':['q_proj','k_proj','v_proj','o_proj',
                'gate_proj','up_proj','down_proj'], 'layers_to_transform':list(range(layers//2,layers)),
            'layers_pattern':'layers'},
        training={'seed':92710, 'learning_rate':5e-6, 'ppo_epochs':2, 'minibatch_size':16,
            'reader_minibatch_size':32, 'clip_range':.2, 'kl_beta':.01, 'max_length':24576,
            'reader_max_response':1024, 'reader_turns_per_episode':2, 'max_grad_norm':1.,
            'max_behavior_logp_mae':.15, 'unusable_writer_penalty':.1,
            'advantage_clip':3., 'scale':'max(1,RMS of shared train source rewards), frozen before PPO',
            'objective':'PPO-style token clipping with terminal paired advantages; no learned value critic',
            'writer_advantage':'mean(Rcandidate-Rkeep) - .5 mean(max(0,Rkeep-Rcandidate)); divide scale, clip; unusable -.1',
            'reader_advantage':'(Rreader_with_text-Rsame_reader_without_text)/scale, clipped; two uniform generated actions per episode',
            'reduction':'mean tokens per sampled action, mean sampled actions per episode, equal domain weighting'},
        collaboration='Collect with both policies frozen; update writer and then reader from their own on-policy actions; reload both next block',
        sources='Shared original-Delta/base-actor training histories; fresh public-input bindings, no reused human targets',
        parametric_memory='Reader LoRA accumulates across training blocks. Both LoRAs frozen during evaluation; text memory updates online.',
        final_rule='One fixed final checkpoint per training arm. Run full held-out comparison only if writer-only or dual passes development gate. No test tuning.')
    plan['training'].update(correction)
    shutil.copytree(parent['initial_adapter'], root/'adapters/original_delta')
    save(root/'plan.json', plan)
    save(root/'split_audit.json', dict(original_audit, inherited_plan_sha256=sha_file(split_plan),
        shared_with_experience_lab=split_plan.parent != root/'split_source', input_histories=len(histories),
        new_supervision='Fresh automatically verified paired official rewards, not old labels'))
    save(root/'split_supplier.json', {'plan':str(split_plan), 'sha256':sha_file(split_plan)})
    target = root/'source/ttcl'; target.mkdir(parents=True)
    (target/'__init__.py').write_text('')
    shutil.copy2(WORKSPACE/'ttcl/paths.py', target/'paths.py')
    for name in ['experience_coop','experience_lab','experience_evolution','experience_v2',
        'alfworld_comparison','experience_feedback','experience_repair','reflexion_expel',
        'common','structured_memory','llm_memory']:
        shutil.copytree(WORKSPACE/'ttcl'/name, target/name,
                        ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    paths = [p for d in ['source','adapters'] for p in (root/d).rglob('*') if p.is_file()]
    paths += [root/'plan.json',root/'split_audit.json',root/'split_supplier.json']
    # Include all supplier task files and CLBench source, but not mutable outputs.
    for p in read(split_plan.parent/'input_hashes.json'):
        path=Path(p)
        if str(path).startswith(str(Path(plan['data_root']))) or str(path).startswith(str(BENCH/'src')):
            paths.append(path)
    paths += [Path(p) for p in original_audit.get('lineage_input_hashes', {})]
    paths += [p for p in Path(plan['model']).iterdir() if p.is_file() and
              (p.suffix in {'.safetensors','.json'} or p.name in {'vocab.json','merges.txt'})]
    save(root/'input_hashes.json',{str(p):sha_file(p) for p in sorted(set(paths))})
    save(root/'status.json',{'phase':'prepared','histories':len(histories),
        'writer_candidates_per_arm':len(histories)*8,'blocks_per_arm':len(blocks)})
    return read(root/'status.json')
