"""Freeze two independent exploration rounds without modifying older runs."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import random
import shutil
import time

from ttcl.alfworld_comparison.splits import _inventory, _historical_usage, REQUIRED_PROJECTS, _games
from ttcl.experience_evolution.core import FAMILIES, read, save, seed
from ttcl.experience_v2.common import WORKSPACE, MODEL, OLD, BENCH, sha_file

DEFAULT_ROOT = WORKSPACE/'results/experience_lab/20260924_r01'
TOTALS = {'blind_spectrum_monitoring': 90, 'exploitable_poker': 120,
          'database_exploration': 20, 'cohort_studies': 20}


def audit_lineage(results_root, old, inventory, fresh_lineage=False):
    """Missing old experiments are explicit; never accept an opaque initial writer."""
    training, evaluation, manifests, covered = _historical_usage(results_root, not fresh_lineage)
    paths = [results_root/m['path'] for m in manifests]
    if fresh_lineage:
        # A copied adapter with no task provenance cannot establish isolation.
        freeze = read(old/'freeze.json')
        for name, key in [('plan.json', 'plan_sha256'), ('data_hashes.json', 'data_hashes_sha256')]:
            if sha_file(old/name) != freeze[key]:
                raise ValueError('Initial Delta frozen provenance changed: '+name)
        status = read(old/'training/delta/status.json')
        if status.get('phase') != 'complete' or not status.get('audit', {}).get('base_unchanged'):
            raise ValueError('Finish and audit initial Delta training before preparing a fresh lineage')
        games = set(_games(read(old/'plan.json')))
        hashes = read(old/'data_hashes.json')
        if not games or not games.issubset(hashes):
            raise ValueError('Initial Delta task provenance is incomplete')
        for game in games:
            if game not in inventory or inventory[game]['sha256'] != hashes[game]:
                raise ValueError('Initial Delta task content changed: '+game)
        if not games.issubset(set(training) | set(evaluation)):
            raise ValueError('Initial Delta tasks were not included in the historical exclusion audit')
        adapter = old/'training/delta/adapter'
        if not (adapter/'adapter_config.json').is_file() or not (adapter/'adapter_model.safetensors').is_file():
            raise FileNotFoundError('Initial Delta adapter is missing: '+str(adapter))
        paths += [old/n for n in ['freeze.json', 'plan.json', 'data_hashes.json', 'training/delta/status.json']]
    audit = {'mode': 'fresh_local_lineage' if fresh_lineage else 'complete_historical_lineage',
             'covered_projects': sorted(covered),
             'missing_historical_projects': sorted(REQUIRED_PROJECTS-covered),
             'scope': 'Exclude every locally recorded prior task; fresh mode does not reconstruct the old server split'}
    return training, evaluation, manifests, audit, {str(p): sha_file(p) for p in sorted(set(paths))}


def prepare(root, gpu=1, port=18277, rounds=2, *, fresh_lineage=False):
    root = Path(root).resolve()
    if root.exists():
        raise FileExistsError('Each exploration cycle needs a new output directory')
    if rounds not in (1, 2):
        raise ValueError('This reviewed cycle permits one or two rounds')
    data = WORKSPACE/'ttcl/data/alfworld_delta'
    inventory = _inventory(data)
    historical, historical_eval, manifests, lineage, lineage_hashes = audit_lineage(
        WORKSPACE/'ttcl/results', OLD, inventory, fresh_lineage)
    reserved = {inventory[g]['normalized_problem_sha256'] for g in
                set(historical) | set(historical_eval) if g in inventory}
    reserved.update(v['normalized_problem_sha256'] for v in inventory.values() if v['split'] != 'train')
    pools = defaultdict(list)
    for game, item in sorted(inventory.items()):
        if (item['split'] == 'train' and item['solvable'] and
                item['normalized_problem_sha256'] not in reserved and
                'movable' not in game and 'Sliced' not in game):
            pools[item['family']].append(game)
    rng = random.Random(92471)
    used_hashes = set(reserved)
    for pool in pools.values():
        rng.shuffle(pool)
    def take(family):
        while pools[family]:
            game = pools[family].pop()
            h = inventory[game]['normalized_problem_sha256']
            if h and h not in used_hashes:
                used_hashes.add(h)
                return {'domain': 'alfworld', 'family': family, 'game': game, 'id': game}
        raise ValueError(f'Insufficient independent training scenes: {family}')
    development = [{'domain': 'alfworld', 'family': family,
                    'id': f'alf_{family}', 'tasks': [take(family) for _ in range(8)]}
                   for family in FAMILIES]
    cl_splits = {}
    for domain, total in TOTALS.items():
        prefix = total//5
        n_dev = max(2, prefix//3)
        cl_splits[domain] = {'train': list(range(prefix-n_dev)),
                             'development': list(range(prefix-n_dev, prefix)),
                             'test': list(range(prefix, total)), 'total': total}
        development.append({'domain': domain, 'id': domain,
            'tasks': [{'domain': domain, 'index': i, 'id': f'{domain}:{i}'}
                      for i in cl_splits[domain]['development']]})
    round_plans = []
    for ri in range(rounds):
        histories = []
        for family in FAMILIES:
            for i in range(8):
                histories.append({'domain': 'alfworld', 'family': family,
                    'warmup': take(family), 'source': take(family),
                    'probes': [take(family), take(family)]})
        for domain, split in cl_splits.items():
            source_order = split['train'].copy()
            random.Random(seed(92471, ri, domain)).shuffle(source_order)
            for i in range(8):
                source = source_order[i % len(source_order)]
                others = [v for v in source_order if v != source]
                rng.shuffle(others)
                spec = lambda j: {'domain': domain, 'index': j, 'id': f'{domain}:{j}'}
                # Tiny Database/Cohort prefixes have only two training tasks.
                # Warmup repeats the source; the other task remains the future probe.
                histories.append({'domain': domain, 'warmup': spec(source),
                    'source': spec(source), 'probes': [spec(j) for j in others[:2]]})
        # Interleave domains so partial progress cannot masquerade as a balanced run.
        groups = defaultdict(list)
        for h in histories: groups[h['domain']].append(h)
        ordered = []
        while any(groups.values()):
            for domain in sorted(groups):
                if groups[domain]: ordered.append(groups[domain].pop(0))
        for i, item in enumerate(ordered):
            item.update(id=f'r{ri+1:02}_h{i:03}', source_seed=seed(92471, ri, i, 'source'),
                        warmup_seed=seed(92471, ri, i, 'warmup'))
        round_plans.append({'id': f'round_{ri+1:03}', 'histories': ordered,
                            'probe_seeds': [seed(92471, ri, 'screen'), seed(92471, ri, 'confirm')]})
    final_sequences = [{'domain': 'alfworld', 'family': family, 'id': f'alf_{family}',
        'tasks': [{'domain': 'alfworld', 'family': family, 'game': g, 'id': g}
                  for g, v in sorted(inventory.items()) if v['split']=='valid_unseen' and v['family']==family]}
        for family in FAMILIES]
    for domain, split in cl_splits.items():
        final_sequences.append({'domain': domain, 'id': domain,
            'tasks': [{'domain': domain, 'index': i, 'id': f'{domain}:{i}'} for i in split['test']]})
    plan = {'created_at': time.time(), 'model': str(MODEL), 'data_root': str(data),
        'actor_url': f'http://127.0.0.1:{port}', 'port': port, 'gpu': gpu,
        'context': 65536, 'actor_temperature': .7, 'actor_max_tokens': 64, 'max_steps': 50,
        'writer_tokens': 512, 'memory_tokens': 2048, 'environment_seed': 42,
        'rounds': round_plans, 'development': development, 'final_sequences': final_sequences,
        'development_seeds': [924711, 924712], 'final_seeds': [92601, 92602, 92603],
        'clbench_splits': cl_splits, 'initial_adapter': str(root/'adapters/original_delta'),
        'training': {'epochs': 2, 'accumulation': 4, 'learning_rate': 5e-6,
                     'seed': 92471, 'max_length': 16384, 'dpo_beta': .1, 'chosen_nll_weight': .1},
        'training_arms': ['filtered_sft', 'random_sft', 'dpo'], 'min_pairs': 12,
        'candidate_methods': ['parent_greedy', 'parent_sample', 'base_evidence'],
        'replay': 'All accepted previous-round pairs retained; balanced by domain',
        'checkpoint_rule': 'Fixed final of each arm; selection only on development trajectories',
        'gate': 'ALF mean improves over original Delta; all four CLBench domain means are nonnegative vs Delta, at least one strictly positive; complete grid required',
        'final_rule': 'At most one selected writer; no training or selection from final outcomes',
        'already_exposed_test': 'ALF valid_unseen and CLBench suffix were evaluated historically; no claim of researcher-unseen tests',
        'scope': 'Single-attempt online experience updates, not the separate three-attempt comparison'}
    root.mkdir(parents=True)
    shutil.copytree(OLD/'training/delta/adapter', root/'adapters/original_delta')
    target = root/'source/ttcl'; target.mkdir(parents=True)
    (target/'__init__.py').write_text('')
    shutil.copy2(WORKSPACE/'ttcl/paths.py', target/'paths.py')
    for name in ['experience_lab', 'experience_evolution', 'experience_v2', 'alfworld_comparison',
                 'experience_feedback', 'experience_repair', 'reflexion_expel', 'common',
                 'structured_memory', 'llm_memory']:
        shutil.copytree(WORKSPACE/'ttcl'/name, target/name,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    save(root/'plan.json', plan)
    train_games = [s['game'] for r in round_plans for h in r['histories']
                   for s in [h['warmup'], h['source'], *h['probes']] if h['domain']=='alfworld']
    dev_games = [t['game'] for seq in development if seq['domain']=='alfworld' for t in seq['tasks']]
    test_games = [t['game'] for seq in final_sequences if seq['domain']=='alfworld' for t in seq['tasks']]
    roles = {name: {inventory[g]['normalized_problem_sha256'] for g in games}
             for name, games in [('train', train_games), ('development', dev_games), ('test', test_games)]}
    if any(roles[a] & roles[b] for a,b in [('train','development'),('train','test'),('development','test')]):
        raise AssertionError('Scene-level train/development/test collision')
    audit = {'train_games': len(set(train_games)), 'development_games': len(dev_games),
             'test_games': len(test_games), 'full_scene_hash_disjoint': True,
             'historical_manifests': manifests, 'lineage': lineage,
             'lineage_input_hashes': lineage_hashes, 'clbench_splits': cl_splits,
             'clbench_small_pools': 'Database/Cohort each have two train and two development instances; seeds do not increase unique tasks',
             'dev_excludes_historical_alf_training': True}
    save(root/'split_audit.json', audit)
    paths = [p for d in ['source', 'adapters'] for p in (root/d).rglob('*') if p.is_file()]
    paths += [root/'plan.json', root/'split_audit.json']
    paths += [data/g for g in sorted(set(train_games+dev_games+test_games))]
    paths += list((BENCH/'src').rglob('*.py'))
    paths += [Path(p) for p in lineage_hashes]
    save(root/'input_hashes.json', {str(p): sha_file(p) for p in paths})
    save(root/'status.json', {'phase': 'prepared', 'rounds': rounds, 'histories_per_round': 80})
    return audit
