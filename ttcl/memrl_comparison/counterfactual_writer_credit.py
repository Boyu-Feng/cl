"""CPU-only credit for a source-bound, full-horizon experience-writer update.

The caller must collect every branch from the same frozen pre-target memory
snapshot, with the candidate document generated before the target is exposed.
This module audits that branch grid and computes writer credit; it never runs
an actor, trains a model, or turns a missing/failed run into zero reward.
"""
from __future__ import annotations

import hashlib
import json
import math
import statistics


ARMS = frozenset({'candidate', 'previous', 'empty'})
REQUIRED = frozenset({
    'source_trace_sha256', 'source_input_sha256', 'target_input_sha256',
    'reviewed_annotation_sha256',
    'start_snapshot_sha256', 'actor_seed', 'arm', 'memory_sha256',
    'official_reward', 'actor_calls', 'status',
})
EMPTY_SHA256 = hashlib.sha256(b'').hexdigest()


def _digest(value: str) -> bool:
    return (isinstance(value, str) and len(value) == 64 and
            all(char in '0123456789abcdef' for char in value))


def score_writer_branches(rows: list[dict], *, reviewed_targets: dict[str, str],
                          reward_scale: float, call_budget: int,
                          cost_weight: float = 0.0) -> dict:
    """Score a complete paired candidate/previous/empty training grid.

    ``reviewed_targets`` maps each target input-content SHA-256 to its new,
    independently reviewed annotation SHA-256; the collector must verify
    these files and freeze them before rollout. Each row is an official
    *terminal* target outcome. ``actor_calls`` covers the entire predeclared
    target horizon. ``reward_scale`` and ``cost_weight``
    must be frozen from training data before these branch outcomes are read.
    The gradient label is candidate minus previous; candidate minus empty is
    a separate diagnostic, not a second independently credited reward.
    """
    if not rows or not math.isfinite(reward_scale) or reward_scale <= 0:
        raise ValueError('Missing rows or invalid reward scale')
    if (not isinstance(reviewed_targets, dict) or not reviewed_targets or
            not all(_digest(target) and _digest(annotation)
                    for target, annotation in reviewed_targets.items())):
        raise ValueError('Missing reviewed target input-content bindings')
    if (isinstance(call_budget, bool) or not isinstance(call_budget, int) or
            call_budget <= 0 or not math.isfinite(cost_weight) or cost_weight < 0):
        raise ValueError('Invalid frozen call budget or cost weight')
    grid = {}
    sources = set()
    memory_hashes = {arm: set() for arm in ARMS}
    for row in rows:
        if not isinstance(row, dict) or not REQUIRED <= row.keys():
            raise ValueError('Incomplete branch row')
        source, source_input, target, snapshot = (
            row[name] for name in ('source_trace_sha256', 'source_input_sha256',
                                   'target_input_sha256', 'start_snapshot_sha256'))
        if not all(map(_digest, (source, source_input, target, snapshot))):
            raise ValueError('Missing input-content or snapshot binding')
        if (not _digest(row['reviewed_annotation_sha256']) or
                row['reviewed_annotation_sha256'] != reviewed_targets.get(target)):
            raise ValueError('Target differs from reviewed annotation binding')
        if source_input == target:
            raise ValueError('A source trajectory cannot be its own target')
        if row['status'] != 'complete':
            raise ValueError('Incomplete branch is not a zero outcome')
        arm = row['arm']
        if arm not in ARMS or not _digest(row['memory_sha256']):
            raise ValueError('Unknown arm or invalid memory binding')
        if arm == 'empty' and row['memory_sha256'] != EMPTY_SHA256:
            raise ValueError('Empty control contains memory text')
        reward = row['official_reward']
        calls = row['actor_calls']
        if (isinstance(reward, bool) or not isinstance(reward, (int, float)) or
                not math.isfinite(reward) or isinstance(calls, bool) or
                not isinstance(calls, int) or not 0 <= calls <= call_budget):
            raise ValueError('Invalid official terminal reward or call count')
        seed = row['actor_seed']
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError('Invalid actor seed')
        sources.add((source, source_input))
        memory_hashes[arm].add(row['memory_sha256'])
        key = (target, seed, arm)
        if key in grid:
            raise ValueError('Duplicate target, seed, and arm')
        grid[key] = row
    if len(sources) != 1 or any(len(values) != 1 for values in memory_hashes.values()):
        raise ValueError('Source or branch document changed within grid')
    pairs = sorted({(target, seed) for target, seed, _ in grid})
    targets = {target for target, _ in pairs}
    seeds = {seed for _, seed in pairs}
    if targets != set(reviewed_targets):
        raise ValueError('Target grid differs from reviewed annotation targets')
    if len(targets) < 1 or len(seeds) < 2 or len(pairs) != len(targets) * len(seeds):
        raise ValueError('Need a complete target-by-seed grid with two seeds')
    if set(grid) != {(target, seed, arm) for target, seed in pairs for arm in ARMS}:
        raise ValueError('Missing paired candidate, previous, or empty branch')
    for target in targets:
        snapshots = {grid[target, seed, arm]['start_snapshot_sha256']
                     for seed in seeds for arm in ARMS}
        if len(snapshots) != 1:
            raise ValueError('Target seeds or arms did not share a frozen snapshot')
    values = []
    for target, seed in pairs:
        branches = {arm: grid[target, seed, arm] for arm in ARMS}
        if len({row['start_snapshot_sha256'] for row in branches.values()}) != 1:
            raise ValueError('Paired arms did not start from the same snapshot')
        candidate, previous, empty = (branches[name] for name in
                                      ('candidate', 'previous', 'empty'))
        def delta(control):
            return ((candidate['official_reward'] - control['official_reward']) /
                    reward_scale - cost_weight *
                    (candidate['actor_calls'] - control['actor_calls']) / call_budget)
        values.append(dict(target_input_sha256=target, actor_seed=seed,
                           vs_previous=delta(previous), vs_empty=delta(empty),
                           raw_vs_previous=candidate['official_reward'] -
                           previous['official_reward'],
                           raw_vs_empty=candidate['official_reward'] -
                           empty['official_reward']))
    advantages = [row['vs_previous'] for row in values]
    seed_means = {seed: statistics.fmean(row['vs_previous'] for row in values
                                         if row['actor_seed'] == seed)
                  for seed in sorted(seeds)}
    same_document = memory_hashes['candidate'] == memory_hashes['previous']
    return dict(source_trace_sha256=next(iter(sources))[0],
                source_input_sha256=next(iter(sources))[1],
                reviewed_targets_sha256=hashlib.sha256(json.dumps(
                    reviewed_targets, sort_keys=True, separators=(',', ':'))
                    .encode()).hexdigest(),
                memory_sha256={arm: next(iter(memory_hashes[arm])) for arm in
                               sorted(ARMS)},
                target_count=len(targets), seed_count=len(seeds),
                paired_count=len(pairs),
                advantage=(0.0 if same_document else
                           max(-1.0, min(1.0, statistics.fmean(advantages)))),
                identical_candidate_and_previous=same_document,
                mean_unclipped_advantage=statistics.fmean(advantages),
                mean_vs_empty=statistics.fmean(row['vs_empty'] for row in values),
                nonzero_pairs=sum(abs(value) > 1e-12 for value in advantages),
                seed_means=seed_means,
                seed_sign_conflict=min(seed_means.values()) < 0 < max(seed_means.values()),
                pairs=values)


ONLINE_REQUIRED = REQUIRED | frozenset({
    'target_position', 'chain_seed', 'root_snapshot_sha256',
    'end_snapshot_sha256', 'writer_freeze_sha256',
})


def score_writer_online_branches(
        rows: list[dict], *, reviewed_targets: list[tuple[str, str]],
        reward_scale: float, call_budget: int, cost_weight: float = 0.0,
        discount: float = 1.0) -> dict:
    """Credit one frozen writer document over a whole evolving task chain.

    All arms start from the same *root* bank. After document injection, each
    arm has its own memory state; its next task must start from its own exact
    previous end snapshot. Paired arms share future target content and actor
    seed at every position. Rewards include every task through the declared
    horizon, so later effects of changed writes/retrieval are credited to the
    initial document. The reviewed-target order and writer freeze must be
    established before any target rollout by the external collector.
    """
    if (not rows or not isinstance(reviewed_targets, list) or
            len(reviewed_targets) < 2 or not all(
                isinstance(item, (list, tuple)) and len(item) == 2 and
                _digest(item[0]) and _digest(item[1]) for item in reviewed_targets) or
            len({item[0] for item in reviewed_targets}) != len(reviewed_targets)):
        raise ValueError('Missing ordered, distinct reviewed target bindings')
    if (not math.isfinite(reward_scale) or reward_scale <= 0 or
            isinstance(call_budget, bool) or not isinstance(call_budget, int) or
            call_budget <= 0 or not math.isfinite(cost_weight) or cost_weight < 0 or
            not math.isfinite(discount) or not 0 < discount <= 1):
        raise ValueError('Invalid frozen reward scale, cost, or horizon discount')
    source_bindings, roots, writer_freezes = set(), set(), set()
    documents = {arm: set() for arm in ARMS}
    grid = {}
    for row in rows:
        if not isinstance(row, dict) or not ONLINE_REQUIRED <= row.keys():
            raise ValueError('Incomplete online branch row')
        for name in ('source_trace_sha256', 'source_input_sha256',
                     'target_input_sha256', 'root_snapshot_sha256',
                     'start_snapshot_sha256', 'end_snapshot_sha256',
                     'writer_freeze_sha256', 'memory_sha256'):
            if not _digest(row[name]):
                raise ValueError(f'Missing content binding: {name}')
        position = row['target_position']
        if (isinstance(position, bool) or not isinstance(position, int) or
                not 0 <= position < len(reviewed_targets) or
                row['target_input_sha256'] != reviewed_targets[position][0]):
            raise ValueError('Target order differs from reviewed manifest')
        if (not _digest(row['reviewed_annotation_sha256']) or
                row['reviewed_annotation_sha256'] != reviewed_targets[position][1]):
            raise ValueError('Target differs from reviewed annotation binding')
        if row['source_input_sha256'] == row['target_input_sha256']:
            raise ValueError('Source trajectory cannot be its own target')
        for name in ('chain_seed', 'actor_seed'):
            if isinstance(row[name], bool) or not isinstance(row[name], int):
                raise ValueError(f'Invalid {name}')
        arm = row['arm']
        if arm not in ARMS or (arm == 'empty' and
                               row['memory_sha256'] != EMPTY_SHA256):
            raise ValueError('Unknown arm or nonempty empty-memory control')
        if row['status'] != 'complete':
            raise ValueError('Incomplete branch is not a zero outcome')
        reward, calls = row['official_reward'], row['actor_calls']
        if (isinstance(reward, bool) or not isinstance(reward, (int, float)) or
                not math.isfinite(reward) or isinstance(calls, bool) or
                not isinstance(calls, int) or not 0 <= calls <= call_budget):
            raise ValueError('Invalid official terminal reward or call count')
        source_bindings.add((row['source_trace_sha256'],
                             row['source_input_sha256']))
        roots.add(row['root_snapshot_sha256'])
        writer_freezes.add(row['writer_freeze_sha256'])
        documents[arm].add(row['memory_sha256'])
        key = (row['chain_seed'], position, arm)
        if key in grid:
            raise ValueError('Duplicate online chain cell')
        grid[key] = row
    if (len(source_bindings) != 1 or len(roots) != 1 or
            len(writer_freezes) != 1 or
            any(len(value) != 1 for value in documents.values())):
        raise ValueError('Source, root snapshot, writer freeze, or document changed')
    seeds = sorted({seed for seed, _, _ in grid})
    if len(seeds) < 2 or set(grid) != {
            (seed, position, arm) for seed in seeds
            for position in range(len(reviewed_targets)) for arm in ARMS}:
        raise ValueError('Incomplete online chain grid; need two paired seeds')
    weights = [discount ** position for position in range(len(reviewed_targets))]
    denominator = sum(weights)
    by_seed, effects = {}, []
    for seed in seeds:
        per_task = []
        for position, (target, _) in enumerate(reviewed_targets):
            branches = {arm: grid[seed, position, arm] for arm in ARMS}
            if len({row['actor_seed'] for row in branches.values()}) != 1:
                raise ValueError('Paired actor seeds differ at a target')
            if position:
                for arm, row in branches.items():
                    previous = grid[seed, position - 1, arm]
                    if row['start_snapshot_sha256'] != previous['end_snapshot_sha256']:
                        raise ValueError('Online branch memory transition is broken')
            candidate, previous, empty = (branches[arm] for arm in
                                          ('candidate', 'previous', 'empty'))
            def delta(control):
                return ((candidate['official_reward'] - control['official_reward']) /
                        reward_scale - cost_weight *
                        (candidate['actor_calls'] - control['actor_calls']) /
                        call_budget)
            per_task.append(dict(position=position, target_input_sha256=target,
                                 actor_seed=candidate['actor_seed'],
                                 vs_previous=delta(previous),
                                 vs_empty=delta(empty),
                                 raw_vs_previous=(candidate['official_reward'] -
                                                  previous['official_reward'])))
        def weighted(name):
            return sum(weight * item[name] for weight, item in
                       zip(weights, per_task)) / denominator
        immediate = weights[0] * per_task[0]['vs_previous'] / denominator
        delayed = sum(weight * item['vs_previous'] for weight, item in
                      zip(weights[1:], per_task[1:])) / denominator
        by_seed[seed] = dict(vs_previous=weighted('vs_previous'),
                             vs_empty=weighted('vs_empty'),
                             immediate_vs_previous=immediate,
                             delayed_vs_previous=delayed)
        effects.extend(dict(chain_seed=seed, **item) for item in per_task)
    advantages = [value['vs_previous'] for value in by_seed.values()]
    same_document = documents['candidate'] == documents['previous']
    source = next(iter(source_bindings))
    return dict(source_trace_sha256=source[0], source_input_sha256=source[1],
                root_snapshot_sha256=next(iter(roots)),
                writer_freeze_sha256=next(iter(writer_freezes)),
                reviewed_target_order_sha256=hashlib.sha256(json.dumps(
                    reviewed_targets, separators=(',', ':')).encode()).hexdigest(),
                memory_sha256={arm: next(iter(documents[arm])) for arm in
                               sorted(ARMS)},
                target_count=len(reviewed_targets), seed_count=len(seeds),
                paired_cells=len(grid), discount=discount,
                advantage=(0.0 if same_document else max(-1.0, min(
                    1.0, statistics.fmean(advantages)))),
                mean_unclipped_advantage=statistics.fmean(advantages),
                mean_vs_empty=statistics.fmean(value['vs_empty']
                                                for value in by_seed.values()),
                immediate_mean_vs_previous=statistics.fmean(
                    value['immediate_vs_previous'] for value in by_seed.values()),
                delayed_mean_vs_previous=statistics.fmean(
                    value['delayed_vs_previous'] for value in by_seed.values()),
                identical_candidate_and_previous=same_document,
                seed_sign_conflict=min(advantages) < 0 < max(advantages),
                nonzero_seed_effects=sum(abs(value) > 1e-12
                                         for value in advantages),
                seed_effects=by_seed, per_target_effects=effects)


def bind_online_writer_sample(sample: dict, report: dict) -> dict:
    """Attach a replicated online-chain advantage to one frozen writer rollout.

    A single positive seed, an opposed pair of seed effects, or an unchanged
    document receives no policy-gradient reward. The paired estimate remains
    in the audit for later evidence review; the optimizer can still apply its
    frozen-reference KL term to the returned zero-advantage sample.
    """
    if not isinstance(sample, dict) or not isinstance(report, dict):
        raise ValueError('Missing writer sample or online-chain report')
    text = sample.get('text')
    messages = sample.get('messages')
    if not isinstance(text, str) or not isinstance(messages, list):
        raise ValueError('Missing generated document or bound writer input')
    input_sha256 = hashlib.sha256(json.dumps(
        messages, ensure_ascii=False, sort_keys=True,
        separators=(',', ':')).encode()).hexdigest()
    document_sha256 = hashlib.sha256(text.encode()).hexdigest()
    if (sample.get('source_input_sha256') != input_sha256 or
            report.get('source_input_sha256') != input_sha256 or
            sample.get('source_trace_sha256') !=
            report.get('source_trace_sha256') or
            sample.get('writer_freeze_sha256') !=
            report.get('writer_freeze_sha256') or
            report.get('memory_sha256', {}).get('candidate') != document_sha256):
        raise ValueError('Writer sample differs from frozen source or candidate')
    if report.get('seed_count', 0) < 2 or not isinstance(
            report.get('seed_effects'), dict):
        raise ValueError('Missing replicated online-chain evidence')
    effects = [value['vs_previous'] for value in report['seed_effects'].values()]
    replicated = (not report['identical_candidate_and_previous'] and
                  len(effects) == report['seed_count'] and
                  (all(value > 1e-12 for value in effects) or
                   all(value < -1e-12 for value in effects)))
    status = ('unchanged_document' if report['identical_candidate_and_previous']
              else 'replicated_effect' if replicated else 'needs_more_evidence')
    return dict(sample, advantage=(report['advantage'] if replicated else 0.0),
                delta=report['mean_vs_empty'],
                paired_credit_status=status,
                paired_credit_unclipped=report['mean_unclipped_advantage'],
                reviewed_target_order_sha256=report['reviewed_target_order_sha256'])
