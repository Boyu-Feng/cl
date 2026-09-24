"""Pure selection logic; no task answers or test outcomes are accepted here."""
from __future__ import annotations

import hashlib
import json
import math
import statistics

EVIDENCE_SUFFIX = """
Before writing, distinguish executed actions and observed results from guesses.
Retain earlier supported procedures unless the new evidence contradicts them.
One unsuccessful trajectory does not establish that a procedure never works.
Keep concrete tool syntax and necessary preconditions, but scope instance-specific
facts to their own task. If no reliable addition is supported, return the previous
document unchanged. Do not invent a successful action, rule, or causal explanation.
"""


def binding(messages):
    return hashlib.sha256(json.dumps(messages, ensure_ascii=False,
                                     sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def finite(values):
    return bool(values) and all(v is not None and math.isfinite(float(v)) for v in values)


def utility(values, previous):
    if len(values) != len(previous) or not finite(values) or not finite(previous):
        raise ValueError('Utility requires complete finite paired official rewards')
    changes = [a-b for a, b in zip(values, previous)]
    return statistics.mean(changes) - .5*statistics.mean(max(0., -x) for x in changes)


def select_pair(texts, rewards, screening_seed, confirmation_seed):
    """Select using seed A, confirm using seed B; never relabel ties as wins.

    rewards[branch][seed] are aligned vectors over the same future TRAIN tasks.
    The candidate is selected without access to the confirmation outcomes.
    """
    seeds = [str(screening_seed), str(confirmation_seed)]
    if seeds[0] == seeds[1]:
        raise ValueError('Screening and confirmation seeds must differ')
    if not {'keep', 'empty'} <= rewards.keys():
        raise ValueError('Missing counterfactual controls')
    lengths = {len(values) for group in rewards.values() for values in group.values()}
    if len(lengths) != 1 or not lengths or 0 in lengths:
        return {'accepted': False, 'reason': 'incomplete_paired_grid'}
    if any(set(group) != set(seeds) or not all(finite(v) for v in group.values())
           for group in rewards.values()):
        return {'accepted': False, 'reason': 'missing_official_reward'}
    eligible = [key for key, text in texts.items() if key != 'empty' and text.strip()]
    if len(eligible) < 2:
        return {'accepted': False, 'reason': 'too_few_nonempty_candidates'}
    first, confirm = seeds
    values = {key: utility(rewards[key][first], rewards['keep'][first]) for key in eligible}
    order = sorted(eligible, key=lambda key: (-values[key], key))
    chosen, rejected = order[0], order[-1]
    audit = {'accepted': False, 'chosen': chosen, 'rejected': rejected,
             'screen_utility': values, 'screening_seed': first, 'confirmation_seed': confirm}
    if texts[chosen] == texts[rejected] or values[chosen]-values[rejected] <= 1e-8:
        return dict(audit, reason='screen_tie')
    gaps = [a-b for a, b in zip(rewards[chosen][confirm], rewards[rejected][confirm])]
    audit['confirmation_margin'] = statistics.mean(gaps)
    audit['confirmation_vs_keep'] = utility(rewards[chosen][confirm], rewards['keep'][confirm])
    audit['confirmation_vs_empty'] = statistics.mean(a-b for a, b in
        zip(rewards[chosen][confirm], rewards['empty'][confirm]))
    if audit['confirmation_margin'] <= 1e-8:
        return dict(audit, reason='unconfirmed_preference')
    if audit['confirmation_vs_keep'] < -1e-8 or audit['confirmation_vs_empty'] < -1e-8:
        return dict(audit, reason='confirmed_but_harmful')
    return dict(audit, accepted=True, reason='independent_seed_confirmed')


def balanced_order(rows, random_seed):
    """Equal domain sampling; repeated samples are explicitly counted as repeats."""
    import random
    from collections import defaultdict
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[row['domain']].append(i)
    rng = random.Random(random_seed)
    for group in groups.values():
        rng.shuffle(group)
    maximum = max(map(len, groups.values()))
    result = []
    for i in range(maximum):
        domains = sorted(groups)
        rng.shuffle(domains)
        result += [groups[d][i % len(groups[d])] for d in domains]
    return result
