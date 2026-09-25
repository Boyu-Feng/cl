"""Pure reward and routing rules for an immutable cooperative experiment."""
from __future__ import annotations

import math
import statistics

from ttcl.experience_lab.protocol import binding


def paired_advantages(rewards, scale):
    """Keep the signed previous-memory baseline; do not center away all harm."""
    if not math.isfinite(scale) or scale < 1:
        raise ValueError('Training-only reward scale must be finite and >= 1')
    branches = ['empty', 'keep', *[f'candidate_{i}' for i in range(8)]]
    if set(rewards) != set(branches):
        raise ValueError('Exactly eight sampled candidates and two controls are required')
    lengths = {len(rewards[b]) for b in branches}
    if len(lengths) != 1 or not next(iter(lengths)):
        raise ValueError('Missing paired probe grid')
    if any(x is None or not math.isfinite(float(x)) for b in branches for x in rewards[b]):
        raise ValueError('Missing official reward is not zero')
    output = {}
    for branch in branches[2:]:
        delta = [r-p for r,p in zip(rewards[branch], rewards['keep'])]
        # A training-frozen scale can reduce large domain units, but never amplify
        # near-zero differences. Official rewards in result files are unchanged.
        utility = statistics.mean(delta) - .5*statistics.mean(max(0., -d) for d in delta)
        output[branch] = {'writer': max(-3., min(3., utility/scale)),
            'reader': [max(-3., min(3., (r-e)/scale))
                       for r,e in zip(rewards[branch], rewards['empty'])],
            'raw_delta_keep': statistics.mean(delta), 'raw_utility': utility}
    return output


def evaluation_routes():
    """Fixed components are crossed at evaluation; these are removal ablations."""
    return {
        'none': {'writer': None, 'reader': 'frozen-actor'},
        'untrained': {'writer': 'frozen-actor', 'reader': 'frozen-actor'},
        'original_delta': {'writer': 'original_delta', 'reader': 'frozen-actor'},
        'ppo8_writer': {'writer': 'writer_only', 'reader': 'frozen-actor'},
        'dual_writer_base': {'writer': 'dual_writer', 'reader': 'frozen-actor'},
        'old_writer_reader': {'writer': 'original_delta', 'reader': 'dual_reader'},
        'dual': {'writer': 'dual_writer', 'reader': 'dual_reader'},
        'reader_no_text': {'writer': None, 'reader': 'dual_reader'},
    }


def validate_sample(sample, expected_binding=None):
    if sample.get('temperature') != 1. or sample.get('top_p') != 1.:
        raise ValueError('PPO samples must come from the untruncated temperature-1 policy')
    n = len(sample['input_ids']) - sample['prompt_length']
    if sample['prompt_length'] < 1 or n < 1 or n != len(sample['old_logp']):
        raise ValueError('Token IDs and behavior probabilities are not aligned')
    if any(not math.isfinite(x) or x > 1e-5 for x in sample['old_logp']):
        raise ValueError('Invalid behavior log probabilities')
    if expected_binding and sample['input_binding'] != expected_binding:
        raise ValueError('Rollout is bound to a different public input')
    if sample['sample_binding'] != binding({k: sample[k] for k in
            ['input_ids', 'prompt_length', 'old_logp', 'input_binding', 'model', 'seed']}):
        raise ValueError('PPO sample content binding failed')


def correction_settings(mode='strict'):
    """Declare the sampling/backend correction before freezing the run."""
    if mode == 'strict':
        return {'rollout_correction': 'strict'}
    if mode != 'decoupled_token_is':
        raise ValueError('Unknown rollout correction: '+mode)
    return dict(rollout_correction='decoupled_token_is',is_max_weight=2.,
        reject_token_ratio=4.,max_rejected_fraction=.2,max_domain_rejected_fraction=.4,
        correction_rule='Freeze training-backend old logps before every update; token IS=min(2,exp(old_train-old_rollout)); reject whole generated action if any token ratio outside [1/4,4]. Retain original denominators and report every rejection. No reward-based filtering.',
        original_mae_threshold='0.15 retained as a diagnostic. Strict old mode remains unchanged; corrected mode has explicit ratio and rejected-fraction guards.')
