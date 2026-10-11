"""Negative supervision on an observed wrong object span, with no invented action.

Callers must bind annotations, replay-check their episodes, enforce chronological
sources, and include matched successful-action supervision in a training run.
This primitive alone is not a complete training protocol.
"""
from __future__ import annotations

import torch


def object_span_tokens(tokenizer, annotation):
    action = annotation['observed_action']
    label = annotation['supervision']
    if label['label'] != 'off_goal_transformation' or label['actual_object_type'] == label['expected_object_type']:
        raise ValueError('Expected an explicit off-goal operation annotation')
    start = len(label['operation']) + 1
    end = start + len(label['actual_object_type'])
    if action[:start] != label['operation']+' ' or action[start:end] != label['actual_object_type']:
        raise ValueError('Observed action and annotated object span differ')
    encoded = tokenizer(action+tokenizer.eos_token, add_special_tokens=False,
                        return_offsets_mapping=True)
    indices=[]
    covered=set()
    for i,(left,right) in enumerate(encoded.offset_mapping):
        if left < end and right > start:
            outside = action[left:start] if left < start else ''
            outside += action[end:right] if right > end else ''
            if outside.strip():
                raise ValueError('A token spans non-object content; do not penalize it')
            indices.append(i)
            covered.update(range(max(left,start),min(right,end)))
    if not indices or covered != set(range(start,end)):
        raise ValueError('Tokenizer did not fully cover the annotated object')
    return encoded.input_ids, indices


def sequence_unlikelihood(log_probability):
    # -log(1-p), stable near p=1. Float32 avoids low-precision cancellation.
    value=log_probability.float()
    if not torch.isfinite(value).all() or (value > 0).any():
        raise ValueError('Expected finite log probability <= 0')
    return -torch.log((-torch.expm1(value)).clamp_min(torch.finfo(value.dtype).tiny))


def rejected_object_loss(agent, tokenizer, annotation, prefix, device):
    if not prefix:
        raise ValueError('Actor prefix is required')
    answer,indices=object_span_tokens(tokenizer,annotation)
    ids=torch.tensor([prefix+answer],device=device)
    logits=agent.model(input_ids=ids,use_cache=False).logits
    positions=torch.tensor([len(prefix)+i-1 for i in indices],device=device)
    targets=torch.tensor([answer[i] for i in indices],device=device)
    selected=logits[0,positions].float().log_softmax(dim=-1)
    log_probability=selected.gather(1,targets[:,None]).sum()
    return sequence_unlikelihood(log_probability), log_probability.detach()
