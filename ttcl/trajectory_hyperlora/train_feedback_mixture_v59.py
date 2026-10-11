"""Matched joint positive/negative updates for H and direct LoRA.

The caller supplies a fixed chronological schedule and already reviewed inputs.
No sampling, label inference, checkpoint loading, or task scoring occurs here.
"""
from __future__ import annotations

import torch

from ttcl.trajectory_hyperlora.train_alf_next_task import target_loss
from ttcl.trajectory_hyperlora.off_goal_unlikelihood_v58 import rejected_object_loss


def fit_feedback_mixture(agent, tokenizer, optimizer, schedule, *, device,
                         activate, parameters, negative_weight=.1):
    """Each schedule entry has one approved positive and one off-goal negative.

    activate(example) must install the appropriate generated or static factors.
    Prefixes and source encodings are prepared before training by the caller.
    Both arms must use identical target IDs, pair order, weight, and step count.
    """
    if not schedule or not 0 < negative_weight <= 1:
        raise ValueError('Nonempty declared schedule and weight in (0,1] required')
    params=list(parameters)
    if not params:
        raise ValueError('No trainable parameters')
    previous_mode=agent.training
    agent.train()
    rows=[]
    try:
        for positive,negative in schedule:
            if positive['available_after_index'] > positive['update_index'] or negative['available_after_index'] > negative['update_index']:
                raise ValueError('Future target used in an earlier update')
            if positive['update_index'] != negative['update_index']:
                raise ValueError('Mismatched chronological update')
            optimizer.zero_grad(set_to_none=True)
            activate(positive)
            positive_loss=target_loss(agent,tokenizer,positive['annotation'],positive['prefix'],device)
            if not torch.isfinite(positive_loss):
                raise ValueError('Nonfinite positive loss')
            # Backprop separately to bound activation memory; one optimizer step.
            positive_loss.backward()
            activate(negative)
            negative_loss,logp=rejected_object_loss(agent,tokenizer,negative['annotation'],negative['prefix'],device)
            if not torch.isfinite(negative_loss):
                raise ValueError('Nonfinite negative loss')
            (negative_weight*negative_loss).backward()
            norm=torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
            optimizer.step()
            rows.append({'update_index':positive['update_index'],
                'positive_binding':positive['input_content_sha256'],
                'negative_binding':negative['input_content_sha256'],
                'positive_ce':float(positive_loss.detach()),
                'negative_unlikelihood':float(negative_loss.detach()),
                'wrong_object_probability':float(logp.exp()),'gradient_norm':float(norm)})
            agent.set_source(None)
    finally:
        agent.set_source(None)
        agent.train(previous_mode)
    return {'optimizer_steps':len(rows),'positive_target_passes':len(rows),
            'negative_target_passes':len(rows),'negative_weight':negative_weight,
            'rows':rows}
