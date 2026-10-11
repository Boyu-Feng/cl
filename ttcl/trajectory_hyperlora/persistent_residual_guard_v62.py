"""Additive persistent residual with a past-success loss guard.

The scorer may read only already reviewed past successful actions. Its losses
are a proxy for retention, not proof of environment success. The caller must
keep A fixed and bind the scorer's current generator state and anchor contents.
"""
from __future__ import annotations

import math
import torch


def guarded_add(state, delta, *, norm, reference_norm, score,
                max_relative=.25, tolerance=.01,
                candidate_steps=(1., .5, .25, .125)):
    if reference_norm<=0 or not math.isfinite(reference_norm):
        raise ValueError('A positive frozen reference norm is required')
    if not 0<max_relative<=1 or tolerance<0 or not candidate_steps:
        raise ValueError('Invalid predeclared guard parameters')
    if any(not 0<eta<=1 for eta in candidate_steps):
        raise ValueError('Candidate write steps must lie in (0,1]')
    delta=[x.detach().cpu().float().clone() for x in delta]
    if not delta or any(not torch.isfinite(x).all() for x in delta):
        raise ValueError('Invalid residual direction')
    previous=[torch.zeros_like(x) for x in delta] if state is None else [x.detach().cpu().float().clone() for x in state]
    if len(previous)!=len(delta) or any(a.shape!=b.shape for a,b in zip(previous,delta,strict=True)):
        raise ValueError('Persistent LoRA coordinates changed')
    if any(not torch.isfinite(x).all() for x in previous):
        raise ValueError('Nonfinite persistent state')
    baseline=[float(x) for x in score(previous)]
    if not baseline or any(not math.isfinite(x) or x<0 for x in baseline):
        raise ValueError('At least one finite reviewed success loss is required')
    trace={'reference_norm':reference_norm,'max_relative':max_relative,'tolerance':tolerance,
        'baseline_losses':baseline,'trials':[],'accepted_step':0.,'changed':False}
    limit=max_relative*reference_norm
    for eta in candidate_steps:
        proposed=[a+eta*b for a,b in zip(previous,delta,strict=True)]
        magnitude=float(norm(proposed))
        if not math.isfinite(magnitude) or magnitude<0:
            raise ValueError('Invalid effective residual norm')
        scale=min(1.,limit/magnitude) if magnitude else 1.
        proposed=[x*scale for x in proposed]
        change=float(norm([a-b for a,b in zip(proposed,previous,strict=True)]))
        if change<=1e-8*reference_norm:
            trace['trials'].append({'step':eta,'scale':scale,'skipped':'negligible_effective_change'})
            continue
        losses=[float(x) for x in score(proposed)]
        if len(losses)!=len(baseline) or any(not math.isfinite(x) or x<0 for x in losses):
            raise ValueError('Changed or nonfinite anchor evaluation')
        accepted=all(after<=before+tolerance for before,after in zip(baseline,losses,strict=True))
        trace['trials'].append({'step':eta,'scale':scale,'losses':losses,'accepted':accepted,
            'effective_norm':float(norm(proposed)),'effective_change_norm':change})
        if accepted:
            trace.update(accepted_step=eta,changed=True)
            return proposed,trace
    return state,trace
