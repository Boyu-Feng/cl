"""Generate only an experience-dependent increment over a shared LoRA.

B(s,q) = B_shared + cap(H(s,q) - H(empty,q)). For empty input the
increment is exactly zero; generator head biases cancel algebraically.
"""
from contextlib import contextmanager

import torch

from ttcl.trajectory_hyperlora.alfworld_failure_retry_residual_v31b import effective_norm


def centered_factors(agent,fields,target_fields,empty_fields,shared,*,max_relative=.25):
    if not 0<max_relative<=1:raise ValueError('Invalid correction cap')
    if len(shared)!=len(agent.adapters):raise ValueError('Shared factor count changed')
    agent.set_source(fields,target_fields=target_fields)
    actual=[x.b for x in agent.adapters]
    agent.set_source(empty_fields,target_fields=target_fields)
    empty=[x.b for x in agent.adapters]
    delta=[a-b for a,b in zip(actual,empty,strict=True)]
    if any(s.shape!=d.shape for s,d in zip(shared,delta,strict=True)):
        raise ValueError('Shared and generated factors have different shapes')
    reference=effective_norm(agent,shared)
    magnitude=effective_norm(agent,delta)
    if reference<=0:raise ValueError('Train a nonzero shared LoRA before conditional updates')
    # Treat clipping scale as a trust-region coefficient; preserve gradients
    # through both H branches, including when zero-initialized delta has norm0.
    scale=min(1.,max_relative*reference/magnitude) if magnitude else 1.
    output=[s.to(d)+scale*d for s,d in zip(shared,delta,strict=True)]
    for layer,b in zip(agent.adapters,output,strict=True):layer.b=b
    return output,{'reference_norm':reference,'unclipped_delta_norm':magnitude,
                   'scale':scale,'applied_delta_norm':scale*magnitude}


@contextmanager
def centered_reader(agent,empty_fields,shared,*,max_relative=.25):
    """Wrap actor set_source; source encodings still disable all adapters."""
    original=agent.set_source
    def patched(fields,*args,**kwargs):
        if fields is None:return original(fields,*args,**kwargs)
        if args:raise ValueError('Use explicit target_fields for centered reader')
        query=kwargs.get('target_fields')
        if query is None:raise ValueError('Current task context is required')
        # Temporarily restore the original method so factor construction does
        # not recursively invoke this wrapper.
        agent.set_source=original
        try:return centered_factors(agent,fields,query,empty_fields,shared,max_relative=max_relative)
        finally:agent.set_source=patched
    agent.set_source=patched
    try:yield
    finally:agent.set_source=original;original(None)
