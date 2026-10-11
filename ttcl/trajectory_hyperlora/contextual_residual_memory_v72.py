"""Causal retrieval of individually utility-verified residuals in a fixed A basis.

This module does not infer utility from failure. Callers must supply reviewed
paired rewards and a feature encoder binding. Real online efficacy is untested.
"""
import math

import torch


def unit_key(value):
    value=torch.as_tensor(value).detach().cpu().float().flatten().clone()
    magnitude=float(value.norm())
    if not value.numel() or not torch.isfinite(value).all() or magnitude<=0:
        raise ValueError('A finite nonzero context key is required')
    return value/magnitude


class ContextualResidualMemory:
    def __init__(self, *, basis_sha256, encoder_sha256, top_k=3, max_relative=.25):
        if not basis_sha256 or not encoder_sha256 or top_k<1 or not 0<max_relative<=1:
            raise ValueError('Invalid memory configuration')
        self.basis_sha256=basis_sha256
        self.encoder_sha256=encoder_sha256
        self.top_k=top_k
        self.max_relative=max_relative
        self.entries=[]

    def check_coordinates(self,basis_sha256,encoder_sha256):
        if basis_sha256!=self.basis_sha256 or encoder_sha256!=self.encoder_sha256:
            raise ValueError('Residual basis or context encoder changed')

    def add(self, *, key, delta, evidence_id, audit_sha256, source_task,
            available_before_task, paired_gain, gain_over_random,
            basis_sha256, encoder_sha256):
        self.check_coordinates(basis_sha256,encoder_sha256)
        if not evidence_id or not audit_sha256 or any(e['evidence_id']==evidence_id for e in self.entries):
            raise ValueError('Missing evidence binding or duplicate write')
        if source_task<0 or available_before_task<=source_task:
            raise ValueError('A correction is available only after its source task and retries')
        if paired_gain not in (-1.,0.,1.) or gain_over_random not in (-1.,0.,1.):
            raise ValueError('Expected differences of binary paired environment rewards')
        # A random-direction rescue does not establish this candidate's utility.
        if paired_gain<=0 or gain_over_random<=0:
            return False
        normalized=unit_key(key)
        factors=[x.detach().cpu().float().clone() for x in delta]
        if not factors or any(not torch.isfinite(x).all() for x in factors):
            raise ValueError('Invalid residual factors')
        if self.entries:
            first=self.entries[0]
            if normalized.shape!=first['key'].shape or [x.shape for x in factors]!=[x.shape for x in first['delta']]:
                raise ValueError('Context or residual dimensions changed')
        self.entries.append({'key':normalized,'delta':factors,'evidence_id':evidence_id,
            'audit_sha256':audit_sha256,'source_task':source_task,
            'available_before_task':available_before_task,'paired_gain':paired_gain,
            'gain_over_random':gain_over_random})
        return True

    def read(self, *, key, current_task, reference_factors, effective_norm,
             basis_sha256, encoder_sha256):
        self.check_coordinates(basis_sha256,encoder_sha256)
        query=unit_key(key)
        if current_task<0:raise ValueError('Invalid task index')
        ranked=[]
        for e in self.entries:
            if e['available_before_task']>current_task or e['source_task']>=current_task:
                continue
            if query.shape!=e['key'].shape:raise ValueError('Context dimension changed')
            similarity=float(query@e['key'])
            if similarity>0:ranked.append((similarity,e))
        ranked.sort(key=lambda x:(-x[0],x[1]['evidence_id']))
        ranked=ranked[:self.top_k]
        if not ranked:return None,{'selected':[],'gate':0.,'read_scale':0.}
        total=sum(x[0] for x in ranked)
        gate=min(1.,ranked[0][0])
        if [x.shape for x in reference_factors]!=[x.shape for x in ranked[0][1]['delta']]:
            raise ValueError('Current adapter dimensions changed')
        result=[torch.zeros_like(x,device='cpu',dtype=torch.float32) for x in reference_factors]
        selected=[]
        for similarity,e in ranked:
            weight=similarity/total
            result=[a+gate*weight*b for a,b in zip(result,e['delta'],strict=True)]
            selected.append({'evidence_id':e['evidence_id'],'similarity':similarity,'weight':weight})
        reference=float(effective_norm(reference_factors));magnitude=float(effective_norm(result))
        if any(not math.isfinite(x) or x<0 for x in (reference,magnitude)):
            raise ValueError('Invalid effective adapter norm')
        scale=min(1.,self.max_relative*reference/magnitude) if magnitude else 0.
        return [x*scale for x in result],{'selected':selected,'gate':gate,'read_scale':scale,
                'reference_effective_norm':reference,'applied_effective_norm':magnitude*scale}
