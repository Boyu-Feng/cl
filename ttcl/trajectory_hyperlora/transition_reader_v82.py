"""Query-conditioned transition memory to bounded LoRA factors.

Input features come from a frozen encoder. Each row is one completed past
observation/action/feedback transition. Chronology is enforced by the data
builder, not by this tensor module. No test outcome or action label is an input.
"""
from collections.abc import Sequence
import math
import torch
from torch import nn


def effective_squared_norm(values, bases):
    """Squared norm of concatenated effective BA/r matrices, without forming BA."""
    terms=[]
    for b,a in zip(values,bases,strict=True):
        a=a.float();b=b.float()
        gram=a@a.T
        terms.append(((b@gram)*b).sum()/a.shape[0]**2)
    return torch.stack(terms).sum().clamp_min(0.)


class TransitionReader(nn.Module):
    def __init__(self, feature_dim:int, factor_shapes:Sequence[tuple[int,int]], width:int=64):
        super().__init__()
        if not factor_shapes or feature_dim<1 or width<1:raise ValueError('Invalid dimensions')
        self.feature_dim=feature_dim;self.factor_shapes=list(factor_shapes);self.width=width
        self.query=nn.Sequential(nn.LayerNorm(feature_dim),nn.Linear(feature_dim,width,bias=False))
        self.key=nn.Sequential(nn.LayerNorm(feature_dim),nn.Linear(feature_dim,width,bias=False))
        self.value=nn.Sequential(nn.LayerNorm(2*feature_dim),nn.Linear(2*feature_dim,width),nn.Tanh())
        self.read_norm=nn.LayerNorm(width)
        self.heads=nn.ModuleList([nn.Linear(width,out*rank,bias=False) for out,rank in factor_shapes])
        for head in self.heads:nn.init.zeros_(head.weight)

    def forward(self, query, observations, actions, feedback, mask=None):
        if query.shape!=(1,self.feature_dim):raise ValueError('Exactly one current-state query required')
        if observations.ndim!=2 or observations.shape[1]!=self.feature_dim:raise ValueError('Invalid observations')
        if actions.shape!=observations.shape or feedback.shape!=observations.shape:raise ValueError('Unpaired transition fields')
        n=len(observations)
        if mask is None:mask=torch.ones(n,device=observations.device,dtype=torch.bool)
        if mask.shape!=(n,) or mask.dtype!=torch.bool:raise ValueError('Invalid transition mask')
        if not n or not bool(mask.any()):
            return [query.new_zeros((1,out,rank)) for out,rank in self.factor_shapes],query.new_zeros((1,n))
        # Select before encoding so masked padding cannot contaminate normalization.
        indices=mask.nonzero(as_tuple=True)[0]
        keys=self.key(observations[indices].float())
        values=self.value(torch.cat((actions[indices],feedback[indices]),dim=-1).float())
        scores=self.query(query.float())@keys.T/math.sqrt(self.width)
        weights=torch.softmax(scores,dim=-1)
        read=self.read_norm(weights@values)
        factors=[head(read).reshape(1,out,rank) for head,(out,rank) in zip(self.heads,self.factor_shapes,strict=True)]
        full=weights.new_zeros((1,n)).index_copy(1,indices,weights)
        return factors,full

    def adapted_factors(self, query, observations, actions, feedback, shared, bases, max_relative=.25, mask=None):
        if not 0<=max_relative<=1:raise ValueError('Invalid norm cap')
        if len(shared)!=len(self.factor_shapes) or len(bases)!=len(shared):raise ValueError('Invalid shared factors')
        for b,a,(out,rank) in zip(shared,bases,self.factor_shapes,strict=True):
            if b.shape!=(1,out,rank) or a.ndim!=2 or a.shape[0]!=rank:raise ValueError('Incompatible LoRA basis')
        delta,weights=self(query,observations,actions,feedback,mask)
        reference=effective_squared_norm(shared,bases).sqrt().detach()
        magnitude=effective_squared_norm(delta,bases).clamp_min(1e-20).sqrt()
        scale=(max_relative*reference/magnitude).clamp(max=1.)
        adapted=[b+d*scale for b,d in zip(shared,delta,strict=True)]
        return adapted,{'weights':weights,'scale':scale,'raw_effective_norm':magnitude,'reference_effective_norm':reference}
