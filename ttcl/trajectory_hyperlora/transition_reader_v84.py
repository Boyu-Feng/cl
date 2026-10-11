"""Frozen semantic selection with centered transition values.

Candidate motivated by v82/v83 nearly query-invariant effective updates.
No query labels or rewards are used by this reader. Centering removes a common
value contribution, but cannot by itself prove useful experience or transfer.
"""
import torch
from torch import nn
from torch.nn import functional as F
from ttcl.trajectory_hyperlora.transition_reader_v82 import TransitionReader as BaseReader


class TransitionReader(BaseReader):
    def __init__(self, feature_dim, factor_shapes, width=64, top_k=8, temperature=.05):
        super().__init__(feature_dim, factor_shapes, width)
        if top_k<1 or temperature<=0:raise ValueError('Invalid semantic selection settings')
        del self.query
        del self.key
        self.top_k=top_k;self.temperature=temperature
        # Learned bias here would reintroduce a source-independent update.
        self.read_norm=nn.LayerNorm(width,elementwise_affine=False)

    def forward(self, query, observations, actions, feedback, mask=None):
        if query.shape!=(1,self.feature_dim):raise ValueError('Exactly one current query required')
        if observations.ndim!=2 or observations.shape[1]!=self.feature_dim:raise ValueError('Invalid observations')
        if actions.shape!=observations.shape or feedback.shape!=observations.shape:raise ValueError('Unpaired fields')
        n=len(observations)
        if mask is None:mask=torch.ones(n,device=observations.device,dtype=torch.bool)
        if mask.shape!=(n,) or mask.dtype!=torch.bool:raise ValueError('Invalid mask')
        if not n or not bool(mask.any()):
            return [query.new_zeros((1,out,rank)) for out,rank in self.factor_shapes],query.new_zeros((1,n))
        indices=mask.nonzero(as_tuple=True)[0]
        scores=F.normalize(query.float(),dim=-1)@F.normalize(observations[indices].float(),dim=-1).T
        # Include ties at the kth score so a joint row permutation remains invariant.
        threshold=scores.topk(min(self.top_k,len(indices)),dim=-1).values[:,-1:]
        weights=torch.softmax((scores/self.temperature).masked_fill(scores<threshold,float('-inf')),dim=-1)
        values=self.value(torch.cat((actions[indices],feedback[indices]),dim=-1).float())
        # Subtract before summation: an identical value bank yields exact zero.
        offsets=values-values[:1]
        centered=offsets-offsets.mean(dim=0,keepdim=True)
        read=self.read_norm(weights@centered)
        factors=[head(read).reshape(1,out,rank) for head,(out,rank) in zip(self.heads,self.factor_shapes,strict=True)]
        return factors,weights.new_zeros((1,n)).index_copy(1,indices,weights)
