"""Persistent trajectory-to-LoRA update rule.

The updater reads only completed public transitions. A GRU state carries
experience across episodes; a gated generator emits a bounded residual B.
Training should use held-out future-query loss, not reconstruction of a target B.
"""
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F

class TransitionEncoder(nn.Module):
    def __init__(self, state_dim: int = 52, hidden: int = 128):
        super().__init__()
        # state_before, action, state_after, reward, done, legal/feedback change
        self.action=nn.Embedding(6,16)
        self.net=nn.Sequential(nn.LayerNorm(2*state_dim+16+4),nn.Linear(2*state_dim+20,hidden),nn.GELU(),nn.Linear(hidden,hidden))
    def forward(self,before,action,after,reward,done,feedback_change=None):
        if feedback_change is None: feedback_change=torch.zeros_like(reward)
        x=torch.cat((before.float(),self.action(action),after.float(),reward.float().unsqueeze(-1),done.float().unsqueeze(-1),feedback_change.float().unsqueeze(-1),torch.ones_like(reward).unsqueeze(-1)),dim=-1)
        return self.net(x)

class PersistentUpdateRule(nn.Module):
    def __init__(self, shapes, state_dim=52, width=128, rank=4):
        super().__init__();self.shapes=tuple(shapes);self.rank=rank
        self.transition=TransitionEncoder(state_dim,width)
        self.memory=nn.GRUCell(width,width)
        self.state_norm=nn.LayerNorm(width)
        self.heads=nn.ModuleList([nn.Linear(width,o*r) for o,r in shapes])
        self.gate=nn.Sequential(nn.LayerNorm(width),nn.Linear(width,1))
        self.confidence=nn.Sequential(nn.LayerNorm(width),nn.Linear(width,1))
        for h in self.heads:nn.init.zeros_(h.weight);nn.init.zeros_(h.bias)
        nn.init.zeros_(self.gate[1].weight);nn.init.constant_(self.gate[1].bias,-2.0)
        nn.init.zeros_(self.confidence[1].weight);nn.init.constant_(self.confidence[1].bias,-2.0)
    def step_memory(self,m,episode):
        """episode fields are completed public transitions, in chronological order."""
        for row in episode:
            z=self.transition(row['before'],row['action'],row['after'],row['reward'],row['done'],row.get('feedback_change'))
            m=self.memory(z,m)
        return self.state_norm(m)
    def bound(self,b,a,limit):
        z=b.squeeze(0);n=(((z.T@z)*(a@a.T)).sum().clamp_min(0)+1e-16).sqrt()/a.shape[0]
        return b*(limit/n).clamp(max=1)
    def emit(self,m,old,aa,base_norms):
        m=self.state_norm(m);gate=torch.sigmoid(self.gate(m));confidence=torch.sigmoid(self.confidence(m))
        out=[]
        for head,shape,b,a,n in zip(self.heads,self.shapes,old,aa,base_norms,strict=True):
            delta=head(m).reshape(1,*shape)
            delta=self.bound(delta,a,.05*n)
            out.append(self.bound(b+gate.reshape(1,1,1)*confidence.reshape(1,1,1)*delta,a,.15*n))
        return out,gate,confidence
    def update(self,m,episode,old,aa,base_norms):
        m=self.step_memory(m,episode);return m,*self.emit(m,old,aa,base_norms)

def transition_rows(trajectory,device):
    """Convert saved public XLand transitions; reward is never inferred."""
    out=[];prev=None
    for row in trajectory:
        before=torch.tensor([sum(([x[0],x[1]] for r in row['observation'] for x in r),[])] ,device=device).float()
        after=torch.tensor([sum(([x[0],x[1]] for r in row['next_observation'] for x in r),[])],device=device).float()
        out.append(dict(before=before,action=torch.tensor([row['action']],device=device),after=after,reward=torch.tensor([row['reward']],device=device),done=torch.tensor([float(row['done'])],device=device),feedback_change=torch.tensor([float(prev is not None and row['observation']!=prev)],device=device)))
        prev=row['next_observation']
    return out
