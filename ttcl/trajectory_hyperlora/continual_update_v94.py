"""A fixed-A, persistent-B learned update rule. No target query in the updater."""
from __future__ import annotations
import torch
from torch import nn

def effective_norm(b, a):
    """Frobenius norm of B A / rank without materializing dense weights."""
    b=b.squeeze(0)
    return ((b.T@b)*(a@a.T)).sum().clamp_min(0).sqrt()/a.shape[0]

def bound(b, a, limit):
    # Add epsilon before sqrt so gradients at zero B are finite.
    z=b.squeeze(0)
    norm=(((z.T@z)*(a@a.T)).sum().clamp_min(0)+1e-16).sqrt()/a.shape[0]
    return b*(limit/norm).clamp(max=1)

class UpdateRule(nn.Module):
    def __init__(self, hidden, shapes):
        super().__init__();self.shapes=shapes
        self.encoder=nn.Sequential(nn.LayerNorm(2*hidden),nn.Linear(2*hidden,128),nn.Tanh())
        self.state_encoder=nn.Sequential(nn.Linear(4*len(shapes),128),nn.Tanh())
        self.heads=nn.ModuleList([nn.Linear(128,o*r) for o,r in shapes])
        self.gate=nn.Linear(128,1)
        nn.init.zeros_(self.gate.weight);nn.init.constant_(self.gate.bias,-2)
        for h in self.heads:nn.init.zeros_(h.weight);nn.init.zeros_(h.bias)
    def forward(self, fields, old, aa, weight_norms):
        stats=torch.stack([v for b in old for v in (b.mean(),(b.square().mean()+1e-16).sqrt(),b.abs().max(),b.abs().mean())]).unsqueeze(0)
        latent=self.encoder(torch.cat([fields['last'],fields['mean']],-1))+self.state_encoder(stats)
        gate=self.gate(latent).sigmoid()
        out=[]
        for head,shape,b,a,n in zip(self.heads,self.shapes,old,aa,weight_norms,strict=True):
            delta=head(latent).reshape(1,*shape)
            delta=bound(delta,a,0.05*n)
            out.append(bound(b+gate.reshape(1,1,1)*delta,a,0.15*n))
        return out,gate

def trajectory_text(ep):
    """Exact observations/actions/feedback, no hidden rules or expert labels."""
    lines=['Completed own XLand attempt. Official return: '+str(ep['return'])]
    for i,s in enumerate(ep['trajectory']):
        def grid(obs):return '/'.join(' '.join(f'{t[0]},{t[1]}' for t in row) for row in obs)
        lines.append(f"{i}: view={grid(s['observation'])}; action={s['action']}; next={grid(s['next_observation'])}; reward={s['reward']}; done={s['done']}")
    return '\n'.join(lines)

def positive_targets(episodes, max_each=8):
    """Only environment-confirmed rewarding transitions, with preceding context.

No guessed correction label is assigned to failed trajectories. Include up to
three preceding actions of each observed positive reward, capped per episode.
"""
    rows=[]
    for ep in episodes:
        indices=set()
        for i,s in enumerate(ep['trajectory']):
            if s['reward']>0:indices.update(range(max(0,i-3),i+1))
        for i in sorted(indices)[-max_each:]:
            s=ep['trajectory'][i]
            rows.append(dict(observation=s['observation'],action=s['action'],episode_sha256=ep['content_sha256'],step=i))
    return rows
