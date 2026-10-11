"""Training-only attention mass loss for reviewed past-action matches."""
import torch


def attention_match_loss(weights, positive_positions):
    if weights.ndim!=2 or weights.shape[0]!=1:raise ValueError('One memory query required')
    if len(set(positive_positions))!=len(positive_positions):raise ValueError('Duplicate attention labels')
    if any(not isinstance(k,int) or k<0 or k>=weights.shape[1] for k in positive_positions):raise ValueError('Out-of-range attention label')
    if not positive_positions:return weights.sum()*0.
    return -torch.log(weights[:,positive_positions].sum().clamp_min(1e-12))
