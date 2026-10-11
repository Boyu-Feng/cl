"""Tensor invariants needed before spending model/rollout compute."""
import torch
from ttcl.trajectory_hyperlora.transition_reader_v82 import TransitionReader,effective_squared_norm

def main():
    torch.manual_seed(82)
    reader=TransitionReader(8,[(6,2),(5,2)],width=4)
    q=torch.randn(1,8);obs=torch.randn(5,8);act=torch.randn(5,8);fb=torch.randn(5,8)
    shared=[torch.randn(1,6,2),torch.randn(1,5,2)];bases=[torch.randn(2,9),torch.randn(2,7)]
    initial,_=reader.adapted_factors(q,obs,act,fb,shared,bases)
    assert all(torch.equal(a,b) for a,b in zip(initial,shared))
    # Zero-initialized heads must still have finite useful training gradients.
    sum(x.sum() for x in initial).backward()
    assert all(h.weight.grad is not None and torch.isfinite(h.weight.grad).all() and h.weight.grad.abs().sum()>0 for h in reader.heads)
    with torch.no_grad():
        for h in reader.heads:h.weight.normal_()
    reader.zero_grad(set_to_none=True)
    actual,_=reader(q,obs,act,fb)
    p=torch.tensor([4,2,0,3,1]);joint,_=reader(q,obs[p],act[p],fb[p]);wrong,_=reader(q,obs,act[p],fb[p])
    assert all(torch.allclose(a,b,atol=1e-6) for a,b in zip(actual,joint))
    assert any(not torch.allclose(a,b,atol=1e-5) for a,b in zip(actual,wrong))
    empty,_=reader.adapted_factors(q,obs[:0],act[:0],fb[:0],shared,bases)
    masked,_=reader.adapted_factors(q,obs,act,fb,shared,bases,mask=torch.zeros(5,dtype=torch.bool))
    assert all(torch.equal(a,b) and torch.equal(c,b) for a,b,c in zip(empty,shared,masked))
    adapted,info=reader.adapted_factors(q,obs,act,fb,shared,bases)
    delta=[a-b for a,b in zip(adapted,shared)]
    assert effective_squared_norm(delta,bases).sqrt()<=.25*effective_squared_norm(shared,bases).sqrt()+1e-5
    sum(x.square().sum() for x in adapted).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in reader.parameters())
    assert torch.allclose(info['weights'].sum(),torch.tensor(1.))
    print('PASS: zero initialization gradients, paired permutation, mispair sensitivity, empty/masked identity, effective norm cap, finite gradients')

if __name__=='__main__':main()
