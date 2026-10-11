"""Check semantic selection and removal of a common source component."""
import torch
from ttcl.trajectory_hyperlora.transition_reader_v84 import TransitionReader
from ttcl.trajectory_hyperlora.transition_reader_v82 import effective_squared_norm

def main():
    torch.manual_seed(84)
    reader=TransitionReader(8,[(6,2),(5,2)],4,top_k=2)
    obs=torch.eye(8);q=obs[:1];act=torch.randn(8,8);fb=torch.randn(8,8)
    shared=[torch.randn(1,6,2),torch.randn(1,5,2)];bases=[torch.randn(2,9),torch.randn(2,7)]
    initial,_=reader.adapted_factors(q,obs,act,fb,shared,bases)
    assert all(torch.equal(x,y) for x,y in zip(initial,shared))
    sum(x.sum() for x in initial).backward()
    assert all(h.weight.grad.abs().sum()>0 for h in reader.heads)
    with torch.no_grad():
        for h in reader.heads:h.weight.normal_()
    actual,w=reader(q,obs,act,fb);other,_=reader(obs[1:2],obs,act,fb)
    assert w.argmax()==0 and any(not torch.allclose(x,y) for x,y in zip(actual,other))
    perm=torch.randperm(8);joint,_=reader(q,obs[perm],act[perm],fb[perm]);wrong,_=reader(q,obs,act.roll(1,0),fb.roll(1,0))
    assert all(torch.allclose(x,y,atol=1e-5) for x,y in zip(actual,joint))
    assert any(not torch.allclose(x,y) for x,y in zip(actual,wrong))
    same,_=reader(q,obs,act[:1].expand_as(act),fb[:1].expand_as(fb))
    assert all(torch.allclose(x,torch.zeros_like(x),atol=1e-5) for x in same)
    empty,_=reader.adapted_factors(q,obs[:0],act[:0],fb[:0],shared,bases)
    assert all(torch.equal(x,y) for x,y in zip(empty,shared))
    reader.zero_grad(set_to_none=True)
    adapted,_=reader.adapted_factors(q,obs,act,fb,shared,bases)
    delta=[x-y for x,y in zip(adapted,shared)]
    assert effective_squared_norm(delta,bases).sqrt()<=.25*effective_squared_norm(shared,bases).sqrt()+1e-5
    sum(x.square().sum() for x in adapted).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in reader.parameters())
    print('PASS: zero-head gradients, query sensitivity, joint permutation including ties, mismatch sensitivity, common-value cancellation, empty identity, norm bound, finite gradients')

if __name__=='__main__':main()
