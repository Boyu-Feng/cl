"""Preserve failed v94; restart training with verified original initialization/data."""
import argparse
import torch
from ttcl.trajectory_hyperlora import run_continual_xland_v94 as original
from ttcl.trajectory_hyperlora.joint_domains_v93 import read,write,filehash,digest
OLD=original.OUT
NEW=OLD.with_name('continual_xland_v94b_20261010')

class Runner(original.Runner):
    def mount(self,bb):
        # Optimizer-owned Parameters must not become registered adapter parameters.
        # Multiplication preserves the gradient edge while yielding an ordinary tensor.
        for adapter,b in zip(self.agent.adapters,bb,strict=True):
            adapter.b=b*1.0

def main(device):
    if (NEW/'initial.pt').exists():raise RuntimeError('Existing recovery; refuse overwrite')
    collection=read(OLD/'collection.json');assert collection['complete']
    plan=read(OLD/'plan.json')
    for split,n in [('train',4),('test',7)]:
        assert len(collection[split])==len(plan[split])==8
        for ti,group in enumerate(collection[split]):
            assert len(group)==n
            for j,ep in enumerate(group):
                assert ep==read(OLD/f'collection_{split}_{ti}_{j}.json')
                assert ep['content_sha256']==digest({k:v for k,v in ep.items() if k!='content_sha256'})
                assert ep['environment_replay_passed']
                assert ep['ruleset_id']==plan[split][ti]['ruleset_id']
    NEW.mkdir(parents=True,exist_ok=True)
    write(NEW/'plan.json',plan)
    write(NEW/'recovery.json',dict(original_directory=str(OLD),source_hashes={f:filehash(OLD/f) for f in ['plan.json','initial.pt','collection.json','failure.json','training.json']},recovery_code_sha256=filehash(__file__),reason='Parameter B registration prevented subsequent generated Tensor mounting',training_restart_step=0,reused_collection_episodes=88))
    write(NEW/'collection.json',collection)
    original.OUT=NEW
    runner=None
    try:
        runner=Runner(device)
        before=torch.load(OLD/'initial.pt',map_location='cpu',weights_only=True)
        after=torch.load(NEW/'initial.pt',map_location='cpu',weights_only=True)
        assert all(torch.equal(a,b) for a,b in zip(before['a'],after['a'],strict=True))
        assert before['norms']==after['norms']
        assert before['initial_rule'].keys()==after['initial_rule'].keys()
        assert all(torch.equal(v,after['initial_rule'][k]) for k,v in before['initial_rule'].items())
        write(NEW/'initialization_audit.json',dict(passed=True,fixed_a_equal=True,update_rule_equal=True,collection_hash=filehash(NEW/'collection.json')))
        runner.status('recovered_training',initialization_verified=True,reused_episodes=88)
        prior=runner.train(collection)
        runner.evaluate(collection,prior)
    except Exception as e:
        write(NEW/'failure.json',dict(type=type(e).__name__,message=str(e)))
        write(NEW/'status.json',dict(stage='failed',error=str(e)));raise
    finally:
        if runner is not None:runner.worker.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--device',default='cuda:2');args=p.parse_args();main(args.device)
