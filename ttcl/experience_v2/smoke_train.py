"""One discarded update to validate continuation before the long training run."""
from pathlib import Path
import statistics

from .common import ROOT, read, save, seed
from ttcl.experience_evolution.core import writer_messages
from ttcl.experience_evolution.environment import Actor
from ttcl.experience_evolution.writer import Writer
from ttcl.experience_evolution.run import check_pair


def main():
    root=ROOT;plan=read(root/'training_plan.json');out=root/'training_smoke'
    episodes=sorted((root/'training/screening').glob('*/episode.json'))
    if len(episodes)<4:raise RuntimeError('Screening examples not ready')
    source=read(episodes[0]);target=next(x['game'] for x in plan['screening'] if x['game']!=source['game'])
    writer=Writer(plan,adapter=root/'adapters/delta',train=True)
    sample=writer.generate([writer_messages('',source)],seed(923,'smoke'),[out/'writer'])[0]
    jobs=[{'game':target,'memory':sample['text'] if branch=='with' else '',
           'seed':seed(923,'smoke_target',k),'output':str(out/f'{k}/{branch}')}
          for k in range(2) for branch in ['with','without']]
    outcomes=Actor(plan).run_many(jobs)
    deltas=[]
    for yes,no in zip(outcomes[::2],outcomes[1::2]):
        check_pair(yes,no);deltas.append(yes['reward']-no['reward'])
    sample.update(advantage=statistics.mean(deltas),delta=statistics.mean(deltas))
    metrics=writer.update([sample]);audit=writer.save(out)
    assert audit['base_unchanged']
    save(out/'result.json',{'passed':True,'discarded':True,'metrics':metrics,'audit':audit})
    print('Continuation smoke update passed; checkpoint excluded from experiments.',flush=True)


if __name__=='__main__':main()
