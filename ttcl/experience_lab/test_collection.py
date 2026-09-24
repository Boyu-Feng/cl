"""Validate counterfactual identity/reuse and public-input boundaries without a GPU."""
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

from ttcl.experience_evolution.core import save, read
from .collect import Environments, messages_for


class CollectionTests(unittest.TestCase):
    def test_identical_memory_executes_once_and_changed_input_refused(self):
        actor=Environments.__new__(Environments)
        calls=[]
        def run(jobs):
            calls.extend(jobs)
            result=[]
            for j in jobs:
                value={'reward':1.,'game':j['game'],'steps':2,'initial_observation':'public task',
                       'trajectory':[],'hidden_answer':'DO NOT EXPOSE'}
                save(Path(j['output'])/'episode.json',value);result.append(value)
            return result
        actor.alf=SimpleNamespace(run_many=run)
        spec={'domain':'alfworld','game':'train/game.tw-pddl','id':'x'}
        with tempfile.TemporaryDirectory() as temp:
            one,two=Path(temp)/'one',Path(temp)/'two'
            jobs=[(spec,'same memory',7,one),(spec,'same memory',7,two)]
            results=actor.run_many(jobs)
            self.assertEqual(len(calls),1)
            self.assertEqual(results[1]['reused_from'],str(one))
            actor.run_many(jobs);self.assertEqual(len(calls),1)
            with self.assertRaises(ValueError):actor.run_many([(spec,'different',7,one)])
            messages=messages_for('old',spec,results[0])
            self.assertNotIn('hidden_answer',messages[1]['content'])
            self.assertNotIn('DO NOT EXPOSE',messages[1]['content'])

    def test_unequal_memory_cannot_reuse(self):
        actor=Environments.__new__(Environments);calls=[]
        def run(jobs):
            calls.extend(jobs)
            return [{'reward':0.,'game':j['game'],'steps':1,'initial_observation':'public'} for j in jobs]
        actor.alf=SimpleNamespace(run_many=run)
        spec={'domain':'alfworld','game':'train/game.tw-pddl','id':'x'}
        with tempfile.TemporaryDirectory() as temp:
            actor.run_many([(spec,'one',7,Path(temp)/'one'),(spec,'two',7,Path(temp)/'two')])
            self.assertEqual(len(calls),2)


if __name__=='__main__':unittest.main()
