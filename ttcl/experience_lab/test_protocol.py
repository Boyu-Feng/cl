from __future__ import annotations
import unittest

from .protocol import select_pair, binding, balanced_order
from .evaluate import promote


class SelectionTests(unittest.TestCase):
    def data(self):
        texts={'empty':'','keep':'old','good':'supported improvement','bad':'unsupported rewrite'}
        rewards={'empty':{'1':[0.,0.],'2':[0.,0.]},'keep':{'1':[0.,0.],'2':[0.,0.]},
                 'good':{'1':[1.,1.],'2':[1.,1.]},'bad':{'1':[0.,0.],'2':[0.,0.]}}
        return texts,rewards

    def test_requires_independent_confirmation(self):
        texts,rewards=self.data()
        self.assertTrue(select_pair(texts,rewards,1,2)['accepted'])
        rewards['good']['2']=[0.,0.]
        self.assertFalse(select_pair(texts,rewards,1,2)['accepted'])
        with self.assertRaises(ValueError):select_pair(texts,rewards,1,1)

    def test_missing_not_zero(self):
        texts,rewards=self.data();rewards['good']['2'][0]=None
        self.assertEqual(select_pair(texts,rewards,1,2)['reason'],'missing_official_reward')
        rewards['good']['2']=[1.]
        self.assertEqual(select_pair(texts,rewards,1,2)['reason'],'incomplete_paired_grid')

    def test_keep_can_win(self):
        texts,rewards=self.data()
        for s in ['1','2']:rewards['keep'][s]=[2.,2.]
        result=select_pair(texts,rewards,1,2)
        self.assertTrue(result['accepted']);self.assertEqual(result['chosen'],'keep')

    def test_confirmation_does_not_reselect(self):
        texts,rewards=self.data()
        rewards['bad']['1']=[-1.,-1.]
        rewards['bad']['2']=[5.,5.]
        result=select_pair(texts,rewards,1,2)
        self.assertEqual(result['chosen'],'good');self.assertFalse(result['accepted'])

    def test_ties_are_not_labels(self):
        texts,rewards=self.data()
        for g in rewards.values():g['1']=[0.,0.];g['2']=[0.,0.]
        self.assertFalse(select_pair(texts,rewards,1,2)['accepted'])

    def test_harmful_vs_empty_rejected(self):
        texts,rewards=self.data();rewards['empty']['2']=[2.,2.]
        self.assertEqual(select_pair(texts,rewards,1,2)['reason'],'confirmed_but_harmful')

    def test_binding_and_domain_balance(self):
        self.assertNotEqual(binding([{'content':'old'}]),binding([{'content':'new'}]))
        rows=[{'domain':'a'},{'domain':'a'},{'domain':'b'}]
        order=balanced_order(rows,4)
        self.assertEqual(sum(rows[i]['domain']=='a' for i in order),sum(rows[i]['domain']=='b' for i in order))

    def test_promotion_requires_both_benchmarks(self):
        domains=['alfworld','blind_spectrum_monitoring','exploitable_poker','database_exploration','cohort_studies']
        value={'vs_original_delta':.1,'by_seed_delta':{'1':.1,'2':.1},'wins':2,'losses':0,'paired_n':5}
        summary={'complete':True,'missing_scores':0,'domains':{d:{'new':dict(value)} for d in domains}}
        self.assertEqual(promote(summary,['new']),'new')
        summary['domains']['database_exploration']['new']['vs_original_delta']=-.01
        self.assertIsNone(promote(summary,['new']))


if __name__=='__main__':unittest.main()
