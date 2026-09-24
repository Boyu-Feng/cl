import copy
import unittest

from .client import decode_sample
from .protocol import binding, evaluation_routes, paired_advantages, validate_sample


class ProtocolTests(unittest.TestCase):
    def test_eight_candidates_preserve_negative_credit(self):
        rewards={'empty':[0.,0.],'keep':[1.,1.],**{f'candidate_{i}':[0.,0.] for i in range(8)}}
        values=paired_advantages(rewards,1.)
        self.assertEqual(len(values),8)
        self.assertTrue(all(v['writer']==-1.5 for v in values.values()))
        self.assertTrue(all(v['reader']==[0.,0.] for v in values.values()))
        rewards['candidate_3']=[1.,0.]
        self.assertEqual(paired_advantages(rewards,1.)['candidate_3']['writer'],-.75)

    def test_missing_rewards_are_not_zero_and_scale_does_not_flip_sign(self):
        rewards={'empty':[0.],'keep':[0.],**{f'candidate_{i}':[float(i)] for i in range(8)}}
        self.assertEqual(paired_advantages(rewards,10.)['candidate_5']['writer'],.5)
        rewards['candidate_0']=[None]
        with self.assertRaises(ValueError): paired_advantages(rewards,1.)
        del rewards['candidate_7']
        with self.assertRaises(ValueError): paired_advantages(rewards,1.)

    def test_exact_tokens_and_probability_binding(self):
        choice={'finish_reason':'stop','logprobs':{'tokens':['token_id:7','token_id:3'],
                                                  'token_logprobs':[-.4,-.2]}}
        result=decode_sample(choice,[1,2],[{'role':'user','content':'public past'}],
                             'writer_current',11,1.,1.)
        self.assertEqual(result['input_ids'],[1,2,7,3]);validate_sample(result)
        changed=copy.deepcopy(result);changed['old_logp'][0]=-.8
        with self.assertRaises(ValueError): validate_sample(changed)
        changed=copy.deepcopy(result);changed['temperature']=.7
        with self.assertRaises(ValueError): validate_sample(changed)
        choice['logprobs']['tokens']=['some text','token_id:3']
        with self.assertRaises(ValueError): decode_sample(choice,[1],[], 'reader_current',1,1.,1.)

    def test_role_removals_are_separate_routes(self):
        routes=evaluation_routes()
        self.assertEqual(routes['dual']['writer'],routes['dual_writer_base']['writer'])
        self.assertEqual(routes['dual']['reader'],routes['old_writer_reader']['reader'])
        self.assertEqual(routes['dual']['reader'],routes['reader_no_text']['reader'])
        self.assertIsNone(routes['reader_no_text']['writer'])
        self.assertEqual(routes['ppo8_writer']['reader'],'frozen-actor')


if __name__=='__main__': unittest.main()
