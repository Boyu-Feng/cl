import json
import math
import unittest
from collections import Counter

from ttcl.experience_evolution.core import writer_messages
from .data import PILOT,alf_rows,synthetic_rows
from .environments import world,task,LocalTask,scripted_episode,parse_action
from .learning import reward_label


class DesignTests(unittest.TestCase):
    def test_dataset_categories_keep_and_input_isolation(self):
        if not (PILOT / 'histories.json').is_file():
            self.skipTest('Integration test requires locally generated and reviewed ALF repair histories; these data are not shipped in Git.')
        rows=alf_rows()+synthetic_rows('sql',dict(correction=18,success=12,keep=9,scope=9))+synthetic_rows('tools',dict(correction=12,success=8,keep=6,scope=6))
        self.assertEqual(len(rows),128)
        self.assertEqual(Counter(r['category'] for r in rows),dict(correction=48,success=32,keep=24,scope=24))
        self.assertEqual(len({r['id'] for r in rows}),128)
        for r in rows:
            self.assertEqual(r['messages'],writer_messages(r['previous'],r['episode']))
            payload=json.loads(r['messages'][1]['content'])
            self.assertEqual(set(payload),{'previous_experience','completed_interaction'})
            self.assertLessEqual(len(r['target'].split()),200)
            if r['category']=='keep':self.assertEqual(r['previous'],r['target'])
        self.assertTrue(any(r['episode']['reward']==0 and r['category']=='correction' for r in rows))

    def test_all_scripted_successes_execute(self):
        for domain in ['sql','tools']:
            for i in range(9):
                spec=task(world(domain,i,'unit'),i)
                self.assertEqual(scripted_episode(spec,'success')['reward'],1)
                self.assertEqual(scripted_episode(spec,'scope')['reward'],1)
                self.assertEqual(scripted_episode(spec,'ambiguous')['reward'],0)
                self.assertEqual(scripted_episode(spec,'correction')['reward'],float(i%2==1))

    def test_sql_read_only_and_correct_answer(self):
        spec=task(world('sql',0,'unit'),1);e=LocalTask(spec)
        try:
            obs,valid=e.step({'action':'QUERY','sql':'DROP TABLE '+spec['world']['table']})
            self.assertFalse(valid)
            obs,valid=e.step({'action':'QUERY','sql':e.solution_query()})
            self.assertTrue(valid)
            answer=json.loads(obs)['rows'][0][0]
            e.step({'action':'ANSWER','value':answer})
            self.assertEqual(e.reward,1)
        finally:e.close()

    def test_tools_require_actual_fetch_and_conversion(self):
        spec=task(world('tools',1,'unit'),2);e=LocalTask(spec)
        try:
            e.step({'tool':'submit','arguments':{'answer':e.expected}})
            self.assertEqual(e.reward,0)
        finally:e.close()

    def test_reward_baselines_use_matched_vectors(self):
        new=[1,0,1,0];old=[1,1,0,0];empty=[0,0,0,0]
        self.assertEqual(reward_label(new,old,empty,'absolute'),.5)
        self.assertEqual(reward_label(new,old,empty,'empty'),.5)
        self.assertEqual(reward_label(new,old,empty,'previous'),0)
        self.assertEqual(reward_label([0,0],[1,0],[0,0],'previous'),-.5)
        for bad in [[None,0],[float('nan'),0],[float('inf'),0]]:
            with self.assertRaises(ValueError):reward_label(bad,[0,0],[0,0],'absolute')
        with self.assertRaises(ValueError):reward_label([1],[0,0],[0],'empty')

    def test_independent_worlds_and_parser(self):
        a=world('sql',1,'train');b=world('sql',1,'test')
        self.assertNotEqual(a['id'],b['id']);self.assertNotEqual(a['table'],b['table'])
        self.assertEqual(parse_action('```json\n{"action":"ANSWER","value":3}\n```')['value'],3)
        with self.assertRaises(ValueError):parse_action('not an action')


if __name__=='__main__':unittest.main()
