import json
import unittest
from pathlib import Path

from .common import select_locomo_questions, public_messages, WORKSPACE
from .evaluate import session_messages
from .train import choose_sequences
from ttcl.experience_evolution.core import FAMILIES


class ProtocolTests(unittest.TestCase):
    def test_no_hidden_feedback_in_clbench_writer(self):
        ep=dict(public_task_brief='brief',initial_public_query='query',response_schemas=[],
                steps=[{'public_feedback':'observed'}],completed=True,format_failures=[],
                reward='SECRET',metadata='SECRET',future_task='SECRET')
        text=json.dumps(public_messages('old',ep))
        self.assertNotIn('SECRET',text)
        self.assertIn('observed',text)

    def test_fixed_300_question_selection_ignores_answers(self):
        data=json.loads((WORKSPACE/'current_work/delta-Mem/data/locomo10.json').read_text())
        selected=select_locomo_questions(data)
        self.assertEqual(sum(map(len,selected.values())),300)
        self.assertTrue(all(len(set(v))==30 for v in selected.values()))
        for sample in data:
            for q in sample['qa']:
                q['answer']='changed';q['evidence']=['changed']
        self.assertEqual(selected,select_locomo_questions(data))

    def test_locomo_writer_only_sees_session_and_prior_memory(self):
        result=json.loads(session_messages('old','DATE: yesterday; public dialogue')[1]['content'])
        self.assertEqual(set(result),{'previous_experience','completed_interaction'})
        self.assertNotIn('qa',result['completed_interaction'])

    def test_curriculum_never_repeats_task_within_chain(self):
        items=[{'family':f,'game':f'{f}/{i}'} for f in FAMILIES for i in range(24)]
        scores={x['game']:[0,1] for x in items}
        sequences,_=choose_sequences(items,scores)
        self.assertEqual(len(sequences),48)
        self.assertTrue(all(len(set(s['games']))==4 for s in sequences))
        self.assertTrue(all(all(g.startswith(s['family']+'/') for g in s['games']) for s in sequences))


if __name__=='__main__':unittest.main()
