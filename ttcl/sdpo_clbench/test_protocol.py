import copy
import unittest
from .protocol import binding, selected_calls, teacher_messages


class ProtocolTests(unittest.TestCase):
    def test_feedback_does_not_include_hidden_metadata(self):
        event={'messages':[{'role':'user','content':'Choose a tool.'}], 'parse_error':None,
               'hidden_answer':'SECRET_A','instance_id':'SECRET_ID'}
        steps=[{'step':1,'public_feedback':'Tool failed: missing column.', 'hidden_state':'SECRET_B'}]
        messages,audit=teacher_messages(event,steps,-.7)
        text=str(messages)
        self.assertNotIn('SECRET',text);self.assertIn('-0.7',text)
        self.assertIn('missing column',text)
        self.assertEqual(event['messages'][0]['content'],'Choose a tool.')
        self.assertEqual(audit['teacher_input_binding'],binding(messages))

    def test_action_selection_is_not_outcome_selection(self):
        events=[{'reward':i,'response':'x'*i} for i in range(12)]
        first=selected_calls(events,303,4)
        for e in events:e.update(reward=0,response='invalid')
        self.assertEqual(first,selected_calls(events,303,4))

    def test_fresh_feedback_changes_input_binding(self):
        event={'messages':[{'role':'user','content':'Answer.'}]}
        _,a=teacher_messages(event,[{'step':1,'public_feedback':'Incorrect.'}],0.)
        _,b=teacher_messages(event,[{'step':1,'public_feedback':'Correct.'}],1.)
        self.assertNotEqual(a['teacher_input_binding'],b['teacher_input_binding'])

    def test_truncation_is_explicit(self):
        event={'messages':[{'role':'user','content':'Answer.'}]}
        text,a=teacher_messages(event,[{'step':1,'public_feedback':'x'*1000}],0.,100)
        self.assertGreater(a['public_feedback_omitted_chars'],0)
        self.assertIn('omitted',str(text))


if __name__=='__main__':unittest.main()
