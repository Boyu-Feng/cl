import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ttcl.experience_evolution.core import save
from .annotations import corrected, annotation_input_sha256, annotation_path
from .run import V2, select_histories, matched, validate_episode, encode_training, report


class RepairTests(unittest.TestCase):
    def test_histories_and_evidence(self):
        if not annotation_path().is_file() or not (V2 / 'training/curriculum.json').is_file() or not any(
                (V2 / 'training/continued_k1').glob('batch_*/seq_*/update_*/writer.json')):
            self.skipTest('Integration test requires local V2 curriculum, continued_k1 trajectories and reviewed annotation JSON; these data are not shipped in Git.')
        rows=select_histories()
        self.assertEqual(len(rows),32)
        self.assertEqual(len({r['game'] for r in rows}),32)
        self.assertEqual(sum(r['episode']['reward'] for r in rows),16)
        keeps=0
        for row in rows:
            payload=json.loads(row['messages'][1]['content'])
            self.assertEqual(set(payload),{'previous_experience','completed_interaction'})
            self.assertTrue(row['previous'])
            self.assertIn('/train/',row['game'])
            text,operation,steps=corrected(row)
            self.assertTrue(text)
            self.assertLessEqual(len(text.split()),200)
            self.assertTrue(all(0<i<=len(row['episode']['trajectory']) for i in steps))
            if operation=='keep':
                keeps+=1;self.assertEqual(text,row['previous'])
        self.assertEqual(keeps,2)

    def test_annotation_rejects_changed_input_with_same_id(self):
        # A synthetic fixture tests content binding without bundling real histories.
        row={'id':'fixture','previous':'Keep observed facts.',
             'episode':{'initial_observation':'A test observation.',
                        'trajectory':[{'action':'inspect','observation':'Observed.'}],
                        'reward':0,'steps':1}}
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'reviewed.json'
            save(path, {'schema_version':1, 'annotations':{'fixture':{
                'input_sha256':annotation_input_sha256(row),'evidence_steps':[1],
                'observed':'KEEP','revision':'','procedure':''}}})
            with patch.dict(os.environ, {'TTCL_REPAIR_ANNOTATIONS':str(path)}):
                self.assertEqual(corrected(row), (row['previous'], 'keep', [1]))
                variants=[]
                other=copy.deepcopy(row);other['previous']='A changed previous memory.';variants.append(other)
                for key,value in [('initial_observation','A new observation.'),('reward',1),('steps',2)]:
                    other=copy.deepcopy(row);other['episode'][key]=value;variants.append(other)
                for key in ['action','observation']:
                    other=copy.deepcopy(row);other['episode']['trajectory'][0][key]='Changed';variants.append(other)
                for other in variants:
                    with self.subTest(other=other), self.assertRaisesRegex(ValueError, 'fresh evidence review'):
                        corrected(other)

    def test_missing_annotations_require_review(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(
                os.environ, {'TTCL_REPAIR_ANNOTATIONS':str(Path(directory)/'missing.json')}):
            with self.assertRaisesRegex(FileNotFoundError, 'First collect and review'):
                corrected({'id':'fixture'})

    def test_pairing_rejects_different_resets(self):
        ep={'game':'a','seed':1,'initial_observation':'x','initial_commands_sha256':'z'}
        matched([ep,dict(ep)])
        for key in ep:
            other=dict(ep);other[key]='different'
            with self.assertRaises(ValueError):matched([ep,other])

    def test_resume_rejects_changed_memory(self):
        job={'game':'a','seed':1,'memory':'old'}
        ep=dict(job,status='complete',actor_adapter_enabled=False)
        validate_episode(ep,job)
        with self.assertRaises(ValueError):validate_episode(ep,dict(job,memory='new'))
        with self.assertRaises(ValueError):validate_episode(dict(ep,actor_adapter_enabled=True),job)

    def test_target_boundary_and_overflow(self):
        class Tokenizer:
            def apply_chat_template(self,messages,tokenize,add_generation_prompt=False):
                if add_generation_prompt:return [1,2,3]
                return [1,2,3,4,5]
        row={'id':'x','messages':[],'target':'text'}
        self.assertEqual(encode_training(Tokenizer(),row,5),{'ids':[1,2,3,4,5],'target_length':2})
        with self.assertRaises(ValueError):encode_training(Tokenizer(),row,4)

    def test_report_uses_paired_values_and_family_macro(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);rows=[]
            for h,family in [('h0','a'),('h1','a'),('h2','b')]:
                for arm in ['empty','corrected']:
                    rows.append({'history':h,'target':0,'repeat':1,'family':family,'arm':arm,
                                 'reward':int(arm=='corrected' and family=='b'),'steps':2,
                                 'actor_calls':2,'input_tokens':10,'output_tokens':2,'invalid_commands':0})
            save(root/'diagnostic_rows.json',rows);report(root)
            v=json.loads((root/'summary.json').read_text())['diagnostic']['arms']['corrected']
            self.assertAlmostEqual(v['delta_baseline'],1/3)
            self.assertEqual(v['family_macro_reward'],0.5)
            self.assertEqual(v['vs_empty']['wins'],1)


if __name__=='__main__':unittest.main()
