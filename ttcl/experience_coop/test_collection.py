from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ttcl.experience_evolution.core import read, save
from ttcl.experience_v2.common import sha_file
from .protocol import binding


def sample(model,messages):
    value={'input_ids':[1,2,3,4],'prompt_length':2,'old_logp':[-.2,-.3],
           'input_binding':binding(messages),'model':model,'seed':1,'temperature':1.,'top_p':1.}
    value['sample_binding']=binding({k:value[k] for k in
        ['input_ids','prompt_length','old_logp','input_binding','model','seed']})
    return value


class CollectionTests(unittest.TestCase):
    def test_full_paired_grid_labels_both_roles_without_future_writer_input(self):
        from .collect import collect_history, freeze_block
        calls=[]
        class Env:
            def __init__(self,*args,**kwargs): self.client=None
            def close(self): pass
            def run(self,spec,memory,repeat,dest):
                reward=0. if not memory else (.5 if memory=='KEEP' else int(memory[-1])/7)
                action=sample('reader_current',[spec['id'],memory])
                value={'reward':reward,'task_identity':spec['id'],'initial_query_hash':'same',
                       'sampled_actions':[action,action],'row':{'status':'complete'}}
                save(Path(dest)/'completed.json',value);return value
        def fake_writer(client,messages,model,path,repeat,plan,sample=False):
            calls.append(messages)
            self.assertEqual(messages,[{'role':'user','content':'Only source observations'}])
            index=int(Path(path).stem.split('_')[-1])
            value={'raw_response':f'MEM{index}','usable':True,'sample':globals()['sample'](model,messages)}
            save(path,value);return value
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);hid='h0';source=root/'sources'/hid/'input.json'
            h={'id':hid,'domain':'fixture','probes':[{'id':'future A'},{'id':'future B'}]}
            messages=[{'role':'user','content':'Only source observations'}]
            save(source,{'messages':messages,'previous':'KEEP','input_binding':binding(messages)})
            save(root/'source_audit.json',{'hashes':{str(source):sha_file(source)},'reward_scales':{'fixture':1.}})
            save(root/'plan.json',{'histories':[h],'blocks':[{'id':'block_000','histories':[hid]}],
                'probe_seeds':[1,2],'training':{'max_length':64,'reader_max_response':8,'unusable_writer_penalty':.1}})
            with patch('ttcl.experience_coop.collect.Environments',Env),patch('ttcl.experience_coop.collect.writer',fake_writer):
                collect_history(root,'dual','block_000',hid)
            labels=read(root/'training/dual/block_000/histories/h0/labels.json')
            self.assertEqual(len(calls),8)
            self.assertEqual(len(labels['writer_rows']),8)
            self.assertEqual(len(labels['reader_rows']),64)
            self.assertEqual(labels['logical_probe_episodes'],40)
            self.assertLess(labels['writer_rows'][0]['advantage'],0)
            self.assertGreater(labels['writer_rows'][-1]['advantage'],0)
            counts=freeze_block(root,'dual','block_000')
            self.assertEqual(counts,{'writer':8,'reader':64})


if __name__=='__main__': unittest.main()
