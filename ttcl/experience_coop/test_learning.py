"""Real CPU Qwen3/PEFT gradients; fixtures never enter experiment metrics."""
import copy
from pathlib import Path
import tempfile
import unittest

from ttcl.experience_evolution.core import read, save
from .protocol import binding


class LearningTests(unittest.TestCase):
    def test_clipping_and_zero_advantage(self):
        import torch
        from .learning import ppo_loss
        selected=torch.tensor([0.,0.],requires_grad=True)
        old=torch.tensor([0.,0.]);reference=torch.tensor([0.,0.])
        loss,_,_,_=ppo_loss(selected,old,reference,0.)
        loss.backward();self.assertEqual(float(selected.grad.abs().max()),0.)
        selected=torch.tensor([.5],requires_grad=True)
        loss,policy,_,fraction=ppo_loss(selected,torch.tensor([0.]),torch.tensor([.5]),1.,beta=0.)
        self.assertAlmostEqual(float(policy.detach()),-1.2,places=5)
        self.assertEqual(fraction,1.)

    def test_writer_and_reader_optimize_only_their_own_lora(self):
        import torch
        from transformers import Qwen3Config, Qwen3ForCausalLM, AutoModelForCausalLM
        from peft import LoraConfig, get_peft_model, PeftModel
        from .learning import train, initialize_reader, selected_logps
        from .run import checkpoint_hashes
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);base_path=root/'base';original=root/'original'
            torch.manual_seed(321)
            cfg=Qwen3Config(vocab_size=32,hidden_size=16,intermediate_size=32,num_hidden_layers=2,
                           num_attention_heads=2,num_key_value_heads=2,head_dim=8)
            base=Qwen3ForCausalLM(cfg);base.save_pretrained(base_path)
            adapted=get_peft_model(copy.deepcopy(base),LoraConfig(r=2,lora_alpha=4,
                target_modules=['q_proj','k_proj','v_proj','o_proj'],lora_dropout=0.,task_type='CAUSAL_LM'))
            adapted.save_pretrained(original)
            plan={'model':str(base_path),'initial_adapter':str(original),'initial_reader':str(root/'reader'),
                'arms':['writer_only','dual'],'reader_lora':{'r':2,'lora_alpha':4,'lora_dropout':0.,
                    'target_modules':['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'],
                    'layers_to_transform':[1],'layers_pattern':'layers','task_type':'CAUSAL_LM'},
                'training':{'seed':321,'learning_rate':.003,'ppo_epochs':2,'minibatch_size':2,
                    'reader_minibatch_size':2,'clip_range':.2,'kl_beta':.01,'max_length':64,
                    'max_grad_norm':1.,'max_behavior_logp_mae':.15}}
            save(root/'plan.json',plan);audit=initialize_reader(root)
            self.assertTrue(audit['zero_residual']);self.assertEqual(audit['layers'],[1])
            directory=root/'training/dual/block_000'
            inputs={'writer':str(original),'reader':plan['initial_reader'],
                    'writer_optimizer':None,'reader_optimizer':None,
                    'hashes':checkpoint_hashes([original,plan['initial_reader']])}
            save(directory/'block_input.json',inputs)
            datasets={}
            for role in ['writer','reader']:
                model=PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(base_path,
                    torch_dtype=torch.bfloat16,attn_implementation='sdpa'),inputs[role]).eval()
                rows=[]
                for i,adv in enumerate([1.,-.5,0.,.25]):
                    sample={'input_ids':[1,2,4,7+i,9,3],'prompt_length':3,'input_binding':binding(['fixture',i]),
                            'model':role+'_current','seed':i,'temperature':1.,'top_p':1.,'finish_reason':'stop'}
                    with torch.no_grad(): sample['old_logp']=selected_logps(model,sample,'cpu').tolist()
                    sample['sample_binding']=binding({k:sample[k] for k in
                        ['input_ids','prompt_length','old_logp','input_binding','model','seed']})
                    path=directory/'fixtures'/f'{role}_{i}.json';save(path,{'sample':sample})
                    rows.append({'path':str(path),'sample_key':'sample','advantage':adv,'domain':'fixture'})
                datasets[role]=rows;save(directory/f'{role}_dataset.json',rows)
            hashes=checkpoint_hashes([directory/'writer_dataset.json',directory/'reader_dataset.json',directory/'fixtures'])
            save(directory/'dataset_audit.json',{'hashes':hashes})
            for role in ['writer','reader']:
                train(root,'dual','block_000',role,device='cpu')
                output=directory/role
                result=read(output/'audit.json')
                self.assertTrue(result['base_unchanged']);self.assertTrue(result['reference_unchanged'])
                self.assertTrue(result['other_role_checkpoint_unchanged'])
                self.assertNotEqual(result['adapter_before'],result['adapter_after'])
                self.assertEqual(read(output/'status.json')['steps'],4)
                self.assertTrue((output/'optimizer.pt').exists())
            for p,h in inputs['hashes'].items():
                from ttcl.experience_v2.common import sha_file
                self.assertEqual(sha_file(p),h)


if __name__=='__main__': unittest.main()
