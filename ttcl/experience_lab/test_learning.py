"""CPU integration: real tiny Qwen/PEFT optimization; no benchmark labels or scores."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ttcl.experience_evolution.core import read, save
from ttcl.experience_v2.common import sha_file
from .protocol import binding


class LearningIntegration(unittest.TestCase):
    def test_real_sft_and_dpo_update_lora_only(self):
        import torch
        from transformers import Qwen3Config, Qwen3ForCausalLM
        from peft import LoraConfig, get_peft_model
        from .learning import train
        torch.set_num_threads(2)
        config=Qwen3Config(vocab_size=32,hidden_size=16,intermediate_size=32,num_hidden_layers=1,
                          num_attention_heads=2,num_key_value_heads=2,head_dim=8)
        class Tokenizer:
            def apply_chat_template(self,messages,**kw):
                prefix=[1,4,5,2]
                if kw.get('add_generation_prompt'):return prefix
                return prefix+([8,9] if messages[-1]['content']=='chosen' else [10,11])+[3]
            def save_pretrained(self,path):save(Path(path)/'tokenizer_fixture.json',{'unit_test_only':True})
        real_tensor=torch.tensor
        def cpu_tensor(*args,**kwargs):
            if str(kwargs.get('device','')).startswith('cuda'):kwargs['device']='cpu'
            return real_tensor(*args,**kwargs)
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);rd=root/'round_001';initial=root/'initial'
            torch.manual_seed(12)
            base=Qwen3ForCausalLM(config)
            adapter=get_peft_model(base,LoraConfig(r=2,lora_alpha=4,lora_dropout=0.,
                target_modules=['q_proj','v_proj'],task_type='CAUSAL_LM',bias='none'))
            adapter.save_pretrained(initial)
            messages=[{'role':'user','content':'Unit test public trajectory; not experiment data'}]
            rows=[{'id':'fixture','domain':'fixture','messages':messages,'input_binding':binding(messages),
                   'chosen':'chosen','rejected':'rejected','random_target':'rejected'}]
            save(root/'plan.json',{'model':'unit-fixture','min_pairs':1,'training_arms':['filtered_sft','dpo'],
                 'training':{'epochs':1,'accumulation':1,'learning_rate':.001,'seed':12,'max_length':32,
                             'dpo_beta':.1,'chosen_nll_weight':.1}})
            save(rd/'dataset.json',rows)
            save(rd/'dataset_audit.json',{'dataset_sha256':sha_file(rd/'dataset.json'),'provenance_hashes':{}})
            save(rd/'round_input.json',{'parent_adapter':str(initial)})
            for arm in ['filtered_sft','dpo']:
                torch.manual_seed(12)
                model=Qwen3ForCausalLM(config)
                with patch('transformers.AutoTokenizer.from_pretrained',return_value=Tokenizer()), \
                     patch('transformers.AutoModelForCausalLM.from_pretrained',return_value=model), \
                     patch.object(model,'to',return_value=model),patch('torch.tensor',side_effect=cpu_tensor):
                    train(root,'round_001',arm)
                out=rd/'training'/arm
                self.assertEqual(read(out/'status.json')['phase'],'complete')
                self.assertTrue(read(out/'audit.json')['base_unchanged'])
                self.assertTrue(read(out/'audit.json')['reference_unchanged'])
                self.assertTrue((out/'adapter/adapter_model.safetensors').is_file())
                import json
                metric=json.loads((out/'training.jsonl').read_text().splitlines()[0])
                self.assertGreater(metric['gradient_norm'],0.)


if __name__=='__main__':unittest.main()
