import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM
from .model import OnlineModel
from .protocol import binding
from .run import defaults
from .upstream import load_loss


class ToyTokenizer:
    def apply_chat_template(self,messages,**kwargs):
        return '|'.join(m['role']+':'+m['content'] for m in messages)+'|assistant:'
    def encode(self,text,**kwargs):return [3+ord(c)%60 for c in text]


class LearningTests(unittest.TestCase):
    def test_official_loss_and_gradient_match_jsd(self):
        torch.manual_seed(7)
        logits=torch.randn(1,4,7,requires_grad=True)
        teacher=torch.log_softmax(torch.randn(1,4,7),-1)
        student=logits.log_softmax(-1)
        cfg=SimpleNamespace(full_logit_distillation=True,distillation_topk=None,
                            distillation_add_tail=True,alpha=.5,is_clip=None)
        dummy=torch.zeros(1,4)
        loss,_=load_loss()(dummy,dummy,torch.ones_like(dummy),cfg,
                          student_all_log_probs=student,teacher_all_log_probs=teacher)
        mix=(student.exp()+teacher.exp())/2
        expected=(.5*(student.exp()*(student-mix.log())).sum(-1)
                 +.5*(teacher.exp()*(teacher-mix.log())).sum(-1)).mean()
        self.assertTrue(torch.allclose(loss,expected,atol=1e-6))
        grad=torch.autograd.grad(loss,logits,retain_graph=True)[0]
        reference=torch.autograd.grad(expected,logits)[0]
        self.assertTrue(torch.allclose(grad,reference,atol=1e-6))

    def test_lora_changes_base_and_teacher_gradients_are_frozen(self):
        plan=defaults();plan.update(device='cpu',training_context=4096,logit_chunk=2,learning_rate=1e-3)
        cfg=Qwen3Config(vocab_size=64,hidden_size=32,intermediate_size=64,num_hidden_layers=2,
                       num_attention_heads=4,num_key_value_heads=2,head_dim=8,max_position_embeddings=4096)
        base=Qwen3ForCausalLM(cfg)
        model=OnlineModel(plan,303,model=base,tokenizer=ToyTokenizer())
        messages=[{'role':'user','content':'Return 2.'}];_,ids=model.render(messages)
        sample={'messages':messages,'prompt_ids':ids,'response_ids':[4,5,6,2],
                'raw_response':'synthetic','content_binding':binding([ids,[4,5,6,2]])}
        event={'messages':messages,'raw_response':'synthetic','parse_error':None}
        fingerprint=model.fingerprint('base');initial_teacher=model.fingerprint('teacher')
        with tempfile.TemporaryDirectory() as d:
            audit=model.update([sample],[event],[{'step':1,'public_feedback':'Incorrect; return 2.'}],0.,[0],Path(d)/'update')
        self.assertTrue(audit['adapter_changed']);self.assertEqual(model.updates,1)
        self.assertEqual(model.fingerprint('base'),fingerprint)
        self.assertNotEqual(model.fingerprint('teacher'),initial_teacher)
        self.assertGreater(audit['gradient_norm'],0)
        self.assertTrue(all(p.grad is None for n,p in model.model.named_parameters() if '.teacher.' in n or 'lora_' not in n))


if __name__=='__main__':unittest.main()
