from __future__ import annotations

from contextlib import nullcontext
import copy
import hashlib
from pathlib import Path
import time
from types import SimpleNamespace

from .protocol import binding, save, seed, teacher_messages
from .upstream import load_loss


class OnlineModel:
    def __init__(self, plan, repeat, model=None, tokenizer=None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import LoraConfig, get_peft_model
        self.torch = torch; self.plan = plan; self.repeat = repeat
        self.device = plan.get('device', 'cuda:0')
        torch.manual_seed(seed('sdpo_init', repeat))
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(plan['model'], local_files_only=True)
        if model is None:
            model = AutoModelForCausalLM.from_pretrained(plan['model'], local_files_only=True,
                        torch_dtype=torch.bfloat16, attn_implementation='sdpa').to(self.device)
        cfg = LoraConfig(r=plan['lora_rank'], lora_alpha=plan['lora_alpha'], lora_dropout=0.,
                         target_modules=plan['lora_targets'], task_type='CAUSAL_LM')
        self.model = get_peft_model(model, cfg)
        self.model.add_adapter('teacher', copy.deepcopy(cfg))
        self.model.set_adapter('default')
        for name, param in self.model.named_parameters():
            param.requires_grad_('.default.' in name and 'lora_' in name)
        self.parameters = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(self.parameters, lr=plan['learning_rate'], weight_decay=0.)
        self.update_teacher(rate=1.)
        self.loss_fn = load_loss()
        self.loss_cfg = SimpleNamespace(full_logit_distillation=True, distillation_topk=100,
            distillation_add_tail=True, alpha=.5, is_clip=None)
        self.updates = 0; self.arm = 'frozen'; self.captures = []
        self.base_fingerprint = self.fingerprint('base')
        self.initial_adapter = self.fingerprint('default')
        self.model.eval()

    def fingerprint(self, role):
        h = hashlib.sha256()
        for name, p in self.model.named_parameters():
            select = ('lora_' not in name) if role == 'base' else ('lora_' in name and f'.{role}.' in name)
            if select:
                h.update(name.encode());h.update(p.detach().cpu().contiguous().view(self.torch.uint8).numpy().tobytes())
        return h.hexdigest()

    def update_teacher(self, rate):
        params = dict(self.model.named_parameters())
        with self.torch.no_grad():
            for name, p in params.items():
                if 'lora_' in name and '.teacher.' in name:
                    p.mul_(1-rate).add_(params[name.replace('.teacher.', '.default.')], alpha=rate)

    def render(self, messages):
        rendered = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return rendered, self.tokenizer.encode(rendered, add_special_tokens=False)

    def generate(self, messages, random_seed):
        torch = self.torch
        rendered, prompt_ids = self.render(messages)
        if len(prompt_ids) + self.plan['actor_max_tokens'] > self.plan['context']:
            raise ValueError('Context overflow; no silent prompt truncation')
        random_seed = seed(random_seed, self.repeat)
        torch.manual_seed(random_seed)
        if str(self.device).startswith('cuda'): torch.cuda.manual_seed_all(random_seed)
        self.model.set_adapter('default'); self.model.eval()
        self.model.gradient_checkpointing_disable()
        ids = torch.tensor([prompt_ids], device=self.device)
        start = time.monotonic()
        with torch.inference_mode(), (self.model.disable_adapter() if self.arm=='frozen' else nullcontext()):
            output = self.model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                max_new_tokens=self.plan['actor_max_tokens'], do_sample=True,
                temperature=1., top_p=1., top_k=0, repetition_penalty=1., use_cache=True,
                pad_token_id=self.tokenizer.eos_token_id)
        generated = output[0,len(prompt_ids):].tolist()
        raw = self.tokenizer.decode(generated, skip_special_tokens=True)
        self.captures.append({'messages':copy.deepcopy(messages), 'prompt_ids':prompt_ids,
            'response_ids':generated, 'raw_response':raw, 'input_binding':binding(messages),
            'content_binding':binding([prompt_ids,generated]), 'updates_before':self.updates,
            'actual_generation_seed':random_seed})
        eos = self.model.generation_config.eos_token_id
        eos = [eos] if isinstance(eos, int) else eos or []
        return {'raw_response':raw, 'input_tokens':len(prompt_ids), 'output_tokens':len(generated),
            'finish_reason':'stop' if generated and generated[-1] in eos else 'length',
            'rendered_prompt_sha256':hashlib.sha256(rendered.encode()).hexdigest(),
            'actual_generation_seed':random_seed, 'seconds':time.monotonic()-start,
            'adapter_enabled':self.arm!='frozen', 'parameter_updates_before':self.updates if self.arm!='frozen' else 0}

    def hidden(self, prompt_ids, response_ids):
        torch = self.torch
        ids = torch.tensor([prompt_ids + response_ids[:-1]], device=self.device)
        backbone = self.model.get_base_model().model
        output = backbone(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
        return output.last_hidden_state[0,len(prompt_ids)-1:]

    def update(self, captures, events, public_steps, reward, indices, output):
        """One on-policy optimizer step, followed by EMA. No task replay or rescore."""
        torch = self.torch
        from torch.utils.checkpoint import checkpoint
        output = Path(output); output.mkdir(parents=True, exist_ok=False)
        before = self.fingerprint('default'); teacher_before = self.fingerprint('teacher')
        samples = []; exclusions = []
        for i in indices:
            sample = captures[i]; event = events[i]
            if sample['raw_response'] != event['raw_response'] or binding(sample['messages']) != binding(event['messages']):
                raise ValueError('Action/feedback binding mismatch')
            if sample['content_binding'] != binding([sample['prompt_ids'], sample['response_ids']]):
                raise ValueError('Captured tokens changed')
            messages, feedback_audit = teacher_messages(event, public_steps, reward, self.plan['feedback_chars'])
            _, tids = self.render(messages)
            lengths = [len(sample['prompt_ids'])+len(sample['response_ids']), len(tids)+len(sample['response_ids'])]
            if max(lengths)>self.plan['training_context']:
                exclusions.append({'call':i+1, 'reason':'training_context_budget', 'lengths':lengths});continue
            samples.append((sample,tids))
            save(output/f'sample_{i+1:03}.json',dict(sample, teacher_messages=messages,
                teacher_prompt_ids=tids, **feedback_audit, official_episode_reward=reward,
                supervision='fresh feedback-conditioned teacher probabilities; exact tokens bound; no historical labels'))
        if not samples:
            result={'status':'skipped','reason':'no_eligible_actions','selected_calls':indices,'exclusions':exclusions,
                    'updates_before':self.updates,'updates_after':self.updates}
            save(output/'audit.json',result);return result
        self.model.set_adapter('default'); self.model.eval()
        self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        self.model.enable_input_require_grads()
        self.optimizer.zero_grad(set_to_none=True)
        head=self.model.get_base_model().lm_head
        losses=[]; teacher_gaps=[]; started=time.monotonic()
        for sample,tids in samples:
            # Teacher forward finishes before student checkpointed forward.
            self.model.set_adapter('teacher')
            with torch.no_grad():
                teacher_hidden=self.hidden(tids,sample['response_ids']).detach()
            self.model.set_adapter('default')
            student_hidden=self.hidden(sample['prompt_ids'],sample['response_ids'])
            n=len(sample['response_ids']); total=0.
            for start in range(0,n,self.plan['logit_chunk']):
                stop=min(n,start+self.plan['logit_chunk']);hs=student_hidden[start:stop]
                with torch.no_grad():
                    sl=head(hs.detach()).float()
                    top_indices=sl.topk(min(100,sl.shape[-1]),dim=-1).indices
                    tl=head(teacher_hidden[start:stop]).float()
                    teacher_top=tl.gather(-1,top_indices)-tl.logsumexp(-1,keepdim=True)
                    labels=torch.tensor(sample['response_ids'][start:stop],device=self.device).unsqueeze(-1)
                    gap=(tl.gather(-1,labels)-tl.logsumexp(-1,keepdim=True))-(sl.gather(-1,labels)-sl.logsumexp(-1,keepdim=True))
                    teacher_gaps.append(float(gap.mean()))
                    del sl,tl,gap
                def chunk_loss(hidden_chunk, target, chosen):
                    logits=head(hidden_chunk).float()
                    student_top=logits.gather(-1,chosen)-logits.logsumexp(-1,keepdim=True)
                    dummy=torch.zeros((1,len(hidden_chunk)),device=hidden_chunk.device)
                    loss,_=self.loss_fn(dummy,dummy,torch.ones_like(dummy),self.loss_cfg,
                        student_topk_log_probs=student_top.unsqueeze(0),teacher_topk_log_probs=target.unsqueeze(0))
                    return loss
                loss=checkpoint(chunk_loss,hs,teacher_top,top_indices,use_reentrant=False)
                weighted=loss*(stop-start)/n/len(samples)
                total+=float(weighted.detach())
                # Retain only checkpointed head tensors; one backbone backward per action.
                if start==0:action_loss=weighted
                else:action_loss=action_loss+weighted
            if not torch.isfinite(action_loss):raise FloatingPointError('Nonfinite SDPO loss')
            action_loss.backward();losses.append(total)
            del action_loss,student_hidden,teacher_hidden
        if any(p.grad is not None for name,p in self.model.named_parameters() if '.teacher.' in name or 'lora_' not in name):
            raise AssertionError('Gradient reached frozen teacher/base')
        norm=float(torch.nn.utils.clip_grad_norm_(self.parameters,self.plan['max_grad_norm'],error_if_nonfinite=True))
        self.optimizer.step();self.optimizer.zero_grad(set_to_none=True)
        self.update_teacher(self.plan['teacher_update_rate']);self.updates+=1
        after=self.fingerprint('default')
        self.model.gradient_checkpointing_disable();self.model.eval()
        result={'status':'updated','updates_before':self.updates-1,'updates_after':self.updates,
            'selected_calls':indices,'trained_actions':len(samples),'exclusions':exclusions,
            'loss':sum(losses),'gradient_norm':norm,'mean_teacher_sampled_logp_gap':sum(teacher_gaps)/len(teacher_gaps),
            'adapter_before':before,'adapter_after':after,'adapter_changed':after!=before,
            'teacher_before':teacher_before,'teacher_after':self.fingerprint('teacher'),
            'base_has_grad':False,'teacher_has_grad':False,'seconds':time.monotonic()-started,
            'loss_source':'unchanged official compute_self_distillation_loss; top100+tail, JSD alpha=0.5',
            'scalar_reward_role':'teacher feedback context; no invented scalar policy-gradient term'}
        save(output/'audit.json',result)
        return result

    def checkpoint(self, destination, index):
        from peft import get_peft_model_state_dict
        destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
        state={'student':get_peft_model_state_dict(self.model,adapter_name='default'),
               'teacher':get_peft_model_state_dict(self.model,adapter_name='teacher'),
               'optimizer':self.optimizer.state_dict(),'updates':self.updates,'last_scored_index':index,
               'base_fingerprint':self.base_fingerprint,'plan_binding':binding(self.plan)}
        temp=destination/'latest.pt.tmp';self.torch.save(state,temp);temp.replace(destination/'latest.pt')
