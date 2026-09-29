"""Full-parameter GRPO with CPU FP32 AdamW states; no PEFT or adapters."""
from __future__ import annotations
import gc
import math
from pathlib import Path
import time
from .common import binding,save,seed
from .upstream import grpo


class FullPolicy:
    def __init__(self,path,device='cuda:0',model=None,tokenizer=None):
        import torch
        from transformers import AutoModelForCausalLM,AutoTokenizer
        self.torch=torch;self.device=device
        self.tokenizer=tokenizer or AutoTokenizer.from_pretrained(path,local_files_only=True,padding_side='left')
        self.tokenizer.padding_side='left'
        if self.tokenizer.pad_token_id is None:self.tokenizer.pad_token_id=self.tokenizer.eos_token_id
        self.model=model or AutoModelForCausalLM.from_pretrained(path,local_files_only=True,
            torch_dtype=torch.bfloat16,attn_implementation='sdpa')
        self.model.requires_grad_(True);self.model.config.use_cache=False
        self.model.generation_config.top_k=0
        self.master=None;self.optimizer=None;self.updates=0;self.official=grpo()
        self.to(device)

    def to(self,device):
        self.model.to(device);self.device=device
        if str(device)=='cpu':gc.collect();self.torch.cuda.empty_cache()
        return self

    def render(self,messages):
        rendered=self.tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
        return self.tokenizer.encode(rendered,add_special_tokens=False)

    def generate(self,messages,random_seed,max_tokens,context,temperature=1.):
        return self.generate_batch([messages],random_seed,max_tokens,context,temperature)[0]

    def generate_batch(self,messages,random_seed,max_tokens,context,temperature=1.):
        from transformers import LogitsProcessor,LogitsProcessorList,StoppingCriteria,StoppingCriteriaList
        torch=self.torch;prompts=[self.render(m) for m in messages]
        if any(len(p)+max_tokens>context for p in prompts):
            raise ValueError('Declared generation context exceeded; no prompt truncation')
        # Bound worst-case KV allocation; full per-response budget remains unchanged.
        if len(prompts)>1 and str(self.device).startswith('cuda'):
            cfg=self.model.config
            kv_per_token=2*cfg.num_hidden_layers*cfg.num_key_value_heads*getattr(cfg,'head_dim',cfg.hidden_size//cfg.num_attention_heads)*2
            free,_=torch.cuda.mem_get_info()
            cap=max(1,int((free-3*2**30)//(kv_per_token*(max(map(len,prompts))+max_tokens))))
            if len(prompts)>cap:
                out=[]
                for i in range(0,len(prompts),cap):
                    out.extend(self.generate_batch(messages[i:i+cap],seed(random_seed,i),max_tokens,context,temperature))
                return out
        class Capture(LogitsProcessor):
            def __call__(self,ids,scores):
                self.logits=scores.float()
                self.lognorm=self.logits.logsumexp(-1)
                return scores
        capture=Capture();saved=[]
        class Record(StoppingCriteria):
            def __call__(self,ids,scores,**kwargs):
                token=ids[:,-1]
                logp=capture.logits.gather(1,token[:,None]).squeeze(1)-capture.lognorm
                saved.append(logp.detach().cpu())
                return torch.zeros(ids.shape[0],dtype=torch.bool,device=ids.device)
        ids=self.tokenizer.pad({'input_ids':prompts},return_tensors='pt').to(self.device)
        torch.manual_seed(random_seed)
        if str(self.device).startswith('cuda'):torch.cuda.manual_seed_all(random_seed)
        self.model.eval();self.model.gradient_checkpointing_disable()
        start=time.monotonic()
        with torch.inference_mode():
            result=self.model.generate(**ids,max_new_tokens=max_tokens,do_sample=temperature>0,
                temperature=temperature if temperature>0 else 1.,top_p=1.,top_k=0,
                repetition_penalty=1.,use_cache=True,pad_token_id=self.tokenizer.pad_token_id,
                logits_processor=LogitsProcessorList([capture]),stopping_criteria=StoppingCriteriaList([Record()]))
        eos=self.model.generation_config.eos_token_id
        eos=set(eos if isinstance(eos,list) else [eos]);probs=torch.stack(saved,dim=1).tolist()
        generated=result[:,ids['input_ids'].shape[1]:].tolist();rows=[]
        for i,tokens in enumerate(generated):
            end=next((j+1 for j,t in enumerate(tokens) if t in eos),len(tokens));tokens=tokens[:end]
            rows.append(dict(prompt_ids=prompts[i],response_ids=tokens,old_log_probs=probs[i][:end],
                raw_response=self.tokenizer.decode(tokens,skip_special_tokens=True),
                input_tokens=len(prompts[i]),output_tokens=len(tokens),
                finish_reason='stop' if tokens and tokens[-1] in eos else 'length',
                input_binding=binding(messages[i]),token_binding=binding([prompts[i],tokens]),
                sampling_temperature=temperature,actual_generation_seed=random_seed,batch_position=i,
                updates_before=self.updates,seconds=time.monotonic()-start))
        del capture,ids,result
        return rows

    def log_probs(self,event):
        torch=self.torch
        if event['token_binding']!=binding([event['prompt_ids'],event['response_ids']]):
            raise ValueError('Captured token binding changed')
        prompt=event['prompt_ids'];response=event['response_ids']
        ids=torch.tensor([prompt+response[:-1]],device=self.device)
        hidden=self.model.model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False).last_hidden_state
        hidden=hidden[0,len(prompt)-1:]
        targets=torch.tensor(response,device=self.device)
        from torch.utils.checkpoint import checkpoint
        def head_logp(h,labels):
            logits=self.model.lm_head(h).float()
            return logits.gather(1,labels[:,None]).squeeze(1)-logits.logsumexp(-1)
        chunks=[]
        for start in range(0,len(response),64):
            h=hidden[start:start+64];labels=targets[start:start+64]
            chunks.append(checkpoint(head_logp,h,labels,use_reentrant=False) if torch.is_grad_enabled() else head_logp(h,labels))
        return torch.cat(chunks)

    def configure_optimizer(self,lr):
        torch=self.torch
        if self.master is None:
            self.master=[torch.nn.Parameter(p.detach().float().cpu().clone()) for p in self.model.parameters()]
            self.optimizer=torch.optim.AdamW(self.master,lr=lr,betas=(.9,.999),eps=1e-8,weight_decay=.01,foreach=False)
        else:
            for group in self.optimizer.param_groups:group['lr']=lr

    def update(self,trajectories,lr,output=None):
        """One complete on-policy GRPO batch. Zero-variance groups retain records."""
        torch=self.torch;self.configure_optimizer(lr);start=time.monotonic()
        rewards=torch.tensor([[t['reward']] for t in trajectories],dtype=torch.float32)
        import numpy as np
        groups=np.asarray([t['group_id'] for t in trajectories])
        advantages,_=self.official.compute_grpo_outcome_advantage(rewards,torch.ones_like(rewards),groups)
        total_tokens=sum(len(e['response_ids']) for t in trajectories for e in t['events'])
        if total_tokens==0:raise ValueError('No policy tokens in training batch')
        self.model.train();self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        self.model.config.use_cache=False;self.model.zero_grad(set_to_none=True)
        self.optimizer.zero_grad(set_to_none=True)
        losses=[];kls=[];diffs=[];active=0
        # Token mean across the complete batch matches the official default reduction.
        for t,adv in zip(trajectories,advantages[:,0].tolist()):
            if adv==0:continue
            active+=1
            for event in t['events']:
                if event.get('sampling_temperature',1.)!=1.:raise ValueError('Training rollouts must use temperature 1')
                logp=self.log_probs(event)
                old=torch.tensor(event['old_log_probs'],device=self.device,dtype=torch.float32)
                if old.shape!=logp.shape:raise ValueError('Rollout probabilities length mismatch')
                delta=(logp.detach()-old).abs()
                diffs.append(float(delta.mean()))
                # Same on-policy weights and backend; never silently correct large mismatch.
                if float(delta.mean())>.15 or not torch.isfinite(logp).all():
                    raise ValueError(f'On-policy probability mismatch: {float(delta.mean())}')
                n=logp.numel();mask=torch.ones_like(logp)[None]
                loss,_,kl,_=self.official.compute_policy_loss(old[None],logp[None],
                    torch.full_like(logp,adv)[None],mask,cliprange=.2,loss_agg_mode='token-mean')
                (loss*n/total_tokens).backward()
                losses.append(float(loss.detach())*n/total_tokens);kls.append(float(kl))
                del logp,old,loss
        if not active:
            result={'status':'zero_advantage','updated':False,'updates':self.updates,
                    'trajectories':len(trajectories),'mean_reward':float(rewards.mean()),'seconds':time.monotonic()-start}
            if output:save(output,result)
            return result
        # CPU master parameters retain FP32 Adam moments while every model weight trains.
        params=list(self.model.parameters())
        for p,master in zip(params,self.master):
            master.grad=p.grad.detach().float().cpu() if p.grad is not None else None
            p.grad=None
        norm=torch.nn.utils.clip_grad_norm_(self.master,1.,error_if_nonfinite=True)
        self.optimizer.step()
        changed=0.;grad_params=0;changed_tensors=0;master_changed_tensors=0
        with torch.no_grad():
            for p,master in zip(params,self.master):
                if master.grad is not None:grad_params+=p.numel()
                delta=(master-p.detach().float().cpu()).abs().max().item()
                changed=max(changed,delta)
                master_changed_tensors+=int(delta>0)
                changed_tensors+=int(bool((master.to(dtype=p.dtype)!=p.detach().cpu()).any()))
                p.copy_(master.to(device=p.device,dtype=p.dtype));master.grad=None
        self.updates+=1
        result={'status':'updated','updated':True,'updates':self.updates,'loss':sum(losses),
            'mean_reward':float(rewards.mean()),'gradient_norm':float(norm),'max_master_delta':changed,
            'bf16_changed_tensors':changed_tensors,'master_changed_tensors':master_changed_tensors,
            'trainable_parameters':sum(p.numel() for p in params if p.requires_grad),
            'total_parameters':sum(p.numel() for p in params),'parameters_with_grad':grad_params,
            'trajectories':len(trajectories),'nonzero_advantage_trajectories':active,
            'policy_tokens':total_tokens,'old_probability_mean_abs_error':max(diffs,default=0.),
            'mean_approx_kl':sum(kls)/len(kls) if kls else 0.,'seconds':time.monotonic()-start,
            'lora':False,'optimizer':'FP32 CPU AdamW masters; BF16 model; all parameters'}
        if output:save(output,result)
        return result

    def save_model(self,directory):
        directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
        self.model.save_pretrained(directory,safe_serialization=True,max_shard_size='4GB')
        self.tokenizer.save_pretrained(directory)
        save(directory/'training_state.json',{'updates':self.updates,'full_parameter':True,'lora':False,
            'optimizer_state_included':False,'note':'Inference checkpoint; do not resume Adam from this artifact as if optimizer state were preserved.'})
