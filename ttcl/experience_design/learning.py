from __future__ import annotations

import hashlib
import math
from pathlib import Path
import random
import statistics
import time

from ttcl.experience_evolution.core import read,save,append
from ttcl.experience_repair.run import encode_training


def reward_label(candidate,previous,empty,objective):
    if not candidate or not (len(candidate)==len(previous)==len(empty)):
        raise ValueError('Incomplete paired reward vectors')
    if any(x is None or not math.isfinite(float(x)) for values in [candidate,previous,empty] for x in values):
        raise ValueError('Missing/nonfinite outcomes cannot be scored as zero')
    if objective=='absolute':return statistics.mean(candidate)
    if objective=='empty':return statistics.mean(a-b for a,b in zip(candidate,empty))
    if objective=='previous':return statistics.mean(a-b for a,b in zip(candidate,previous))
    raise ValueError(objective)


def adapter_hash(model,name):
    digest=hashlib.sha256()
    for key,p in model.named_parameters():
        if f'.{name}.' in key and 'lora_' in key:
            digest.update(key.encode());digest.update(p.detach().contiguous().view(__import__('torch').uint8).cpu().numpy().tobytes())
    return digest.hexdigest()


def make_writer(plan,adapter):
    from ttcl.experience_evolution.writer import Writer
    writer=Writer(plan,adapter=adapter,train=True)
    writer.model.load_adapter(str(adapter),adapter_name='reference',is_trainable=False)
    writer.model.set_adapter('default')
    writer.reference_hash=adapter_hash(writer.model,'reference')
    assert all('lora_' in n and '.default.' in n for n,p in writer.model.named_parameters() if p.requires_grad)
    return writer


def update(writer,samples):
    import torch
    from ttcl.experience_evolution.writer import clipped_loss
    metrics=[]
    for epoch in range(2):
        writer.optimizer.zero_grad(set_to_none=True);policies=[];kls=[];mismatches=[]
        for sample in samples:
            writer.model.set_adapter('reference');writer.model.eval()
            with torch.no_grad():reference,_=writer.distribution(sample)
            writer.model.set_adapter('default');writer.model.train()
            logp,selected=writer.distribution(sample)
            old=torch.tensor(sample['old_logp'],device='cuda:0')
            mismatch=float((selected.detach()-old).abs().mean());mismatches.append(mismatch)
            if epoch==0 and mismatch>.15:raise ValueError(f'Rollout/training logprob mismatch {mismatch}')
            policy=clipped_loss(selected,old,sample['advantage'])
            kl=(logp.exp()*(logp-reference)).sum(-1).mean()
            loss=(policy+writer.plan['kl_beta']*kl)/len(samples)
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
            policies.append(float(policy.detach()));kls.append(float(kl.detach()))
            loss.backward();del reference,logp,selected,old,policy,kl,loss
        norm=float(torch.nn.utils.clip_grad_norm_(writer.parameters,1.,error_if_nonfinite=True))
        writer.optimizer.step()
        metrics.append({'epoch':epoch,'policy_loss':statistics.mean(policies),'reference_kl':statistics.mean(kls),
                        'gradient_norm':norm,'max_rollout_logprob_mae':max(mismatches)})
    writer.model.eval()
    if adapter_hash(writer.model,'reference')!=writer.reference_hash:raise AssertionError('Reference adapter changed')
    return {'epochs':metrics,'examples':len(samples),'positive':sum(s['advantage']>0 for s in samples),
            'negative':sum(s['advantage']<0 for s in samples),'zero':sum(s['advantage']==0 for s in samples)}


def train_sft(root,arm):
    import torch
    from transformers import AutoTokenizer,AutoModelForCausalLM
    from peft import PeftModel
    from ttcl.experience_evolution.writer import fingerprint
    plan=read(root/'plan.json');cfg=plan['training'];out=root/'training'/arm
    if (out/'status.json').exists() and read(out/'status.json')['phase']=='complete':return
    if out.exists():raise RuntimeError('Partial training requires checkpoint recovery')
    out.mkdir(parents=True);save(out/'status.json',{'phase':'loading'})
    tokenizer=AutoTokenizer.from_pretrained(plan['model'],local_files_only=True)
    data=[]
    for name in plan['sft_arms']:
        values=read(root/'training_data'/f'{name}.json')
        assert len(values)==128
        encoded=[encode_training(tokenizer,row,cfg['max_length']) for row in values]
        if name==arm:data=encoded
    torch.manual_seed(cfg['seed'])
    base=AutoModelForCausalLM.from_pretrained(plan['model'],local_files_only=True,torch_dtype=torch.bfloat16,
                                            attn_implementation='sdpa').to('cuda:0')
    model=PeftModel.from_pretrained(base,str(root/'adapters/delta'),is_trainable=True)
    model.config.use_cache=False;model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    model.enable_input_require_grads();parameters=[p for p in model.parameters() if p.requires_grad]
    assert parameters and all('lora_' in n for n,p in model.named_parameters() if p.requires_grad)
    before=fingerprint(model);initial=fingerprint(model,True)
    optimizer=torch.optim.AdamW(parameters,lr=cfg['learning_rate'],weight_decay=0.)
    rng=random.Random(cfg['seed']);step=0;model.train()
    for epoch in range(cfg['epochs']):
        order=list(range(len(data)));rng.shuffle(order)
        for offset in range(0,len(order),cfg['accumulation']):
            batch=order[offset:offset+cfg['accumulation']];optimizer.zero_grad(set_to_none=True);losses=[]
            for i in batch:
                item=data[i];ids=torch.tensor([item['ids'][:-1]],device='cuda:0')
                targets=torch.tensor([item['ids'][-item['target_length']:]],device='cuda:0')
                logits=model(input_ids=ids,attention_mask=torch.ones_like(ids),logits_to_keep=item['target_length'],use_cache=False).logits
                loss=torch.nn.functional.cross_entropy(logits.float().reshape(-1,logits.shape[-1]),targets.reshape(-1))
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
                losses.append(float(loss.detach()));(loss/len(batch)).backward()
                del ids,targets,logits,loss
            norm=float(torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True));optimizer.step();step+=1
            value={'phase':'training','epoch':epoch+1,'step':step,'expected_steps':64,'mean_loss':statistics.mean(losses),
                   'gradient_norm':norm,'updated_at':time.time()}
            append(out/'training.jsonl',value);save(out/'status.json',value);print(value,flush=True)
    assert fingerprint(model)==before
    model.save_pretrained(out/'adapter');tokenizer.save_pretrained(out/'adapter')
    save(out/'audit.json',{'base_unchanged':True,'initial_adapter':initial,'final_adapter':fingerprint(model,True),
                         'examples':128,'assistant_only_loss':True})
    save(out/'status.json',{'phase':'complete','steps':step,'examples':128})
