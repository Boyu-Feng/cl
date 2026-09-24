"""LoRA-only PPO-style updates with exact behavior tokens and immutable references."""
from __future__ import annotations

from collections import defaultdict
import hashlib
import math
from pathlib import Path
import random
import statistics
import time

from ttcl.experience_evolution.core import read, save, append, seed
from ttcl.experience_v2.common import sha_file
from .protocol import validate_sample


def initialize_reader(root):
    """CPU-only preparation; zero B matrices make the initial reader the base policy."""
    import torch
    from transformers import AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model
    root=Path(root); plan=read(root/'plan.json'); output=Path(plan['initial_reader'])
    if output.exists(): raise FileExistsError(output)
    torch.manual_seed(plan['training']['seed'])
    base=AutoModelForCausalLM.from_pretrained(plan['model'],local_files_only=True,
        torch_dtype=torch.bfloat16,attn_implementation='sdpa')
    model=get_peft_model(base,LoraConfig(**plan['reader_lora']))
    parameters={n:p for n,p in model.named_parameters() if p.requires_grad}
    if not parameters or any('lora_' not in n for n in parameters):
        raise AssertionError('Reader must consist only of LoRA parameters')
    if any(torch.count_nonzero(p).item() for n,p in parameters.items() if 'lora_B' in n):
        raise AssertionError('Initial reader must have zero residual')
    model.save_pretrained(output)
    save(output/'initialization_audit.json',{'zero_residual':True,'device':'cpu',
        'trainable_parameters':sum(p.numel() for p in parameters.values()),
        'layers':plan['reader_lora']['layers_to_transform'],
        'target_modules':plan['reader_lora']['target_modules']})
    save(root/'reader_initial_hashes.json',{str(p):sha_file(p) for p in output.rglob('*') if p.is_file()})
    return read(output/'initialization_audit.json')


def selected_logps(model, sample, device):
    import torch
    import torch.nn.functional as F
    ids=torch.tensor([sample['input_ids'][:-1]],device=device)
    labels=torch.tensor(sample['input_ids'][sample['prompt_length']:],device=device)
    logits=model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False,
                 logits_to_keep=len(labels)).logits[0].float()
    return -F.cross_entropy(logits,labels,reduction='none')


def ppo_loss(selected, old, reference, advantage, clip=.2, beta=.01):
    import torch
    log_ratio=selected-old
    if not torch.isfinite(log_ratio).all() or log_ratio.detach().abs().max()>20:
        raise FloatingPointError('Unstable PPO probability ratio')
    ratio=log_ratio.exp()
    policy=-torch.minimum(ratio*advantage,ratio.clamp(1-clip,1+clip)*advantage).mean()
    # k3 sampled-token KL estimator, not a claim to compute full-vocabulary KL.
    difference=reference-selected
    if difference.detach().abs().max()>20:
        raise FloatingPointError('Unstable reference probability ratio')
    kl=(difference.exp()-difference-1).mean()
    return policy+beta*kl, policy, kl, float(((ratio-1).abs()>clip).float().mean().detach())


def train(root, arm, block_id, role, device='cuda:0'):
    import torch
    from transformers import AutoModelForCausalLM
    from peft import PeftModel
    from ttcl.experience_evolution.writer import fingerprint
    root=Path(root); plan=read(root/'plan.json'); cfg=plan['training']
    if arm not in plan['arms'] or role not in ['writer','reader'] or (arm=='writer_only' and role=='reader'):
        raise ValueError('Invalid optimizer role')
    directory=root/'training'/arm/block_id; output=directory/role
    output.mkdir(parents=True,exist_ok=False)
    inputs=read(directory/'block_input.json')
    for p,h in {**inputs['hashes'],**read(directory/'dataset_audit.json')['hashes']}.items():
        if sha_file(p)!=h: raise ValueError('Training input changed: '+p)
    rows=read(directory/f'{role}_dataset.json')
    if not rows: raise ValueError('No usable on-policy samples; refusing an empty update')
    samples=[]
    for row in rows:
        value=read(row['path'])[row['sample_key']]
        sample=value[row['sample_index']] if 'sample_index' in row else value
        validate_sample(sample,row.get('input_binding'))
        if sample['model'] != ('writer_current' if role=='writer' else 'reader_current'):
            raise ValueError('Wrong policy supplied to optimizer')
        if len(sample['input_ids'])>cfg['max_length']:
            raise ValueError('Unreviewed context overflow')
        samples.append(sample)
    torch.manual_seed(cfg['seed'])
    base=AutoModelForCausalLM.from_pretrained(plan['model'],local_files_only=True,
        torch_dtype=torch.bfloat16,attn_implementation='sdpa').to(device)
    model=PeftModel.from_pretrained(base,inputs[role],is_trainable=True)
    if role=='writer':
        model.load_adapter(plan['initial_adapter'],adapter_name='reference',is_trainable=False)
        model.set_adapter('default')
    model.config.use_cache=False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    model.enable_input_require_grads()
    parameters=[p for n,p in model.named_parameters() if p.requires_grad]
    if not parameters or any('lora_' not in n or '.default.' not in n
                              for n,p in model.named_parameters() if p.requires_grad):
        raise AssertionError('Only the selected role LoRA may receive gradients')
    before=fingerprint(model)
    adapter_before=fingerprint(model,True)
    def reference_hash():
        result=hashlib.sha256()
        for n,p in model.named_parameters():
            if '.reference.' in n:
                result.update(n.encode());result.update(p.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes())
        return result.hexdigest()
    frozen_reference=reference_hash()
    optimizer=torch.optim.AdamW(parameters,lr=cfg['learning_rate'],weight_decay=0.)
    previous_optimizer=inputs.get(role+'_optimizer')
    if previous_optimizer:
        optimizer.load_state_dict(torch.load(previous_optimizer,map_location=device,weights_only=True))
    references=[]; mismatches=[]
    # Validate the entire rollout while it still corresponds to the exact current
    # checkpoint. Checking after minibatch updates would confuse learning with drift.
    model.eval()
    for index,sample in enumerate(samples):
        with torch.no_grad():
            current=selected_logps(model,sample,device)
            old=torch.tensor(sample['old_logp'],device=device)
            mismatch=float((current-old).abs().mean())
            if mismatch>cfg['max_behavior_logp_mae']:
                raise ValueError(f'Behavior/training probability mismatch at {index}: {mismatch}')
            mismatches.append(mismatch)
            if role=='writer':
                model.set_adapter('reference')
                references.append(selected_logps(model,sample,device).cpu())
                model.set_adapter('default')
            else:
                with model.disable_adapter():
                    references.append(selected_logps(model,sample,device).cpu())
        if index%16==0:
            save(output/'status.json',{'phase':'checking_behavior_probabilities','checked':index+1,
                                      'total':len(samples),'time':time.time()})
    domain_mass=defaultdict(float)
    for row in rows: domain_mass[row['domain']]+=row.get('within_episode_weight',1.)
    weights=[row.get('within_episode_weight',1.)/domain_mass[row['domain']]/len(domain_mass) for row in rows]
    batch_size=cfg['reader_minibatch_size'] if role=='reader' else cfg['minibatch_size']
    step=0
    for epoch in range(cfg['ppo_epochs']):
        order=list(range(len(rows)));random.Random(seed(cfg['seed'],arm,block_id,role,epoch)).shuffle(order)
        for start in range(0,len(order),batch_size):
            batch=order[start:start+batch_size];optimizer.zero_grad(set_to_none=True)
            metrics=[]
            for i in batch:
                model.set_adapter('default');model.train()
                selected=selected_logps(model,samples[i],device)
                old=torch.tensor(samples[i]['old_logp'],device=device)
                loss,policy,kl,clipped=ppo_loss(selected,old,references[i].to(device),rows[i]['advantage'],
                                               cfg['clip_range'],cfg['kl_beta'])
                if not torch.isfinite(loss): raise FloatingPointError('Nonfinite PPO loss')
                (loss*len(rows)*weights[i]/len(batch)).backward()
                metrics.append((float(policy.detach()),float(kl.detach()),clipped))
                del loss,policy,kl,selected,old
            norm=float(torch.nn.utils.clip_grad_norm_(parameters,cfg['max_grad_norm'],error_if_nonfinite=True))
            optimizer.step();step+=1
            event={'phase':'training','role':role,'epoch':epoch+1,'step':step,
                'expected_steps':cfg['ppo_epochs']*math.ceil(len(rows)/batch_size),
                'policy_loss':statistics.mean(v[0] for v in metrics),
                'sampled_reference_kl':statistics.mean(v[1] for v in metrics),
                'clip_fraction':statistics.mean(v[2] for v in metrics),
                'gradient_norm':norm,'time':time.time()}
            append(output/'training.jsonl',event);save(output/'status.json',event)
    if fingerprint(model)!=before: raise AssertionError('Frozen base parameters changed')
    if reference_hash()!=frozen_reference: raise AssertionError('Reference writer changed')
    for p,h in inputs['hashes'].items():
        if sha_file(p)!=h: raise AssertionError('Other role or initial checkpoint changed')
    model.set_adapter('default');model.save_pretrained(output/'adapter',selected_adapters=['default'])
    torch.save(optimizer.state_dict(),output/'optimizer.pt')
    save(output/'audit.json',{'base_unchanged':True,'reference_unchanged':True,
        'base_sha256':before,'adapter_before':adapter_before,'adapter_after':fingerprint(model,True),
        'other_role_checkpoint_unchanged':True,'role':role,'sample_count':len(samples),
        'positive':sum(r['advantage']>0 for r in rows),'negative':sum(r['advantage']<0 for r in rows),
        'zero':sum(r['advantage']==0 for r in rows),'domains':dict(domain_mass),
        'max_behavior_logp_mae':max(mismatches),'mean_behavior_logp_mae':statistics.mean(mismatches),
        'trainable_parameters':sum(p.numel() for p in parameters),
        'assistant_tokens_only':True,'future_task_text_excluded_from_writer':True})
    save(output/'status.json',{'phase':'complete','role':role,'steps':step,'samples':len(samples)})
