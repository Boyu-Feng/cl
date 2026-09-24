"""Matched SFT controls and reference-regularized preference training."""
from __future__ import annotations

from pathlib import Path
import math
import statistics
import time

from ttcl.experience_evolution.core import read, save, append, seed
from ttcl.experience_v2.common import sha_file
from .protocol import balanced_order, binding


def train(root, round_id, arm):
    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    from ttcl.experience_evolution.writer import fingerprint
    from ttcl.experience_repair.run import encode_training
    root = Path(root); directory = root/round_id
    plan = read(root/'plan.json'); cfg = plan['training']
    if arm not in plan['training_arms']: raise ValueError(arm)
    output = directory/'training'/arm
    output.mkdir(parents=True, exist_ok=False)
    save(output/'status.json', {'phase': 'loading'})
    audit = read(directory/'dataset_audit.json')
    if sha_file(directory/'dataset.json') != audit['dataset_sha256']:
        raise ValueError('Dataset changed after automatic annotation review')
    for p, h in audit['provenance_hashes'].items():
        if sha_file(p) != h: raise ValueError('Label source changed')
    rows = read(directory/'dataset.json')
    if len(rows) < plan['min_pairs']: raise ValueError('Insufficient confirmed preferences')
    tokenizer = AutoTokenizer.from_pretrained(plan['model'], local_files_only=True)
    encoded = []
    for row in rows:
        if binding(row['messages']) != row['input_binding']: raise ValueError('Input binding mismatch')
        encoded.append({key: encode_training(tokenizer, {'id': row['id'], 'messages': row['messages'], 'target': row[key]},
                          cfg['max_length']) for key in ['chosen', 'rejected', 'random_target']})
    torch.manual_seed(cfg['seed'])
    base = AutoModelForCausalLM.from_pretrained(plan['model'], local_files_only=True,
        torch_dtype=torch.bfloat16, attn_implementation='sdpa').to('cuda:0')
    parent = read(directory/'round_input.json')['parent_adapter']
    model = PeftModel.from_pretrained(base, parent, is_trainable=True)
    if arm=='dpo':
        model.load_adapter(parent, adapter_name='reference', is_trainable=False)
        model.set_adapter('default')
    model.config.use_cache=False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    model.enable_input_require_grads()
    parameters=[p for p in model.parameters() if p.requires_grad]
    if not parameters or any('lora_' not in n or '.default.' not in n
                              for n,p in model.named_parameters() if p.requires_grad):
        raise AssertionError('Only writer default LoRA can be optimized')
    def reference_hash():
        import hashlib
        h=hashlib.sha256()
        for n,p in model.named_parameters():
            if 'lora_' in n and '.reference.' in n:
                h.update(n.encode());h.update(p.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes())
        return h.hexdigest()
    reference_before=reference_hash()
    before=fingerprint(model)
    optimizer=torch.optim.AdamW(parameters, lr=cfg['learning_rate'], weight_decay=0.)
    def log_probability(item):
        ids=torch.tensor([item['ids'][:-1]], device='cuda:0')
        labels=torch.tensor(item['ids'][-item['target_length']:], device='cuda:0')
        logits=model(input_ids=ids, attention_mask=torch.ones_like(ids),
                     logits_to_keep=item['target_length'], use_cache=False).logits[0].float()
        return -F.cross_entropy(logits, labels, reduction='none')
    step=0
    expected_order=balanced_order(rows, cfg['seed'])
    expected=cfg['epochs']*math.ceil(len(expected_order)/cfg['accumulation'])
    for epoch in range(cfg['epochs']):
        order=balanced_order(rows, seed(cfg['seed'], epoch))
        for offset in range(0,len(order),cfg['accumulation']):
            batch=order[offset:offset+cfg['accumulation']]
            optimizer.zero_grad(set_to_none=True); losses=[]; margins=[]
            for index in batch:
                item=encoded[index]
                model.set_adapter('default');model.train()
                if arm=='dpo':
                    model.set_adapter('reference');model.eval()
                    with torch.no_grad():
                        rc=log_probability(item['chosen']).sum()
                        rr=log_probability(item['rejected']).sum()
                    model.set_adapter('default');model.train()
                    chosen=log_probability(item['chosen'])
                    rejected=log_probability(item['rejected'])
                    margin=cfg['dpo_beta']*((chosen.sum()-rejected.sum())-(rc-rr))
                    loss=-F.logsigmoid(margin)-cfg['chosen_nll_weight']*chosen.mean()
                    margins.append(float(margin.detach()))
                else:
                    key='chosen' if arm=='filtered_sft' else 'random_target'
                    loss=-log_probability(item[key]).mean()
                if not torch.isfinite(loss): raise FloatingPointError('Nonfinite writer loss')
                losses.append(float(loss.detach()));(loss/len(batch)).backward()
                if arm=='dpo': del chosen,rejected,margin,rc,rr
                del loss
            norm=float(torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True))
            optimizer.step();step+=1
            result={'phase':'training','epoch':epoch+1,'step':step,'expected_steps':expected,
                    'mean_loss':statistics.mean(losses),'gradient_norm':norm,
                    'dpo_margin':statistics.mean(margins) if margins else None,'updated_at':time.time()}
            save(output/'status.json',result);append(output/'training.jsonl',result)
    if fingerprint(model)!=before: raise AssertionError('Frozen actor/base weights changed')
    if reference_hash()!=reference_before: raise AssertionError('DPO reference adapter changed')
    model.set_adapter('default')
    model.save_pretrained(output/'adapter', selected_adapters=['default'])
    tokenizer.save_pretrained(output/'adapter')
    save(output/'audit.json', {'base_unchanged':True,'base_sha256':before,'reference_unchanged':True,
        'initial_adapter':parent,'unique_histories':len(rows),'domain_balanced_samples_per_epoch':len(expected_order),
        'assistant_only_loss':True,'fixed_final_checkpoint':True,
        'trainable_parameters':sum(p.numel() for p in parameters)})
    save(output/'status.json', {'phase':'complete','steps':step,'examples':len(rows)})
