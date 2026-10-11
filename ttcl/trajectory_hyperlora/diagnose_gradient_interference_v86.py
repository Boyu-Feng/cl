"""Training-only first-order interference at one fixed common LoRA state."""
import argparse
from collections import Counter
import json
from pathlib import Path
import torch
from ttcl.trajectory_hyperlora.train_transition_reader_v82 import checked_cache
from ttcl.trajectory_hyperlora.prepare_transition_reader_v82 import BOOT,REVIEW as TRAIN_REVIEW
from ttcl.trajectory_hyperlora.cache_transition_reader_v82 import MODEL
from ttcl.trajectory_hyperlora.alfworld_online_from_empty import digest,file_hash
from ttcl.trajectory_hyperlora.alfworld_online_accumulate_lora_v1 import factors_hash
from ttcl.trajectory_hyperlora.alfworld_zero_shot_probe import load_agent
from ttcl.trajectory_hyperlora.train_alf_next_task import target_loss
from ttcl.trajectory_hyperlora.alfworld_persistent_failure_delta_v36 import save
ROOT=Path('results/trajectory_hyperlora/alfworld_gradient_interference_v86_20261010')
REVIEW=Path('data/annotations/alfworld_gradient_interference_v86_20261010.json')


def main():
    p=argparse.ArgumentParser();p.add_argument('--device',default='cuda:1');args=p.parse_args()
    torch.set_num_threads(2)
    if ROOT.exists() or REVIEW.exists():raise FileExistsError('Frozen output already exists')
    train,targets,cache,encoding=checked_cache()
    groups=[]
    for g in train['groups']:
        counts=Counter(g['update_order'])
        groups.append({'index':g['stream_index'],'parent_group_binding':g['input_content_sha256'],
            'family':targets[g['query_indices'][0]]['game'].split('/')[2].split('-')[0],
            'targets':[{'index':k,'count':n,'input_content_sha256':targets[k]['input_content_sha256']} for k,n in sorted(counts.items())]})
    review={'groups':groups,'train_review_sha256':file_hash(TRAIN_REVIEW),'bootstrap_sha256':file_hash(BOOT),
        'model_weights':encoding['model_weights'],'loader_checkpoint_sha256':file_hash(BOOT.parent/'hyper.pt'),
        'dependencies':{n:file_hash(Path(__file__).with_name(n)) for n in ['diagnose_gradient_interference_v86.py','train_transition_reader_v82.py','train_alf_next_task.py','alfworld_zero_shot_probe.py']},
        'gradient_space':'B-only Euclidean parameter gradient; A frozen at common bootstrap',
        'training_updates':0,'loss_weighting':'Each of the existing 20 scheduled action occurrences has equal weight',
        'norm_threshold':1e-8,'scope':'All20 postbootstrap own-success train-prefix groups at common state. Pairwise gradient dot products predict infinitesimal B-only SGD interaction, not Adam/finite-step or environment success. No dev/unseen targets.'}
    save(REVIEW,review);ROOT.mkdir()
    for path,h in review['model_weights'].items():assert file_hash(Path(path))==h
    agent,tok=load_agent(MODEL,BOOT.parent/'hyper.pt',args.device,.6);agent.requires_grad_(False);agent.eval();agent.set_source(None)
    boot=torch.load(BOOT,map_location=args.device,weights_only=True)
    with torch.no_grad():
        for layer,a in zip(agent.adapters,boot['A'],strict=True):layer.a.copy_(a.to(layer.a))
    factors=[b.detach().clone().requires_grad_(True) for b in boot['B']]
    for layer,b in zip(agent.adapters,factors,strict=True):layer.b=b
    initial=factors_hash(factors);gradients=[]
    result={'complete':False,'review_sha256':file_hash(REVIEW),'rows':[],'failures':[]}
    try:
        for g in groups:
            total=sum(x['count'] for x in g['targets']);assert total==20
            accumulated=[torch.zeros_like(b) for b in factors];loss_mean=0.
            for item in g['targets']:
                t=targets[item['index']];assert digest({k:v for k,v in t.items() if k not in ('input_content_sha256','reviewed_target')})==t['input_content_sha256']
                text=tok.apply_chat_template(t['actor_messages'],tokenize=False,add_generation_prompt=True)
                prefix=tok(text,add_special_tokens=False).input_ids;assert len(prefix)<=3000
                loss=target_loss(agent,tok,t,prefix,args.device);assert torch.isfinite(loss)
                grad=torch.autograd.grad(loss,factors)
                weight=item['count']/total;loss_mean+=float(loss.detach())*weight
                for acc,value in zip(accumulated,grad,strict=True):acc.add_(value.detach(),alpha=weight)
            vector=torch.cat([x.flatten().cpu() for x in accumulated]).double();assert torch.isfinite(vector).all()
            gradients.append(vector);row={'index':g['index'],'family':g['family'],'loss_mean':loss_mean,'gradient_norm':float(vector.norm())}
            result['rows'].append(row);save(ROOT/'report.json',result);print(json.dumps(row),flush=True)
        assert factors_hash(factors)==initial==factors_hash(boot['B'])
        assert factors_hash([layer.a for layer in agent.adapters])==factors_hash(boot['A'])
        matrix=torch.stack(gradients);torch.save(matrix,ROOT/'gradients.pt');pairs=[]
        for i in range(len(groups)):
            for j in range(i):
                norm=float(matrix[i].norm()*matrix[j].norm())
                valid=min(float(matrix[i].norm()),float(matrix[j].norm()))>review['norm_threshold']
                dot=float(matrix[i]@matrix[j])
                pairs.append({'indices':[groups[j]['index'],groups[i]['index']],'same_family':groups[i]['family']==groups[j]['family'],
                    'dot':dot,'cosine':dot/norm if norm>0 else None,'above_threshold':valid})
        def summary(same):
            selected=[p for p in pairs if p['above_threshold'] and p['same_family']==same]
            return {'pairs':len(selected),'negative_dot_pairs':sum(p['dot']<0 for p in selected),
                'mean_cosine':sum(p['cosine'] for p in selected)/len(selected) if selected else None}
        result.update(complete=True,gradient_file_sha256=file_hash(ROOT/'gradients.pt'),pairs=pairs,
            summary={'within_family':summary(True),'between_family':summary(False),'near_zero_group_indices':[r['index'] for r in result['rows'] if r['gradient_norm']<=review['norm_threshold']]})
        save(ROOT/'report.json',result);print(json.dumps(result['summary']),flush=True)
    except Exception as exc:
        result['failures'].append(repr(exc));save(ROOT/'report.json',result);raise

if __name__=='__main__':main()
