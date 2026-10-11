"""Train an auxiliary-supervised attention H; reuse the exact v82 direct control."""
import argparse
import json
from pathlib import Path
import torch
from ttcl.trajectory_hyperlora.prepare_transition_reader_v82 import build,REVIEW,BOOT
from ttcl.trajectory_hyperlora.cache_transition_reader_v82 import ROOT,MODEL,FIELDS
from ttcl.trajectory_hyperlora.transition_reader_v82 import TransitionReader
from ttcl.trajectory_hyperlora.alfworld_online_from_empty import digest,file_hash
from ttcl.trajectory_hyperlora.alfworld_online_accumulate_lora_v1 import factors_hash
from ttcl.trajectory_hyperlora.alfworld_persistent_failure_delta_v36 import save
from ttcl.trajectory_hyperlora.alfworld_zero_shot_probe import load_agent
from ttcl.trajectory_hyperlora.train_alf_next_task import target_loss


from ttcl.trajectory_hyperlora.train_transition_reader_v82 import checked_cache
from ttcl.trajectory_hyperlora.prepare_attention_supervision_v83 import prepare as auxiliary_review, REVIEW as AUX_REVIEW
from ttcl.trajectory_hyperlora.attention_supervision_v83 import attention_match_loss
import shutil
CACHE_ROOT=ROOT
ROOT=Path('results/trajectory_hyperlora/alfworld_transition_reader_v83_20261010')

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--device',default='cuda:3');args=p.parse_args()
    output=ROOT/'training.json'
    if output.exists() or (ROOT/'training_manifest.json').exists():raise FileExistsError(output)
    review,targets,cache,encoding=checked_cache()
    auxiliary=auxiliary_review();assert auxiliary==json.loads(AUX_REVIEW.read_text())
    auxiliary_groups={g['stream_index']:g for g in auxiliary['groups']}
    ROOT.mkdir(parents=True,exist_ok=False)
    for path,h in encoding['model_weights'].items():assert file_hash(Path(path))==h
    manifest={'review_sha256':file_hash(REVIEW),'cache_sha256':file_hash(CACHE_ROOT/'features.pt'),
        'bootstrap_sha256':file_hash(BOOT),'runner_sha256':file_hash(Path(__file__)),
        'reader_sha256':file_hash(Path(__file__).with_name('transition_reader_v82.py')),
        'loss_sha256':file_hash(Path(__file__).with_name('train_alf_next_task.py')),
        'seed':82,'width':64,'max_relative':.25,'lr':1e-5,'weight_decay':.01,
        'budget':review['budget'],'auxiliary_review_sha256':file_hash(AUX_REVIEW),'auxiliary_weight':auxiliary['auxiliary_weight'],
        'auxiliary_loss_sha256':file_hash(Path(__file__).with_name('attention_supervision_v83.py')),
        'static_training_report_sha256':file_hash(CACHE_ROOT/'training.json'),
        'loss':'Action CE plus 0.1 past exact-action attention mass loss when covered; no wrong-source maximization',
        'scope':'Chronological replay training; equal target schedule and optimizer steps, different trainable parameter counts and compute. H additionally reads the previously completed trajectory memory.'}
    save(ROOT/'training_manifest.json',manifest)
    torch.manual_seed(82)
    agent,tok=load_agent(MODEL,BOOT.parent/'hyper.pt',args.device,.6)
    agent.requires_grad_(False);agent.eval();agent.set_source(None)
    boot=torch.load(BOOT,map_location=args.device,weights_only=True)
    with torch.no_grad():
        for layer,a in zip(agent.adapters,boot['A'],strict=True):layer.a.copy_(a.to(layer.a))
    shared=[b.clone() for b in boot['B']];bases=[a.a for a in agent.adapters]
    shapes=[(b.shape[1],b.shape[2]) for b in shared]
    dim=next(iter(cache['values'].values())).shape[-1]
    reader=TransitionReader(dim,shapes,64).to(args.device)
    vectors={k:v.to(args.device) for k,v in cache['values'].items()}
    def field(k,name):return vectors[cache['rows'][k][name]]
    def memory(indices):return [torch.cat([field(k,f) for k in indices],dim=0) for f in FIELDS]
    prefixes={}
    for g in review['groups']:
        for k in g['query_indices']:
            text=tok.apply_chat_template(targets[k]['actor_messages'],tokenize=False,add_generation_prompt=True)
            prefixes[k]=tok(text,add_special_tokens=False).input_ids
            assert len(prefixes[k])<=3000
    result={'complete':False,'manifest_sha256':file_hash(ROOT/'training_manifest.json'),'groups':[],'failures':[],
        'shared_A_sha256':factors_hash(bases),'shared_B_sha256':factors_hash(shared),
        'actor_ids_sha256':digest(prefixes),'hyper_parameters':sum(p.numel() for p in reader.parameters())}
    try:
        optimizer=torch.optim.AdamW(reader.parameters(),lr=manifest['lr'],weight_decay=manifest['weight_decay'])
        for group in review['groups']:
            obs,act,fb=memory(group['memory_indices']);losses=[];actor_losses=[];aux_losses=[]
            ag=auxiliary_groups[group['stream_index']];assert ag['parent_group_binding']==group['input_content_sha256']
            positives={a['query_index']:a['positive_positions'] for a in ag['annotations']}
            for k in group['update_order']:
                reader.train();agent.eval()
                values,info=reader.adapted_factors(field(k,'key_text'),obs,act,fb,shared,bases)
                for layer,b in zip(agent.adapters,values,strict=True):layer.b=b
                actor_loss=target_loss(agent,tok,targets[k],prefixes[k],args.device)
                auxiliary_loss=attention_match_loss(info['weights'],positives[k])
                loss=actor_loss+auxiliary['auxiliary_weight']*auxiliary_loss
                actor_losses.append(float(actor_loss.detach()));aux_losses.append(float(auxiliary_loss.detach()))
                assert torch.isfinite(loss)
                optimizer.zero_grad(set_to_none=True);loss.backward()
                torch.nn.utils.clip_grad_norm_(reader.parameters(),1.,error_if_nonfinite=True)
                optimizer.step();agent.set_source(None);losses.append(float(loss.detach()))
            assert factors_hash(bases)==result['shared_A_sha256'] and factors_hash(shared)==result['shared_B_sha256']
            weights=info['weights'].detach().clamp_min(1e-12)
            row={'arm':'hyper','stream_index':group['stream_index'],'group_binding':group['input_content_sha256'],
                'steps':len(losses),'order_sha256':digest(group['update_order']),'loss_mean':sum(losses)/len(losses),
                'auxiliary_group_binding':ag['input_content_sha256'],'actor_loss_mean':sum(actor_losses)/len(actor_losses),
                'auxiliary_loss_mean':sum(aux_losses)/len(aux_losses),'last_attention_entropy':float(-(weights*weights.log()).sum()),'memory_transitions':len(group['memory_indices']),
                'last_clip_scale':float(info['scale'].detach())}
            result['groups'].append(row);save(output,result);print(json.dumps(row),flush=True)
        torch.save({'state':{k:v.detach().cpu() for k,v in reader.state_dict().items()},'feature_dim':dim,'factor_shapes':shapes,'width':64,
            'shared_A':[a.detach().cpu() for a in bases],'shared_B':[b.cpu() for b in shared],
            'manifest_sha256':result['manifest_sha256'],'bootstrap_sha256':file_hash(BOOT)},ROOT/'reader.pt')
        # Reuse the exact completed v82 direct control: same bootstrap, labels,
        # optimizer schedule and budget. Do not train a new numerical replica.
        old=json.loads((CACHE_ROOT/'training.json').read_text())
        assert old['complete'] and not old['failures']
        assert old['static_sha256']==file_hash(CACHE_ROOT/'static.pt')
        old_groups=[g for g in old['groups'] if g['arm']=='static']
        for g,t in zip(old_groups,review['groups'],strict=True):
            assert g['group_binding']==t['input_content_sha256'] and g['order_sha256']==digest(t['update_order'])
        result['groups'].extend(old_groups)
        result['static_parameters']=old['static_parameters']
        result['static_reused']=True
        result['executed_steps']={'hyper':400,'static':0}
        result['inherited_static_steps']=400
        shutil.copyfile(CACHE_ROOT/'static.pt',ROOT/'static.pt')
        result.update(complete=True,reader_sha256=file_hash(ROOT/'reader.pt'),static_sha256=file_hash(ROOT/'static.pt'))
        save(output,result)
    except Exception as exc:
        result['failures'].append(repr(exc));save(output,result);raise

if __name__=='__main__':main()
