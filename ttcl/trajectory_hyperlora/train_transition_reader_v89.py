"""Train H plus shared A; reuse the exact cleaned-data direct LoRA control."""
import argparse
import json
from pathlib import Path
import torch
from ttcl.trajectory_hyperlora.prepare_transition_reader_v88 import build,REVIEW,BOOT
from ttcl.trajectory_hyperlora.cache_transition_reader_v88 import ROOT,MODEL,FIELDS
from ttcl.trajectory_hyperlora.transition_reader_v84 import TransitionReader
from ttcl.trajectory_hyperlora.alfworld_online_from_empty import digest,file_hash
from ttcl.trajectory_hyperlora.alfworld_online_accumulate_lora_v1 import factors_hash
from ttcl.trajectory_hyperlora.alfworld_persistent_failure_delta_v36 import save
from ttcl.trajectory_hyperlora.alfworld_zero_shot_probe import load_agent
from ttcl.trajectory_hyperlora.train_alf_next_task import target_loss


from ttcl.trajectory_hyperlora.train_transition_reader_v88 import checked_cache
import shutil
CACHE_ROOT=ROOT
ROOT=Path('results/trajectory_hyperlora/alfworld_transition_reader_v89_20261010')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--device',default='cuda:3');args=p.parse_args()
    output=ROOT/'training.json'
    if output.exists() or (ROOT/'training_manifest.json').exists():raise FileExistsError(output)
    review,targets,cache,encoding=checked_cache()
    ROOT.mkdir(parents=True,exist_ok=False)
    for path,h in encoding['model_weights'].items():assert file_hash(Path(path))==h
    manifest={'review_sha256':file_hash(REVIEW),'cache_sha256':file_hash(CACHE_ROOT/'features.pt'),
        'bootstrap_sha256':file_hash(BOOT),'runner_sha256':file_hash(Path(__file__)),
        'reader_sha256':file_hash(Path(__file__).with_name('transition_reader_v84.py')),
        'reader_parent_sha256':file_hash(Path(__file__).with_name('transition_reader_v82.py')),
        'initial_reader_sha256':file_hash(Path('results/trajectory_hyperlora/alfworld_transition_reader_v84_20261010/initial_reader.pt')),
        'top_k':8,'temperature':.05,
        'loss_sha256':file_hash(Path(__file__).with_name('train_alf_next_task.py')),
        'seed':82,'width':64,'max_relative':.25,'lr':1e-5,'weight_decay':.01,
        'budget':review['budget'],'train_shared_A':True,
        'static_training_report_sha256':file_hash(CACHE_ROOT/'training.json'),
        'norm_scope':'25% bounds B increment at the current learned A; total BA change from bootstrap is not capped at25%',
        'loss':'Mean target action token cross entropy including EOS; no wrong-source loss maximization',
        'scope':'Cleaned chronological replay training; exact v84 initial H state, equal target schedule and optimizer steps, different trainable parameter counts and compute. H additionally reads the previously completed trajectory memory.'}
    save(ROOT/'training_manifest.json',manifest)
    torch.manual_seed(82)
    agent,tok=load_agent(MODEL,BOOT.parent/'hyper.pt',args.device,.6)
    agent.requires_grad_(False);agent.eval();agent.set_source(None)
    boot=torch.load(BOOT,map_location=args.device,weights_only=True)
    with torch.no_grad():
        for layer,a in zip(agent.adapters,boot['A'],strict=True):layer.a.copy_(a.to(layer.a))
    shared=[b.clone() for b in boot['B']];bases=[a.a for a in agent.adapters]
    for a in bases:a.requires_grad_(True)
    shapes=[(b.shape[1],b.shape[2]) for b in shared]
    dim=next(iter(cache['values'].values())).shape[-1]
    reader=TransitionReader(dim,shapes,64,top_k=8,temperature=.05).to(args.device)
    initial_path=Path('results/trajectory_hyperlora/alfworld_transition_reader_v84_20261010/initial_reader.pt')
    reader.load_state_dict(torch.load(initial_path,map_location=args.device,weights_only=True))
    assert all(torch.count_nonzero(h.weight)==0 for h in reader.heads)
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
        'actor_ids_sha256':digest(prefixes),'hyper_parameters':sum(p.numel() for p in reader.parameters()),
        'shared_A_trainable_parameters':sum(a.numel() for a in bases)}
    try:
        params=list(reader.parameters())+bases
        optimizer=torch.optim.AdamW(params,lr=manifest['lr'],weight_decay=manifest['weight_decay'])
        for group in review['groups']:
            obs,act,fb=memory(group['memory_indices']);losses=[]
            for k in group['update_order']:
                reader.train();agent.eval()
                values,info=reader.adapted_factors(field(k,'key_text'),obs,act,fb,shared,bases)
                for layer,b in zip(agent.adapters,values,strict=True):layer.b=b
                loss=target_loss(agent,tok,targets[k],prefixes[k],args.device)
                assert torch.isfinite(loss)
                optimizer.zero_grad(set_to_none=True);loss.backward()
                assert all(a.grad is not None and torch.isfinite(a.grad).all() for a in bases)
                torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
                optimizer.step();agent.set_source(None);losses.append(float(loss.detach()))
            assert factors_hash(shared)==result['shared_B_sha256']
            weights=info['weights'].detach().clamp_min(1e-12)
            row={'arm':'hyper','stream_index':group['stream_index'],'group_binding':group['input_content_sha256'],
                'steps':len(losses),'order_sha256':digest(group['update_order']),'loss_mean':sum(losses)/len(losses),
                'last_attention_entropy':float(-(weights*weights.log()).sum()),'memory_transitions':len(group['memory_indices']),
                'last_clip_scale':float(info['scale'].detach()),'learned_A_sha256':factors_hash(bases)}
            result['groups'].append(row);save(output,result);print(json.dumps(row),flush=True)
        torch.save({'state':{k:v.detach().cpu() for k,v in reader.state_dict().items()},'feature_dim':dim,'factor_shapes':shapes,'width':64,
            'top_k':8,'temperature':.05,
            'shared_A':[a.detach().cpu() for a in bases],'shared_B':[b.cpu() for b in shared],
            'manifest_sha256':result['manifest_sha256'],'bootstrap_sha256':file_hash(BOOT)},ROOT/'reader.pt')
        result['learned_A_sha256']=factors_hash(bases)
        assert result['learned_A_sha256']!=result['shared_A_sha256']
        old=json.loads((CACHE_ROOT/'training.json').read_text())
        assert old['complete'] and not old['failures'] and old['static_sha256']==file_hash(CACHE_ROOT/'static.pt')
        for key in ('shared_A_sha256','shared_B_sha256','actor_ids_sha256'):assert old[key]==result[key]
        old_groups=[g for g in old['groups'] if g['arm']=='static']
        for g,t in zip(old_groups,review['groups'],strict=True):
            assert g['group_binding']==t['input_content_sha256'] and g['order_sha256']==digest(t['update_order'])
        result['groups'].extend(old_groups);result['static_parameters']=old['static_parameters']
        result['static_reused']=True;result['executed_steps']={'hyper':400,'static':0};result['inherited_static_steps']=400
        shutil.copyfile(CACHE_ROOT/'static.pt',ROOT/'static.pt')
        result.update(complete=True,reader_sha256=file_hash(ROOT/'reader.pt'),static_sha256=file_hash(ROOT/'static.pt'))
        save(output,result)
    except Exception as exc:
        result['failures'].append(repr(exc));save(output,result);raise

if __name__=='__main__':main()
