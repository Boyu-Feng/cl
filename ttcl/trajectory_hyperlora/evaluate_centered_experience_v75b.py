"""Five-arm next24 evaluation of the shared-bootstrap centered increment."""
import argparse
from contextlib import nullcontext
import json
from pathlib import Path

import torch

from ttcl.trajectory_hyperlora.evaluate_causal_prefix_v70 import episode,audit,ARMS,EMPTY
from ttcl.trajectory_hyperlora.centered_experience_update_v75 import centered_reader
from ttcl.trajectory_hyperlora.alfworld_zero_shot_probe import load_agent
from ttcl.trajectory_hyperlora.contextual_alf_source import contextual_text_fields
from ttcl.trajectory_hyperlora.alfworld_online_from_empty import digest,file_hash
from ttcl.trajectory_hyperlora.alfworld_persistent_failure_delta_v36 import save
from ttcl.trajectory_hyperlora.alfworld_online_accumulate_lora_v1 import factors_hash


def bindings(args):
    manifest=json.loads((args.training/'manifest.json').read_text())
    result=json.loads((args.training/'training.json').read_text())
    assert result['complete'] and not result['failures']
    assert result['manifest_sha256']==file_hash(args.training/'manifest.json')
    assert result['bootstrap_sha256']==file_hash(args.training/'bootstrap.pt')
    for name,value in manifest['dependencies'].items():assert file_hash(Path(__file__).with_name(name))==value
    assert manifest['empty_text']==EMPTY
    for arm in ('hyper','static'):
        assert result[arm+'_sha256']==file_hash(args.training/f'{arm}.pt')
        groups=[g for g in result['groups'] if g['arm']==arm]
        assert sum(g['steps'] for g in groups)==400
        assert [g['target_order_sha256'] for g in groups]==[g['target_order_sha256'] for g in manifest['schedule'][1:]]
    assert sum(g['steps'] for g in result['groups'] if g['arm']=='bootstrap')==300
    previous=Path('data/annotations/alfworld_causal_prefix_v70_eval_20261010.json')
    parent=json.loads(previous.read_text())
    assert parent['model_config_sha256']==manifest['model_config_sha256']==file_hash(args.model/'config.json')
    for path,value in parent['model_weights'].items():assert file_hash(Path(path))==value
    targets=[]
    for t in parent['targets']:
        assert file_hash(args.data_root/t['game'])==t['game_sha256']
        c={k:v for k,v in t.items() if k not in ('reviewed_target','input_content_sha256')}
        c['training_manifest_sha256']=result['manifest_sha256']
        targets.append({**c,'input_content_sha256':digest(c),'reviewed_target':True})
    return {'targets':targets,'targets_sha256':digest(targets),'arms':list(ARMS),
        'training_manifest_sha256':result['manifest_sha256'],'training_sha256':file_hash(args.training/'training.json'),
        'checkpoints':{a:result[a+'_sha256'] for a in ('hyper','static')},
        'runner_sha256':file_hash(Path(__file__)),
        'rollout_sha256':file_hash(Path(__file__).with_name('evaluate_causal_prefix_v70.py')),
        'scope':'Frozen evaluation after chronological replay; H increment over one exact shared bootstrap vs continued A+B',
        'max_steps':30,'max_new_tokens':64,'history_turns':2,'loop_guard_max':2,'activation_step':3,
        'empty_control':'Exactly bootstrap LoRA when centered increment is zero',
        'limitation':'More H parameters and computation; same targets and optimizer steps, not equal FLOPs'}


def evaluate(args,review):
    if args.output.exists():raise FileExistsError(args.output)
    agent,tok=load_agent(args.model,args.training/'hyper.pt',args.device,.4)
    agent.requires_grad_(False);agent.eval()
    checkpoint=torch.load(args.training/'hyper.pt',map_location='cpu',weights_only=True)
    assert checkpoint['centered_experience'] and checkpoint['empty_text']==EMPTY
    hyper_a=[x.a.detach().clone() for x in agent.adapters]
    assert factors_hash(hyper_a)==factors_hash(checkpoint['shared_A'])
    shared=[x.to(args.device) for x in checkpoint['shared_B']]
    static=torch.load(args.training/'static.pt',map_location='cpu',weights_only=True)
    assert static['bootstrap_sha256']==checkpoint['bootstrap_sha256']
    empty=contextual_text_fields(agent,tok,EMPTY,args.device,2048,pooling='both')
    result={'complete':False,'rows':[],'failures':[],'review_sha256':file_hash(args.review)}
    for target in review['targets']:
        row={'index':target['index'],'game':target['game'],'input_content_sha256':target['input_content_sha256'],'arms':{}}
        try:
            for arm in ARMS:
                with torch.no_grad():
                    for layer,a in zip(agent.adapters,static['A'] if arm=='static' else hyper_a,strict=True):layer.a.copy_(a.to(layer.a))
                context=centered_reader(agent,empty,shared) if arm in ('real','empty','shifted') else nullcontext()
                with context:ep=episode(agent,tok,args.data_root/target['game'],arm,static,args.device)
                assert digest(ep['initial_observation'])==target['initial_observation_sha256']
                row['arms'][arm]=ep;save(args.output,{**result,'in_progress_row':row})
            result['rows'].append(row)
            result['summary']={'n':len(result['rows']),**{a:sum(r['arms'][a]['reward'] for r in result['rows']) for a in ARMS}}
            save(args.output,result);print(json.dumps(result['summary']),flush=True)
        except Exception as exc:
            result['failures'].append({'index':target['index'],'error':repr(exc),'partial_row':row});save(args.output,result);raise
    result['complete']=True;save(args.output,result)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('prepare','evaluate','audit'))
    p.add_argument('--training',type=Path,default=Path('results/trajectory_hyperlora/alfworld_centered_experience_v75b_20261010'))
    p.add_argument('--review',type=Path,default=Path('data/annotations/alfworld_centered_experience_v75b_eval_20261010.json'))
    p.add_argument('--output',type=Path,default=Path('results/trajectory_hyperlora/alfworld_centered_experience_v75b_eval_20261010.json'))
    p.add_argument('--model',type=Path,default=Path('current_work/delta-Mem/model/Qwen3-4B-Instruct-2507'))
    p.add_argument('--data-root',type=Path,default=Path('ttcl/data/alfworld_delta'))
    p.add_argument('--device',default='cuda:1')
    args=p.parse_args();review=bindings(args)
    if args.command=='prepare':
        if args.review.exists():raise FileExistsError(args.review)
        save(args.review,review)
    else:
        assert json.loads(args.review.read_text())==review
        (evaluate if args.command=='evaluate' else audit)(args,review)


if __name__=='__main__':main()
