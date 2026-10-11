"""Does retaining full within-task history help the same next24 base tasks?

Diagnostic only; extra actor context/token cost is not budget matched to v70.
"""
import argparse
import json
from pathlib import Path

from ttcl.trajectory_hyperlora.alfworld_zero_shot_probe import load_agent,run_episode
from ttcl.trajectory_hyperlora.alfworld_online_from_empty import digest,file_hash
from ttcl.trajectory_hyperlora.alfworld_persistent_failure_delta_v36 import save
from ttcl.trajectory_hyperlora.audit_my_online_update_v28_reward import replay


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent',type=Path,default=Path('results/trajectory_hyperlora/alfworld_causal_prefix_v70_eval_20261010.json'))
    p.add_argument('--parent-review',type=Path,default=Path('data/annotations/alfworld_causal_prefix_v70_eval_20261010.json'))
    p.add_argument('--output',type=Path,default=Path('results/trajectory_hyperlora/alfworld_full_history_v74_20261010.json'))
    p.add_argument('--review',type=Path,default=Path('data/annotations/alfworld_full_history_v74_20261010.json'))
    p.add_argument('--checkpoint',type=Path,default=Path('results/trajectory_hyperlora/alfworld_causal_prefix_v70_20261010/hyper.pt'))
    p.add_argument('--model',type=Path,default=Path('current_work/delta-Mem/model/Qwen3-4B-Instruct-2507'))
    p.add_argument('--data-root',type=Path,default=Path('ttcl/data/alfworld_delta'))
    p.add_argument('--device',default='cuda:1')
    args=p.parse_args()
    if args.output.exists() or args.review.exists():raise FileExistsError('Use new output/review paths')
    parent=json.loads(args.parent.read_text());audit=json.loads(args.parent.with_suffix('.audit.json').read_text())
    prev=json.loads(args.parent_review.read_text())
    assert parent['complete'] and not parent['failures'] and audit['passed'] and audit['episodes_replayed']==120
    assert audit['report_sha256']==file_hash(args.parent) and parent['review_sha256']==file_hash(args.parent_review)
    assert file_hash(args.checkpoint)==prev['checkpoints']['hyper']
    for name,value in prev['dependencies'].items():assert file_hash(Path(__file__).with_name(name))==value
    for path,value in prev['model_weights'].items():assert file_hash(Path(path))==value
    for target in prev['targets']:assert file_hash(args.data_root/target['game'])==target['game_sha256']
    review={'parent_sha256':file_hash(args.parent),'parent_audit_sha256':file_hash(args.parent.with_suffix('.audit.json')),
            'runner_sha256':file_hash(Path(__file__)),'parent_review_sha256':file_hash(args.parent_review),
            'targets':prev['targets'],'targets_sha256':prev['targets_sha256'],
            'adapter':False,'training_updates':0,'max_steps':30,'max_new_tokens':64,'loop_guard_max':2,
            'actor_history_turns':None,'scope':'Exposed next24 full-history diagnostic; more context and token cost, not a parameter-learning method'}
    save(args.review,review)
    agent,tok=load_agent(args.model,args.checkpoint,args.device,.40);agent.requires_grad_(False);agent.eval()
    result={'complete':False,'rows':[],'failures':[],'review_sha256':file_hash(args.review)}
    for target,prior in zip(review['targets'],parent['rows'],strict=True):
        try:
            ep=run_episode(agent,tok,args.data_root/target['game'],{},adapter=False,device=args.device,
                max_steps=30,max_new_tokens=64,constrain_actions=True,actor_history_turns=None,loop_guard_max=2)
            assert ep['status']=='complete' and digest(ep['initial_observation'])==target['initial_observation_sha256']
            result['rows'].append({'index':target['index'],'game':target['game'],'input_content_sha256':target['input_content_sha256'],
                'episode':ep,'short_history_reward':prior['arms']['base']['reward']})
            result['summary']={'n':len(result['rows']),'full_history':sum(r['episode']['reward'] for r in result['rows']),
                               'short_history':sum(r['short_history_reward'] for r in result['rows'])}
            save(args.output,result);print(json.dumps(result['summary']),flush=True)
        except Exception as exc:
            result['failures'].append({'index':target['index'],'error':repr(exc)});save(args.output,result);raise
    result['complete']=True;save(args.output,result)
    for row in result['rows']:replay(args.data_root/row['game'],row['episode'])
    save(args.output.with_suffix('.audit.json'),{'passed':True,'episodes_replayed':24,'report_sha256':file_hash(args.output),
        'summary':result['summary'],'scope':'Bound parent comparison and environment replay; not equal token budgets'})


if __name__=='__main__':main()
