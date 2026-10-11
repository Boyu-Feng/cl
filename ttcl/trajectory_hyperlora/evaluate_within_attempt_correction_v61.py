"""First-attempt prefix correction on the old 68/134 method's fixed TRAIN dev24."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ttcl.trajectory_hyperlora.evaluate_alfworld_failure_retry_v31b import checked_input
from ttcl.trajectory_hyperlora.alfworld_online_from_empty import digest, file_hash
from ttcl.trajectory_hyperlora.alfworld_zero_shot_probe import load_agent
from ttcl.trajectory_hyperlora.contextual_alf_source import contextual_source_fields
from ttcl.trajectory_hyperlora.within_attempt_correction_v61 import corrected_episode, trigger_reason
from ttcl.trajectory_hyperlora.alfworld_persistent_failure_delta_v36 import save
from ttcl.trajectory_hyperlora.audit_my_online_update_v28_reward import replay
from ttcl.trajectory_hyperlora.analyze_alfworld_online_shared_trajectory_v53 import paired


def parser():
    p=argparse.ArgumentParser()
    p.add_argument('command',choices=('prepare','evaluate','audit'))
    defaults={
        'reviewed_pairs':'data/annotations/alf_reward_router_train72_reviewed_20261007.json',
        'candidates':'results/trajectory_hyperlora/alf_sibling_train600_candidates_20261006.json',
        'source_review':'data/annotations/alf_sibling_train600_reviewed_20261006.json',
        'retry_candidates':'results/trajectory_hyperlora/alf_same_game_retry_candidates_20261006.json',
        'retry_review':'data/annotations/alf_same_game_retry_reviewed_20261006.json',
        'all_source_review':'data/annotations/alf_expert_first_attempts_reviewed_20261006.json',
        'first_report':'results/trajectory_hyperlora/alf_parent_loop2_dev24_20261007.json',
        'checkpoint':'results/trajectory_hyperlora/alf_sibling_simple40_onpolicy600_20261006.pt',
        'model':'current_work/delta-Mem/model/Qwen3-4B-Instruct-2507',
        'data_root':'ttcl/data/alfworld_delta',
        'review':'data/annotations/alfworld_within_attempt_correction_v61_dev24_20261010.json',
        'output':'results/trajectory_hyperlora/alfworld_within_attempt_correction_v61_dev24_20261010.json'}
    for name,path in defaults.items(): p.add_argument('--'+name.replace('_','-'),type=Path,default=Path(path))
    p.add_argument('--max-relative',type=float,default=.25)
    p.add_argument('--device',default='cuda:2')
    p.add_argument('--gpu-fraction',type=float,default=.65)
    return p


def bound(args):
    assert args.max_relative==.25
    pairs,first,parent,old=checked_input(args)
    fixed={'parent_inputs_sha256':digest(parent),'runner_sha256':file_hash(Path(__file__)),
        'correction_sha256':file_hash(Path(__file__).with_name('within_attempt_correction_v61.py')),
        'prefix_encoder_sha256':file_hash(Path(__file__).with_name('alfworld_online_own_trajectory_v52b.py')),
        'goal_parser_sha256':file_hash(Path(__file__).with_name('prepare_alfworld_off_goal_feedback_v57.py')),
        'seed':42,'max_steps':50,'max_new_tokens':64,'max_relative':.25,
        'trigger':'First public-goal wrong-object transformation, or after >=12 steps at least two duplicated observation/action pairs in last8',
        'correction':'One fixed away-from-prefix B residual capped at 25% of current source-adapter effective norm; no reset or extra environment actions',
        'arms':['none','away','random']}
    targets=[]
    for previous in old:
        content={'index':previous['index'],'game':previous['game'],
            'parent_input_content_sha256':previous['input_content_sha256'],'protocol_sha256':digest(fixed)}
        targets.append({**content,'input_content_sha256':digest(content),'reviewed_target':True})
    return pairs,first,fixed,targets


def checked(args):
    values=bound(args); review=json.loads(args.review.read_text())
    assert review['targets']==values[3] and review['targets_sha256']==digest(values[3])
    assert all(review[k]==v for k,v in values[2].items())
    return values


def main():
    args=parser().parse_args()
    if args.command=='prepare':
        if args.review.exists(): raise FileExistsError(args.review)
        _,_,fixed,targets=bound(args)
        save(args.review,{'protocol':'Exposed official TRAIN dev24; frozen old model and sources; all arms one 50-step attempt; corrected arms use only observed ongoing prefixes; no training',
            **fixed,'targets':targets,'targets_sha256':digest(targets)})
        print(json.dumps({'reviewed_targets':24}),flush=True); return
    pairs,first,fixed,targets=checked(args)
    if args.command=='evaluate':
        if args.output.exists(): raise FileExistsError(args.output)
        agent,tok=load_agent(args.model,args.checkpoint,args.device,args.gpu_fraction)
        result={'review_sha256':file_hash(args.review),**fixed,'rows':[],'failures':[]}
        for i,(pair,target) in enumerate(zip(pairs,targets,strict=True)):
            fields=contextual_source_fields(agent,tok,pair['source_records'],args.device,2048,pooling='both')
            row={'index':i,'game':pair['target_game'],'input_content_sha256':target['input_content_sha256']}
            for arm in fixed['arms']:
                ep=corrected_episode(agent,tok,args.data_root/pair['target_game'],fields,
                    device=args.device,mode=arm,seed=42,max_steps=50)
                row[arm]=ep
                if ep['status']!='complete':
                    result['failures'].append({'index':i,'arm':arm,'episode':ep})
                    result['rows'].append(row); save(args.output,result)
                    raise RuntimeError('Incomplete correction episode')
                if arm=='none':
                    baseline=first['games'][i]['new']
                    if any(ep[k]!=baseline[k] for k in ('trajectory','reward','steps','initial_observation','initial_commands_sha256','invalid_commands','termination')):
                        result['failures'].append({'index':i,'reason':'No-op wrapper changed frozen baseline'})
                        result['rows'].append(row); save(args.output,result)
                        raise RuntimeError('No-op actor equivalence failed')
            result['rows'].append(row)
            result['summary']={'n':len(result['rows']),**{arm:sum(r[arm]['reward'] for r in result['rows']) for arm in fixed['arms']}}
            save(args.output,result); print(json.dumps(result['summary']),flush=True)
        return
    result=json.loads(args.output.read_text())
    assert result['review_sha256']==file_hash(args.review) and not result['failures']
    assert len(result['rows'])==24
    for i,row in enumerate(result['rows']):
        assert row['index']==i and row['input_content_sha256']==targets[i]['input_content_sha256']
        assert row['game']==pairs[i]['target_game']
        for arm in fixed['arms']:
            ep=row[arm]; replay(args.data_root/row['game'],ep)
            event=ep['within_attempt_correction']
            if arm=='none':
                assert event is None and ep['trajectory']==first['games'][i]['new']['trajectory']
            records=[]; observation=ep['initial_observation']; first_trigger=None
            for turn,step in enumerate(ep['trajectory']):
                reason=trigger_reason(ep['initial_observation'],records)
                if first_trigger is None and reason is not None: first_trigger=(turn,reason,list(records))
                records.append({'observation':observation,'action':step['command'],'feedback':step['observation']})
                observation=step['observation']
            if arm!='none':
                if first_trigger is None: assert event is None
                else:
                    turn,reason,prefix=first_trigger
                    assert event['before_turn']==turn and event['reason']==reason and event['mode']==arm
                    assert event['prefix_records']==prefix and event['prefix_sha256']==digest(prefix)
                    assert event['applied_effective_norm']<=.25*event['metrics']['expert_effective_norm']+1e-6
        ae=row['away']['within_attempt_correction']; re=row['random']['within_attempt_correction']
        assert (ae is None)==(re is None)
        if ae:
            assert ae['prefix_sha256']==re['prefix_sha256'] and ae['before_turn']==re['before_turn']
            assert abs(ae['applied_effective_norm']-re['applied_effective_norm'])<max(1e-7,1e-5*ae['applied_effective_norm'])
        print(json.dumps({'replayed_tasks':i+1}),flush=True)
    comparisons=[paired(result['rows'],'away',arm) for arm in ('none','random')]
    running=0.
    for i,item in enumerate(sorted(comparisons,key=lambda x:x['mcnemar_exact_two_sided_p'])):
        running=max(running,min(1.,(2-i)*item['mcnemar_exact_two_sided_p'])); item['holm_adjusted_p']=running
    save(args.output.with_name(args.output.stem+'_audit.json'),{'passed':True,'report_sha256':file_hash(args.output),
        'replayed_episodes':72,'comparisons':comparisons,'limitation':'Exposed dev24, one random seed, additional prefix encoding compute; no independent generalization claim'})


if __name__=='__main__':
    main()
