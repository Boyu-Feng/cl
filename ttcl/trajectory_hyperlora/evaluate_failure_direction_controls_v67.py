"""Matched-norm controls for the old failure-retry direction on fixed dev24.

Directions use either the existing good B or a false failure tag on the good
source. Only their norm is matched using the real failed-trajectory residual.
This is an old-model, at-most-two-attempt diagnostic, not a cold-start result.
"""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from ttcl.trajectory_hyperlora.evaluate_within_attempt_correction_v61 import parser as parent_parser
from ttcl.trajectory_hyperlora.evaluate_alfworld_failure_retry_v31b import (
    checked_input, source_factors, retry,
)
from ttcl.trajectory_hyperlora.alfworld_success_gated_factors_v27 import episode_factor
from ttcl.trajectory_hyperlora.alfworld_failure_retry_residual_v31b import effective_norm, bounded_residual
from ttcl.trajectory_hyperlora.alfworld_online_accumulate_lora_v1 import factors_hash
from ttcl.trajectory_hyperlora.alfworld_online_from_empty import digest, file_hash
from ttcl.trajectory_hyperlora.alfworld_zero_shot_probe import load_agent
from ttcl.trajectory_hyperlora.contextual_alf_source import contextual_source_fields
from ttcl.trajectory_hyperlora.alfworld_persistent_failure_delta_v36 import save
from ttcl.trajectory_hyperlora.audit_my_online_update_v28_reward import replay
from ttcl.trajectory_hyperlora.analyze_alfworld_online_shared_trajectory_v53 import paired

ARMS=('parallel_good','false_failure_tag')


def false_tag(records):
    result=copy.deepcopy(records)
    old=result[-1]['feedback']
    if not old.startswith('task completed. '):raise ValueError('Source lacks reviewed success tag')
    result[-1]['feedback']='task not completed. '+old[len('task completed. '):]
    return result


def match_norm(agent,direction,target):
    magnitude=effective_norm(agent,direction)
    if magnitude<=1e-10 and target>0:raise ValueError('Degenerate control direction')
    scale=target/magnitude if magnitude else 0.
    result=[x.detach().cpu().float()*scale for x in direction]
    actual=effective_norm(agent,result)
    if abs(actual-target)>max(1e-7,1e-5*target):raise ValueError('Control norm mismatch')
    return result,{'unscaled_norm':magnitude,'target_norm':target,'scale':scale,'actual_norm':actual}


def bound(args):
    pairs,first,parent,old_targets=checked_input(args)
    old=json.loads(args.old_retry.read_text());audit=json.loads(args.old_audit.read_text())
    assert args.max_relative==.25
    assert len(old['games'])==24 and not old['failures'] and audit['audit']=='passed'
    assert audit['summary']==old['summary'] and audit['static_retries_identical_to_first']==13
    assert all(old[k]==v for k,v in parent.items())
    # Historical audit schemas use report_sha256 for this run.
    assert audit['report_sha256']==file_hash(args.old_retry)
    assert old['checkpoint_sha256']==parent['checkpoint_sha256']
    assert old['first_report_sha256']==parent['first_report_sha256']
    names=['evaluate_failure_direction_controls_v67.py','evaluate_within_attempt_correction_v61.py',
        'evaluate_alfworld_failure_retry_v31b.py','alfworld_failure_retry_residual_v31b.py',
        'alfworld_success_gated_factors_v27.py','contextual_alf_source.py','alfworld_zero_shot_probe.py']
    fixed={'method':'v67 real failure direction versus matched-norm source-only directions',
        'parent_inputs_sha256':digest(parent),'old_retry_sha256':file_hash(args.old_retry),
        'old_audit_sha256':file_hash(args.old_audit),
        'code_sha256':{name:file_hash(Path(__file__).with_name(name)) for name in names},
        'arms':list(ARMS),'max_steps_per_attempt':50,'max_new_tokens':64,'history_turns':2,
        'loop_guard_max':2,'max_relative':.25,
        'norm_source':'real complete failed trajectory, same bounded residual norm as v31b',
        'parallel_direction':'positive initial good-source B',
        'tag_direction':'initial good-source B minus same source with only final status tag changed',
        'primary_comparisons':'each control versus historical away_from_failure; two exact paired tests, Holm'}
    targets=[]
    for i,(pair,binding,row) in enumerate(zip(pairs,old_targets,old['games'],strict=True)):
        assert row['game']==pair['target_game'] and row['first_episode_sha256']==binding['first_episode_sha256']
        content={'index':i,'game':pair['target_game'],'parent_input_content_sha256':binding['input_content_sha256'],
            'synthetic_tag_source_sha256':digest(false_tag(pair['source_records'])),
            'protocol_sha256':digest(fixed)}
        targets.append({**content,'input_content_sha256':digest(content),'reviewed_target':True,
            'synthetic_source_is_not_training_label':True})
    return pairs,first,old,fixed,targets


def checked(args):
    values=bound(args)
    assert json.loads(args.review.read_text())=={**values[3],'targets':values[4],'targets_sha256':digest(values[4])}
    return values


def evaluate(args):
    if args.output.exists() or args.factors_dir.exists():raise FileExistsError('Use fresh v67 outputs')
    pairs,first,old,fixed,targets=checked(args)
    args.factors_dir.mkdir(parents=True)
    agent,tok=load_agent(args.model,args.checkpoint,args.device,args.gpu_fraction)
    result={**fixed,'review_sha256':file_hash(args.review),'rows':[],'failures':[]}
    for i,(pair,baseline,binding) in enumerate(zip(pairs,first['games'],targets,strict=True)):
        episode=baseline['new']
        row={'index':i,'game':pair['target_game'],'input_content_sha256':binding['input_content_sha256'],
            'first_reward':episode['reward'],'retries':{}}
        try:
            if not episode['reward']:
                fields=contextual_source_fields(agent,tok,pair['source_records'],args.device,2048,pooling='both')
                good=source_factors(agent,tok,fields,episode['initial_observation'],args.device)
                bad,_,_,_=episode_factor(agent,tok,episode,SimpleNamespace(device=args.device,context_tokens=2048))
                toward,info=bounded_residual(agent,good,bad,max_relative=.25)
                prior=old['games'][i]
                assert factors_hash(good)==prior['expert_factor_sha256']
                assert factors_hash(bad)==prior['failed_factor_sha256']
                assert factors_hash(toward)==prior['residual_factor_sha256']
                synthetic=contextual_source_fields(agent,tok,false_tag(pair['source_records']),args.device,2048,pooling='both')
                tagged=source_factors(agent,tok,synthetic,episode['initial_observation'],args.device)
                directions={'parallel_good':good,'false_failure_tag':[g-t for g,t in zip(good,tagged,strict=True)]}
                controls={};metadata={}
                for name in ARMS:
                    controls[name],metadata[name]=match_norm(agent,directions[name],info['residual_effective_norm'])
                    metadata[name]['factor_sha256']=factors_hash(controls[name])
                path=args.factors_dir/f'{i:02d}.pt'
                torch.save({'A':[x.a.detach().cpu().clone() for x in agent.adapters],
                    'good':good,'tagged':tagged,'toward_real_failure':toward,'controls':controls},path)
                row.update(factor_file=str(path),factor_file_sha256=file_hash(path),controls=metadata,
                    real_residual_info=info,original_real_residual_sha256=factors_hash(toward))
                for name in ARMS:
                    trial=retry(agent,tok,args.data_root/pair['target_game'],fields,args.device,controls[name])
                    row['retries'][name]=trial
                    if trial['status']!='complete' or trial['initial_observation']!=episode['initial_observation']:
                        raise ValueError('Incomplete or changed control retry')
            result['rows'].append(row)
            result['summary']={'n':len(result['rows']),
                'first_wins':sum(x['first_reward'] for x in result['rows']),
                'two_attempt_wins':{name:sum(x['first_reward'] or x['retries'][name]['reward'] for x in result['rows']) for name in ARMS}}
            save(args.output,result);print(json.dumps(result['summary']),flush=True)
        except Exception as exc:
            result['failures'].append({'index':i,'error':repr(exc),'partial_row':row})
            save(args.output,result);raise
    result['complete']=True;save(args.output,result)


def audit(args):
    pairs,first,old,fixed,targets=checked(args)
    report=json.loads(args.output.read_text())
    assert report.get('complete') and not report['failures'] and len(report['rows'])==24
    assert report['review_sha256']==file_hash(args.review)
    assert all(report[k]==v for k,v in fixed.items())
    comparison=[];count=0;basis_hash=None
    for i,(row,target) in enumerate(zip(report['rows'],targets,strict=True)):
        baseline=first['games'][i]['new']
        assert row['index']==i and row['game']==target['game']
        assert row['input_content_sha256']==target['input_content_sha256'] and row['first_reward']==baseline['reward']
        item={'away_from_failure':baseline if baseline['reward'] else old['games'][i]['retries']['away_from_failure']}
        if baseline['reward']:
            assert row['retries']=={}
            item.update({name:baseline for name in ARMS})
        else:
            assert file_hash(Path(row['factor_file']))==row['factor_file_sha256']
            state=torch.load(row['factor_file'],map_location='cpu',weights_only=True)
            ah=factors_hash(state['A'])
            if basis_hash is None:basis_hash=ah
            assert ah==basis_hash
            norm_agent=SimpleNamespace(adapters=[SimpleNamespace(a=a,rank=4,base=SimpleNamespace(out_features=b.shape[1]))
                for a,b in zip(state['A'],state['good'],strict=True)])
            assert factors_hash(state['good'])==old['games'][i]['expert_factor_sha256']
            assert factors_hash(state['toward_real_failure'])==old['games'][i]['residual_factor_sha256']
            for name in ARMS:
                delta=state['controls'][name];meta=row['controls'][name]
                direction=state['good'] if name=='parallel_good' else [g-t for g,t in zip(state['good'],state['tagged'],strict=True)]
                assert factors_hash([x*meta['scale'] for x in direction])==factors_hash(delta)==meta['factor_sha256']
                magnitude=effective_norm(norm_agent,delta)
                target_norm=old['games'][i]['residual_info']['residual_effective_norm']
                assert abs(magnitude-target_norm)<=max(1e-7,1e-5*target_norm)
                assert abs(meta['scale']*meta['unscaled_norm']-target_norm)<=max(1e-7,1e-5*target_norm)
                ep=row['retries'][name]
                assert ep['status']=='complete' and ep['steps']<=50
                replay(args.data_root/row['game'],ep);count+=1;item[name]=ep
        comparison.append(item)
        print(json.dumps({'audited_tasks':i+1,'replayed_new_episodes':count}),flush=True)
    comparisons=[paired(comparison,'away_from_failure',name) for name in ARMS]
    running=0.
    for rank,c in enumerate(sorted(comparisons,key=lambda x:x['mcnemar_exact_two_sided_p'])):
        running=max(running,min(1.,(2-rank)*c['mcnemar_exact_two_sided_p']));c['holm_adjusted_p']=running
    assert count==26
    save(args.output.with_suffix('.audit.json'),{'passed':True,'report_sha256':file_hash(args.output),
        'new_episodes_replayed':count,'comparisons':comparisons,
        'limitation':'exposed old-model dev24, at most two attempts; control magnitude still uses real failure; small noninferiority power'})


def parser():
    p=parent_parser()
    p.set_defaults(review=Path('data/annotations/alfworld_failure_direction_controls_v67_dev24_20261010.json'),
        output=Path('results/trajectory_hyperlora/alfworld_failure_direction_controls_v67_dev24_20261010.json'),
        device='cuda:0',gpu_fraction=.35)
    p.add_argument('--old-retry',type=Path,default=Path('results/trajectory_hyperlora/alfworld_failure_retry_v31b_dev24_20261009.json'))
    p.add_argument('--old-audit',type=Path,default=Path('results/trajectory_hyperlora/alfworld_failure_retry_v31b_dev24_audit_20261009.json'))
    p.add_argument('--factors-dir',type=Path,default=Path('results/trajectory_hyperlora/alfworld_failure_direction_controls_v67_factors_20261010'))
    return p


if __name__=='__main__':
    args=parser().parse_args()
    if args.command=='prepare':
        if args.review.exists():raise FileExistsError(args.review)
        _,_,_,fixed,targets=bound(args)
        save(args.review,{**fixed,'targets':targets,'targets_sha256':digest(targets)})
        print(json.dumps({'reviewed_targets':len(targets)}),flush=True)
    elif args.command=='evaluate':evaluate(args)
    else:audit(args)
