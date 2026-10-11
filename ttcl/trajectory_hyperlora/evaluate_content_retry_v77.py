"""Second-attempt retries using components orthogonal to a tag-only direction.

Old audited dev24; 13 first failures, two alternative single-retry arms.
Orthogonality is in effective BA/r space. It is not semantic disentanglement.
"""
import copy
import json
from pathlib import Path

import torch

from ttcl.trajectory_hyperlora import evaluate_failure_direction_controls_v67 as parent
from ttcl.trajectory_hyperlora.alfworld_online_from_empty import digest, file_hash
from ttcl.trajectory_hyperlora.alfworld_online_accumulate_lora_v1 import factors_hash
from ttcl.trajectory_hyperlora.alfworld_persistent_failure_delta_v36 import save
from ttcl.trajectory_hyperlora.alfworld_zero_shot_probe import load_agent
from ttcl.trajectory_hyperlora.contextual_alf_source import contextual_source_fields
from ttcl.trajectory_hyperlora.evaluate_alfworld_failure_retry_v31b import source_factors, retry
from ttcl.trajectory_hyperlora.audit_my_online_update_v28_reward import replay

ARMS = ('orthogonal_component', 'mixed_component')


def inner(xs, ys, basis):
    total = 0.
    for x, y, a in zip(xs, ys, basis, strict=True):
        x, y, a = x.squeeze(0).double().cpu(), y.squeeze(0).double().cpu(), a.double().cpu()
        total += float(((x @ (a @ a.T))*y).sum())/(a.shape[0]**2)
    return total


def directions(state):
    basis = state['A']
    away = [-x.float() for x in state['toward_real_failure']]
    tag = state['controls']['false_failure_tag']
    norm = inner(away, away, basis)**.5
    tag_squared = inner(tag, tag, basis)
    assert norm > 0 and tag_squared > 0
    coefficient = inner(away, tag, basis)/tag_squared
    orth = [a-coefficient*t for a,t in zip(away,tag,strict=True)]
    orth_norm = max(0.,inner(orth,orth,basis))**.5
    if orth_norm <= 1e-6*norm:
        raise ValueError('Orthogonal component is too small to amplify reliably')
    orth = [x*(norm/orth_norm) for x in orth]
    mixture = [a+o for a,o in zip(away,orth,strict=True)]
    mixture_norm = inner(mixture,mixture,basis)**.5
    mixture = [x*(norm/mixture_norm) for x in mixture]
    controls = dict(zip(ARMS,(orth,mixture),strict=True))
    for values in controls.values():
        assert abs(inner(values,values,basis)**.5-norm) < 1e-5*norm
    assert abs(inner(orth,tag,basis)) < 1e-5*norm*(tag_squared**.5)
    return controls, {'target_effective_norm':norm,'projection_coefficient':coefficient,
                      'orthogonal_fraction':orth_norm/norm,
                      'original_tag_cosine':inner(away,tag,basis)/(norm*tag_squared**.5)}


def bindings(args):
    previous = copy.copy(args)
    previous.review = args.v67_review
    previous.output = args.v67_report
    pairs, first, old, fixed, targets = parent.checked(previous)
    report = json.loads(args.v67_report.read_text())
    audit_path = args.v67_report.with_suffix('.audit.json')
    audit = json.loads(audit_path.read_text())
    assert report['complete'] and not report['failures'] and len(report['rows']) == 24
    assert audit['passed'] and audit['new_episodes_replayed'] == 26
    assert audit['report_sha256'] == file_hash(args.v67_report)
    assert report['review_sha256'] == file_hash(args.v67_review)
    new_targets = []
    for row, target in zip(report['rows'],targets,strict=True):
        assert row['input_content_sha256'] == target['input_content_sha256']
        factor_hash = None
        if not row['first_reward']:
            factor_hash = file_hash(Path(row['factor_file']))
            assert factor_hash == row['factor_file_sha256']
        content = {'index':row['index'],'game':row['game'],
                   'parent_target_binding':target['input_content_sha256'],
                   'first_reward':row['first_reward'],'factor_file_sha256':factor_hash}
        new_targets.append({**content,'input_content_sha256':digest(content),'reviewed_target':True})
    review = {'v67_report_sha256':file_hash(args.v67_report),'v67_audit_sha256':file_hash(audit_path),
              'v67_review_sha256':file_hash(args.v67_review),'checkpoint_sha256':file_hash(args.checkpoint),
              'runner_sha256':file_hash(Path(__file__)), 'parent_protocol_sha256':digest(fixed),
              'targets':new_targets,'targets_sha256':digest(new_targets),'arms':list(ARMS),
              'max_steps':50,'max_new_tokens':64,'history_turns':2,'loop_guard_max':2,
              'scope':'Old trained H and expert success sources, exposed train dev24; fixed first attempts, one additional attempt per alternative arm. No training, no test-outcome direction selection.',
              'limitation':'Geometric tag-direction removal does not isolate semantic failure content. Normalized orthogonal components may amplify noise.'}
    return review,pairs,first,old,report


def evaluate(args,review,pairs,first,old,previous):
    if args.output.exists() or args.factors_dir.exists(): raise FileExistsError('Use new outputs')
    args.factors_dir.mkdir(parents=True)
    agent,tok = load_agent(args.model,args.checkpoint,args.device,args.gpu_fraction)
    agent.requires_grad_(False);agent.eval()
    result = {'complete':False,'review_sha256':file_hash(args.review),'rows':[],'failures':[]}
    for pair, base, prior, target in zip(pairs,first['games'],previous['rows'],review['targets'],strict=True):
        row = {k:target[k] for k in ('index','game','input_content_sha256','first_reward')}
        row['retries'] = {}
        try:
            if not row['first_reward']:
                state = torch.load(prior['factor_file'],map_location='cpu',weights_only=True)
                assert factors_hash([x.a for x in agent.adapters]) == factors_hash(state['A'])
                controls, metadata = directions(state)
                fields = contextual_source_fields(agent,tok,pair['source_records'],args.device,2048,pooling='both')
                good = source_factors(agent,tok,fields,base['new']['initial_observation'],args.device)
                assert factors_hash(good) == factors_hash(state['good'])
                path = args.factors_dir/f"{row['index']:02d}.pt"
                torch.save({'controls':controls,'metadata':metadata,'parent_factor_sha256':prior['factor_file_sha256']},path)
                row.update(factor_file=str(path),factor_file_sha256=file_hash(path),metadata=metadata)
                for arm in ARMS:
                    ep = retry(agent,tok,args.data_root/row['game'],fields,args.device,controls[arm])
                    assert ep['status'] == 'complete' and ep['initial_observation'] == base['new']['initial_observation']
                    row['retries'][arm] = ep
                    save(args.output,{**result,'in_progress_row':row})
            result['rows'].append(row)
            result['summary'] = {'n':len(result['rows']),'first_wins':sum(r['first_reward'] for r in result['rows']),
                'rescued':{a:sum(r['retries'][a]['reward'] for r in result['rows'] if not r['first_reward']) for a in ARMS}}
            save(args.output,result);print(json.dumps(result['summary']),flush=True)
        except Exception as exc:
            result['failures'].append({'index':row['index'],'error':repr(exc),'partial_row':row})
            save(args.output,result);raise
    result['complete'] = True;save(args.output,result)


def audit(args,review,pairs,first,old,previous):
    result = json.loads(args.output.read_text())
    assert result['complete'] and not result['failures'] and len(result['rows']) == 24
    assert result['review_sha256'] == file_hash(args.review)
    count = 0
    for row,prior,target in zip(result['rows'],previous['rows'],review['targets'],strict=True):
        assert all(row[k] == target[k] for k in ('index','game','first_reward','input_content_sha256'))
        if row['first_reward']:
            assert not row['retries'];continue
        state = torch.load(prior['factor_file'],map_location='cpu',weights_only=True)
        controls,metadata = directions(state)
        path = Path(row['factor_file']);assert file_hash(path) == row['factor_file_sha256']
        stored = torch.load(path,map_location='cpu',weights_only=True)
        assert stored['parent_factor_sha256'] == prior['factor_file_sha256']
        assert stored['metadata'] == row['metadata'] == metadata
        for arm in ARMS:
            assert factors_hash(controls[arm]) == factors_hash(stored['controls'][arm])
            ep = row['retries'][arm];assert ep['steps'] <= 50
            replay(args.data_root/row['game'],ep);count += 1
    assert count == 26
    rescues = {a:sum(r['retries'][a]['reward'] for r in result['rows'] if not r['first_reward']) for a in ARMS}
    assert result['summary'] == {'n':24,'first_wins':11.,'rescued':rescues}
    save(args.output.with_suffix('.audit.json'),{'passed':True,'report_sha256':file_hash(args.output),
         'new_episodes_replayed':count,'first_failures':13,'rescued':rescues,
         'historical_rescues':{'unchanged':0,'real_failure':2,'false_failure_tag':3},
         'scope':'Input bindings, saved-factor reconstruction/norm/orthogonality checks and environment replay; no independent neural regeneration. Exposed development tasks.'})


def main():
    p = parent.parser()
    p.set_defaults(review=Path('data/annotations/alfworld_content_retry_v77_dev24_20261010.json'),
                   output=Path('results/trajectory_hyperlora/alfworld_content_retry_v77_dev24_20261010.json'),
                   factors_dir=Path('results/trajectory_hyperlora/alfworld_content_retry_v77_factors_20261010'),
                   device='cuda:1',gpu_fraction=.6)
    p.add_argument('--v67-review',type=Path,default=Path('data/annotations/alfworld_failure_direction_controls_v67_dev24_20261010.json'))
    p.add_argument('--v67-report',type=Path,default=Path('results/trajectory_hyperlora/alfworld_failure_direction_controls_v67_dev24_20261010.json'))
    args = p.parse_args(); values = bindings(args)
    if args.command == 'prepare':
        if args.review.exists(): raise FileExistsError(args.review)
        save(args.review,values[0])
    else:
        assert json.loads(args.review.read_text()) == values[0]
        (evaluate if args.command == 'evaluate' else audit)(args,*values)


if __name__ == '__main__': main()
