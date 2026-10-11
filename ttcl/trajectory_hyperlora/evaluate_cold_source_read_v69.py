"""Fixed pre-task H/A/R: original, recent, and same-room own source reads.

All eligible tasks are selected by availability of past own successful sources,
never by the target outcome. Counterfactual wins do not enter the original pool.
"""
import argparse
import json
from pathlib import Path

import torch

from ttcl.trajectory_hyperlora.alfworld_cold_joint_memory_v63 import (
    new_agent, read_memory, bounded_source_fields, trainable_digest, records_from_episode,
    file_hash, digest, save, run_episode, factors_hash,
)
from ttcl.trajectory_hyperlora.audit_my_online_update_v28_reward import replay

ARMS = ('original_source', 'recent_same_family', 'recent_same_room_family')


def family(row):
    return Path(row['game']).parts[2].split('-')[0]


def room(row):
    text = row['episode']['initial_observation']
    if 'Your task is to:' not in text:
        raise ValueError('Missing task boundary in visible initial observation')
    return text.split('Your task is to:', 1)[0].strip()


def bindings(args):
    report = json.loads(args.parent.read_text())
    audit = json.loads(args.parent.with_suffix('.audit.json').read_text())
    assert report['complete'] and not report['failures'] and report['mode']=='hybrid'
    assert audit['passed'] and audit['episodes_replayed']==240 and audit['report_sha256']==file_hash(args.parent)
    parent_review = json.loads(args.parent_review.read_text())
    assert report['review_sha256']==file_hash(args.parent_review)
    for name, expected in report['code_sha256'].items():
        assert file_hash(Path(__file__).with_name(name)) == expected
    assert report['model_config_sha256']==file_hash(args.model/'config.json')
    targets = []
    past, snapshot = [], None
    for row, original_binding in zip(report['rows'], parent_review['targets'], strict=True):
        same = [r for r in past if family(r)==family(row)]
        matching = [r for r in same if room(r)==room(row)]
        if row['adapter_active'] and matching:
            assert snapshot and file_hash(Path(snapshot['path']))==snapshot['sha256']
            selected = {'original_source':row['source_stream_index'],
                        'recent_same_family':same[-1]['index'],
                        'recent_same_room_family':matching[-1]['index']}
            assert all(j<row['index'] and report['rows'][j]['episode']['reward']==1 for j in selected.values())
            assert original_binding['input_content_sha256']==row['target_input_content_sha256']
            assert original_binding['game_sha256']==file_hash(args.data_root/row['game'])
            content = {'index':row['index'], 'game':row['game'], 'family':family(row),
                       'game_sha256':original_binding['game_sha256'],
                       'parent_target_binding':row['target_input_content_sha256'],
                       'initial_observation_sha256':digest(row['episode']['initial_observation']),
                       'room_sha256':digest(room(row)), 'pre_task_snapshot':snapshot,
                       'H_before_sha256':row['H_before_sha256'], 'R_before_sha256':row['R_before_sha256'],
                       'source_indices':selected,
                       'source_episode_sha256':{arm:digest(report['rows'][j]['episode']) for arm,j in selected.items()}}
            targets.append({**content, 'input_content_sha256':digest(content), 'reviewed_target':True})
        if row['episode']['reward']==1:
            past.append(row)
        if 'snapshot' in row:
            snapshot = row['snapshot']
    assert len(targets)==23, 'Availability-defined set changed'
    fixed = {'method':'v69 three source choices with exact same saved pre-task H/A/R',
             'parent_sha256':file_hash(args.parent), 'parent_audit_sha256':file_hash(args.parent.with_suffix('.audit.json')),
             'parent_review_sha256':file_hash(args.parent_review),
             'runner_sha256':file_hash(Path(__file__)), 'dependencies':report['code_sha256'],
             'model_config_sha256':report['model_config_sha256'], 'seed':report['seed'],
             'arms':list(ARMS), 'max_steps':30, 'max_new_tokens':64,
             'history_turns':2, 'loop_guard_max':2, 'constrain_actions':True,
             'target_selection':'All active tasks having an earlier own success with same family and visible initial room description; no target outcome selection',
             'training_updates':0, 'memory_writes':0,
             'scope':'Exposed development counterfactual reads, not an independent online stream',
             'limitation':'Same visible room description does not guarantee same object placement; room selection also changes source age/content'}
    return report, fixed, targets


def evaluate(args, report, review):
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.cuda.set_per_process_memory_fraction(.45, device=args.device)
    agent, tok = new_agent(args)
    agent.eval()
    params = dict(agent.named_parameters())
    result = {'review_sha256':file_hash(args.review), 'rows':[], 'failures':[], 'complete':False}
    for target in review['targets']:
        row = {'index':target['index'], 'game':target['game'],
               'input_content_sha256':target['input_content_sha256'], 'arms':{}}
        try:
            state = torch.load(target['pre_task_snapshot']['path'], map_location='cpu', weights_only=True)
            with torch.no_grad():
                for name,value in state['trainable_state'].items():
                    params[name].copy_(value.to(params[name]))
                for layer,a in zip(agent.adapters,state['A'],strict=True):
                    layer.a.copy_(a.to(layer.a)); layer.a.requires_grad_(False)
            assert trainable_digest(agent)==target['H_before_sha256']
            assert factors_hash(state['R'])==target['R_before_sha256']
            a_hash = factors_hash([x.a for x in agent.adapters])
            for arm in ARMS:
                source = report['rows'][target['source_indices'][arm]]
                fields = bounded_source_fields(agent,tok,records_from_episode(source['episode']),args.device)
                with read_memory(agent, state['R']):
                    ep = run_episode(agent,tok,args.data_root/target['game'],fields,adapter=True,
                        device=args.device,max_steps=30,max_new_tokens=64,constrain_actions=True,
                        actor_history_turns=2,loop_guard_max=2)
                assert ep['status']=='complete' and ep['steps']<=30
                assert digest(ep['initial_observation'])==target['initial_observation_sha256']
                row['arms'][arm] = ep
                save(args.output,{**result,'in_progress_row':row})
            assert trainable_digest(agent)==target['H_before_sha256']
            assert factors_hash([x.a for x in agent.adapters])==a_hash
            assert factors_hash(state['R'])==target['R_before_sha256']
            row['original_episode_reproduced'] = row['arms']['original_source']==report['rows'][target['index']]['episode']
            result['rows'].append(row)
            result['summary'] = {'n':len(result['rows']), **{arm:sum(r['arms'][arm]['reward'] for r in result['rows']) for arm in ARMS},
                                 'original_episodes_reproduced':sum(r['original_episode_reproduced'] for r in result['rows'])}
            save(args.output,result); print(json.dumps(result['summary']),flush=True)
        except Exception as exc:
            result['failures'].append({'index':target['index'],'error':repr(exc),'partial_row':row})
            save(args.output,result)
            raise
    result['complete']=True
    save(args.output,result)


def audit(args, report, review):
    result = json.loads(args.output.read_text())
    assert result['complete'] and not result['failures'] and result['review_sha256']==file_hash(args.review)
    assert len(result['rows'])==len(review['targets'])==23
    for row,target in zip(result['rows'],review['targets'],strict=True):
        assert row['input_content_sha256']==target['input_content_sha256'] and row['index']==target['index']
        for arm in ARMS:
            ep = row['arms'][arm]
            assert ep['status']=='complete' and ep['steps']<=30
            assert digest(ep['initial_observation'])==target['initial_observation_sha256']
            replay(args.data_root/target['game'],ep)
        assert row['original_episode_reproduced']==(row['arms']['original_source']==report['rows'][row['index']]['episode'])
    summary = {'n':23, **{arm:sum(r['arms'][arm]['reward'] for r in result['rows']) for arm in ARMS},
               'original_episodes_reproduced':sum(r['original_episode_reproduced'] for r in result['rows'])}
    assert summary==result['summary']
    contrasts = {arm:{'matched_only':sum(r['arms'][ARMS[2]]['reward']>r['arms'][arm]['reward'] for r in result['rows']),
                     'control_only':sum(r['arms'][arm]['reward']>r['arms'][ARMS[2]]['reward'] for r in result['rows'])}
                 for arm in ARMS[:2]}
    outcome = {'passed':True,'episodes_replayed':69,'summary':summary,'contrasts':contrasts,
               'report_sha256':file_hash(args.output),'review_sha256':file_hash(args.review),
               'audit_sha256':file_hash(Path(__file__)),
               'scope':'All environment replays, causal source selection and fixed snapshot bindings; not independent neural regeneration'}
    save(args.output.with_suffix('.audit.json'),outcome);print(json.dumps(outcome),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('prepare','evaluate','audit'))
    p.add_argument('--parent',type=Path,default=Path('results/trajectory_hyperlora/alfworld_cold_joint_memory_v63_hybrid_20261010.json'))
    p.add_argument('--parent-review',type=Path,default=Path('data/annotations/alfworld_cold_joint_memory_v63_hybrid_20261010_review.json'))
    p.add_argument('--review',type=Path,default=Path('data/annotations/alfworld_cold_source_read_v69_20261010.json'))
    p.add_argument('--output',type=Path,default=Path('results/trajectory_hyperlora/alfworld_cold_source_read_v69_20261010.json'))
    p.add_argument('--model',type=Path,default=Path('current_work/delta-Mem/model/Qwen3-4B-Instruct-2507'))
    p.add_argument('--data-root',type=Path,default=Path('ttcl/data/alfworld_delta'))
    p.add_argument('--device',default='cuda:1')
    args=p.parse_args()
    report,fixed,targets=bindings(args); args.seed=fixed['seed']
    review={**fixed,'targets':targets,'targets_sha256':digest(targets)}
    if args.command=='prepare':
        if args.review.exists():raise FileExistsError(args.review)
        save(args.review,review);print(json.dumps({'reviewed_targets':len(targets),'new_episodes':len(targets)*len(ARMS)}))
    else:
        assert json.loads(args.review.read_text())==review
        (evaluate if args.command=='evaluate' else audit)(args,report,review)


if __name__=='__main__':main()
