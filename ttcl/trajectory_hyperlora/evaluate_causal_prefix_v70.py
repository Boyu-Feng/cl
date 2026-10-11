"""Frozen next-task causal-prefix reading diagnostic, five matched arms."""
import argparse
import json
from pathlib import Path

import torch

from ttcl.experience_evolution.environment import ACTOR_SYSTEM, clean_command, make_env
from ttcl.trajectory_hyperlora.alfworld_zero_shot_probe import load_agent, generate, guarded_commands
from ttcl.trajectory_hyperlora.prepare_causal_prefix_v70 import prefix_records
from ttcl.trajectory_hyperlora.train_causal_prefix_v70 import source_variants
from ttcl.trajectory_hyperlora.train_alf_next_task import compact_messages
from ttcl.trajectory_hyperlora.contextual_alf_source import contextual_text_fields, task_context_text
from ttcl.trajectory_hyperlora.alfworld_online_from_empty import digest, file_hash
from ttcl.trajectory_hyperlora.alfworld_persistent_failure_delta_v36 import save
from ttcl.trajectory_hyperlora.alfworld_online_accumulate_lora_v1 import factors_hash
from ttcl.trajectory_hyperlora.audit_my_online_update_v28_reward import replay

ARMS = ('real', 'empty', 'shifted', 'static', 'base')
EMPTY = 'Earlier observed steps of the current task:\nNo earlier steps supplied.'


def source_read(tok, initial, trajectory, arm):
    cutoff = len(trajectory)-2
    assert cutoff > 0
    records = prefix_records(initial, trajectory[:cutoff])
    texts, indices, counts = source_variants(tok, records)
    text = EMPTY if arm == 'empty' else texts[arm]
    return text, {'source_end_exclusive':cutoff, 'retained_source_indices':indices,
                  'records_sha256':digest(records), 'text_sha256':digest(text),
                  'variant_token_counts':counts}


def episode(agent, tok, game, arm, static, device):
    env = make_env(game)
    trajectory, reads, seen = [], [], {}
    try:
        state = env.reset(); initial = str(state['feedback'])
        initial_commands = list(state['admissible_commands'])
        messages = [{'role':'system','content':ACTOR_SYSTEM}]
        for turn in range(30):
            agent.set_source(None)
            feedback = str(state['feedback'])
            if turn >= 3:
                if arm in ('real','empty','shifted'):
                    text, binding = source_read(tok, initial, trajectory, arm)
                    with torch.no_grad():
                        fields = contextual_text_fields(agent,tok,text,device,2048,pooling='both')
                        query = contextual_text_fields(agent,tok,task_context_text(initial,feedback),device,2048,pooling='both')
                        agent.set_source(fields,target_fields=query)
                    reads.append({'turn':turn, **binding})
                elif arm == 'static':
                    for layer,b in zip(agent.adapters,static['factors'],strict=True):
                        layer.b = b.to(device)
            available = list(state['admissible_commands'])
            allowed = guarded_commands(feedback,available,seen,2)
            messages.append({'role':'user','content':feedback+'\nAvailable commands:\n'+'\n'.join(available)})
            response = generate(agent,tok,compact_messages(messages,2),device,64,allowed)
            command = clean_command(response,available)
            assert command in allowed
            seen[(feedback,command)] = seen.get((feedback,command),0)+1
            state,_,done = env.step(command)
            messages.append({'role':'assistant','content':response})
            trajectory.append({'turn':turn,'response':response,'command':command,
                               'valid':command in available,'observation':str(state['feedback']),
                               'won':bool(state['won'])})
            if done or state['won']: break
        return {'status':'complete','reward':float(bool(state['won'])), 'steps':len(trajectory),
                'initial_observation':initial,'initial_commands_sha256':digest(initial_commands),
                'invalid_commands':sum(not s['valid'] for s in trajectory),'trajectory':trajectory,
                'termination':'success' if state['won'] else 'budget_or_environment_done',
                'source_reads':reads}
    finally:
        env.close(); agent.set_source(None)


def bindings(args):
    training = json.loads((args.training/'manifest.json').read_text())
    done = json.loads((args.training/'completed.json').read_text())
    assert done['complete'] and done['training_steps_per_arm']==700
    assert done['initial_A_equal'] and done['target_order_equal']
    assert done['manifest_sha256']==file_hash(args.training/'manifest.json')
    for name,value in training['dependencies'].items():
        assert file_hash(Path(__file__).with_name(name))==value
    prepared = json.loads(Path('data/annotations/alfworld_causal_prefix_v70_train120_20261010.json').read_text())
    assert training['review_sha256']==file_hash(Path('data/annotations/alfworld_causal_prefix_v70_train120_20261010.json'))
    parent = Path('results/trajectory_hyperlora/alfworld_cold_joint_memory_v63_hybrid_20261010.json')
    assert file_hash(parent)==prepared['parent_sha256']
    rows = json.loads(parent.read_text())['rows'][120:144]
    targets=[]
    for r in rows:
        c={'index':r['index'],'game':r['game'],'game_sha256':file_hash(args.data_root/r['game']),
           'initial_observation_sha256':digest(r['episode']['initial_observation'])}
        targets.append({**c,'input_content_sha256':digest(c),'reviewed_target':True})
    assert len(targets)==24 and not set(r['game'] for r in targets)&set(r['game'] for r in prepared['targets'])
    for arm in ('hyper','static'):
        state=json.loads((args.training/f'{arm}_training.json').read_text())
        assert state['complete'] and state['checkpoint_sha256']==file_hash(args.training/f'{arm}.pt')
    names=['evaluate_causal_prefix_v70.py','audit_my_online_update_v28_reward.py',
           'alfworld_online_from_empty.py','alfworld_online_accumulate_lora_v1.py']
    return {'targets':targets,'targets_sha256':digest(targets),'arms':list(ARMS),
            'dependencies':{**training['dependencies'],**{n:file_hash(Path(__file__).with_name(n)) for n in names}},
            'checkpoints':{arm:file_hash(args.training/f'{arm}.pt') for arm in ('hyper','static')},
            'training_manifest_sha256':done['manifest_sha256'],
            'model_config_sha256':file_hash(args.model/'config.json'),
            'model_weights':{str(p):file_hash(p) for p in sorted(args.model.glob('*.safetensors'))},
            'max_steps':30,'history_turns':2,'max_new_tokens':64,'loop_guard_max':2,
            'activation_step':3,'training_updates':0,
            'scope':'Exposed development next24; frozen models after chronological own-trajectory replay, not live online updating',
            'source_policy':'Only current episode steps strictly before last two completed steps; no future outcomes',
            'empty_text':EMPTY}


def evaluate(args, review):
    if args.output.exists(): raise FileExistsError(args.output)
    torch.manual_seed(42)
    agent,tok=load_agent(args.model,args.training/'hyper.pt',args.device,.45)
    agent.requires_grad_(False);agent.eval()
    hyper_a=[x.a.detach().clone() for x in agent.adapters]
    static=torch.load(args.training/'static.pt',map_location='cpu',weights_only=True)
    result={'review_sha256':file_hash(args.review),'rows':[],'failures':[],'complete':False}
    for target in review['targets']:
        row={'index':target['index'],'game':target['game'],'input_content_sha256':target['input_content_sha256'],'arms':{}}
        try:
            for arm in ARMS:
                basis=static['A'] if arm=='static' else hyper_a
                with torch.no_grad():
                    for layer,a in zip(agent.adapters,basis,strict=True):layer.a.copy_(a.to(layer.a))
                before=factors_hash([x.a for x in agent.adapters])
                ep=episode(agent,tok,args.data_root/target['game'],arm,static,args.device)
                assert digest(ep['initial_observation'])==target['initial_observation_sha256']
                assert before==factors_hash([x.a for x in agent.adapters])
                row['arms'][arm]=ep
                save(args.output,{**result,'in_progress_row':row})
            result['rows'].append(row)
            result['summary']={'n':len(result['rows']),**{a:sum(r['arms'][a]['reward'] for r in result['rows']) for a in ARMS}}
            save(args.output,result);print(json.dumps(result['summary']),flush=True)
        except Exception as exc:
            result['failures'].append({'index':target['index'],'error':repr(exc),'partial_row':row})
            save(args.output,result);raise
    result['complete']=True;save(args.output,result)


def audit(args,review):
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(args.model,local_files_only=True)
    result=json.loads(args.output.read_text())
    assert result['complete'] and not result['failures'] and len(result['rows'])==24
    assert result['review_sha256']==file_hash(args.review)
    for row,target in zip(result['rows'],review['targets'],strict=True):
        assert row['index']==target['index'] and row['game']==target['game']
        assert row['input_content_sha256']==target['input_content_sha256']
        for arm in ARMS:
            ep=row['arms'][arm];assert ep['steps']<=30
            replay(args.data_root/row['game'],ep)
            expected=[]
            if arm in ('real','empty','shifted'):
                for turn in range(3,ep['steps']):
                    _,bound=source_read(tok,ep['initial_observation'],ep['trajectory'][:turn],arm)
                    expected.append({'turn':turn,**bound})
            assert ep['source_reads']==expected
        # All methods are disabled for the first three actions.
        assert all(row['arms'][a]['trajectory'][:3]==row['arms']['base']['trajectory'][:3] for a in ARMS)
    summary={'n':24,**{a:sum(r['arms'][a]['reward'] for r in result['rows']) for a in ARMS}}
    assert summary==result['summary']
    save(args.output.with_suffix('.audit.json'),{'passed':True,'episodes_replayed':120,
         'summary':summary,'report_sha256':file_hash(args.output),'review_sha256':file_hash(args.review),
         'scope':'Environment replay and causal source-text reconstruction; no independent neural regeneration'})
    print(json.dumps(summary),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('prepare','evaluate','audit'))
    p.add_argument('--training',type=Path,default=Path('results/trajectory_hyperlora/alfworld_causal_prefix_v70_20261010'))
    p.add_argument('--review',type=Path,default=Path('data/annotations/alfworld_causal_prefix_v70_eval_20261010.json'))
    p.add_argument('--output',type=Path,default=Path('results/trajectory_hyperlora/alfworld_causal_prefix_v70_eval_20261010.json'))
    p.add_argument('--model',type=Path,default=Path('current_work/delta-Mem/model/Qwen3-4B-Instruct-2507'))
    p.add_argument('--data-root',type=Path,default=Path('ttcl/data/alfworld_delta'))
    p.add_argument('--device',default='cuda:1')
    args=p.parse_args();review=bindings(args)
    if args.command=='prepare':
        if args.review.exists():raise FileExistsError(args.review)
        save(args.review,review);print('Reviewed 24 tasks, 120 episodes')
    else:
        assert json.loads(args.review.read_text())==review
        (evaluate if args.command=='evaluate' else audit)(args,review)


if __name__=='__main__':main()
