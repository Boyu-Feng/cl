"""Build a contextual bank only after the complete v71 utility audit passes."""
import argparse
import json
from pathlib import Path

import torch

from ttcl.trajectory_hyperlora.evaluate_own_correction_utility_v71 import checked
from ttcl.trajectory_hyperlora.alfworld_cold_joint_memory_v63 import new_agent, factors_hash, digest, file_hash, save
from ttcl.trajectory_hyperlora.contextual_alf_source import contextual_text_fields, task_context_text
from ttcl.trajectory_hyperlora.contextual_residual_memory_v72 import ContextualResidualMemory


def build(args):
    if args.output.exists() or args.output.with_suffix('.json').exists():
        raise FileExistsError(args.output)
    review,parent=checked(args)
    audit_path=args.utility.with_suffix('.audit.json')
    audit=json.loads(audit_path.read_text());utility=json.loads(args.utility.read_text())
    assert utility['complete'] and not utility['failures']
    assert audit['passed'] and audit['training_ready'] and audit['episodes_replayed']==102
    assert audit['report_sha256']==file_hash(args.utility)
    assert utility['review_sha256']==file_hash(args.review)
    assert utility['runner_sha256']==file_hash(Path(__file__).with_name('evaluate_own_correction_utility_v71.py'))
    assert len(audit['labels'])==len(utility['rows'])==len(review['targets'])==34
    selected=[]
    for label,row,target in zip(audit['labels'],utility['rows'],review['targets'],strict=True):
        assert label['index']==row['index']==target['index']
        assert label['target_binding']==row['input_content_sha256']==target['input_content_sha256']
        assert label['retry_episodes_sha256']==digest(row['arms'])
        assert label['paired_gain']==row['arms']['failure_delta']['reward']-row['arms']['unchanged_retry']['reward']
        assert label['gain_over_random']==row['arms']['failure_delta']['reward']-row['arms']['matched_norm_random']['reward']
        if label['paired_gain']>0 and label['gain_over_random']>0:selected.append((target,label))
    # All auxiliary interactions are made available at the prefix boundary,
    # never retroactively during the original first120 trajectory stream.
    snapshot=next(row['snapshot'] for row in reversed(parent['rows'][:120]) if 'snapshot' in row)
    assert file_hash(Path(snapshot['path']))==snapshot['sha256']
    state=torch.load(snapshot['path'],map_location='cpu',weights_only=True)
    basis=factors_hash(state['A'])
    encoder={'model_config_sha256':file_hash(args.model/'config.json'),
             'weights':{str(p):file_hash(p) for p in sorted(args.model.glob('*.safetensors'))},
             'encoder_code_sha256':file_hash(Path(__file__).with_name('contextual_alf_source.py')),
             'input':'Initial observation and initial task context only','pooling':'pair_contextual mean',
             'max_tokens':2048,'adapters_disabled':True}
    assert encoder['weights']
    memory=ContextualResidualMemory(basis_sha256=basis,encoder_sha256=digest(encoder))
    agent=tok=None
    if selected:
        torch.cuda.set_per_process_memory_fraction(.40,device=args.device)
        agent,tok=new_agent(args);agent.eval();agent.set_source(None)
    bindings=[]
    for target,label in selected:
        source_state=torch.load(target['pre_task_snapshot']['path'],map_location='cpu',weights_only=True)
        assert factors_hash(source_state['A'])==basis
        delta=torch.load(target['candidate_delta_path'],map_location='cpu',weights_only=True)
        assert factors_hash(delta)==target['candidate_delta_sha256']
        text=task_context_text(target['initial_observation'],target['initial_observation'])
        fields=contextual_text_fields(agent,tok,text,args.device,2048,pooling='both')
        accepted=memory.add(key=fields['pair_contextual'],delta=delta,
            evidence_id=target['input_content_sha256'],audit_sha256=file_hash(audit_path),
            source_task=target['index'],available_before_task=120,
            paired_gain=label['paired_gain'],gain_over_random=label['gain_over_random'],
            basis_sha256=basis,encoder_sha256=digest(encoder))
        assert accepted
        bindings.append({'source_task':target['index'],'input_content_sha256':target['input_content_sha256'],
                         'key_text_sha256':digest(text),'delta_sha256':target['candidate_delta_sha256']})
    payload={'basis_sha256':basis,'encoder_sha256':digest(encoder),'encoder':encoder,
             'entries':memory.entries,'top_k':3,'max_relative':.25,'prefix_snapshot':snapshot,
             'available_before_task':120,'utility_report_sha256':file_hash(args.utility),
             'utility_audit_sha256':file_hash(audit_path),'bindings':bindings}
    torch.save(payload,args.output)
    save(args.output.with_suffix('.json'),{'complete':True,'accepted_entries':len(memory.entries),
        'bank_sha256':file_hash(args.output),'builder_sha256':file_hash(Path(__file__)),
        'component_sha256':file_hash(Path(__file__).with_name('contextual_residual_memory_v72.py')),
        'utility_audit_sha256':file_hash(audit_path),'basis_sha256':basis,
        'scope':'Bank prepared from audited own train retries; no downstream effectiveness result',
        'extra_training_attempts':102,'available_before_task':120})
    print(json.dumps({'accepted_entries':len(memory.entries),'evaluation_started':False}),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--review',type=Path,default=Path('data/annotations/alfworld_own_correction_utility_v71_20261010.json'))
    p.add_argument('--parent',type=Path,default=Path('results/trajectory_hyperlora/alfworld_cold_joint_memory_v63_hybrid_20261010.json'))
    p.add_argument('--utility',type=Path,default=Path('results/trajectory_hyperlora/alfworld_own_correction_utility_v71_20261010.json'))
    p.add_argument('--output',type=Path,default=Path('results/trajectory_hyperlora/contextual_residual_memory_v72_20261010.pt'))
    p.add_argument('--model',type=Path,default=Path('current_work/delta-Mem/model/Qwen3-4B-Instruct-2507'))
    p.add_argument('--data-root',type=Path,default=Path('ttcl/data/alfworld_delta'))
    p.add_argument('--device',default='cuda:6')
    args=p.parse_args();args.seed=42;build(args)


if __name__=='__main__':main()
