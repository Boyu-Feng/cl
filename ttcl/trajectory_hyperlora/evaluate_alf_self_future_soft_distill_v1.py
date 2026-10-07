"""Offline, ancestry-exposed action-likelihood diagnostic of frozen generator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from ttcl.trajectory_hyperlora.alfworld_online_from_empty import file_hash
from ttcl.trajectory_hyperlora.alfworld_online_persistent_context_lora_v1 import (
    bounded_source_text,
)
from ttcl.trajectory_hyperlora.alfworld_source_utility_dataset_v2 import save
from ttcl.trajectory_hyperlora.alfworld_zero_shot_probe import load_agent
from ttcl.trajectory_hyperlora.centered_feedback_residual_v3 import (
    CenteredEvidenceResidual,
)
from ttcl.trajectory_hyperlora.contextual_alf_source import contextual_text_fields
from ttcl.trajectory_hyperlora.train_alf_self_future_soft_distill_v1 import (
    action_logits, messages_for, prompt_ids,
)


def evaluate(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    bank = json.loads(args.bank.read_text())
    plan = json.loads(args.pairs.read_text())
    ancestry = json.loads(args.ancestry_audit.read_text())
    training = json.loads(args.training_report.read_text())
    audit = json.loads(args.training_audit.read_text())
    features = torch.load(args.features, map_location='cpu', weights_only=True)
    model = torch.load(args.residual, map_location='cpu', weights_only=True)
    if (ancestry['summary']['independent_of_complete_checkpoint'] or
            audit['report_sha256'] != file_hash(args.training_report) or
            audit['model_sha256'] != file_hash(args.residual) or
            training['trained_model_sha256'] != file_hash(args.residual) or
            model['model_kind'] != 'self_future_soft_distill_v1' or
            model['config']['features_sha256'] != file_hash(args.features) or
            features['pairs_review_sha256'] != file_hash(args.pairs) or
            len([p for p in plan['pairs'] if p['split']=='dev']) != 42):
        raise ValueError('Changed or falsely independent frozen proxy')
    rows = {row['game']: row for row in bank['rows']}
    source_index = {row['game']: i for i, row in enumerate(bank['rows'])}
    agent, tokenizer = load_agent(args.model, args.checkpoint,
                                  args.device, args.gpu_fraction)
    agent.model.config.use_cache = False
    correction = CenteredEvidenceResidual.from_state(
        model['correction']).to(args.device)
    correction.load_state_dict(model['correction'])
    correction.eval()
    basis = model['basis'].to(args.device)
    delta = torch.stack([features['deltas'][i, :int(n)].mean(0)
        for i, n in enumerate(features['event_counts'])])
    source_games = sorted({pair['source_game'] for pair in plan['pairs']
                           if pair['split']=='dev'})
    fields, bounded = {}, {}
    with torch.no_grad():
        for game in source_games:
            memory, _, _ = bounded_source_text(tokenizer,
                rows[game]['records'], training['context_tokens'])
            bounded[game] = memory
            fields[game] = contextual_text_fields(agent, tokenizer,
                memory, args.device, training['context_tokens'],
                pooling='both')
    report = {'protocol': 'Ancestry-exposed offline reviewed future-action token CE proxy; same source and target for base, old LoRA, raw text, and generated residual; not official environment reward',
        'bank_review_sha256': file_hash(args.bank),
        'pairs_review_sha256': file_hash(args.pairs),
        'ancestry_audit_sha256': file_hash(args.ancestry_audit),
        'training_audit_sha256': file_hash(args.training_audit),
        'residual_sha256': file_hash(args.residual),
        'checkpoint_sha256': file_hash(args.checkpoint),
        'context_tokens': training['context_tokens'],
        'rows': [], 'failures': []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for pair in plan['pairs']:
        if pair['split'] != 'dev':
            continue
        source = pair['source_game']
        target = rows[pair['target_game']]
        action = target['reviewed_targets'][pair['target_turn']]
        answer = tokenizer(action['target_action'] + tokenizer.eos_token,
            add_special_tokens=False).input_ids
        student = prompt_ids(tokenizer,
            messages_for(target, pair['target_turn']), 2560)
        teacher = prompt_ids(tokenizer,
            messages_for(target, pair['target_turn'], bounded[source]),
            2560)
        q = features['targets'][
            pair['target_action_input_sha256']].to(args.device)
        query = {'contextual': q[None], 'pair_contextual': q[None]}
        i = source_index[source]
        values = {}
        with torch.no_grad():
            agent.set_source(None)
            for arm, prefix in (('base', student), ('raw_text', teacher)):
                logits = action_logits(agent, prefix, answer, args.device)
                ids = torch.tensor(answer, device=args.device)[:, None]
                values[arm] = float(-F.log_softmax(logits, -1)
                    .gather(-1, ids).mean())
            agent.set_source(fields[source], target_fields=query)
            logits = action_logits(agent, student, answer, args.device)
            values['old_lora'] = float(-F.log_softmax(logits, -1)
                .gather(-1, ids).mean())
            mean = correction(features['global'][i].to(args.device),
                delta[i].to(args.device), q)
            shift = mean @ basis.T
            for adapter, head in zip(agent.adapters, agent.b_heads,
                                     strict=True):
                adapter.b = adapter.b + F.linear(
                    shift.to(head.weight.dtype), head.weight).reshape_as(
                        adapter.b)
            logits = action_logits(agent, student, answer, args.device)
            values['generated_lora'] = float(-F.log_softmax(logits, -1)
                .gather(-1, ids).mean())
            agent.set_source(None)
        report['rows'].append({'input_content_sha256':
            pair['input_content_sha256'], 'source_game': source,
            'target_game': target['game'], 'target_turn': pair['target_turn'],
            'action_tokens': len(answer), 'code_norm': float(mean.norm()),
            'ce': values})
        save(args.output, report)
        print(json.dumps({'n': len(report['rows']),
                          'ce': values}), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, default=Path(
        'current_work/delta-Mem/model/Qwen3-4B-Instruct-2507'))
    parser.add_argument('--checkpoint', type=Path, default=Path(
        'results/trajectory_hyperlora/alf_sibling_simple40_onpolicy600_20261006.pt'))
    parser.add_argument('--bank', type=Path, default=Path(
        'data/annotations/alf_self_base57_replay_reviewed_20261007.json'))
    parser.add_argument('--pairs', type=Path, default=Path(
        'data/annotations/alf_self_future_pairs_v2_checkpoint_disjoint_reviewed_20261007.json'))
    parser.add_argument('--ancestry-audit', type=Path, default=Path(
        'results/trajectory_hyperlora/alf_self_future_pairs_v2_ancestry_audited_20261007.json'))
    parser.add_argument('--features', type=Path, default=Path(
        'results/trajectory_hyperlora/alf_self_future_base57_features_20261007.pt'))
    parser.add_argument('--training-report', type=Path, default=Path(
        'results/trajectory_hyperlora/alf_self_future_soft_distill_v1_seed42_64_20261007.json'))
    parser.add_argument('--training-audit', type=Path, default=Path(
        'results/trajectory_hyperlora/alf_self_future_soft_distill_v1_seed42_64_audited_20261007.json'))
    parser.add_argument('--residual', type=Path, default=Path(
        'results/trajectory_hyperlora/alf_self_future_soft_distill_v1_seed42_64_20261007.pt'))
    parser.add_argument('--output', type=Path, default=Path(
        'results/trajectory_hyperlora/alf_self_future_soft_distill_v1_dev42_20261007.json'))
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--gpu-fraction', type=float, default=.65)
    evaluate(parser.parse_args())
