"""Shared-own-history official online comparison of projected LoRA and text.

The same frozen no-memory actor supplies the history stream to every arm.
This isolates memory use and later updates; a separate method-on-policy study
is still needed to measure each arm's full feedback loop.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ttcl.experience_evolution.environment import make_env
from ttcl.trajectory_hyperlora.alfworld_online_from_empty import (
    digest, file_hash, records_from_episode,
)
from ttcl.trajectory_hyperlora.alfworld_online_persistent_context_lora_v1 import (
    bounded_source_text,
)
from ttcl.trajectory_hyperlora.alfworld_online_context_vector_mean_v1 import (
    update_fields, vector_hash,
)
from ttcl.trajectory_hyperlora.alfworld_source_utility_dataset_v2 import save
from ttcl.trajectory_hyperlora.alfworld_zero_shot_probe import load_agent, run_episode
from ttcl.trajectory_hyperlora.centered_feedback_residual_v3 import CenteredEvidenceResidual
from ttcl.trajectory_hyperlora.contextual_alf_source import (
    contextual_text_fields, source_text, task_context_text,
)
from ttcl.trajectory_hyperlora.encode_alf_future_utility_features_v1 import event_indices
from ttcl.trajectory_hyperlora.freeze_alf_fresh_seen_v6 import expected_review
from ttcl.trajectory_hyperlora.train_alf_feedback_residual_v4_antithetic import ResidualInjection


def encode_source(agent, tokenizer, records, device):
    bounded, original, retained = bounded_source_text(tokenizer, records, 2048)
    with torch.no_grad():
        fields = contextual_text_fields(agent, tokenizer, bounded,
                                       device, 2048, pooling='both')
        global_vector = fields['pair_contextual'].float().flatten()
        deltas = []
        for index in event_indices(len(records)):
            event = records[index]
            before = f"Action: {event['action']}\nBefore: {event['observation']}"
            after = (f"Action: {event['action']}\n"
                     f"Environment reply: {event['feedback']}")
            a = contextual_text_fields(agent, tokenizer, before,
                device, 2048, pooling='both')['pair_contextual'].float().flatten()
            b = contextual_text_fields(agent, tokenizer, after,
                device, 2048, pooling='both')['pair_contextual'].float().flatten()
            deltas.append(b-a)
    if not deltas or len(deltas) > 16 or retained > 2048:
        raise ValueError('Unencodable own-success interaction')
    return fields, global_vector, torch.stack(deltas).mean(0), original, retained


def target_feature(agent, tokenizer, initial, device):
    with torch.no_grad():
        return contextual_text_fields(agent, tokenizer,
            task_context_text(initial, initial), device, 2048,
            pooling='both')['pair_contextual'].float().flatten()


def bounded_history_text(tokenizer, sources, limit=2048):
    text = '\n\n'.join(
        f"Own successful episode {index + 1}:\n{source_text(item['records'])}"
        for index, item in enumerate(sources))
    ids = tokenizer(text, add_special_tokens=False).input_ids
    original = len(ids)
    if original > limit:
        head = limit // 2
        ids = ids[:head] + ids[-(limit-head):]
    bounded = tokenizer.decode(ids, skip_special_tokens=False)
    retained = len(tokenizer(bounded, add_special_tokens=False).input_ids)
    if retained > limit:
        raise ValueError('Truncated raw-history text exceeds context budget')
    return bounded, original, retained


def checked_inputs(args):
    review = json.loads(args.review.read_text())
    binding_args = argparse.Namespace(output=args.plan,
        data_root=args.data_root, checkpoint=args.checkpoint)
    if review != expected_review(binding_args):
        raise ValueError('Changed reviewed official game or input binding')
    training = json.loads(args.training_report.read_text())
    audit = json.loads(args.training_audit.read_text())
    model = torch.load(args.residual, map_location='cpu', weights_only=True)
    if (training['trained_model_sha256'] != file_hash(args.residual) or
            training['checkpoint_sha256'] != file_hash(args.checkpoint) or
            training['max_pairs'] != 72 or training['failures'] or
            audit['raw_report_sha256'] != file_hash(args.training_report) or
            audit['trained_model_sha256'] != file_hash(args.residual) or
            audit['summary']['pairs'] != 72 or
            audit['summary']['failed_replays'] != 0 or
            model['model_kind'] !=
                'centered_bilinear_action_projected_v6' or
            model['config']['checkpoint_sha256'] !=
                file_hash(args.checkpoint) or
            model['config']['train_review_sha256'] !=
                training['train_review_sha256'] or
            model['basis'].shape != (128, 8) or
            len(review['targets']) != 36):
        raise ValueError('Changed or incomplete frozen reward-training lineage')
    return review, model


def run(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    review, model = checked_inputs(args)
    agent, tokenizer = load_agent(args.model, args.checkpoint,
                                  args.device, args.gpu_fraction)
    if (agent.encoder_kind != 'contextual' or
            agent.task_context_scope != 'current' or
            agent.task_pair_pooling != 'mean'):
        raise ValueError('Changed frozen actor')
    for parameter in agent.parameters():
        parameter.requires_grad_(False)
    correction = CenteredEvidenceResidual.from_state(
        model['correction']).to(args.device)
    correction.load_state_dict(model['correction'])
    correction.eval()
    basis = model['basis'].to(args.device).float()
    sources = []
    all_fields = None
    all_global = all_delta = None
    report = {
        'protocol': 'Official valid_seen frozen prospective order, shared no-memory own-success source stream; continuous average source-vector and projected-score residual LoRA versus fixed first success LoRA, all-success raw text and no memory; one official 50-step greedy constrained attempt each per task; identical source evidence, text context compute not matched; not method-on-policy online control',
        'plan_sha256': file_hash(args.plan),
        'review_sha256': file_hash(args.review),
        'checkpoint_sha256': file_hash(args.checkpoint),
        'residual_sha256': file_hash(args.residual),
        'training_report_sha256': file_hash(args.training_report),
        'training_audit_sha256': file_hash(args.training_audit),
        'source_tokens': 2048,
        'max_steps': 50, 'max_new_tokens': 64,
        'actor_history_turns': 2, 'loop_guard_max': 2,
        'games': [], 'failures': [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save(args.output, report)
    for row in review['targets']:
        game = args.data_root / row['game']
        if file_hash(game) != row['game_sha256']:
            raise ValueError('Changed reviewed official game content')
        env = make_env(game)
        try:
            initial = str(env.reset()['feedback'])
        finally:
            env.close()
        prior = [x['records_sha256'] for x in sources]
        entry = {'index': row['index'], 'family': row['family'],
                 'game': row['game'], 'game_sha256': row['game_sha256'],
                 'input_content_sha256': row['input_content_sha256'],
                 'prior_success_count': len(sources),
                 'prior_source_records_sha256': prior,
                 'prior_source_chain_sha256': digest(prior),
                 'source_vector_sha256': vector_hash(all_fields)}
        try:
            base = run_episode(agent, tokenizer, game, {}, adapter=False,
                device=args.device, max_steps=50, max_new_tokens=64,
                constrain_actions=True, actor_history_turns=2,
                loop_guard_max=2)
            entry['base'] = base
            if sources:
                q = target_feature(agent, tokenizer, initial, args.device)
                raw_text, original, retained = bounded_history_text(
                    tokenizer, sources, 2048)
                entry['raw_text_original_tokens'] = original
                entry['raw_text_retained_tokens'] = retained
                entry['raw_text_sha256'] = digest(raw_text)
                for arm, fields, global_vector, delta_vector in (
                    ('continuous', all_fields, all_global, all_delta),
                    ('fixed_first', sources[0]['fields'],
                     sources[0]['global'], sources[0]['delta'])):
                    with torch.no_grad():
                        mean = correction(global_vector, delta_vector, q)
                        shift = mean @ basis.T
                    entry[f'{arm}_code_norm'] = float(mean.norm())
                    with ResidualInjection(agent, shift):
                        entry[arm] = run_episode(agent, tokenizer, game,
                            fields, adapter=True, device=args.device,
                            max_steps=50, max_new_tokens=64,
                            constrain_actions=True, actor_history_turns=2,
                            loop_guard_max=2)
                entry['raw_text'] = run_episode(agent, tokenizer, game,
                    {}, adapter=False, device=args.device,
                    max_steps=50, max_new_tokens=64,
                    constrain_actions=True, actor_history_turns=2,
                    loop_guard_max=2, memory_text=raw_text)
            else:
                # Empty memory makes every arm exactly the same deterministic actor.
                entry['continuous'] = base
                entry['fixed_first'] = base
                entry['raw_text'] = base
                entry['raw_text_original_tokens'] = 0
                entry['raw_text_retained_tokens'] = 0
                entry['raw_text_sha256'] = digest('')
                entry['continuous_code_norm'] = 0.
                entry['fixed_first_code_norm'] = 0.
            for arm in ('base', 'continuous', 'fixed_first', 'raw_text'):
                episode = entry[arm]
                if (episode['status'] != 'complete' or
                        episode['invalid_commands'] != 0 or
                        episode['initial_observation'] != initial):
                    raise ValueError(f'Changed or incomplete {arm} episode')
            if base['reward']:
                records = records_from_episode(base)
                fields, global_vector, delta_vector, original, retained = (
                    encode_source(agent, tokenizer, records, args.device))
                count = len(sources)
                all_fields = update_fields(all_fields, fields, count)
                if count:
                    all_global += (global_vector-all_global)/(count+1)
                    all_delta += (delta_vector-all_delta)/(count+1)
                else:
                    all_global = global_vector.clone()
                    all_delta = delta_vector.clone()
                sources.append({'records': records,
                    'records_sha256': digest(records),
                    'fields': fields, 'global': global_vector,
                    'delta': delta_vector})
                entry['new_source_records_sha256'] = digest(records)
                entry['new_source_original_tokens'] = original
                entry['new_source_retained_tokens'] = retained
            entry['source_count_after'] = len(sources)
            entry['source_vector_sha256_after'] = vector_hash(all_fields)
            report['games'].append(entry)
            save(args.output, report)
            print(json.dumps({'done': len(report['games']),
                'base': sum(x['base']['reward'] for x in report['games']),
                'continuous': sum(x['continuous']['reward'] for x in report['games']),
                'fixed_first': sum(x['fixed_first']['reward'] for x in report['games']),
                'raw_text': sum(x['raw_text']['reward'] for x in report['games']),
                'shared_success_sources': len(sources),
                'failures': len(report['failures'])}), flush=True)
        except Exception as exc:
            report['failures'].append({'index': row['index'],
                                       'reason': repr(exc), 'partial': entry})
            save(args.output, report)
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, default=Path(
        'current_work/delta-Mem/model/Qwen3-4B-Instruct-2507'))
    parser.add_argument('--checkpoint', type=Path, default=Path(
        'results/trajectory_hyperlora/alf_sibling_simple40_onpolicy600_20261006.pt'))
    parser.add_argument('--residual', type=Path, default=Path(
        'results/trajectory_hyperlora/alf_projected_reward_train72_v6_20261007.pt'))
    parser.add_argument('--training-report', type=Path, default=Path(
        'results/trajectory_hyperlora/alf_projected_reward_train72_v6_20261007.json'))
    parser.add_argument('--training-audit', type=Path, default=Path(
        'results/trajectory_hyperlora/alf_projected_reward_train72_v6_audited_20261007.json'))
    parser.add_argument('--plan', type=Path, default=Path(
        'ttcl/trajectory_hyperlora/alf_fresh_seen36_v6_plan.json'))
    parser.add_argument('--review', type=Path, default=Path(
        'data/annotations/alf_fresh_seen36_online_v6_reviewed_20261007.json'))
    parser.add_argument('--data-root', type=Path, default=Path(
        'ttcl/data/alfworld_delta'))
    parser.add_argument('--output', type=Path, default=Path(
        'results/trajectory_hyperlora/alf_fresh_seen36_projected_shared_online_v7_20261007.json'))
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--gpu-fraction', type=float, default=.7)
    run(parser.parse_args())
