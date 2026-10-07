"""Distill replay-verified own-only-success future actions into a LoRA hypernetwork.

Exploratory offline action-level objective on official train games; the target
actor had an expert-sibling LoRA and the checkpoint ancestor saw these games.
No independent or online reward claim follows from this proxy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

import torch
from torch.nn import functional as F

from ttcl.experience_evolution.environment import ACTOR_SYSTEM
from ttcl.trajectory_hyperlora.alfworld_online_from_empty import file_hash
from ttcl.trajectory_hyperlora.alfworld_online_persistent_context_lora_v1 import (
    bounded_source_text,
)
from ttcl.trajectory_hyperlora.alfworld_source_utility_dataset_v2 import save
from ttcl.trajectory_hyperlora.alfworld_zero_shot_probe import load_agent
from ttcl.trajectory_hyperlora.centered_feedback_residual_v3 import (
    from_encoded_features,
)
from ttcl.trajectory_hyperlora.contextual_alf_source import contextual_text_fields
from ttcl.trajectory_hyperlora.analyze_alfworld_online_lora_geometry_v1 import (
    source_and_pair,
)
from ttcl.trajectory_hyperlora.train_alf_next_task import compact_messages


def messages_for(target: dict, turn: int, memory: str | None = None):
    system = ACTOR_SYSTEM + (
        '\n\nPrior attempt record:\n' + memory if memory else '')
    messages = [{'role': 'system', 'content': system}]
    for index, row in enumerate(target['reviewed_targets'][:turn+1]):
        messages.append({'role': 'user', 'content': row['observation'] +
            '\nAvailable commands:\n' +
            '\n'.join(row['admissible_commands'])})
        if index < turn:
            messages.append({'role': 'assistant',
                             'content': row['target_action']})
    return compact_messages(messages, history_turns=2)


def prompt_ids(tokenizer, messages, limit):
    text = tokenizer.apply_chat_template(messages, tokenize=False,
                                         add_generation_prompt=True)
    ids = tokenizer(text, add_special_tokens=False).input_ids
    if len(ids) > limit:
        raise ValueError(f'Reviewed actor prompt exceeds token budget {len(ids)}')
    return ids


def action_logits(agent, prefix, answer, device):
    ids = torch.tensor([prefix + answer], device=device)
    logits = agent.model(input_ids=ids, use_cache=False).logits
    start = len(prefix)-1
    return logits[0, start:start+len(answer)].float()


def train(args):
    if args.output.exists() or args.save_model.exists():
        raise FileExistsError('Use fresh distillation output and model paths')
    if not (0 < args.steps <= 234 and 0 < args.lr and
            0 <= args.teacher_weight <= 1):
        raise ValueError('Invalid frozen pilot budget')
    bank = json.loads(args.bank.read_text())
    plan = json.loads(args.pairs.read_text())
    audit = json.loads(args.pair_audit.read_text())
    target_bank = json.loads(args.target_bank.read_text())
    features = torch.load(args.features, map_location='cpu', weights_only=True)
    if (features['source_bank_sha256'] != file_hash(args.bank) or
            features['target_bank_sha256'] != file_hash(args.target_bank) or
            features['pairs_review_sha256'] != file_hash(args.pairs) or
            features['checkpoint_sha256'] != file_hash(args.checkpoint) or
            audit['pairs_review_sha256'] != file_hash(args.pairs) or
            audit['summary']['different_game_action_pairs'] != 234 or
            len(bank['rows']) != 57 or len(target_bank['rows']) != 42 or
            len(plan['pairs']) != 234):
        raise ValueError('Changed or falsely independent self-future inputs')
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    agent, tokenizer = load_agent(args.model, args.checkpoint,
                                  args.device, args.gpu_fraction)
    for parameter in agent.parameters():
        parameter.requires_grad_(False)
    agent.model.config.use_cache = False
    rows = {row['game']: row for row in target_bank['rows']}
    source_index = {row['game']: i for i, row in enumerate(bank['rows'])}
    bounded = {}
    fields = {}
    with torch.no_grad():
        for game in sorted({p['source_game'] for p in plan['pairs']}):
            text, _, _ = bounded_source_text(tokenizer,
                next(r for r in bank['rows'] if r['game'] == game)['records'], args.context_tokens)
            bounded[game] = text
            fields[game] = contextual_text_fields(agent, tokenizer,
                text, args.device, args.context_tokens, pooling='both')
    train_pairs = list(plan['pairs'])
    latent = []
    with torch.no_grad():
        for pair in train_pairs:
            q = features['targets'][
                pair['target_action_input_sha256']].to(args.device)[None]
            query = {'contextual': q, 'pair_contextual': q}
            s, t = source_and_pair(agent, fields[pair['source_game']], query)
            latent.append((s+t).flatten().float().cpu())
    matrix = torch.stack(latent)
    _, _, vh = torch.linalg.svd(matrix-matrix.mean(0),
                                 full_matrices=False)
    basis = vh[:args.code_dim].T.contiguous().to(args.device)
    train_games = {p['source_game'] for p in train_pairs}
    train_indices = [source_index[game] for game in sorted(train_games)]
    target_keys = sorted({pair['target_action_input_sha256']
                          for pair in train_pairs})
    encoded = {'global': features['global'][train_indices],
        'deltas': features['deltas'][train_indices],
        'event_counts': features['event_counts'][train_indices],
        'targets': torch.stack([features['targets'][key]
                                for key in target_keys])}
    correction = from_encoded_features(encoded, args.feature_rank,
                                       args.code_dim).to(args.device)
    optimizer = torch.optim.AdamW(correction.parameters(), lr=args.lr,
                                  weight_decay=0)
    delta = torch.stack([features['deltas'][i, :int(n)].mean(0)
        for i, n in enumerate(features['event_counts'])])
    rng.shuffle(train_pairs)
    selected = train_pairs[:args.steps]
    report = {'protocol': 'Offline exploratory future-action CE plus frozen raw-trajectory teacher soft distillation into source-conditioned centered LoRA; own-LoRA-success/base-failure official train targets, ancestor-exposed, no independent environment-RL or online-success claim',
        'seed': args.seed, 'steps': args.steps, 'lr': args.lr,
        'teacher_weight': args.teacher_weight,
        'context_tokens': args.context_tokens,
        'feature_rank': args.feature_rank, 'code_dim': args.code_dim,
        'bank_review_sha256': file_hash(args.bank),
        'target_bank_review_sha256': file_hash(args.target_bank),
        'pairs_review_sha256': file_hash(args.pairs),
        'pair_audit_sha256': file_hash(args.pair_audit),
        'features_sha256': file_hash(args.features),
        'checkpoint_sha256': file_hash(args.checkpoint),
        'train_games': len(train_games), 'train': [], 'failures': []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    completed = False
    try:
        for step, pair in enumerate(selected):
            source = pair['source_game']
            target = rows[pair['target_game']]
            action = target['reviewed_targets'][pair['target_turn']]
            answer = tokenizer(action['target_action'] + tokenizer.eos_token,
                add_special_tokens=False).input_ids
            student = prompt_ids(tokenizer,
                messages_for(target, pair['target_turn']),
                args.max_prompt_tokens)
            teacher = prompt_ids(tokenizer,
                messages_for(target, pair['target_turn'], bounded[source]),
                args.max_prompt_tokens)
            q = features['targets'][
                pair['target_action_input_sha256']].to(args.device)
            i = source_index[source]
            mean = correction(features['global'][i].to(args.device),
                delta[i].to(args.device), q)
            shift = mean @ basis.T
            with torch.no_grad():
                agent.set_source(None)
                teacher_logits = action_logits(agent, teacher, answer,
                                               args.device).detach()
                query = {'contextual': q[None], 'pair_contextual': q[None]}
                agent.set_source(fields[source], target_fields=query)
            for adapter, head in zip(agent.adapters, agent.b_heads,
                                     strict=True):
                adapter.b = adapter.b.detach() + F.linear(
                    shift.to(head.weight.dtype), head.weight).reshape_as(
                        adapter.b)
            student_logits = action_logits(agent, student, answer,
                                           args.device)
            target_ids = torch.tensor(answer, device=args.device)[:, None]
            ce = -F.log_softmax(student_logits, -1).gather(
                -1, target_ids).mean()
            kl = F.kl_div(F.log_softmax(student_logits, -1),
                          F.softmax(teacher_logits, -1),
                          reduction='batchmean')
            loss = ((1-args.teacher_weight)*ce +
                    args.teacher_weight*kl +
                    args.code_penalty*mean.square().sum())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                correction.parameters(), 1.)
            optimizer.step()
            with torch.no_grad():
                agent.set_source(None)
            report['train'].append({'step': step,
                'input_content_sha256': pair['input_content_sha256'],
                'source_game': source, 'target_game': target['game'],
                'target_turn': pair['target_turn'],
                'student_prompt_tokens': len(student),
                'teacher_prompt_tokens': len(teacher),
                'answer_tokens': len(answer),
                'ce': float(ce.detach()), 'kl': float(kl.detach()),
                'loss': float(loss.detach()),
                'code_norm': float(mean.detach().norm()),
                'grad_norm': float(grad_norm)})
            save(args.output, report)
            print(json.dumps({'step': step+1,
                'ce': float(ce.detach()), 'kl': float(kl.detach()),
                'code_norm': float(mean.detach().norm())}), flush=True)
        completed = True
    except Exception as exc:
        report['failures'].append({'step': len(report['train']),
                                   'reason': repr(exc)})
        raise
    finally:
        if completed:
            args.save_model.parent.mkdir(parents=True, exist_ok=True)
            torch.save({'model_kind': 'own_only_soft_distill_v2',
                'correction': correction.cpu().state_dict(),
                'basis': basis.detach().cpu(),
                'config': {key: report[key] for key in (
                    'protocol', 'seed', 'steps', 'lr', 'teacher_weight',
                    'context_tokens', 'feature_rank', 'code_dim',
                    'bank_review_sha256', 'target_bank_review_sha256', 'pairs_review_sha256',
                    'features_sha256', 'checkpoint_sha256')}},
                args.save_model)
            report['trained_model_sha256'] = file_hash(args.save_model)
        save(args.output, report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, default=Path(
        'current_work/delta-Mem/model/Qwen3-4B-Instruct-2507'))
    parser.add_argument('--checkpoint', type=Path, default=Path(
        'results/trajectory_hyperlora/alf_sibling_simple40_onpolicy600_20261006.pt'))
    parser.add_argument('--bank', type=Path, default=Path(
        'data/annotations/alf_self_base57_replay_reviewed_20261007.json'))
    parser.add_argument('--pairs', type=Path, default=Path(
        'data/annotations/alf_own_only_future_pairs_reviewed_20261007.json'))
    parser.add_argument('--pair-audit', type=Path, default=Path(
        'results/trajectory_hyperlora/alf_own_only_future_pairs_audited_20261007.json'))
    parser.add_argument('--target-bank', type=Path, default=Path(
        'data/annotations/alf_own_only42_replay_reviewed_20261007.json'))
    parser.add_argument('--features', type=Path, default=Path(
        'results/trajectory_hyperlora/alf_own_only_future_features_v2_20261007.pt'))
    parser.add_argument('--output', type=Path, default=Path(
        'results/trajectory_hyperlora/alf_own_only_soft_distill_v2_20261007.json'))
    parser.add_argument('--save-model', type=Path, default=Path(
        'results/trajectory_hyperlora/alf_own_only_soft_distill_v2_20261007.pt'))
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--steps', type=int, default=64)
    parser.add_argument('--feature-rank', type=int, default=12)
    parser.add_argument('--code-dim', type=int, default=8)
    parser.add_argument('--teacher-weight', type=float, default=.25)
    parser.add_argument('--code-penalty', type=float, default=.001)
    parser.add_argument('--lr', type=float, default=2e-3)
    parser.add_argument('--context-tokens', type=int, default=2048)
    parser.add_argument('--max-prompt-tokens', type=int, default=4096)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--gpu-fraction', type=float, default=.8)
    train(parser.parse_args())
