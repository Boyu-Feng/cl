"""Train a feedback-conditioned LoRA residual with real ALFWorld reward.

Only a small residual mean network is trained. A Gaussian code is sampled
once per episode and inserted after each frozen source/task latent update.
The actor remains deterministic given that code, so the score-function loss
is over the stochastic parameter policy, not fictitious actor log-probabilities.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from ttcl.trajectory_hyperlora.alfworld_online_from_empty import digest, file_hash
from ttcl.trajectory_hyperlora.alfworld_source_utility_dataset_v2 import checked, save
from ttcl.trajectory_hyperlora.alfworld_online_persistent_context_lora_v1 import bounded_source_text
from ttcl.trajectory_hyperlora.alfworld_zero_shot_probe import load_agent, run_episode
from ttcl.trajectory_hyperlora.contextual_alf_source import contextual_text_fields, task_context_text
from ttcl.trajectory_hyperlora.analyze_alfworld_online_lora_geometry_v1 import source_and_pair
from ttcl.trajectory_hyperlora.train_alf_future_utility_selector_v1 import target_reward_matrix
from ttcl.trajectory_hyperlora.trajectory_lora_rl import latent_adapter_reinforce


class FeedbackResidual(nn.Module):
    def __init__(self, width: int, code_dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(width * 3),
            nn.Linear(width * 3, 32), nn.Tanh(), nn.Linear(32, code_dim))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, source: torch.Tensor, delta: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        normalized = [F.layer_norm(item.float(), item.shape[-1:])
                      for item in (source, delta, target)]
        return self.net(torch.cat(normalized))


class ResidualInjection:
    """Insert a fixed per-episode latent residual after each current-state B update."""

    def __init__(self, agent, latent_shift: torch.Tensor):
        self.agent = agent
        self.latent_shift = latent_shift.detach()
        self.original = agent.set_source

    def __enter__(self):
        def patched(fields, oracle_bits=None, target_fields=None):
            self.original(fields, oracle_bits=oracle_bits,
                          target_fields=target_fields)
            if fields is None:
                return
            with torch.no_grad():
                for adapter, head in zip(self.agent.adapters,
                                         self.agent.b_heads, strict=True):
                    correction = F.linear(self.latent_shift.to(
                        head.weight.device, dtype=head.weight.dtype),
                        head.weight).reshape_as(adapter.b)
                    adapter.b = adapter.b + correction
        self.agent.set_source = patched
        return self

    def __exit__(self, *_):
        self.agent.set_source = self.original
        self.original(None)


def source_fields(agent, tokenizer, sources, args):
    fields = []
    with torch.no_grad():
        for source in sources:
            bounded, _, _ = bounded_source_text(tokenizer,
                source['records'], 2048)
            fields.append(contextual_text_fields(agent, tokenizer, bounded,
                args.device, 2048, pooling='both'))
    return fields


def latent_basis(agent, tokenizer, sources, targets, matrix, args):
    fields = source_fields(agent, tokenizer, sources, args)
    vectors = []
    with torch.no_grad():
        for target, base in zip(targets, matrix['base'], strict=True):
            if target['target_id'] != base['target_id']:
                raise ValueError('Changed target order')
            observation = base['episode']['initial_observation']
            query = contextual_text_fields(agent, tokenizer,
                task_context_text(observation, observation), args.device,
                2048, pooling='both')
            for source in fields:
                a, b = source_and_pair(agent, source, query)
                vectors.append((a + b).flatten().float())
    matrix_latent = torch.stack(vectors)
    _, _, vh = torch.linalg.svd(matrix_latent - matrix_latent.mean(0),
                                 full_matrices=False)
    if args.code_dim > vh.shape[0]:
        raise ValueError('Requested latent subspace exceeds sample rank')
    return fields, vh[:args.code_dim].T.contiguous()


def run(args):
    if args.output.exists() or args.save_model.exists():
        raise FileExistsError('Use fresh reward pilot and checkpoint paths')
    if not (args.epochs >= 1 and args.sigma > 0 and args.code_dim >= 1 and
            args.lr > 0 and args.max_rollouts >= 1 and args.kl_weight >= 0):
        raise ValueError('Invalid frozen training budget')
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    sources, targets, _ = checked(args)
    matrix = json.loads(args.matrix.read_text())
    audit = json.loads(args.audit.read_text())
    features = torch.load(args.features, map_location='cpu', weights_only=True)
    if (args.target_policy != 'train18' or
            audit['target_policy'] != 'train18' or
            audit['review_sha256'] != file_hash(args.review) or
            audit['raw_report_sha256'] != file_hash(args.matrix) or
            audit['checkpoint_sha256'] != file_hash(args.checkpoint) or
            features['review_sha256'] != audit['review_sha256'] or
            features['checkpoint_sha256'] != audit['checkpoint_sha256'] or
            features['target_policy'] != 'train18' or
            matrix['failures'] or len(sources) != 8 or len(targets) != 18 or
            len(matrix['base']) != 18 or len(matrix['pairs']) != 144 or
            digest([row['episode']['initial_observation']
                    for row in matrix['base']]) !=
                    features['base_observations_sha256']):
        raise ValueError('Changed or unaudited reward-training inputs')
    prior = target_reward_matrix(matrix, 8, 18)
    agent, tokenizer = load_agent(args.model, args.checkpoint,
                                  args.device, args.gpu_fraction)
    if (agent.encoder_kind != 'contextual' or
            agent.task_context_scope != 'current' or
            agent.task_pair_pooling != 'mean'):
        raise ValueError('Expected task-conditioned contextual hypernetwork')
    agent.eval()
    for parameter in agent.parameters():
        parameter.requires_grad_(False)
    fields, basis = latent_basis(agent, tokenizer, sources, targets,
                                 matrix, args)
    width = features['global'].shape[-1]
    correction = FeedbackResidual(width, args.code_dim).to(args.device)
    optimizer = torch.optim.AdamW(correction.parameters(), lr=args.lr,
                                  weight_decay=0)
    deltas = torch.stack([features['deltas'][i, :int(count)].mean(0)
        for i, count in enumerate(features['event_counts'])])
    pair_ids = [(si, ti) for ti in range(18) for si in range(8)]
    rng = random.Random(args.seed)
    result = {'protocol': 'Train-only sampled low-dimensional feedback-conditioned residual on frozen contextual hyper-LoRA; actual official 50-step reward and frozen old-source paired reward baseline; no actor-probability claim',
        'seed': args.seed, 'epochs': args.epochs,
        'max_rollouts': args.max_rollouts,
        'sigma': args.sigma, 'lr': args.lr,
        'kl_weight': args.kl_weight, 'code_dim': args.code_dim,
        'review_sha256': file_hash(args.review),
        'audit_sha256': file_hash(args.audit),
        'features_sha256': file_hash(args.features),
        'checkpoint_sha256': file_hash(args.checkpoint),
        'base_observations_sha256': features['base_observations_sha256'],
        'train': [], 'failures': []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    completed = False
    try:
        for epoch in range(args.epochs):
            rng.shuffle(pair_ids)
            for source_id, target_id in pair_ids:
                if len(result['train']) >= args.max_rollouts:
                    completed = True
                    break
                target = targets[target_id]
                inp = (features['global'][source_id].to(args.device),
                       deltas[source_id].to(args.device),
                       features['targets'][target_id].to(args.device))
                mean = correction(*inp)
                sampled = (mean.detach() + args.sigma * torch.randn_like(mean))
                shift = sampled @ basis.T
                with ResidualInjection(agent, shift):
                    episode = run_episode(agent, tokenizer,
                        args.data_root / target['game'], fields[source_id],
                        adapter=True, device=args.device, max_steps=50,
                        max_new_tokens=64, constrain_actions=True,
                        actor_history_turns=2, loop_guard_max=2)
                if (episode['status'] != 'complete' or
                        episode['initial_observation'] !=
                        matrix['base'][target_id]['episode']['initial_observation']):
                    result['failures'].append({'epoch': epoch,
                        'source_id': source_id, 'target_id': target_id,
                        'episode': episode})
                    save(args.output, result)
                    raise RuntimeError('Failed or changed ALFWorld reward sample')
                baseline = float(prior[target_id, source_id])
                reward = float(episode['reward'])
                loss = (latent_adapter_reinforce(mean, sampled, args.sigma,
                    reward, baseline) + args.kl_weight * mean.square().sum() /
                    (2 * args.sigma * args.sigma))
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(correction.parameters(), 1.0)
                optimizer.step()
                result['train'].append({'epoch': epoch,
                    'source_id': source_id, 'target_id': target_id,
                    'input_content_sha256': json.loads(args.review.read_text())
                        ['pair_bindings'][source_id * 18 + target_id]
                        ['input_content_sha256'],
                    'baseline_reward': baseline,
                    'sampled_reward': reward,
                    'sampled_code': sampled.detach().cpu().tolist(),
                    'mean_code_norm': float(mean.detach().norm()),
                    'steps': episode['steps'],
                    'loss': float(loss.detach())})
                save(args.output, result)
                print(json.dumps({'rollouts': len(result['train']),
                    'sampled_reward': reward, 'baseline_reward': baseline,
                    'failures': len(result['failures'])}), flush=True)
            if completed:
                break
        completed = True
    finally:
        if completed:
            args.save_model.parent.mkdir(parents=True, exist_ok=True)
            torch.save({'correction': correction.cpu().state_dict(),
                'basis': basis.detach().cpu(), 'config': {key: result[key]
                for key in ('protocol', 'seed', 'epochs', 'max_rollouts',
                    'sigma', 'lr', 'kl_weight', 'code_dim', 'review_sha256',
                    'audit_sha256', 'features_sha256',
                    'checkpoint_sha256')}}, args.save_model)
            result['trained_model_sha256'] = file_hash(args.save_model)
        save(args.output, result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, default=Path('current_work/delta-Mem/model/Qwen3-4B-Instruct-2507'))
    parser.add_argument('--checkpoint', type=Path, default=Path('results/trajectory_hyperlora/alf_sibling_simple40_onpolicy600_20261006.pt'))
    parser.add_argument('--source-report-seq0', type=Path, default=Path('results/trajectory_hyperlora/alf_online_context_vector_reward_gate_train_seq0_6_20261007.json'))
    parser.add_argument('--source-report-seq6', type=Path, default=Path('results/trajectory_hyperlora/alf_online_context_vector_reward_gate_train_seq6_6_20261007.json'))
    parser.add_argument('--plan', type=Path, default=Path('ttcl/results/experience_evolution/alfworld_delta_20260922/plan.json'))
    parser.add_argument('--data-root', type=Path, default=Path('ttcl/data/alfworld_delta'))
    parser.add_argument('--target-policy', choices=('train18',), default='train18')
    parser.add_argument('--checkpoint-training-candidates', type=Path, default=Path('results/trajectory_hyperlora/alf_sibling_train600_candidates_20261006.json'))
    parser.add_argument('--warmstart-candidates', type=Path, default=Path('results/trajectory_hyperlora/alf_next_task_candidates_20261005.json'))
    parser.add_argument('--prior-train-review', type=Path, default=Path('data/annotations/alf_own_success_future_reward_8x18_reviewed_20261007.json'))
    parser.add_argument('--review', type=Path, default=Path('data/annotations/alf_own_success_future_reward_8x18_reviewed_20261007.json'))
    parser.add_argument('--matrix', type=Path, default=Path('results/trajectory_hyperlora/alf_own_success_future_reward_8x18_20261007.json'))
    parser.add_argument('--audit', type=Path, default=Path('results/trajectory_hyperlora/alf_own_success_future_reward_8x18_lineage_audited_20261007.json'))
    parser.add_argument('--features', type=Path, default=Path('results/trajectory_hyperlora/alf_future_reward_train8x18_features_20261007.pt'))
    parser.add_argument('--output', type=Path, default=Path('results/trajectory_hyperlora/alf_feedback_residual_train18_pilot_20261007.json'))
    parser.add_argument('--save-model', type=Path, default=Path('results/trajectory_hyperlora/alf_feedback_residual_train18_pilot_20261007.pt'))
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--max-rollouts', type=int, default=36)
    parser.add_argument('--code-dim', type=int, default=8)
    parser.add_argument('--sigma', type=float, default=.5)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--kl-weight', type=float, default=.001)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--gpu-fraction', type=float, default=.65)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
