"""Audit own-only-success offline LoRA distillation bindings and weights."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ttcl.trajectory_hyperlora.alfworld_online_from_empty import file_hash


def audit(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    run = json.loads(args.report.read_text())
    plan = json.loads(args.pairs.read_text())
    pair_audit = json.loads(args.pair_audit.read_text())
    model = torch.load(args.model, map_location='cpu', weights_only=True)
    if (run['failures'] or len(run['train']) != run['steps'] or
            run['pairs_review_sha256'] != file_hash(args.pairs) or
            run['target_bank_review_sha256'] != file_hash(args.target_bank) or
            run['features_sha256'] != file_hash(args.features) or
            run['trained_model_sha256'] != file_hash(args.model) or
            pair_audit['pairs_review_sha256'] != file_hash(args.pairs) or
            pair_audit['summary']['different_game_action_pairs'] != 234 or
            model['model_kind'] != 'own_only_soft_distill_v2' or
            model['config']['pairs_review_sha256'] !=
                file_hash(args.pairs) or
            model['config']['features_sha256'] !=
                file_hash(args.features)):
        raise ValueError('Changed self-future distillation lineage')
    allowed = {row['input_content_sha256']: row for row in plan['pairs']}
    seen = set()
    for index, row in enumerate(run['train']):
        pair = allowed.get(row['input_content_sha256'])
        if (pair is None or row['input_content_sha256'] in seen or
                row['step'] != index or
                row['source_game'] != pair['source_game'] or
                row['target_game'] != pair['target_game'] or
                row['target_turn'] != pair['target_turn'] or
                not 0 < row['student_prompt_tokens'] <= 4096 or
                not 0 < row['teacher_prompt_tokens'] <= 4096 or
                row['answer_tokens'] < 1 or
                not all(torch.isfinite(torch.tensor(row[key])) for key in
                        ('ce', 'kl', 'loss', 'code_norm', 'grad_norm'))):
            raise ValueError(f'Changed offline action target {index}')
        seen.add(row['input_content_sha256'])
    basis = model['basis'].float()
    if (basis.shape != (128, run['code_dim']) or
            not torch.allclose(basis.T @ basis,
                torch.eye(run['code_dim']), atol=1e-4, rtol=1e-4) or
            not all(torch.isfinite(x).all()
                    for x in model['correction'].values())):
        raise ValueError('Invalid frozen soft-distilled LoRA generator')
    summary = {'steps': run['steps'],
        'distinct_train_pairs': len(seen),
        'mean_ce': sum(row['ce'] for row in run['train'])/run['steps'],
        'mean_kl': sum(row['kl'] for row in run['train'])/run['steps'],
        'mean_code_norm': sum(row['code_norm']
            for row in run['train'])/run['steps'],
        'checkpoint_ancestor_exposed_train_only': True}
    result = {'protocol': 'Content-bound offline soft-distillation training audit; no independent outcome or environment reward',
        'report_sha256': file_hash(args.report),
        'model_sha256': file_hash(args.model),
        'pairs_review_sha256': file_hash(args.pairs),
        'pair_audit_sha256': file_hash(args.pair_audit),
        'summary': summary}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--report', type=Path, default=Path(
        'results/trajectory_hyperlora/alf_own_only_soft_distill_v2_seed42_64_retry_20261007.json'))
    parser.add_argument('--model', type=Path, default=Path(
        'results/trajectory_hyperlora/alf_own_only_soft_distill_v2_seed42_64_retry_20261007.pt'))
    parser.add_argument('--pairs', type=Path, default=Path(
        'data/annotations/alf_own_only_future_pairs_reviewed_20261007.json'))
    parser.add_argument('--pair-audit', type=Path, default=Path(
        'results/trajectory_hyperlora/alf_own_only_future_pairs_audited_20261007.json'))
    parser.add_argument('--target-bank', type=Path, default=Path(
        'data/annotations/alf_own_only42_replay_reviewed_20261007.json'))
    parser.add_argument('--features', type=Path, default=Path(
        'results/trajectory_hyperlora/alf_own_only_future_features_v2_20261007.pt'))
    parser.add_argument('--output', type=Path, default=Path(
        'results/trajectory_hyperlora/alf_own_only_soft_distill_v2_seed42_64_retry_audited_20261007.json'))
    audit(parser.parse_args())
