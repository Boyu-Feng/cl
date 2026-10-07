"""Audit the source/target lineage and numerical state of the distillation pilot."""

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
    ancestry = json.loads(args.ancestry_audit.read_text())
    model = torch.load(args.model, map_location='cpu', weights_only=True)
    if (run['failures'] or len(run['train']) != run['steps'] or
            run['pairs_review_sha256'] != file_hash(args.pairs) or
            run['ancestry_audit_sha256'] !=
                file_hash(args.ancestry_audit) or
            run['features_sha256'] != file_hash(args.features) or
            run['trained_model_sha256'] != file_hash(args.model) or
            ancestry['pairs_review_sha256'] != file_hash(args.pairs) or
            ancestry['summary']['independent_of_complete_checkpoint'] or
            model['model_kind'] != 'self_future_soft_distill_v1' or
            model['config']['pairs_review_sha256'] !=
                file_hash(args.pairs) or
            model['config']['features_sha256'] !=
                file_hash(args.features)):
        raise ValueError('Changed self-future distillation lineage')
    allowed = {row['input_content_sha256']: row for row in plan['pairs']
               if row['split'] == 'train'}
    seen = set()
    for index, row in enumerate(run['train']):
        pair = allowed.get(row['input_content_sha256'])
        if (pair is None or row['input_content_sha256'] in seen or
                row['step'] != index or
                row['source_game'] != pair['source_game'] or
                row['target_game'] != pair['target_game'] or
                row['target_turn'] != pair['target_turn'] or
                not 0 < row['student_prompt_tokens'] <= 2560 or
                not 0 < row['teacher_prompt_tokens'] <= 2560 or
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
        'ancestor_exposed_internal_dev': True}
    result = {'protocol': 'Content-bound offline soft-distillation training audit; no independent outcome or environment reward',
        'report_sha256': file_hash(args.report),
        'model_sha256': file_hash(args.model),
        'pairs_review_sha256': file_hash(args.pairs),
        'ancestry_audit_sha256': file_hash(args.ancestry_audit),
        'summary': summary}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--report', type=Path, default=Path(
        'results/trajectory_hyperlora/alf_self_future_soft_distill_v1_seed42_64_20261007.json'))
    parser.add_argument('--model', type=Path, default=Path(
        'results/trajectory_hyperlora/alf_self_future_soft_distill_v1_seed42_64_20261007.pt'))
    parser.add_argument('--pairs', type=Path, default=Path(
        'data/annotations/alf_self_future_pairs_v2_checkpoint_disjoint_reviewed_20261007.json'))
    parser.add_argument('--ancestry-audit', type=Path, default=Path(
        'results/trajectory_hyperlora/alf_self_future_pairs_v2_ancestry_audited_20261007.json'))
    parser.add_argument('--features', type=Path, default=Path(
        'results/trajectory_hyperlora/alf_self_future_base57_features_20261007.pt'))
    parser.add_argument('--output', type=Path, default=Path(
        'results/trajectory_hyperlora/alf_self_future_soft_distill_v1_seed42_64_audited_20261007.json'))
    audit(parser.parse_args())
