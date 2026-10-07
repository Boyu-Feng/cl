"""Audit ancestry-exposed reviewed action likelihood diagnostics."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from ttcl.trajectory_hyperlora.alfworld_online_from_empty import file_hash


def audit(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    run = json.loads(args.report.read_text())
    plan = json.loads(args.pairs.read_text())
    training = json.loads(args.training_report.read_text())
    parent = json.loads(args.training_audit.read_text())
    if (run['failures'] or len(run['rows']) != 42 or
            run['pairs_review_sha256'] != file_hash(args.pairs) or
            run['training_audit_sha256'] !=
                file_hash(args.training_audit) or
            run['residual_sha256'] != file_hash(args.residual) or
            parent['report_sha256'] != file_hash(args.training_report) or
            training['trained_model_sha256'] != file_hash(args.residual)):
        raise ValueError('Changed soft-distillation action-likelihood proxy')
    expected = [pair for pair in plan['pairs'] if pair['split']=='dev']
    arms = ('base', 'old_lora', 'raw_text', 'generated_lora')
    for index, (row, pair) in enumerate(zip(run['rows'], expected,
                                            strict=True)):
        if (row['input_content_sha256'] !=
                pair['input_content_sha256'] or
                row['source_game'] != pair['source_game'] or
                row['target_game'] != pair['target_game'] or
                row['target_turn'] != pair['target_turn'] or
                row['action_tokens'] < 1 or
                not math.isfinite(row['code_norm']) or
                set(row['ce']) != set(arms) or
                any(not math.isfinite(row['ce'][arm]) or
                    row['ce'][arm] < 0 for arm in arms)):
            raise ValueError(f'Changed reviewed proxy target {index}')
    summary = {'pairs': 42, 'ancestor_exposed': True,
        'mean_ce': {arm: sum(row['ce'][arm] for row in run['rows'])/42
                    for arm in arms},
        'generated_better_than_old_pairs': sum(
            row['ce']['generated_lora'] < row['ce']['old_lora']
            for row in run['rows']),
        'generated_worse_than_old_pairs': sum(
            row['ce']['generated_lora'] > row['ce']['old_lora']
            for row in run['rows'])}
    result = {'protocol': 'Content-bound but ancestry-exposed offline token CE diagnostic; not official environment reward',
        'report_sha256': file_hash(args.report),
        'training_audit_sha256': file_hash(args.training_audit),
        'pairs_review_sha256': file_hash(args.pairs),
        'summary': summary}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--report', type=Path, default=Path(
        'results/trajectory_hyperlora/alf_self_future_soft_distill_v1_dev42_20261007.json'))
    parser.add_argument('--pairs', type=Path, default=Path(
        'data/annotations/alf_self_future_pairs_v2_checkpoint_disjoint_reviewed_20261007.json'))
    parser.add_argument('--training-report', type=Path, default=Path(
        'results/trajectory_hyperlora/alf_self_future_soft_distill_v1_seed42_64_20261007.json'))
    parser.add_argument('--training-audit', type=Path, default=Path(
        'results/trajectory_hyperlora/alf_self_future_soft_distill_v1_seed42_64_audited_20261007.json'))
    parser.add_argument('--residual', type=Path, default=Path(
        'results/trajectory_hyperlora/alf_self_future_soft_distill_v1_seed42_64_20261007.pt'))
    parser.add_argument('--output', type=Path, default=Path(
        'results/trajectory_hyperlora/alf_self_future_soft_distill_v1_dev42_audited_20261007.json'))
    audit(parser.parse_args())
