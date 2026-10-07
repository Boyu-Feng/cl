"""Replay frozen reward-trained LoRA and raw-text development outcomes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ttcl.trajectory_hyperlora.alfworld_online_from_empty import file_hash
from ttcl.trajectory_hyperlora.alfworld_source_utility_dataset_v2 import checked
from ttcl.trajectory_hyperlora.audit_alf_action_sensitive_reward_v5 import replay


def audit(args):
    if args.audit_output.exists():
        raise FileExistsError(args.audit_output)
    sources, targets, bindings = checked(args)
    report = json.loads(args.output.read_text())
    matrix = json.loads(args.matrix.read_text())
    matrix_audit = json.loads(args.audit.read_text())
    training_audit = json.loads(args.training_audit.read_text())
    if (report['review_sha256'] != file_hash(args.review) or
            report['matrix_sha256'] != file_hash(args.matrix) or
            report['audit_sha256'] != file_hash(args.audit) or
            report['features_sha256'] != file_hash(args.features) or
            report['training_report_sha256'] !=
                file_hash(args.training_report) or
            report['training_audit_sha256'] !=
                file_hash(args.training_audit) or
            report['residual_sha256'] != file_hash(args.residual) or
            report['checkpoint_sha256'] != file_hash(args.checkpoint) or
            matrix_audit['raw_report_sha256'] != file_hash(args.matrix) or
            training_audit['raw_report_sha256'] !=
                file_hash(args.training_report) or
            training_audit['trained_model_sha256'] !=
                file_hash(args.residual) or
            report['source_id'] != args.source_id or
            report['source_records_sha256'] !=
                sources[args.source_id]['records_sha256'] or
            report['source_tokens_retained'] > 2048 or
            report['max_steps'] != 50 or
            report['max_new_tokens'] != 64 or
            report['actor_history_turns'] != 2 or
            report['loop_guard_max'] != 2 or
            report['failures'] or len(report['games']) != 12 or
            len(targets) != 12 or len(bindings) != 96):
        raise ValueError('Changed or incomplete development comparison')
    totals = {'base': 0., 'old_lora': 0.,
              'new_lora': 0., 'raw_text': 0.}
    new_only_vs_old = old_only_vs_new = 0
    new_only_vs_text = text_only_vs_new = 0
    for target, row in zip(targets, report['games'], strict=True):
        ti = target['target_id']
        si = args.source_id
        original = matrix['pairs'][si*12+ti]
        baseline = matrix['base'][ti]['episode']
        if (row['target_id'] != ti or row['game'] != target['game'] or
                row['game_sha256'] != target['game_sha256'] or
                file_hash(args.data_root / target['game']) !=
                    target['game_sha256'] or
                row['source_target_binding'] !=
                    bindings[si*12+ti]['input_content_sha256'] or
                original['input_content_sha256'] !=
                    row['source_target_binding'] or
                row['base_reward'] != baseline['reward'] or
                row['old_lora_reward'] != original['episode']['reward']):
            raise ValueError('Changed paired official task or source')
        initial = baseline['initial_observation']
        new = replay(args.data_root / target['game'], row['new'], initial)
        text = replay(args.data_root / target['game'], row['raw_text'], initial)
        totals['base'] += baseline['reward']
        totals['old_lora'] += row['old_lora_reward']
        totals['new_lora'] += new
        totals['raw_text'] += text
        new_only_vs_old += new > row['old_lora_reward']
        old_only_vs_new += new < row['old_lora_reward']
        new_only_vs_text += new > text
        text_only_vs_new += new < text
    summary = {'targets': 12, **totals,
        'new_only_vs_old': new_only_vs_old,
        'old_only_vs_new': old_only_vs_new,
        'new_only_vs_text': new_only_vs_text,
        'text_only_vs_new': text_only_vs_new,
        'failed_replays': 0}
    result = {'protocol': 'Independent original-ALFWorld replay of new LoRA and raw same-source text on twelve disjoint but previously exposed official train development games; base and old LoRA bound to prior full-matrix audit',
        'raw_report_sha256': file_hash(args.output),
        'matrix_audit_sha256': file_hash(args.audit),
        'training_audit_sha256': file_hash(args.training_audit),
        'residual_sha256': file_hash(args.residual),
        'summary': summary}
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.write_text(json.dumps(result, ensure_ascii=False,
                                             indent=2)+'\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, default=Path('results/trajectory_hyperlora/alf_sibling_simple40_onpolicy600_20261006.pt'))
    parser.add_argument('--residual', type=Path, default=Path('results/trajectory_hyperlora/alf_action_sensitive_reward_train72_v5_20261007.pt'))
    parser.add_argument('--training-report', type=Path, default=Path('results/trajectory_hyperlora/alf_action_sensitive_reward_train72_v5_20261007.json'))
    parser.add_argument('--training-audit', type=Path, default=Path('results/trajectory_hyperlora/alf_action_sensitive_reward_train72_v5_audited_20261007.json'))
    parser.add_argument('--source-report-seq0', type=Path, default=Path('results/trajectory_hyperlora/alf_online_context_vector_reward_gate_train_seq0_6_20261007.json'))
    parser.add_argument('--source-report-seq6', type=Path, default=Path('results/trajectory_hyperlora/alf_online_context_vector_reward_gate_train_seq6_6_20261007.json'))
    parser.add_argument('--plan', type=Path, default=Path('ttcl/results/experience_evolution/alfworld_delta_20260922/plan.json'))
    parser.add_argument('--data-root', type=Path, default=Path('ttcl/data/alfworld_delta'))
    parser.add_argument('--target-policy', choices=('independent12',), default='independent12')
    parser.add_argument('--checkpoint-training-candidates', type=Path, default=Path('results/trajectory_hyperlora/alf_sibling_train600_candidates_20261006.json'))
    parser.add_argument('--warmstart-candidates', type=Path, default=Path('results/trajectory_hyperlora/alf_next_task_candidates_20261005.json'))
    parser.add_argument('--prior-train-review', type=Path, default=Path('data/annotations/alf_own_success_future_reward_8x18_reviewed_20261007.json'))
    parser.add_argument('--review', type=Path, default=Path('data/annotations/alf_own_success_future_reward_holdout_8x12_v3c_reviewed_20261007.json'))
    parser.add_argument('--matrix', type=Path, default=Path('results/trajectory_hyperlora/alf_own_success_future_reward_holdout_8x12_v3c_merged_20261007.json'))
    parser.add_argument('--audit', type=Path, default=Path('results/trajectory_hyperlora/alf_own_success_future_reward_holdout_8x12_v3c_merged_audited_20261007.json'))
    parser.add_argument('--features', type=Path, default=Path('results/trajectory_hyperlora/alf_future_reward_holdout8x12_v3c_features_20261007.pt'))
    parser.add_argument('--output', type=Path, default=Path('results/trajectory_hyperlora/alf_action_sensitive_reward_dev12_v5_20261007.json'))
    parser.add_argument('--audit-output', type=Path, default=Path('results/trajectory_hyperlora/alf_action_sensitive_reward_dev12_v5_audited_20261007.json'))
    parser.add_argument('--source-id', type=int, default=3)
    args = parser.parse_args()
    audit(args)
