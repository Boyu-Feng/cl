"""Independently replay shared-history official ALFWorld online comparison."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from transformers import AutoTokenizer

from ttcl.trajectory_hyperlora.alfworld_online_from_empty import (
    digest, file_hash, records_from_episode,
)
from ttcl.trajectory_hyperlora.alfworld_online_context_vector_mean_v1 import vector_hash
from ttcl.trajectory_hyperlora.audit_alf_action_sensitive_reward_v5 import replay
from ttcl.trajectory_hyperlora.evaluate_alf_fresh_seen_online_v6 import (
    bounded_history_text, checked_inputs,
)


def audit(args):
    if args.audit_output.exists():
        raise FileExistsError(args.audit_output)
    review, _ = checked_inputs(args)
    report = json.loads(args.output.read_text())
    if (report['plan_sha256'] != file_hash(args.plan) or
            report['review_sha256'] != file_hash(args.review) or
            report['checkpoint_sha256'] != file_hash(args.checkpoint) or
            report['residual_sha256'] != file_hash(args.residual) or
            report['training_report_sha256'] !=
                file_hash(args.training_report) or
            report['training_audit_sha256'] !=
                file_hash(args.training_audit) or
            report['source_tokens'] != 2048 or
            report['max_steps'] != 50 or
            report['max_new_tokens'] != 64 or
            report['actor_history_turns'] != 2 or
            report['loop_guard_max'] != 2 or
            report['failures'] or len(report['games']) != 36):
        raise ValueError('Changed or incomplete prospective online comparison')
    tokenizer = AutoTokenizer.from_pretrained(str(args.model),
                                              local_files_only=True)
    prior = []
    source_records = []
    totals = {arm: 0. for arm in ('base', 'continuous',
                                  'fixed_first', 'raw_text')}
    paired = {'continuous_only_vs_fixed': 0,
              'fixed_only_vs_continuous': 0,
              'continuous_only_vs_text': 0,
              'text_only_vs_continuous': 0}
    executed_episodes = 0
    previous_vector = vector_hash(None)
    for target, row in zip(review['targets'], report['games'], strict=True):
        if (row['index'] != target['index'] or
                row['family'] != target['family'] or
                row['game'] != target['game'] or
                row['game_sha256'] != target['game_sha256'] or
                file_hash(args.data_root / target['game']) !=
                    target['game_sha256'] or
                row['input_content_sha256'] !=
                    target['input_content_sha256'] or
                row['prior_success_count'] != len(prior) or
                row['prior_source_records_sha256'] != prior or
                row['prior_source_chain_sha256'] != digest(prior) or
                row['source_vector_sha256'] != previous_vector):
            raise ValueError('Changed game, past history, or source-vector chain')
        if any(not math.isfinite(row[f'{arm}_code_norm'])
               for arm in ('continuous', 'fixed_first')):
            raise ValueError('Nonfinite generated code')
        if not prior and (row['continuous'] != row['base'] or
                          row['fixed_first'] != row['base'] or
                          row['raw_text'] != row['base'] or
                          row['continuous_code_norm'] != 0 or
                          row['fixed_first_code_norm'] != 0):
            raise ValueError('First empty-memory episode changed actor')
        if (row['raw_text_retained_tokens'] > 2048 or
                row['raw_text_retained_tokens'] < 0 or
                row['raw_text_original_tokens'] <
                    row['raw_text_retained_tokens']):
            raise ValueError('Changed raw-text history budget')
        if source_records:
            bounded, original, retained = bounded_history_text(
                tokenizer, [{'records': records} for records in source_records],
                2048)
            if (row['raw_text_sha256'] != digest(bounded) or
                    row['raw_text_original_tokens'] != original or
                    row['raw_text_retained_tokens'] != retained):
                raise ValueError('Changed same-source raw-history text')
        elif (row['raw_text_sha256'] != digest('') or
              row['raw_text_original_tokens'] != 0 or
              row['raw_text_retained_tokens'] != 0):
            raise ValueError('Empty history has nonempty text')
        initial = row['base']['initial_observation']
        for arm in totals:
            totals[arm] += replay(args.data_root / target['game'],
                                  row[arm], initial)
        executed_episodes += 4 if row['prior_success_count'] else 1
        paired['continuous_only_vs_fixed'] += (
            row['continuous']['reward'] > row['fixed_first']['reward'])
        paired['fixed_only_vs_continuous'] += (
            row['continuous']['reward'] < row['fixed_first']['reward'])
        paired['continuous_only_vs_text'] += (
            row['continuous']['reward'] > row['raw_text']['reward'])
        paired['text_only_vs_continuous'] += (
            row['continuous']['reward'] < row['raw_text']['reward'])
        if row['base']['reward']:
            records = records_from_episode(row['base'])
            source_hash = digest(records)
            if row.get('new_source_records_sha256') != source_hash:
                raise ValueError('Changed self-success history write')
            prior.append(source_hash)
            source_records.append(records)
        elif 'new_source_records_sha256' in row:
            raise ValueError('Failure entered success-gated source memory')
        if row['source_count_after'] != len(prior):
            raise ValueError('Changed source update count')
        previous_vector = row['source_vector_sha256_after']
    summary = {'games': 36, **totals, **paired,
               'shared_success_sources': len(prior),
               'audited_arm_records': 144,
               'executed_episodes': executed_episodes,
               'failed_replays': 0}
    value = {'protocol': 'Independent original ALFWorld replay of every frozen fresh valid_seen online arm, paired game hashes and common own-success source chronology',
             'raw_report_sha256': file_hash(args.output),
             'review_sha256': file_hash(args.review),
             'residual_sha256': file_hash(args.residual),
             'summary': summary}
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.write_text(json.dumps(value, ensure_ascii=False,
                                            indent=2) + '\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, default=Path(
        'current_work/delta-Mem/model/Qwen3-4B-Instruct-2507'))
    parser.add_argument('--checkpoint', type=Path, default=Path(
        'results/trajectory_hyperlora/alf_sibling_simple40_onpolicy600_20261006.pt'))
    parser.add_argument('--residual', type=Path, default=Path(
        'results/trajectory_hyperlora/alf_action_sensitive_reward_train72_v5_20261007.pt'))
    parser.add_argument('--training-report', type=Path, default=Path(
        'results/trajectory_hyperlora/alf_action_sensitive_reward_train72_v5_20261007.json'))
    parser.add_argument('--training-audit', type=Path, default=Path(
        'results/trajectory_hyperlora/alf_action_sensitive_reward_train72_v5_audited_20261007.json'))
    parser.add_argument('--plan', type=Path, default=Path(
        'ttcl/trajectory_hyperlora/alf_fresh_seen36_v6_plan.json'))
    parser.add_argument('--review', type=Path, default=Path(
        'data/annotations/alf_fresh_seen36_online_v6_reviewed_20261007.json'))
    parser.add_argument('--data-root', type=Path, default=Path(
        'ttcl/data/alfworld_delta'))
    parser.add_argument('--output', type=Path, default=Path(
        'results/trajectory_hyperlora/alf_fresh_seen36_shared_online_v6_20261007.json'))
    parser.add_argument('--audit-output', type=Path, default=Path(
        'results/trajectory_hyperlora/alf_fresh_seen36_shared_online_v6_audited_20261007.json'))
    audit(parser.parse_args())
