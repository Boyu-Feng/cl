"""Replay and check provenance of a full method-on-policy ALFWorld arm."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from transformers import AutoTokenizer

from ttcl.trajectory_hyperlora.alfworld_online_from_empty import (
    digest, file_hash, records_from_episode,
)
from ttcl.trajectory_hyperlora.alfworld_online_context_vector_mean_v1 import vector_hash
from ttcl.trajectory_hyperlora.audit_alf_action_sensitive_reward_v5 import replay
from ttcl.trajectory_hyperlora.evaluate_alf_fresh_seen_online_v6 import bounded_history_text
from ttcl.trajectory_hyperlora.freeze_alf_onpolicy_v19 import METHODS, expected


def audit(args):
    if args.audit_output.exists():
        raise FileExistsError(args.audit_output)
    review = json.loads(args.review.read_text())
    report = json.loads(args.report.read_text())
    if (review != expected(args) or args.method not in METHODS or
            not 0 <= args.order < 3 or
            report['plan_sha256'] != file_hash(args.plan) or
            report['review_sha256'] != file_hash(args.review) or
            report['checkpoint_sha256'] != file_hash(args.checkpoint) or
            report['residual_sha256'] != file_hash(args.residual) or
            report['order'] != args.order or
            report['method'] != args.method or
            report['max_steps'] != 50 or
            report['max_new_tokens'] != 64 or
            report['actor_history_turns'] != 2 or
            report['loop_guard_max'] != 2 or
            report['source_token_limit'] != 2048 or
            report['failures'] or len(report['rows']) != 30):
        raise ValueError('Changed or incomplete method-on-policy arm')
    targets = [row for row in review['targets']
               if row['order_id'] == args.order and
                  row['method'] == args.method]
    if len(targets) != 30:
        raise ValueError('Changed reviewed target order')
    tokenizer = (AutoTokenizer.from_pretrained(str(args.model),
                 local_files_only=True)
                 if args.method == 'own_success_raw_text' else None)
    prior_hashes = []
    accepted = []
    vector_before = vector_hash(None)
    successes = 0
    writes = 0
    for position, (binding, row) in enumerate(zip(targets, report['rows'], strict=True)):
        if (row['position'] != position or
                row['game'] != binding['game'] or
                row['game_sha256'] != binding['game_sha256'] or
                row['family'] != binding['family'] or
                row['target_binding_sha256'] != binding['input_content_sha256'] or
                row['prior_success_records_sha256'] != prior_hashes or
                row['prior_success_chain_sha256'] != digest(prior_hashes) or
                row['accepted_memory_count_before'] != len(accepted) or
                row['source_vector_sha256_before'] != vector_before or
                row['input_content_sha256'] != digest({
                    'target_binding_sha256': binding['input_content_sha256'],
                    'prior_success_records_sha256': prior_hashes,
                    'accepted_memory_count': len(accepted),
                    'source_vector_sha256': vector_before}) or
                not math.isfinite(row['code_norm'])):
            raise ValueError('Changed target or causal memory input binding')
        should_mount = (args.method in ('continuous_success_mean',
                        'fixed_first_success') and len(accepted) > 0)
        if (row['adapter_mounted'] is not should_mount or
                (not should_mount and row['code_norm'] != 0.)):
            raise ValueError('Changed memory adapter policy')
        if args.method == 'own_success_raw_text' and accepted:
            text, original, retained = bounded_history_text(
                tokenizer, accepted, 2048)
        else:
            text, original, retained = '', 0, 0
        if (row['raw_text'] != text or
                row['raw_text_original_tokens'] != original or
                row['raw_text_retained_tokens'] != retained or
                retained > 2048 or
                row['raw_text_sha256'] != digest(text) or
                row['prompt_memory_sha256'] != hashlib.sha256(
                    text.encode()).hexdigest() or
                row['episode']['memory_sha256'] !=
                    row['prompt_memory_sha256']):
            raise ValueError('Changed raw-text evidence or prompt budget')
        initial = row['episode']['initial_observation']
        replay(args.data_root / binding['game'], row['episode'], initial)
        reward = bool(row['episode']['reward'])
        successes += reward
        should_write = (reward and (args.method in (
            'continuous_success_mean', 'own_success_raw_text') or
            args.method == 'fixed_first_success' and not accepted))
        if row['memory_write'] is not bool(should_write):
            raise ValueError('Changed own-success write eligibility')
        if reward:
            records = records_from_episode(row['episode'])
            record_hash = digest(records)
            if (row['own_success_records'] != records or
                    row['own_success_records_sha256'] != record_hash):
                raise ValueError('Changed own trajectory records')
            prior_hashes.append(record_hash)
            if should_write:
                accepted.append({'records': records})
                writes += 1
        elif ('own_success_records' in row or
              'own_success_records_sha256' in row):
            raise ValueError('Failed episode entered success history')
        if (row['prior_success_records_sha256_after'] != prior_hashes or
                row['accepted_memory_count_after'] != len(accepted)):
            raise ValueError('Changed causal memory state count')
        vector_after = row['source_vector_sha256_after']
        if args.method in ('base', 'own_success_raw_text'):
            if vector_after != vector_hash(None):
                raise ValueError('Unexpected LoRA memory in non-LoRA arm')
        elif not should_write and vector_after != vector_before:
            raise ValueError('LoRA memory changed without eligible write')
        elif should_write and vector_after == vector_before:
            raise ValueError('Eligible write left LoRA vector unchanged')
        vector_before = vector_after
    value = {'protocol': 'Independent original-environment replay and causal own-success source/text-chain audit of all 30 method-on-policy episodes',
        'plan_sha256': file_hash(args.plan),
        'review_sha256': file_hash(args.review),
        'raw_report_sha256': file_hash(args.report),
        'order': args.order, 'method': args.method,
        'episodes': 30, 'successes': successes, 'writes': writes,
        'failed_replays': 0}
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.write_text(json.dumps(value, ensure_ascii=False,
                                           indent=2) + '\n')
    print(json.dumps(value), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, default=Path(
        'current_work/delta-Mem/model/Qwen3-4B-Instruct-2507'))
    parser.add_argument('--checkpoint', type=Path, default=Path(
        'results/trajectory_hyperlora/alf_sibling_simple40_onpolicy600_20261006.pt'))
    parser.add_argument('--residual', type=Path, default=Path(
        'results/trajectory_hyperlora/alf_action_sensitive_reward_train72_v5_20261007.pt'))
    parser.add_argument('--plan', type=Path, default=Path(
        'ttcl/trajectory_hyperlora/alf_valid_unseen_multiorder_v10_plan.json'))
    parser.add_argument('--review', type=Path, default=Path(
        'data/annotations/alf_onpolicy_multiorder_v19_reviewed_20261008.json'))
    parser.add_argument('--data-root', type=Path, default=Path(
        'ttcl/data/alfworld_delta'))
    parser.add_argument('--order', type=int, required=True)
    parser.add_argument('--method', choices=METHODS, required=True)
    parser.add_argument('--report', type=Path)
    parser.add_argument('--audit-output', type=Path)
    args = parser.parse_args()
    if args.report is None:
        args.report = Path(f'results/trajectory_hyperlora/alf_onpolicy_v19_order{args.order}_{args.method}_20261008.json')
    if args.audit_output is None:
        args.audit_output = Path(f'results/trajectory_hyperlora/alf_onpolicy_v19_order{args.order}_{args.method}_audited_20261008.json')
    audit(args)
