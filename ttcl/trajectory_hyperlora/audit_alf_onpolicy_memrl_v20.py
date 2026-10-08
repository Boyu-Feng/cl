"""Replay native MemRL's own ALFWorld chain and audit its memory snapshots."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from transformers import AutoTokenizer

from ttcl.trajectory_hyperlora.alfworld_online_from_empty import (
    digest, file_hash, records_from_episode,
)
from ttcl.trajectory_hyperlora.audit_alf_action_sensitive_reward_v5 import replay
from ttcl.trajectory_hyperlora.evaluate_xland_memrl import plan as native_plan
from ttcl.trajectory_hyperlora.freeze_alf_onpolicy_memrl_v20 import expected


def audit(args):
    if args.audit_output.exists():
        raise FileExistsError(args.audit_output)
    review = json.loads(args.review.read_text())
    report = json.loads(args.report.read_text())
    if (review != expected(args) or not 0 <= args.order < 3 or
            report['plan_sha256'] != file_hash(args.plan) or
            report['parent_review_sha256'] != file_hash(args.parent_review) or
            report['review_sha256'] != file_hash(args.review) or
            report['checkpoint_sha256'] != file_hash(args.checkpoint) or
            report['memory_plan'] != native_plan(args) or
            report['order'] != args.order or
            report['max_steps'] != 50 or report['max_new_tokens'] != 64 or
            report['actor_history_turns'] != 2 or
            report['loop_guard_max'] != 2 or report['failures'] or
            len(report['rows']) != 30):
        raise ValueError('Changed or incomplete own-trajectory MemRL order')
    targets = [row for row in review['targets'] if row['order_id'] == args.order]
    tokenizer = AutoTokenizer.from_pretrained(str(args.model),
                                              local_files_only=True)
    prior_hashes = []
    previous_snapshot_hash = None
    previous_items = set()
    previous_calls = 0
    successes = retrieval_hits = 0
    for position, (binding, row) in enumerate(zip(targets, report['rows'], strict=True)):
        if (row['position'] != position or
                row['family'] != binding['family'] or
                row['game'] != binding['game'] or
                row['game_sha256'] != binding['game_sha256'] or
                row['target_binding_sha256'] != binding['input_content_sha256'] or
                row['prior_success_records_sha256'] != prior_hashes or
                row['prior_success_chain_sha256'] != digest(prior_hashes) or
                row['prior_memory_snapshot_sha256'] !=
                    previous_snapshot_hash or
                row['input_content_sha256'] != digest({
                    'target_binding_sha256': binding['input_content_sha256'],
                    'prior_success_records_sha256': prior_hashes,
                    'prior_memory_snapshot_sha256': previous_snapshot_hash}) or
                file_hash(args.data_root / binding['game']) !=
                    binding['game_sha256']):
            raise ValueError('Changed target or causal MemRL input binding')
        retrieved = row['retrieved']
        context = retrieved['context']
        initial = row['episode']['initial_observation']
        if (retrieved['query'] != initial or
                retrieved['tokens'] != len(tokenizer.encode(
                    context, add_special_tokens=False)) or
                retrieved['tokens'] > args.memory_tokens or
                not set(retrieved['ids']).issubset(previous_items) or
                retrieved['context_sha256'] != hashlib.sha256(
                    context.encode()).hexdigest() or
                row['episode']['memory_sha256'] != hashlib.sha256(
                    context.encode()).hexdigest()):
            raise ValueError('Changed retrieval or actor prompt')
        replay(args.data_root / binding['game'], row['episode'], initial)
        retrieval_hits += bool(retrieved['ids'])
        success = bool(row['episode']['reward'])
        successes += success
        if row['memory_write'] is not success:
            raise ValueError('Changed official-success source eligibility')
        if success:
            records = records_from_episode(row['episode'])
            public_trace = '\n'.join(json.dumps(record,
                ensure_ascii=False) for record in records)
            if (row['own_success_records'] != records or
                    row['own_success_records_sha256'] != digest(records) or
                    row['public_trace_sha256'] != digest(public_trace) or
                    row['memory_update']['input_binding'] !=
                        row['input_content_sha256'] or
                    row['writer_calls_after'] <= previous_calls):
                raise ValueError('Changed native writer source or call')
            prior_hashes.append(digest(records))
        elif (row['writer_calls_after'] != previous_calls or
              'memory_update' in row or 'own_success_records' in row):
            raise ValueError('Failed episode altered success-only writer')
        if (row['prior_success_records_sha256_after'] != prior_hashes or
                row['source_count_after'] != len(prior_hashes)):
            raise ValueError('Changed causal MemRL success chronology')
        snapshot = args.directory / f'memory_after_{position:03d}.json'
        state = json.loads(snapshot.read_text())
        if (row['memory_snapshot_sha256'] != file_hash(snapshot) or
                state['calls'] != row['writer_calls_after'] or
                state['input_tokens'] != row['writer_input_tokens_after'] or
                state['output_tokens'] != row['writer_output_tokens_after'] or
                state['limit_hits'] != row['writer_limit_hits_after'] or
                state['errors'] or not 0 <= len(state['items']) <= successes or
                success and row['memory_update']['new_memory_id'] not in
                    state['items']):
            raise ValueError('Changed native writer state or saved snapshot')
        previous_snapshot_hash = file_hash(snapshot)
        previous_items = set(state['items'])
        previous_calls = row['writer_calls_after']
    value = {'protocol': 'Original-environment replay and content-bound provenance audit of native success-gated MemRL independently collected on-policy history',
        'review_sha256': file_hash(args.review),
        'raw_report_sha256': file_hash(args.report),
        'order': args.order, 'episodes': 30,
        'successes': successes, 'writes': len(prior_hashes),
        'retrieval_hits': retrieval_hits,
        'writer_calls': previous_calls,
        'writer_input_tokens': state['input_tokens'],
        'writer_output_tokens': state['output_tokens'],
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
    parser.add_argument('--plan', type=Path, default=Path(
        'ttcl/trajectory_hyperlora/alf_valid_unseen_multiorder_v10_plan.json'))
    parser.add_argument('--parent-review', type=Path, default=Path(
        'data/annotations/alf_onpolicy_multiorder_v19_reviewed_20261008.json'))
    parser.add_argument('--review', type=Path, default=Path(
        'data/annotations/alf_onpolicy_memrl_v20_reviewed_20261008.json'))
    parser.add_argument('--data-root', type=Path, default=Path(
        'ttcl/data/alfworld_delta'))
    parser.add_argument('--upstream', type=Path, default=Path(
        'current_work/MemRL'))
    parser.add_argument('--embedding', type=Path, default=Path(
        'models/embedding/bge-m3'))
    parser.add_argument('--memory-tokens', type=int, default=2048)
    parser.add_argument('--writer-tokens', type=int, default=256)
    parser.add_argument('--retrieval-threshold', type=float, default=.5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--order', type=int, required=True)
    parser.add_argument('--directory', type=Path)
    parser.add_argument('--report', type=Path)
    parser.add_argument('--audit-output', type=Path)
    args = parser.parse_args()
    if args.directory is None:
        args.directory = Path(f'results/trajectory_hyperlora/alf_onpolicy_memrl_v20_order{args.order}_store_20261008')
    if args.report is None:
        args.report = Path(f'results/trajectory_hyperlora/alf_onpolicy_memrl_v20_order{args.order}_20261008.json')
    if args.audit_output is None:
        args.audit_output = Path(f'results/trajectory_hyperlora/alf_onpolicy_memrl_v20_order{args.order}_audited_20261008.json')
    audit(args)
