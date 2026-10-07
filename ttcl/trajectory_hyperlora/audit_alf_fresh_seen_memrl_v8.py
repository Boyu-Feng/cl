"""Replay same-source native MemRL online comparison and memory lineage."""
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
from ttcl.trajectory_hyperlora.evaluate_alf_fresh_seen_memrl_v8 import expected
from ttcl.trajectory_hyperlora.evaluate_xland_memrl import plan as native_plan


def audit(args):
    if args.audit_output.exists():
        raise FileExistsError(args.audit_output)
    review = json.loads(args.memrl_review.read_text())
    if review != expected(args):
        raise ValueError('Changed MemRL content-bound target review')
    report = json.loads(args.output.read_text())
    parent = json.loads(args.base_result.read_text())
    if (report['plan_sha256'] != file_hash(args.plan) or
            report['parent_review_sha256'] != file_hash(args.review) or
            report['memrl_review_sha256'] != file_hash(args.memrl_review) or
            report['base_report_sha256'] != file_hash(args.base_result) or
            report['base_audit_sha256'] != file_hash(args.base_audit) or
            report['checkpoint_sha256'] != file_hash(args.checkpoint) or
            report['memory_plan'] != native_plan(args) or
            report['max_steps'] != 50 or
            report['max_new_tokens'] != 64 or
            report['actor_history_turns'] != 2 or
            report['loop_guard_max'] != 2 or
            report['failures'] or len(report['games']) != 36):
        raise ValueError('Changed or incomplete native MemRL run')
    tokenizer = AutoTokenizer.from_pretrained(str(args.model),
                                              local_files_only=True)
    prior = []
    base_total = text_total = memrl_total = 0.
    retrieval_hits = memory_writes = 0
    writer_calls_before = 0
    for target, source, row in zip(review['targets'], parent['games'],
                                   report['games'], strict=True):
        if (row['index'] != target['index'] or
                row['family'] != target['family'] or
                row['game'] != target['game'] or
                row['game_sha256'] != target['game_sha256'] or
                file_hash(args.data_root / target['game']) !=
                    target['game_sha256'] or
                row['input_content_sha256'] !=
                    target['input_content_sha256'] or
                row['parent_target_binding'] !=
                    target['parent_target_binding'] or
                row['base_episode_sha256'] !=
                    digest(source['base']) or
                row['prior_success_records_sha256'] != prior or
                row['prior_success_count'] != len(prior) or
                row['base_reward'] != source['base']['reward'] or
                row['raw_text_reward'] != source['raw_text']['reward']):
            raise ValueError('Changed official task or matched source chronology')
        retrieved = row['retrieved']
        context = retrieved['context']
        if (retrieved['query'] != source['base']['initial_observation'] or
                retrieved['tokens'] != len(tokenizer.encode(
                    context, add_special_tokens=False)) or
                retrieved['tokens'] > 2048 or
                retrieved['context_sha256'] !=
                    hashlib.sha256(context.encode()).hexdigest()):
            raise ValueError('Changed retrieved text or actor prompt budget')
        reward = replay(args.data_root / target['game'],
                        row['memrl'], source['base']['initial_observation'])
        base_total += row['base_reward']
        text_total += row['raw_text_reward']
        memrl_total += reward
        retrieval_hits += bool(retrieved['ids'])
        if source['base']['reward']:
            records = records_from_episode(source['base'])
            public_trace = '\n'.join(json.dumps(record,
                ensure_ascii=False) for record in records)
            if (row.get('new_source_records_sha256') != digest(records) or
                    row.get('public_trace_sha256') != digest(public_trace) or
                    row.get('memory_update', {}).get('input_binding') !=
                        target['input_content_sha256']):
                raise ValueError('Changed native writer input or source binding')
            prior.append(digest(records))
            memory_writes += 1
            if row['writer_calls_after'] <= writer_calls_before:
                raise ValueError('Source success did not call native writer')
        elif ('new_source_records_sha256' in row or
              'memory_update' in row or
              row['writer_calls_after'] != writer_calls_before):
            raise ValueError('Failure entered success-only memory')
        if row['source_count_after'] != len(prior):
            raise ValueError('Changed memory update schedule')
        snapshot = args.directory / f'memory_after_{target["index"]:03d}.json'
        state = json.loads(snapshot.read_text())
        if (row['memory_snapshot_sha256'] != file_hash(snapshot) or
                state['calls'] != row['writer_calls_after'] or
                state['input_tokens'] != row['writer_input_tokens_after'] or
                state['output_tokens'] != row['writer_output_tokens_after'] or
                state['limit_hits'] != row['writer_limit_hits_after'] or
                state['errors'] or len(state['items']) != memory_writes or
                source['base']['reward'] and
                    row['memory_update']['new_memory_id'] not in
                        state['items']):
            raise ValueError('Changed native memory snapshot or writer accounting')
        writer_calls_before = row['writer_calls_after']
    summary = {'games': 36, 'base': base_total,
        'raw_text': text_total, 'native_memrl': memrl_total,
        'retrieval_hits': retrieval_hits,
        'shared_success_sources': len(prior),
        'native_memory_writes': memory_writes,
        'writer_calls': writer_calls_before,
        'failed_replays': 0}
    value = {'protocol': 'Independent original ALFWorld replay of shared-own-history native MemRL target episodes, target/source/content bindings, retrieved prompt budget and every persisted native memory snapshot',
             'raw_report_sha256': file_hash(args.output),
             'review_sha256': file_hash(args.memrl_review),
             'base_audit_sha256': file_hash(args.base_audit),
             'summary': summary}
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.write_text(json.dumps(value, ensure_ascii=False,
                                            indent=2)+'\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, default=Path(
        'current_work/delta-Mem/model/Qwen3-4B-Instruct-2507'))
    parser.add_argument('--checkpoint', type=Path, default=Path(
        'results/trajectory_hyperlora/alf_sibling_simple40_onpolicy600_20261006.pt'))
    parser.add_argument('--plan', type=Path, default=Path(
        'ttcl/trajectory_hyperlora/alf_fresh_seen36_v6_plan.json'))
    parser.add_argument('--review', type=Path, default=Path(
        'data/annotations/alf_fresh_seen36_online_v6_reviewed_20261007.json'))
    parser.add_argument('--base-result', type=Path, default=Path(
        'results/trajectory_hyperlora/alf_fresh_seen36_shared_online_v6_20261007.json'))
    parser.add_argument('--base-audit', type=Path, default=Path(
        'results/trajectory_hyperlora/alf_fresh_seen36_shared_online_v6_audited_20261007.json'))
    parser.add_argument('--memrl-review', type=Path, default=Path(
        'data/annotations/alf_fresh_seen36_memrl_v8_reviewed_20261007.json'))
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
    parser.add_argument('--directory', type=Path, default=Path(
        'results/trajectory_hyperlora/alf_fresh_seen36_memrl_v8_store_20261007'))
    parser.add_argument('--output', type=Path, default=Path(
        'results/trajectory_hyperlora/alf_fresh_seen36_memrl_v8_20261007.json'))
    parser.add_argument('--audit-output', type=Path, default=Path(
        'results/trajectory_hyperlora/alf_fresh_seen36_memrl_v8_audited_20261007.json'))
    audit(parser.parse_args())
