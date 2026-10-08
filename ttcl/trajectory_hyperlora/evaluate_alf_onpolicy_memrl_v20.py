"""Success-gated native MemRL on its own from-empty ALFWorld histories."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ttcl.experience_evolution.environment import make_env
from ttcl.memrl_comparison.memory import Embedder, Memory
from ttcl.trajectory_hyperlora.alfworld_online_from_empty import (
    digest, file_hash, records_from_episode,
)
from ttcl.trajectory_hyperlora.alfworld_source_utility_dataset_v2 import save
from ttcl.trajectory_hyperlora.alfworld_zero_shot_probe import load_agent, run_episode
from ttcl.trajectory_hyperlora.evaluate_xland_memrl import LocalQwenClient, plan as native_plan
from ttcl.trajectory_hyperlora.freeze_alf_onpolicy_memrl_v20 import expected


def run(args):
    review = json.loads(args.review.read_text())
    if review != expected(args) or not 0 <= args.order < 3:
        raise ValueError('Changed reviewed native MemRL official order')
    targets = [row for row in review['targets'] if row['order_id'] == args.order]
    if len(targets) != 30:
        raise ValueError('Incomplete order')
    if args.validate_only:
        print(json.dumps({'order': args.order,
            'targets': len(targets),
            'review_sha256': file_hash(args.review)}), flush=True)
        return
    output = args.output or Path(
        f'results/trajectory_hyperlora/alf_onpolicy_memrl_v20_order{args.order}_20261008.json')
    directory = args.directory or Path(
        f'results/trajectory_hyperlora/alf_onpolicy_memrl_v20_order{args.order}_store_20261008')
    if output.exists() or directory.exists():
        raise FileExistsError('Native MemRL requires fresh report and store')
    if not args.upstream.is_dir() or not args.embedding.is_dir():
        raise FileNotFoundError('Missing native MemRL or BGE-M3 asset')
    agent, tokenizer = load_agent(args.model, args.checkpoint,
                                  args.device, args.gpu_fraction)
    client = LocalQwenClient(agent.model, tokenizer, args.device, args.seed)
    plan = native_plan(args)
    memory = Memory(plan, client, directory,
                    {'threshold': args.retrieval_threshold,
                     'mean': 0., 'std': 1.},
                    embedder=Embedder(args.embedding))
    report = {'protocol': 'Frozen official valid_unseen 30-game order, success-gated native MemRL writer/retrieval/Q memory from empty; independent own actions and official-won source trajectories; same Qwen actor and command budget as v19; games previously exposed',
        'plan_sha256': file_hash(args.plan),
        'parent_review_sha256': file_hash(args.parent_review),
        'review_sha256': file_hash(args.review),
        'checkpoint_sha256': file_hash(args.checkpoint),
        'memory_plan': plan,
        'order': args.order,
        'max_steps': 50, 'max_new_tokens': 64,
        'actor_history_turns': 2, 'loop_guard_max': 2,
        'rows': [], 'failures': []}
    output.parent.mkdir(parents=True, exist_ok=True)
    save(output, report)
    prior_hashes = []
    snapshot_hash = None
    for binding in targets:
        entry = {'position': binding['position'],
            'family': binding['family'], 'game': binding['game'],
            'game_sha256': binding['game_sha256'],
            'target_binding_sha256': binding['input_content_sha256'],
            'prior_success_records_sha256': prior_hashes.copy(),
            'prior_success_chain_sha256': digest(prior_hashes),
            'prior_memory_snapshot_sha256': snapshot_hash}
        entry['input_content_sha256'] = digest({
            'target_binding_sha256': binding['input_content_sha256'],
            'prior_success_records_sha256': prior_hashes,
            'prior_memory_snapshot_sha256': snapshot_hash})
        try:
            game = args.data_root / binding['game']
            if file_hash(game) != binding['game_sha256']:
                raise ValueError('Changed reviewed official game')
            env = make_env(game)
            try:
                initial = str(env.reset()['feedback'])
            finally:
                env.close()
            retrieved = memory.retrieve(initial)
            episode = run_episode(agent, tokenizer, game, {},
                adapter=False, device=args.device, max_steps=50,
                max_new_tokens=64, constrain_actions=True,
                actor_history_turns=2, loop_guard_max=2,
                memory_text=retrieved['context'])
            if (episode['status'] != 'complete' or
                    episode['initial_observation'] != initial or
                    episode['invalid_commands'] != 0 or
                    retrieved['tokens'] > args.memory_tokens):
                raise ValueError('Incomplete MemRL episode or prompt budget')
            entry['retrieved'] = retrieved
            entry['episode'] = episode
            entry['memory_write'] = False
            if episode['reward']:
                records = records_from_episode(episode)
                public_trace = '\n'.join(json.dumps(record,
                    ensure_ascii=False) for record in records)
                update = memory.update(initial, public_trace,
                    episode['reward'], True, retrieved,
                    entry['input_content_sha256'])
                prior_hashes.append(digest(records))
                entry['own_success_records'] = records
                entry['own_success_records_sha256'] = digest(records)
                entry['public_trace_sha256'] = digest(public_trace)
                entry['memory_update'] = update
                entry['memory_write'] = True
            entry['prior_success_records_sha256_after'] = prior_hashes.copy()
            entry['source_count_after'] = len(prior_hashes)
            entry['writer_calls_after'] = memory.calls
            entry['writer_input_tokens_after'] = memory.input_tokens
            entry['writer_output_tokens_after'] = memory.output_tokens
            entry['writer_limit_hits_after'] = memory.limit_hits
            snapshot = directory / f'memory_after_{binding["position"]:03d}.json'
            memory.snapshot(snapshot)
            snapshot_hash = file_hash(snapshot)
            entry['memory_snapshot_sha256'] = snapshot_hash
            report['rows'].append(entry)
            save(output, report)
            print(json.dumps({'order': args.order,
                'done': len(report['rows']),
                'success': sum(row['episode']['reward']
                               for row in report['rows']),
                'writes': len(prior_hashes),
                'writer_calls': memory.calls}), flush=True)
        except Exception as exc:
            report['failures'].append({'position': binding['position'],
                'reason': repr(exc), 'partial': entry})
            save(output, report)
            raise


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
    parser.add_argument('--output', type=Path)
    parser.add_argument('--validate-only', action='store_true')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--gpu-fraction', type=float, default=.7)
    run(parser.parse_args())
