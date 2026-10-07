"""Native MemRL on exactly the shared own-success stream of frozen LoRA eval.

Only official-won episodes from the separately audited no-memory actor enter
the memory. This is a matched-source memory-interface comparison, not each
method's independently collected on-policy history.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ttcl.memrl_comparison.memory import Embedder, Memory
from ttcl.trajectory_hyperlora.alfworld_online_from_empty import (
    digest, file_hash, records_from_episode,
)
from ttcl.trajectory_hyperlora.alfworld_source_utility_dataset_v2 import save
from ttcl.trajectory_hyperlora.alfworld_zero_shot_probe import load_agent, run_episode
from ttcl.trajectory_hyperlora.evaluate_xland_memrl import LocalQwenClient, plan as native_plan
from ttcl.trajectory_hyperlora.freeze_alf_fresh_seen_v6 import expected_review


def expected(args):
    parent = json.loads(args.review.read_text())
    binding_args = argparse.Namespace(output=args.plan,
        data_root=args.data_root, checkpoint=args.checkpoint)
    if parent != expected_review(binding_args):
        raise ValueError('Changed original reviewed official target plan')
    base = json.loads(args.base_result.read_text())
    audit = json.loads(args.base_audit.read_text())
    if (audit['raw_report_sha256'] != file_hash(args.base_result) or
            audit['review_sha256'] != file_hash(args.review) or
            audit['summary']['games'] != 36 or
            audit['summary']['failed_replays'] != 0 or
            base['review_sha256'] != file_hash(args.review) or
            base['checkpoint_sha256'] != file_hash(args.checkpoint) or
            base['max_steps'] != 50 or
            base['max_new_tokens'] != 64 or
            base['actor_history_turns'] != 2 or
            base['loop_guard_max'] != 2 or
            base['failures'] or len(base['games']) != 36 or
            args.memory_tokens != 2048 or args.writer_tokens != 256 or
            args.retrieval_threshold != .5 or args.seed != 42):
        raise ValueError('Changed or unaudited common own-source stream')
    prior = []
    rows = []
    for target, game in zip(parent['targets'], base['games'], strict=True):
        if (target['index'] != game['index'] or
                target['game'] != game['game'] or
                target['game_sha256'] != game['game_sha256'] or
                target['input_content_sha256'] !=
                    game['input_content_sha256'] or
                game['prior_source_records_sha256'] != prior or
                game['prior_success_count'] != len(prior)):
            raise ValueError('Changed shared source chronology')
        episode = game['base']
        if (episode['status'] != 'complete' or
                episode['invalid_commands'] != 0):
            raise ValueError('Incomplete official baseline source')
        content = {'index': target['index'], 'game': target['game'],
            'family': target['family'],
            'game_sha256': target['game_sha256'],
            'parent_target_binding': target['input_content_sha256'],
            'base_episode_sha256': digest(episode),
            'prior_success_records_sha256': list(prior),
            'source_report_sha256': file_hash(args.base_result),
            'source_audit_sha256': file_hash(args.base_audit),
            'checkpoint_sha256': file_hash(args.checkpoint),
            'method': 'shared_own_success_native_memrl_v8',
            'memory_tokens': 2048, 'writer_tokens': 256,
            'retrieval_threshold': .5, 'seed': 42,
            'max_steps': 50, 'max_new_tokens': 64,
            'actor_history_turns': 2, 'loop_guard_max': 2}
        rows.append({**content, 'input_content_sha256': digest(content),
                     'reviewed_target': True})
        if episode['reward']:
            records = records_from_episode(episode)
            prior.append(digest(records))
        if game['source_count_after'] != len(prior):
            raise ValueError('Changed success-only memory write schedule')
    return {'protocol': 'New reviewed MemRL targets bound to exactly the frozen official valid_seen common own-success trajectory stream, target content, writer/retrieval/actor budgets; prior game history never drawn from future tasks',
            'plan_sha256': file_hash(args.plan),
            'parent_review_sha256': file_hash(args.review),
            'base_report_sha256': file_hash(args.base_result),
            'base_audit_sha256': file_hash(args.base_audit),
            'checkpoint_sha256': file_hash(args.checkpoint),
            'targets': rows}


def prepare(args):
    if args.memrl_review.exists():
        raise FileExistsError(args.memrl_review)
    review = expected(args)
    args.memrl_review.parent.mkdir(parents=True, exist_ok=True)
    args.memrl_review.write_text(json.dumps(review, ensure_ascii=False,
                                            indent=2)+'\n')
    print(json.dumps({'reviewed_targets': len(review['targets']),
                      'review_sha256': file_hash(args.memrl_review)}),
          flush=True)


def run(args):
    if args.output.exists() or args.directory.exists():
        raise FileExistsError('Use fresh MemRL report and memory directory')
    review = json.loads(args.memrl_review.read_text())
    if review != expected(args):
        raise ValueError('Changed MemRL trajectory annotation or binding')
    base = json.loads(args.base_result.read_text())
    agent, tokenizer = load_agent(args.model, args.checkpoint,
                                  args.device, args.gpu_fraction)
    client = LocalQwenClient(agent.model, tokenizer, args.device, args.seed)
    plan = native_plan(args)
    embedder = Embedder(args.embedding)
    memory = Memory(plan, client, args.directory,
                    {'threshold': args.retrieval_threshold,
                     'mean': 0., 'std': 1.}, embedder=embedder)
    report = {'protocol': 'Native upstream MemRL writer, BGE-M3 retrieval and Q update on exactly the success-gated no-memory own trajectory stream of frozen official valid_seen36 comparison; same frozen Qwen target actor, 50 steps, two history turns; equal target/source/episode budget, writer and retrieval compute explicitly recorded; not method-on-policy',
        'plan_sha256': file_hash(args.plan),
        'parent_review_sha256': file_hash(args.review),
        'memrl_review_sha256': file_hash(args.memrl_review),
        'base_report_sha256': file_hash(args.base_result),
        'base_audit_sha256': file_hash(args.base_audit),
        'checkpoint_sha256': file_hash(args.checkpoint),
        'memory_plan': plan, 'max_steps': 50,
        'max_new_tokens': 64, 'actor_history_turns': 2,
        'loop_guard_max': 2, 'games': [], 'failures': []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save(args.output, report)
    prior = []
    for target, source in zip(review['targets'], base['games'], strict=True):
        entry = {'index': target['index'], 'family': target['family'],
            'game': target['game'], 'game_sha256': target['game_sha256'],
            'input_content_sha256': target['input_content_sha256'],
            'parent_target_binding': target['parent_target_binding'],
            'base_episode_sha256': target['base_episode_sha256'],
            'prior_success_records_sha256': list(prior),
            'prior_success_count': len(prior)}
        try:
            game = args.data_root / target['game']
            if file_hash(game) != target['game_sha256']:
                raise ValueError('Changed official game')
            original = source['base']
            initial = original['initial_observation']
            retrieved = memory.retrieve(initial)
            episode = run_episode(agent, tokenizer, game, {},
                adapter=False, device=args.device, max_steps=50,
                max_new_tokens=64, constrain_actions=True,
                actor_history_turns=2, loop_guard_max=2,
                memory_text=retrieved['context'])
            if (episode['status'] != 'complete' or
                    episode['invalid_commands'] != 0 or
                    episode['initial_observation'] != initial):
                raise ValueError('Changed or incomplete MemRL actor episode')
            entry.update({'base_reward': original['reward'],
                'raw_text_reward': source['raw_text']['reward'],
                'retrieved': retrieved, 'memrl': episode})
            if original['reward']:
                records = records_from_episode(original)
                public_trace = '\n'.join(json.dumps(record,
                    ensure_ascii=False) for record in records)
                update = memory.update(initial, public_trace,
                    original['reward'], True, retrieved,
                    target['input_content_sha256'])
                prior.append(digest(records))
                entry.update({'new_source_records_sha256': digest(records),
                    'public_trace_sha256': digest(public_trace),
                    'memory_update': update})
            entry['source_count_after'] = len(prior)
            entry['writer_calls_after'] = memory.calls
            entry['writer_input_tokens_after'] = memory.input_tokens
            entry['writer_output_tokens_after'] = memory.output_tokens
            entry['writer_limit_hits_after'] = memory.limit_hits
            snapshot = args.directory / f'memory_after_{target["index"]:03d}.json'
            memory.snapshot(snapshot)
            entry['memory_snapshot_sha256'] = file_hash(snapshot)
            report['games'].append(entry)
            save(args.output, report)
            print(json.dumps({'done': len(report['games']),
                'base': sum(x['base_reward'] for x in report['games']),
                'raw_text': sum(x['raw_text_reward'] for x in report['games']),
                'memrl': sum(x['memrl']['reward'] for x in report['games']),
                'success_sources': len(prior),
                'writer_calls': memory.calls,
                'failures': len(report['failures'])}), flush=True)
        except Exception as exc:
            report['failures'].append({'index': target['index'],
                'reason': repr(exc), 'partial': entry})
            save(args.output, report)
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('prepare', 'run'))
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
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--gpu-fraction', type=float, default=.7)
    args = parser.parse_args()
    (prepare if args.command == 'prepare' else run)(args)
