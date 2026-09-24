"""Prepare, generate, score, and report a fixed-history diagnostic."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import traceback

from ttcl.experience_diagnostic.core import ARMS, digest, extract_raw, paired_summary
from ttcl.experience_training.core import restore_bank
from ttcl.experience_training.run import Backend, actor_args, save, update_writer
from ttcl.llm_memory.trajectory_bank import UPDATE_PROMPT, encode
from ttcl.structured_memory import run_benchmark as base
from ttcl.structured_memory.online_bank import read, run_episode

WORKSPACE = Path('/home/fengboyu/cl')
ORIGIN = WORKSPACE / 'ttcl/results/experience_training/next_reward_sft_20260921'


def hash_file(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def prepare(root):
    root.mkdir(parents=True, exist_ok=False)
    prior = read(ORIGIN / 'plan.json')
    plan = {
        'created_unix': time.time(), 'tasks': ['database_exploration', 'cohort_studies'],
        'source_episodes': [13, 17], 'probe_offsets': [1, 2], 'repeats': [505, 606],
        'environment_seed': 42, 'model': prior['model'], 'bank': prior['bank'],
        'arms': ARMS, 'writer_output_tokens': 4096, 'writer_seed': 922,
        'adapter': str(root / 'adapter'), 'source': str(ORIGIN),
        'expected_cells': 80, 'expected_pairs_per_task': 8,
        'audited_origin': 'Assistant in this conversation; manually evidence-audited, not independent API writer.',
    }
    save(root / 'plan.json', plan)
    shutil.copytree(ORIGIN / 'training/utility_sft/adapter', root / 'adapter')
    for task in plan['tasks']:
        for ep in plan['source_episodes']:
            origin = ORIGIN / 'evaluation' / task / 'untrained/303' / f'episode_{ep:03d}'
            dest = root / 'inputs' / task / str(ep)
            dest.mkdir(parents=True)
            shutil.copy2(origin / 'trajectory.json', dest / 'trajectory.json')
            before = read(origin / 'bank_update.json')['bank_before']
            save(dest / 'bank_before.json', before)
            episode = read(dest / 'trajectory.json')
            assert before['last_observed'] == ep - 1 and episode['episode'] == ep
            bank = restore_bank(before, plan['bank'])
            save(dest / 'writer_input.json', [
                {'role': 'system', 'content': UPDATE_PROMPT},
                {'role': 'user', 'content': encode(bank.payload(episode))},
            ])
    package = root / 'source/ttcl'
    package.mkdir(parents=True)
    (package / '__init__.py').write_text('')
    for name in ['common', 'llm_memory', 'structured_memory', 'memory_writer', 'experience_training', 'experience_diagnostic']:
        shutil.copytree(WORKSPACE / 'ttcl' / name, package / name,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copy2(Path(__file__).with_name('PROTOCOL.md'), root / 'PROTOCOL.md')
    (root / 'EXTRACTION_PROMPT.md').write_text(UPDATE_PROMPT)
    files = [p for sub in ['inputs', 'source', 'adapter'] for p in (root / sub).rglob('*') if p.is_file()]
    files += [root / 'plan.json', root / 'PROTOCOL.md', root / 'EXTRACTION_PROMPT.md']
    files += list((base.BENCH / 'src').rglob('*.py'))
    model = Path(plan['model']['model'])
    files += [p for p in model.iterdir() if p.suffix in {'.json', '.safetensors'}]
    save(root / 'input_hashes.json', {str(p): hash_file(p) for p in files})
    save(root / 'status.json', {'phase': 'prepared', 'expected_cells': 80})


def generate(root, task):
    plan = read(root / 'plan.json')
    os.chdir(base.BENCH)
    model = Backend(actor_args(plan, task, 20), adapter=plan['adapter'])
    for ep in plan['source_episodes']:
        src = root / 'inputs' / task / str(ep)
        episode, before = read(src / 'trajectory.json'), read(src / 'bank_before.json')
        out = root / 'candidates' / task / str(ep)
        out.mkdir(parents=True, exist_ok=True)
        contexts = {'keep': restore_bank(before, plan['bank']).context()}
        for arm in ['untrained', 'utility_sft']:
            dest = out / arm
            bank = restore_bank(before, plan['bank'])
            if (dest / 'bank_update.json').exists():
                result = read(dest / 'bank_update.json')
            else:
                # Backend disables adapters only on actor calls. For the base writer,
                # wrap the entire generation in disable_adapter explicitly.
                from contextlib import nullcontext
                with model.inner.model.disable_adapter() if arm == 'untrained' else nullcontext():
                    result = update_writer(bank, episode, model, dest,
                                           base.generation_seed(plan['writer_seed'], task, ep),
                                           0.0, plan['writer_output_tokens'])
                # Correct metadata: the outer context disabled LoRA for untrained.
                if arm == 'untrained':
                    for attempt in result['attempts']:
                        attempt.get('completion', {})['writer_adapter_enabled'] = False
                    save(dest / 'bank_update.json', result)
                    log = dest / 'writer_generations.jsonl'
                    if log.exists():
                        events = [json.loads(s) for s in log.read_text().splitlines()]
                        for event in events:
                            event['writer_adapter_enabled'] = False
                        log.write_text(''.join(json.dumps(e) + '\n' for e in events))
            for attempt in result['attempts']:
                assert attempt['messages'][1]['content'].startswith(read(src / 'writer_input.json')[1]['content'])
            contexts[arm] = restore_bank(result['bank_after'], plan['bank']).context()
        authored = read(src / 'audited_update.json')
        bank = restore_bank(before, plan['bank'])
        bank.entries = bank.candidate(authored, episode, model.count)
        contexts['audited'] = bank.context()
        contexts['raw'], raw_audit = extract_raw(episode, before, read(src / 'raw_spec.json'), model.count)
        save(out / 'raw_audit.json', raw_audit)
        for arm, context in contexts.items():
            assert model.count(context) <= plan['bank']['max_tokens']
            save(out / f'{arm}_context.json', {'context': context, 'tokens': model.count(context), 'sha256': hashlib.sha256(context.encode()).hexdigest()})
        save(out / 'candidate_audit.json', {
            'writer_input_sha256': digest(read(src / 'writer_input.json')),
            'context_tokens': {a: model.count(c) for a, c in contexts.items()},
            'actor_was_not_run': True, 'audited_origin': plan['audited_origin'],
        })
    save(root / 'generation' / f'{task}.json', {'phase': 'complete'})


def freeze(root):
    plan = read(root / 'plan.json')
    for task in plan['tasks']:
        assert read(root / 'generation' / f'{task}.json')['phase'] == 'complete'
    paths = [p for sub in ['inputs', 'candidates'] for p in (root / sub).rglob('*') if p.is_file()]
    save(root / 'candidate_hashes.json', {str(p): hash_file(p) for p in paths})


def score(root, task, repeat):
    plan = read(root / 'plan.json')
    if not (root / 'candidate_hashes.json').exists():
        raise ValueError('All candidates must be frozen before scoring')
    for path, expected in read(root / 'candidate_hashes.json').items():
        assert hash_file(Path(path)) == expected
    os.chdir(base.BENCH)
    args = actor_args(plan, task, 20)
    # Fresh base-only model: no adapter is even loaded in scoring workers.
    model = Backend(args)
    model.repeat = repeat
    cache = {}
    for ep in plan['source_episodes']:
        for offset in plan['probe_offsets']:
            index = ep - 1 + offset
            for arm in ARMS:
                dest = root / 'scores' / task / str(repeat) / str(ep) / str(index + 1) / arm
                if (dest / 'row.json').exists():
                    row = read(dest / 'row.json')
                    cache[(row['bank_context_sha256'], index)] = (row, dest)
                    continue
                context = read(root / 'candidates' / task / str(ep) / f'{arm}_context.json')['context']
                context_hash = hashlib.sha256(context.encode()).hexdigest()
                key = (context_hash, index)
                save(root / 'progress' / f'{task}_{repeat}.json', {
                    'phase': 'scoring', 'source_episode': ep, 'probe_episode': index + 1, 'arm': arm, 'time': time.time()})
                if key in cache:
                    old, path = cache[key]
                    row = dict(old, reused_from=str(path))
                    dest.mkdir(parents=True, exist_ok=True)
                else:
                    if dest.exists():
                        raise RuntimeError(f'Incomplete execution directory retained; do not silently resample: {dest}')
                    row, trajectory = run_episode(args, model, index, dest, context)
                    calls = [s['action'] for s in trajectory['steps']]
                    canon = [json.dumps(a.get('tool_call', a), sort_keys=True) for a in calls]
                    row['repeated_actions'] = len(canon) - len(set(canon))
                    row['public_error_steps'] = sum('ERROR:' in s['public_feedback'] for s in trajectory['steps'])
                    row['actor_adapter_loaded'] = False
                    cache[key] = (row, dest)
                row.update(task=task, arm=arm, repeat=repeat, source_episode=ep)
                save(dest / 'row.json', row)
                print(json.dumps({k: row.get(k) for k in ['task', 'source_episode', 'episode', 'arm', 'repeat', 'status', 'reward', 'actor_calls', 'reused_from']}), flush=True)
    save(root / 'progress' / f'{task}_{repeat}.json', {'phase': 'complete', 'time': time.time()})


def report(root):
    plan = read(root / 'plan.json')
    rows = [read(p) for p in sorted((root / 'scores').rglob('row.json'))]
    all_complete = len(rows) == plan['expected_cells']
    summaries = {task: paired_summary([r for r in rows if r['task'] == task]) for task in plan['tasks']}
    save(root / 'comparison.json', summaries)
    lines = ['# 固定历史经验诊断', '', f'已记录 {len(rows)}/{plan["expected_cells"]} 个组别单元。', '',
             'audited 是本会话assistant编写并核查的辅助诊断候选，不能当作独立API模型排名。keep 是旧库不更新，不一定是无经验。', '',
             '相同上下文的单元显式复用执行。失败不补零；以下均分只包含五组共同完成单元。', '']
    for task, result in summaries.items():
        lines += [f'## {task}', '', f'共同完成配对：{result["paired_count"]}/8（每任务4道目标题×2个采样seed）。', '',
                  '| 组别 | reward均分 | 相对keep | 胜/平/负 | 平均actor调用 |', '|---|---:|---:|---|---:|']
        for arm, metrics in result['arms'].items():
            fmt = lambda x: '—' if x is None else f'{x:.6f}'
            lines += [f'| {arm} | {fmt(metrics["mean_reward"])} | {fmt(metrics["delta_vs_keep"])} | {metrics["wins"]}/{metrics["ties"]}/{metrics["losses"]} | {fmt(metrics["mean_calls"])} |']
        lines += ['', '| 历史题 | 探测题 | seed | keep | untrained | utility_sft | audited | raw |', '|---|---|---|---:|---:|---:|---:|---:|']
        for pair in result['pairs']:
            lines += ['| ' + ' | '.join([str(pair['source_episode']), str(pair['probe_index'] + 1), str(pair['repeat']), *[f'{pair["rewards"][a]:.6f}' for a in ARMS]]) + ' |']
        lines += ['']
    accepted = []
    for path in sorted((root / 'candidates').rglob('bank_update.json')):
        value = read(path)
        accepted.append({'path': str(path), 'accepted': value['accepted'], 'decision': value['decision'], 'error': value['error']})
    save(root / 'writer_acceptance.json', accepted)
    lines += ['## 局限', '', '两个历史点/任务、两个采样重复、既有开发环境；不能据此宣称稳定泛化或无收益。raw使用同一token上限，但实际长度、旧条目保留和信息内容不等同。audited的证据核查不是独立真值验证。所有probe相互隔离，本实验不测长期累计效果。', '',
              '完整定义见 PROTOCOL.md；候选、输入、raw引用位置、逐题轨迹和调用成本均保留。']
    (root / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    if all_complete:
        mismatches = []
        for name in ['input_hashes.json', 'candidate_hashes.json']:
            for path, expected in read(root / name).items():
                if hash_file(Path(path)) != expected:
                    mismatches.append(path)
        save(root / 'integrity.json', {'passed': not mismatches, 'mismatches': mismatches})
    save(root / 'status.json', {'phase': 'complete' if all_complete else 'running', 'recorded_cells': len(rows),
                               'complete_cells': sum(r['status'] == 'complete' for r in rows),
                               'failed_cells': sum(r['status'] != 'complete' for r in rows), 'expected_cells': plan['expected_cells']})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['prepare', 'generate', 'freeze', 'score', 'report'])
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--task')
    parser.add_argument('--repeat', type=int)
    args = parser.parse_args()
    root = args.root.resolve()
    try:
        if args.phase in ['prepare', 'freeze', 'report']:
            globals()[args.phase](root)
        elif args.phase == 'generate':
            generate(root, args.task)
        else:
            score(root, args.task, args.repeat)
    except Exception:
        save(root / 'failures' / f'{args.phase}_{args.task}_{args.repeat}.json', {'traceback': traceback.format_exc()})
        raise


if __name__ == '__main__':
    main()
