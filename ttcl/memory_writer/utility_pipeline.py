"""Dependency-aware, detached queue for bridge SFT and delayed-utility DPO controls."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from ttcl.memory_writer.pipeline import read_optional
from ttcl.memory_writer.train import save_json
from ttcl.memory_writer.utility_data import prepare


def reports(root, jobs, phase):
    now = datetime.now(timezone.utc).isoformat()
    save_json(root / 'status.json', dict(phase=phase, updated_at=now, supervisor_pid=os.getpid(), jobs=jobs))
    lines = ['# 写入器第二轮：补充训练与后续效用偏好', '', f'状态：{phase}；更新：{now}', '',
             '| 组别 | 新单步 exact | 新连续 exact | 记忆答题 | 移除记忆 | 正确记忆诊断 | 原 SGD exact | BSM IoU |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    results = {}
    for name in ('legacy_mixed', 'bridge_sft', 'utility_dpo', 'shuffled_dpo'):
        directory = root / name
        old = read_optional(directory / 'eval/metrics.json')
        utility = read_optional(directory / 'utility_eval/metrics.json')
        values = [utility.get('single_update', {}).get('exact_state'), utility.get('stream', {}).get('exact_state'),
                  utility.get('reader_accuracy'), utility.get('memory_removed_accuracy'), utility.get('oracle_memory_accuracy'),
                  old.get('test_sgd_gold_history', {}).get('exact_state'), old.get('bsm_structured_memory', {}).get('mean_score')]
        lines.append('| ' + name + ' | ' + ' | '.join('—' if v is None else f'{v:.5f}' for v in values) + ' |')
        results[name] = dict(original_eval=old, utility_eval=utility,
                             train=read_optional(directory / 'train/metrics.json'))
    lines.extend(['', '| 后台任务 | 状态 | 进度 |', '|---|---|---|'])
    for job in jobs:
        progress = read_optional(root / job['output'] / 'progress.json')
        step = progress.get('step', progress.get('completed', progress.get('streams', '')))
        lines.append(f"| {job['name']} | {job['status']} | {progress.get('phase', '')} {step} |")
    pairs = read_optional(root / 'preference_data/metrics.json')
    results['feedback_budget'] = pairs
    save_json(root / 'results.json', results)
    lines += ['', '空白格尚未完成，不代表零。基线原评测复用第一轮存档；新测试对所有组重新执行。',
              '训练数据与候选评分仅来自独立合成训练环境及原 SGD train replay，CLBench 不用于梯度或偏好标签。',
              'writer 不看到后续问题/答案；候选的后续问题、冻结 reader、随机种子和生成预算一致。',
              '合法候选至少相差 0.5 后续问答正确率才构成偏好；chosen 另需通过训练环境状态支持检查。',
              'DPO 与 shuffled DPO 使用相同候选对、初始化、步数；shuffled 固定随机翻转一半标签。',
              '所有候选额外反馈单列预算；正式 BSM 仍每题一次回答/评分，memory 不接收外部 scalar reward。',
              '原 BSM 前 12 条为重复诊断，不是新盲测；新合成测试有模板局限。',
              '本轮只有一个训练种子，不能据此主张稳定跨任务提升。',
              'DPO 参考：https://arxiv.org/abs/2305.18290',
              '', '候选评分预算：' + json.dumps(pairs, ensure_ascii=False)]
    path = root / 'RESULT.md.tmp'
    path.write_text('\n'.join(lines) + '\n')
    path.replace(root / 'RESULT.md')


def run(root):
    spec = read_optional(root / 'experiment.json')
    jobs = [dict(j, status='queued') for j in spec['jobs']]
    active, stopped = {}, False

    def stop(signum, frame):
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        while any(j['status'] in ('queued', 'running') for j in jobs):
            if stopped:
                break
            for gpu, (process, log, job) in list(active.items()):
                code = process.poll()
                if code is None:
                    continue
                log.close()
                del active[gpu]
                job.update(status='complete' if code == 0 else 'failed', exit_code=code)
                job.pop('pid', None)
            by_name = {j['name']: j for j in jobs}
            for job in jobs:
                if job['status'] == 'queued' and any(by_name[d]['status'] in ('failed', 'blocked', 'stopped') for d in job['needs']):
                    job.update(status='blocked', reason='Prerequisite failed; see dependency log')
            for gpu in spec['gpus']:
                if gpu in active:
                    continue
                job = next((j for j in jobs if j['status'] == 'queued'
                            and all(by_name[d]['status'] == 'complete' for d in j['needs'])), None)
                if job is None:
                    continue
                log = (root / 'logs' / (job['name'] + '.log')).open('a')
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED='1',
                           TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
                process = subprocess.Popen(job['command'], cwd=root / 'code_snapshot', env=env,
                                           stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
                job.update(status='running', gpu=gpu, pid=process.pid)
                active[gpu] = process, log, job
                print(json.dumps(dict(start=job['name'], gpu=gpu, pid=process.pid)), flush=True)
            reports(root, jobs, 'running')
            time.sleep(10)
    finally:
        for process, log, job in active.values():
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            log.close()
            job['status'] = 'stopped'
        for job in jobs:
            if job['status'] == 'queued':
                job['status'] = 'stopped'
        phase = 'complete' if all(j['status'] == 'complete' for j in jobs) else 'stopped' if stopped else 'finished_with_failures'
        reports(root, jobs, phase)


def launch(args):
    repo = Path(__file__).resolve().parents[2]
    root, previous = args.root.resolve(), args.previous.resolve()
    original_adapter = previous / 'mixed_sft_seed42/train/adapter'
    if not (original_adapter / 'adapter_model.safetensors').exists():
        raise ValueError('First-stage mixed adapter missing')
    if read_optional(previous / 'status.json').get('phase') != 'complete':
        raise ValueError('First-stage queue is not complete')
    gpus = [int(x) for x in args.gpus.split(',')]
    if len(gpus) != len(set(gpus)) or not gpus:
        raise ValueError('GPU list must be nonempty and unique')
    root.mkdir(parents=True, exist_ok=False)
    (root / 'logs').mkdir()
    data, original_data = root / 'data/utility', root / 'data/original'
    shutil.copytree(repo / 'ttcl/data/memory_writer/pilot_v1', original_data)
    prepare(original_data, data)
    # Copy initial adapter as well as code/data: future edits cannot change this run.
    adapter = root / 'initial_adapter'
    shutil.copytree(original_adapter, adapter)
    snapshot = root / 'code_snapshot'
    (snapshot / 'ttcl').mkdir(parents=True)
    shutil.copy2(repo / 'ttcl/__init__.py', snapshot / 'ttcl/__init__.py')
    for name in ('memory_writer', 'common', 'llm_memory'):
        shutil.copytree(repo / 'ttcl' / name, snapshot / 'ttcl' / name, ignore=shutil.ignore_patterns('__pycache__'))
    (root / 'legacy_mixed/eval').mkdir(parents=True)
    shutil.copy2(previous / 'mixed_sft_seed42/eval/metrics.json', root / 'legacy_mixed/eval/metrics.json')
    jobs = []
    model = str(args.model.resolve())

    def add(name, module, arguments, output, needs=()):
        jobs.append(dict(name=name, command=[sys.executable, '-m', 'ttcl.memory_writer.' + module] + list(map(str, arguments)),
                         output=output, needs=list(needs)))

    add('bridge_train', 'train', ['--model', model, '--init-adapter', adapter,
        '--data', data / 'train_bridge.jsonl', '--validation', original_data / 'dev_sgd.jsonl', data / 'dev_utility.jsonl',
        '--output', root / 'bridge_sft/train', '--steps', 128, '--accumulation', 8,
        '--learning-rate', 5e-5, '--max-length', 4096], 'bridge_sft/train')
    add('legacy_utility_eval', 'utility', ['evaluate', '--model', model, '--adapter', adapter,
        '--data', data, '--output', root / 'legacy_mixed/utility_eval'], 'legacy_mixed/utility_eval')
    bridge_adapter = root / 'bridge_sft/train/adapter'
    add('collect_feedback', 'utility', ['collect', '--model', model, '--adapter', bridge_adapter,
        '--data', data, '--output', root / 'preference_data'], 'preference_data', ['bridge_train'])
    for name, shuffled in [('utility_dpo', False), ('shuffled_dpo', True)]:
        options = ['--model', model, '--init-adapter', bridge_adapter, '--pairs', root / 'preference_data/pairs.jsonl',
                   '--output', root / name / 'train', '--steps', 64, '--accumulation', 4]
        if shuffled:
            options += ['--shuffle-labels']
        add(name + '_train', 'preference_train', options, name + '/train', ['collect_feedback'])
    for name, dependency in [('bridge_sft', 'bridge_train'), ('utility_dpo', 'utility_dpo_train'), ('shuffled_dpo', 'shuffled_dpo_train')]:
        own_adapter = root / name / 'train/adapter'
        add(name + '_utility_eval', 'utility', ['evaluate', '--model', model, '--adapter', own_adapter,
            '--data', data, '--output', root / name / 'utility_eval'], name + '/utility_eval', [dependency])
        add(name + '_original_eval', 'evaluate', ['--model', model, '--adapter', own_adapter,
            '--data', original_data, '--output', root / name / 'eval', '--repo-root', repo,
            '--examples', 64, '--bsm-scans', 12], name + '/eval', [dependency])
    save_json(root / 'experiment.json', dict(gpus=gpus, jobs=jobs, prior_run=str(previous),
        model=model, protocol='independent training-prefix future-utility preference; frozen reader',
        dpo_source='https://arxiv.org/abs/2305.18290', created_at=datetime.now(timezone.utc).isoformat()))
    files = list(snapshot.rglob('*.py')) + list((root / 'data').rglob('*.jsonl')) + list(adapter.glob('*'))
    save_json(root / 'source_hashes.json', {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                                          for p in files if p.is_file()})
    reports(root, [dict(j, status='queued') for j in jobs], 'queued')
    with (root / 'supervisor.log').open('w') as log:
        process = subprocess.Popen([sys.executable, '-m', 'ttcl.memory_writer.utility_pipeline', 'run', '--root', str(root)],
                                   cwd=snapshot, env=dict(os.environ, PYTHONUNBUFFERED='1'), stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    (root / 'supervisor.pid').write_text(str(process.pid) + '\n')
    print(json.dumps(dict(root=str(root), pid=process.pid, result=str(root / 'RESULT.md')), indent=2))


def main():
    repo = Path(__file__).resolve().parents[2]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['launch', 'run'])
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--previous', type=Path, default=repo / 'ttcl/results/memory_writer/sgd_mixed_sft_20260918')
    p.add_argument('--model', type=Path, default=repo / 'current_work/delta-Mem/model/Qwen3-4B-Instruct-2507')
    p.add_argument('--gpus', default='0,1')
    args = p.parse_args()
    run(args.root.resolve()) if args.mode == 'run' else launch(args)


if __name__ == '__main__':
    main()
