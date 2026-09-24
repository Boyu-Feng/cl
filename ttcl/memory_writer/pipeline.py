"""Detached two-GPU queue with independent results and automatic status reports."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from ttcl.memory_writer.train import save_json


def read_optional(path):
    return json.loads(path.read_text()) if path.exists() else {}


def report(root, jobs, phase):
    now = datetime.now(timezone.utc).isoformat()
    save_json(root / 'status.json', {'phase': phase, 'updated_at': now, 'supervisor_pid': os.getpid(), 'jobs': jobs})
    lines = ['# Memory writer 实验结果', '', f'状态：{phase}；更新时间：{now}', '',
             '未完成的格子表示尚无最终结果，不代表分数为零。', '',
             '| 实验 | 状态 | SGD 单步 exact | 合成单步 exact | 连续状态 exact | reader probe | BSM IoU |',
             '|---|---|---:|---:|---:|---:|---:|']
    combined = {}
    for job in jobs:
        directory = root / job['name']
        metrics = read_optional(directory / 'eval/metrics.json')
        progress = read_optional(directory / job.get('stage', 'eval') / 'progress.json')
        combined[job['name']] = {'status': job['status'], 'progress': progress, 'metrics': metrics,
                                 'training': read_optional(directory / 'train/metrics.json')}
        cells = []
        for group, field in [('test_sgd_gold_history', 'exact_state'), ('test_synthetic_gold_history', 'exact_state'),
                             ('stream_predicted_history', 'exact_state'), ('reader_probes', 'accuracy'),
                             ('bsm_structured_memory', 'mean_score')]:
            value = metrics.get(group, {}).get(field)
            cells.append('—' if value is None else f'{value:.5f}')
        stage = job.get('stage', '')
        step = progress.get('step', progress.get('completed', progress.get('count', '')))
        lines.append(f"| {job['name']} | {job['status']} {stage} {step} | " + ' | '.join(cells) + ' |')
    lines.extend(['', 'SGD 为自定义严格键值状态指标，不是官方 leaderboard 分数；gold_history 给定正确旧状态，连续测试使用模型自身历史。',
                  '所有 reader 使用冻结底座并关闭 writer LoRA；训练只监督 memory-writer 输出。BSM 使用官方评分，公开反馈不额外注入标量 reward。',
                  '每组 eval/ 下有独立 metrics.json、逐次输入、生成、记忆更新和 BSM 报告；train/ 下有 LoRA、训练曲线和验证损失。',
                  '本轮仅执行第一阶段 SFT；没有把额外候选评分或第二阶段 RL 混入本轮结果。',
                  '数据来源、划分和许可见 dataset_manifest.json；参数见 experiment.json；运行代码快照见 code_snapshot/。'])
    temporary = root / 'RESULT.md.tmp'
    temporary.write_text('\n'.join(lines) + '\n')
    temporary.replace(root / 'RESULT.md')
    save_json(root / 'results.json', combined)


def stage_command(spec, job, root):
    base = [sys.executable, '-m', 'ttcl.memory_writer.' + ('train' if job['stage'] == 'train' else 'evaluate')]
    data, model = spec['data'], spec['model']
    if job['stage'] == 'train':
        return base + ['--model', model, '--data', str(Path(data) / ('train_' + job['dataset'] + '.jsonl')),
                       '--validation', str(Path(data) / 'dev_sgd.jsonl'), str(Path(data) / 'dev_synthetic.jsonl'),
                       '--output', str(root / job['name'] / 'train'), '--seed', str(job['seed']),
                       '--steps', str(spec['steps']), '--accumulation', str(spec['accumulation'])]
    command = base + ['--model', model, '--data', data, '--output', str(root / job['name'] / 'eval'),
                      '--repo-root', spec['repo_root'], '--examples', str(spec['eval_examples']),
                      '--bsm-scans', str(spec['bsm_scans'])]
    if job['dataset']:
        command += ['--adapter', str(root / job['name'] / 'train/adapter')]
    return command


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    spec = json.loads((root / 'experiment.json').read_text())
    jobs = [dict(job, status='queued', stage='train' if job['dataset'] else 'eval') for job in spec['jobs']]
    active = {}
    stopped = False

    def request_stop(signum, frame):
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    while any(j['status'] in ('queued', 'running') for j in jobs):
        if stopped:
            for process, log, job in active.values():
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                log.close()
            for job in jobs:
                if job['status'] in ('queued', 'running'):
                    job['status'] = 'stopped'
            break
        for gpu, (process, log, job) in list(active.items()):
            code = process.poll()
            if code is None:
                continue
            log.close()
            del active[gpu]
            job['last_exit_code'] = code
            job.pop('pid', None)
            if code:
                job['status'] = 'failed'
            elif job['stage'] == 'train':
                job.update(stage='eval', status='queued')
            else:
                job['status'] = 'complete'
        for gpu in spec['gpus']:
            if gpu in active:
                continue
            job = next((j for j in jobs if j['status'] == 'queued'), None)
            if job is None:
                break
            directory = root / job['name']
            directory.mkdir(exist_ok=True)
            command = stage_command(spec, job, root)
            save_json(directory / (job['stage'] + '_command.json'), command)
            environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED='1',
                               TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
            log = (directory / (job['stage'] + '.log')).open('a')
            process = subprocess.Popen(command, cwd=root / 'code_snapshot', env=environment,
                                       stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            job.update(status='running', gpu=gpu, pid=process.pid)
            active[gpu] = process, log, job
            print(json.dumps({'start': job['name'], 'stage': job['stage'], 'gpu': gpu, 'pid': process.pid}), flush=True)
        report(root, jobs, 'running')
        time.sleep(10)
    phase = 'stopped' if stopped else 'complete' if all(j['status'] == 'complete' for j in jobs) else 'finished_with_failures'
    report(root, jobs, phase)


if __name__ == '__main__':
    main()
