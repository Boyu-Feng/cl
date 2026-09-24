"""Create an immutable experiment snapshot and detach the experiment queue."""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def main():
    repo = Path(__file__).resolve().parents[2]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=repo / 'ttcl/results/memory_writer' / datetime.now().strftime('%Y%m%d_%H%M%S'))
    p.add_argument('--data', type=Path, default=repo / 'ttcl/data/memory_writer/pilot_v1')
    p.add_argument('--model', type=Path, default=repo / 'current_work/delta-Mem/model/Qwen3-4B-Instruct-2507')
    p.add_argument('--gpus', default='0,1')
    p.add_argument('--steps', type=int, default=256)
    p.add_argument('--accumulation', type=int, default=8)
    p.add_argument('--eval-examples', type=int, default=64)
    p.add_argument('--bsm-scans', type=int, default=12)
    args = p.parse_args()
    root = args.root.resolve()
    if min(args.steps, args.accumulation, args.eval_examples) < 1 or args.bsm_scans < 0:
        p.error('Invalid experiment sizes')
    gpus = [int(x) for x in args.gpus.split(',')]
    if not gpus or len(set(gpus)) != len(gpus):
        p.error('GPU IDs must be unique')
    manifest = json.loads((args.data / 'manifest.json').read_text())
    for name, record in manifest['datasets'].items():
        actual = hashlib.sha256((args.data / (name + '.jsonl')).read_bytes()).hexdigest()
        if actual != record['sha256']:
            raise ValueError('Dataset changed: ' + name)
    root.mkdir(parents=True, exist_ok=False)
    snapshot = root / 'code_snapshot'
    (snapshot / 'ttcl').mkdir(parents=True)
    shutil.copy2(repo / 'ttcl/__init__.py', snapshot / 'ttcl/__init__.py')
    for name in ('memory_writer', 'common', 'llm_memory'):
        shutil.copytree(repo / 'ttcl' / name, snapshot / 'ttcl' / name, ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copy2(args.data / 'manifest.json', root / 'dataset_manifest.json')
    spec = dict(model=str(args.model.resolve()), data=str(args.data.resolve()), repo_root=str(repo),
                steps=args.steps, accumulation=args.accumulation, gpus=gpus,
                eval_examples=args.eval_examples, bsm_scans=args.bsm_scans,
                jobs=[{'name': 'mixed_sft_seed42', 'dataset': 'mixed', 'seed': 42},
                      {'name': 'untrained_writer', 'dataset': None, 'seed': 42},
                      {'name': 'sgd_sft_seed42', 'dataset': 'sgd', 'seed': 42},
                      {'name': 'mixed_sft_seed43', 'dataset': 'mixed', 'seed': 43}])
    (root / 'experiment.json').write_text(json.dumps(spec, indent=2))
    files = list(snapshot.rglob('*.py'))
    (root / 'source_hashes.json').write_text(json.dumps({str(f.relative_to(snapshot)): hashlib.sha256(f.read_bytes()).hexdigest()
                                                       for f in files}, indent=2))
    with (root / 'supervisor.log').open('w') as log:
        process = subprocess.Popen([sys.executable, '-m', 'ttcl.memory_writer.pipeline', '--root', str(root)],
                                   cwd=snapshot, env=dict(os.environ, PYTHONUNBUFFERED='1'),
                                   stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
    (root / 'supervisor.pid').write_text(str(process.pid) + '\n')
    print(json.dumps({'root': str(root), 'pid': process.pid, 'result': str(root / 'RESULT.md')}, indent=2))


if __name__ == '__main__':
    main()
