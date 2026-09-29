#!/usr/bin/env python3
"""Read-only inventory for APEx reproduction. Does not launch or label a run ready."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path(os.environ.get('TTCL_WORKSPACE', Path(__file__).resolve().parents[1])).resolve()


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def model_inventory(relative):
    directory = ROOT/relative
    present = (directory/'config.json').is_file()
    indices = list(directory.glob('*.index.json')) if directory.is_dir() else []
    if indices:
        shards = set()
        for index in indices:
            shards.update(json.loads(index.read_text()).get('weight_map', {}).values())
        present = present and bool(shards) and all((directory/s).is_file() for s in shards)
    else:
        present = present and ((directory/'model.safetensors').is_file() or
                               (directory/'pytorch_model.bin').is_file())
    return {'directory':relative, 'weight_files_present':present,
            'note':'Presence only; loading and model provenance still require validation.'}


def inspect(verify_data=False):
    config = json.loads((ROOT/'config/apex-reproduction.json').read_text())
    source = ROOT/config['official_source']['path']
    files = {str(p.relative_to(source)):sha(p) for p in sorted(source.rglob('*'))
             if p.is_file() and '__pycache__' not in p.parts and '.git' not in p.parts}
    fingerprint = hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()
    dataset = config['dataset']
    data = []
    for item in dataset['files']:
        p = ROOT/dataset['directory']/item['filename']
        valid = p.is_file() and p.stat().st_size == item['bytes']
        data.append({'path':str(p.relative_to(ROOT)), 'size_matches':valid,
                     'sha256_matches':sha(p) == item['sha256'] if valid and verify_data else None})
    try:
        result = subprocess.run(['nvidia-smi','--query-gpu=index,memory.used,memory.total',
                                 '--format=csv,noheader,nounits'],capture_output=True,text=True,
                                check=True,timeout=15)
        gpus = [dict(zip(('index','used_mib','total_mib'),map(int,line.split(','))))
                for line in result.stdout.splitlines() if line.strip()]
        gpu_error = None
    except (OSError,subprocess.SubprocessError,ValueError) as e:
        gpus, gpu_error = [], str(e)
    threshold = config['runtime']['available_gpu_memory_threshold_mib']
    idle = [g['index'] for g in gpus if g['used_mib'] <= threshold]
    base = [model_inventory(m['directory']) for m in config['base_models']]
    checkpoints = {role:model_inventory(path) for role,path in config['trained_checkpoints'].items()}
    generated = {p:(ROOT/p).is_file() for p in config['generated_training_assets']}
    retrieval = {p:(ROOT/p).is_file() for p in config['retrieval_assets']}
    blockers = []
    if fingerprint != config['official_source']['snapshot_sha256']:
        blockers.append('Official code snapshot differs from the audited commit.')
    if not all(d['size_matches'] and d['sha256_matches'] is not False for d in data):
        blockers.append('Official data prerequisites are incomplete or invalid.')
    if not all(m['weight_files_present'] for m in base):
        blockers.append('Original base-model prerequisites have not all been downloaded.')
    if not all(m['weight_files_present'] for m in checkpoints.values()):
        blockers.append('Trained APEx checkpoints are missing; the three offline stages have not run.')
    if not all(generated.values()):
        blockers.append('Writer trajectory-derived training data and skill repository must be generated.')
    if not all(retrieval.values()):
        blockers.append('wiki25 corpus and its E5 FAISS index are not prepared.')
    if len(idle) < config['runtime']['official_gpu_count']:
        blockers.append('Currently idle GPUs do not meet the official eight-GPU launcher layout.')
    if not config['runtime']['environment_validated']:
        blockers.append('The isolated upstream training environment is not validated.')
    if config['clbench_protocol']['status'] != 'implemented_and_validated':
        blockers.append('CLBench execution/online-update adapter and matched budget protocol remain to be implemented and validated.')
    return {'checked_utc':datetime.now(timezone.utc).isoformat(),
            'status':'not_started', 'ready_to_train_or_evaluate':False,
            'source_files':len(files),'source_sha256':fingerprint,
            'source_matches':fingerprint == config['official_source']['snapshot_sha256'],
            'data':data,'base_models':base,'trained_checkpoints':checkpoints,
            'generated_training_assets':generated,'retrieval_assets':retrieval,
            'gpus':gpus,'idle_gpu_indices':idle,'gpu_query_error':gpu_error,
            'free_disk_gib':round(shutil.disk_usage(ROOT).free/2**30,2),'blockers':blockers,
            'note':'Inventory only; no training, GPU reservation, or evaluation is launched.'}


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--verify-data',action='store_true')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    report=inspect(args.verify_data)
    payload=json.dumps(report,ensure_ascii=False,indent=2)+'\n'
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(payload)
    print(payload,end='')
    raise SystemExit(2)
