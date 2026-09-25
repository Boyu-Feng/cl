#!/usr/bin/env python3
"""Download public prerequisites into ignored directories; never fetch private run data."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
ALF_URL = 'https://github.com/alfworld/alfworld/releases/download/0.4.2/json_2.1.3_tw-pddl.zip'
LOCOMO_URL = 'https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json'


def download(url, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix+'.partial')
    request = urllib.request.Request(url, headers={'User-Agent':'cl-reproducibility/1.0'})
    h = hashlib.sha256()
    with urllib.request.urlopen(request, timeout=120) as response, partial.open('wb') as out:
        while chunk := response.read(1024*1024):
            h.update(chunk)
            out.write(chunk)
    partial.replace(path)
    return h.hexdigest()


def fetch(name):
    if name in {'base', 'embedding'}:
        from huggingface_hub import snapshot_download
        repo, rel = {
            'base':('Qwen/Qwen3-4B-Instruct-2507','models/delta_mem/Qwen3-4B-Instruct-2507'),
            'embedding':('sentence-transformers/all-mpnet-base-v2','models/embedding/all-mpnet-base-v2'),
        }[name]
        snapshot_download(repo_id=repo, local_dir=ROOT/rel)
        print(f'Downloaded {repo} to {rel}')
    elif name == 'clbench-db':
        from huggingface_hub import hf_hub_download
        repo = (os.environ.get('CL_BENCHMARK_DB_REPO') or
                'continual-learning-bench/database-exploration').strip()
        directory = ROOT/'data/clbench/database_exploration'
        directory.mkdir(parents=True, exist_ok=True)
        for filename in ('products.db', 'products_drifted.db'):
            path = directory/filename
            if not path.is_file():
                hf_hub_download(repo_id=repo, repo_type='dataset', filename=filename,
                                local_dir=str(directory), token=os.environ.get('HF_TOKEN'))
            if not path.is_file():
                raise FileNotFoundError(f'Expected {path} after Hugging Face download')
        print(f'CLBench databases ready in {directory}')
    elif name == 'alfworld':
        directory = ROOT/'data/ttcl/alfworld_delta'
        archive = directory/'games_complete.zip'
        sha = download(ALF_URL, archive)
        with zipfile.ZipFile(archive) as z:
            for item in z.infolist():
                q = Path(item.filename)
                if q.is_absolute() or '..' in q.parts or (item.external_attr >> 16) & 0o170000 == 0o120000:
                    raise ValueError(f'Unsafe archive member: {item.filename}')
            if z.testzip() is not None:
                raise ValueError('ALFWorld archive CRC failed')
            z.extractall(directory)
        for split in ['train','valid_unseen']:
            if not (directory/'json_2.1.1'/split).is_dir():
                raise ValueError('Unexpected ALFWorld archive structure; inspect before preparing experiments')
        (directory/'download_provenance.json').write_text(json.dumps({
            'official_url':ALF_URL,'sha256':sha,'bytes':archive.stat().st_size,'zip_crc_verified':True},indent=2))
    elif name == 'locomo':
        path = ROOT/'data/delta_mem/data/locomo10.json'
        sha = download(LOCOMO_URL, path)
        json.loads(path.read_text())
        print(json.dumps({'dataset':'locomo','sha256':sha}))
    elif name == 'clbench-data':
        # The benchmark contains generated/public input data in its original Git tree.
        # Read only the pinned upstream data tree; do not overwrite vendored source changes.
        meta = json.loads((ROOT/'config/upstreams.json').read_text())['repositories']['continual-learning-bench']
        with tempfile.TemporaryDirectory(prefix='clbench-public-data-') as tmp:
            subprocess.run(['git','clone','--filter=blob:none','--no-checkout',meta['url'],tmp],check=True)
            subprocess.run(['git','-C',tmp,'checkout',meta['commit'],'--','data'],check=True)
            dest = ROOT/'data/clbench'
            shutil.copytree(Path(tmp)/'data',dest,dirs_exist_ok=True)
        print('Pinned CLBench public files restored. Download SQLite files separately with: python scripts/fetch_assets.py clbench-db.')
    else:
        raise ValueError(name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('assets', nargs='+', choices=['base','embedding','alfworld','locomo','clbench-data','clbench-db'])
    args = parser.parse_args()
    subprocess.run([sys.executable,str(ROOT/'scripts/prepare_workspace.py')],check=True)
    for asset in args.assets:
        fetch(asset)
