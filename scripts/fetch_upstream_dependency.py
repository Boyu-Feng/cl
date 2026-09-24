#!/usr/bin/env python3
"""Restore an optional dependency at its recorded revision without Git submodules."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def fetch(repo_name, dependency):
    repos = json.loads((ROOT/'config/upstreams.json').read_text())['repositories']
    item = next((x for x in repos[repo_name]['submodules'] if x['path'] == dependency), None)
    if item is None:
        raise ValueError('Dependency must match a path in config/upstreams.json')
    target = ROOT/repos[repo_name]['path']/dependency
    if target.exists() and any(target.iterdir()):
        raise FileExistsError(f'Refusing to overwrite {target}')
    with tempfile.TemporaryDirectory(prefix='cl-dependency-') as tmp:
        subprocess.run(['git','clone','--no-checkout',item['url'],tmp],check=True)
        subprocess.run(['git','-C',tmp,'checkout',item['commit']],check=True)
        shutil.copytree(tmp,target,dirs_exist_ok=True,ignore=shutil.ignore_patterns('.git'))
    print(f'Restored {target} at {item["commit"]}; no nested Git metadata retained.')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('repository')
    p.add_argument('dependency')
    a = p.parse_args()
    fetch(a.repository,a.dependency)
