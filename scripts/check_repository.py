#!/usr/bin/env python3
"""Check the actual Git index before upload, not merely the working directory."""
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = {'.safetensors','.pt','.pth','.ckpt','.gguf','.onnx','.arrow','.parquet','.sqlite','.db'}
TOKEN_PATTERNS = [
    re.compile(rb'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----'),
    re.compile(rb'github_pat_[A-Za-z0-9_]{35,}'),
    re.compile(rb'gh[pousr]_[A-Za-z0-9]{30,}'),
    re.compile(rb'sk-(?:proj-|or-v1-)?[A-Za-z0-9_-]{35,}'),
    re.compile(rb'AKIA[A-Z0-9]{16}'),
]


def check():
    layout = json.loads((ROOT/'config/asset-layout.json').read_text())['aliases']
    excluded = [x['legacy'] for x in layout] + ['migration','ttcl/.runtime','.local']
    entries = subprocess.check_output(['git','ls-files','--stage','-z'],cwd=ROOT).split(b'\0')
    rows = []
    errors = []
    for entry in entries:
        if not entry:
            continue
        header, filename = entry.split(b'\t',1)
        mode, oid, stage = header.decode().split()
        name = filename.decode()
        if stage != '0' or mode == '160000':
            errors.append({'path':name,'reason':'conflict or Git submodule'})
        if any(name == x or name.startswith(x+'/') for x in excluded):
            errors.append({'path':name,'reason':'excluded asset or local-only path'})
        if name.split('/')[0] in {'models','data','results'} and name not in {'models/.gitkeep','data/.gitkeep','results/.gitkeep'}:
            errors.append({'path':name,'reason':'asset content'})
        if Path(name).suffix.lower() in WEIGHTS:
            errors.append({'path':name,'reason':'weight or dataset binary'})
        size = int(subprocess.check_output(['git','cat-file','-s',oid],cwd=ROOT))
        if size > 50*1024*1024:
            errors.append({'path':name,'reason':'file exceeds 50 MiB code-repository limit'})
        # Known vendored security-test fixtures deliberately contain fake credential
        # strings. They were reviewed; this exemption never applies to application code.
        fixture = name.startswith('current_work/reef/tests/')
        if size < 2*1024*1024 and not fixture:
            data = subprocess.check_output(['git','cat-file','blob',oid],cwd=ROOT)
            if any(pattern.search(data) for pattern in TOKEN_PATTERNS):
                errors.append({'path':name,'reason':'credential-like content; inspect privately'})
        basename = Path(name).name
        env_fixture = name == 'current_work/reef/tests/reef_service/data/harness_goldens/dsh/dsh/.env'
        if basename == 'mykey.py' or (basename == '.env' and not env_fixture):
            errors.append({'path':name,'reason':'local credential configuration'})
        rows.append({'path':name,'bytes':size})
    result = {'files':len(rows),'total_mib':round(sum(r['bytes'] for r in rows)/2**20,2),
              'errors':errors,'passed':not errors}
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return not errors


if __name__ == '__main__':
    sys.exit(0 if check() else 1)
