from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import time


def workspace():
    return Path(os.environ.get('TTCL_WORKSPACE',Path(__file__).resolve().parents[2])).resolve()


def read(path):
    return json.loads(Path(path).read_text())


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    temp.replace(path)


def append(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a') as f:f.write(json.dumps(value,ensure_ascii=False,allow_nan=False)+'\n')


def binding(value):
    return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,allow_nan=False).encode()).hexdigest()


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def seed(*parts):
    return int(binding(parts)[:12],16)%(2**31-1)


def status(root,phase,**kwargs):
    save(Path(root)/'status.json',dict(phase=phase,time=time.time(),pid=os.getpid(),**kwargs))
