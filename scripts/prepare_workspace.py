#!/usr/bin/env python3
"""Create ignored asset directories; optionally organize existing local artifacts."""
import argparse
import ctypes
import errno
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def exchange(left, right):
    """Atomically replace a live directory with its compatibility symlink on Linux."""
    libc = ctypes.CDLL(None, use_errno=True)
    call = getattr(libc, 'renameat2', None)
    if call is None:
        raise OSError(errno.ENOSYS, 'Atomic rename exchange is unavailable; no live paths were moved')
    call.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    call.restype = ctypes.c_int
    if call(-100, os.fsencode(left), -100, os.fsencode(right), 2):
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code), str(left))


def place(legacy, storage, kind, organize=False):
    source, target = ROOT / legacy, ROOT / storage
    if not source.is_relative_to(ROOT) or not target.is_relative_to(ROOT):
        raise ValueError('Paths must remain inside this workspace')
    source.parent.mkdir(parents=True, exist_ok=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.is_symlink():
        if source.resolve() != target.resolve():
            raise ValueError(f'Conflicting existing symlink: {legacy}')
        return 'linked'
    if source.exists():
        if not organize:
            raise FileExistsError(f'{legacy} contains local files; use --organize to relocate it')
        if target.exists() or target.is_symlink():
            raise FileExistsError(f'Refusing to merge or overwrite {storage}')
        # The temporary link is meaningful from the OLD path after the exchange.
        target.symlink_to(os.path.relpath(target, source.parent), target_is_directory=(kind == 'directory'))
        try:
            exchange(source, target)
        except BaseException:
            if target.is_symlink():
                target.unlink()
            raise
        return 'moved'
    if kind == 'directory':
        target.mkdir(parents=True, exist_ok=True)
    source.symlink_to(os.path.relpath(target, source.parent), target_is_directory=(kind == 'directory'))
    return 'created'


def prepare(organize=False):
    rows = json.loads((ROOT / 'config/asset-layout.json').read_text())['aliases']
    result = []
    for name in ['models', 'data', 'results']:
        (ROOT/name).mkdir(exist_ok=True)
        (ROOT/name/'.gitkeep').touch(exist_ok=True)
    for row in rows:
        status = place(row['legacy'], row['storage'], row['kind'], organize)
        result.append({**row, 'action':status})
    if organize:
        # Finished model tensors are centralized; metadata/trajectories stay in results.
        # Iteration does not follow symlinks and can therefore be safely repeated.
        for directory, dirs, files in os.walk(ROOT/'results'):
            dirs[:] = [d for d in dirs if not (Path(directory)/d).is_symlink()]
            for name in files:
                p = Path(directory)/name
                if p.is_symlink() or p.suffix.lower() not in {'.safetensors', '.pt', '.pth', '.ckpt'}:
                    continue
                rel = p.relative_to(ROOT/'results')
                destination = 'models/experiments/'+rel.as_posix()
                status = place(p.relative_to(ROOT).as_posix(), destination, 'file', True)
                result.append({'legacy':p.relative_to(ROOT).as_posix(), 'storage':destination, 'action':status})
    log = ROOT/'results/workspace-organization.json'
    log.write_text(json.dumps(result, indent=2))
    print(json.dumps({'aliases':len(rows), 'moved':sum(r['action']=='moved' for r in result),
                      'manifest':str(log)}))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--organize', action='store_true', help='Relocate existing local assets atomically; never delete their contents')
    prepare(p.parse_args().organize)
