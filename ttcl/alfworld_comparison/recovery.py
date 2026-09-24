"""Reuse an already frozen ExpeL bank after an infrastructure-only restart."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import tempfile


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def _read(path):
    return json.loads(Path(path).read_text())


def _verify_manifest(path):
    manifest = _read(path)
    if not isinstance(manifest, dict) or not manifest:
        raise ValueError(f'Empty or invalid hash manifest: {path}')
    for source, expected in manifest.items():
        source_path = Path(source)
        if not source_path.is_absolute():
            raise ValueError(f'Expected frozen absolute source path: {source}')
        if not source_path.is_file() or _sha(source_path) != expected:
            raise ValueError(f'Frozen source hash mismatch: {source}')
    return manifest


def verify_reused_bank(bankdir):
    """Validate the untouched state and its original source identity, read-only."""
    bankdir = Path(bankdir)
    freeze = _read(bankdir / 'freeze.json')
    state_path, hashes_path = bankdir / 'state.json', bankdir / 'source_hashes.json'
    if freeze.get('state_sha256') != _sha(state_path):
        raise ValueError('Frozen ExpeL state hash mismatch')
    if freeze.get('source_hashes_sha256') != _sha(hashes_path):
        raise ValueError('Frozen ExpeL source manifest hash mismatch')
    _verify_manifest(hashes_path)
    state = _read(state_path)
    if (state.get('source_hashes_sha256') != _sha(hashes_path)
            or state.get('test_feedback_used') is not False
            or freeze.get('test_feedback_used') is not False
            or state.get('frozen_before_evaluation') is not True):
        raise ValueError('ExpeL source binding or evaluation freeze flag mismatch')
    identity = dict(state)
    bank_id = identity.pop('bank_id', None)
    actual = hashlib.sha256(json.dumps(identity, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    if not bank_id or bank_id != actual:
        raise ValueError('Frozen ExpeL bank identity mismatch')
    return state


def restore_bank(previous_run, dest_bank, current_upstream):
    """Copy a fully verified bank without regenerating any critic output.

    Original absolute source references intentionally remain unchanged. The
    previous run must be retained so those original files stay verifiable.
    """
    previous_run = Path(previous_run).resolve()
    source_bank = previous_run / 'expel_bank'
    dest_bank = Path(dest_bank).absolute()
    current_upstream = Path(current_upstream).resolve()
    if dest_bank.exists() or dest_bank.is_symlink():
        raise FileExistsError(f'Refusing to overwrite a bank: {dest_bank}')
    if dest_bank.resolve().is_relative_to(previous_run):
        raise ValueError('Recovery destination must not modify the previous run')
    input_path, bank_path = previous_run / 'input_hashes.json', previous_run / 'bank_hashes.json'
    _verify_manifest(input_path)
    bank_hashes = _verify_manifest(bank_path)
    files = [p for p in source_bank.rglob('*') if p.is_file()]
    if any(p.is_symlink() for p in source_bank.rglob('*')):
        raise ValueError('Frozen bank contains an unexpected symbolic link')
    inventory = {str(p.resolve()) for p in files}
    if inventory != set(bank_hashes):
        raise ValueError('Bank hash manifest does not cover exactly the complete bank')
    state = verify_reused_bank(source_bank)
    source_hashes = _read(source_bank / 'source_hashes.json')
    current_hashes = {}
    for name in ['expel.py', 'human.py', 'alfworld.py']:
        original = previous_run / 'upstream/expel' / name
        expected = source_hashes.get(str(original.resolve()))
        current = current_upstream / 'expel' / name
        if expected is None or not current.is_file() or _sha(current) != expected:
            raise ValueError(f'Recovery upstream ExpeL source differs: {name}')
        current_hashes[str(current)] = expected
    relative_hashes = {p.relative_to(source_bank).as_posix(): bank_hashes[str(p.resolve())]
                       for p in files}
    dest_bank.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.expel-recovery-', dir=dest_bank.parent) as temp:
        staged = Path(temp) / 'bank'
        shutil.copytree(source_bank, staged)
        for relative, expected in relative_hashes.items():
            if _sha(staged / relative) != expected:
                raise ValueError(f'Recovered bank file hash mismatch: {relative}')
        restored = verify_reused_bank(staged)
        if restored['bank_id'] != state['bank_id']:
            raise ValueError('Recovery changed bank identity')
        staged.rename(dest_bank)
    return {'origin_directory': str(previous_run), 'origin_bank_directory': str(source_bank),
            'bank_id': state['bank_id'], 'copied_files': len(files),
            'copied_relative_hashes': relative_hashes,
            'input_manifest_sha256': _sha(input_path), 'bank_manifest_sha256': _sha(bank_path),
            'current_upstream_hashes': current_hashes,
            'new_critic_calls': 0, 'new_environment_episodes': 0,
            'source_references_preserved': True}
