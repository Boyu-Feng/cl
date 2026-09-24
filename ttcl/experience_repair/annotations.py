"""Load reviewed supervision from local data and bind it to exact public inputs.

No historical training targets are shipped in this module. New trajectories need
fresh evidence review; history IDs and source filenames are not content bindings.
"""

import hashlib
import json
import os
from pathlib import Path

from ttcl.experience_evolution.core import writer_messages, workspace_root


def annotation_path():
    override = os.environ.get('TTCL_REPAIR_ANNOTATIONS')
    if override:
        return Path(override).expanduser().resolve()
    return workspace_root() / 'data/annotations/experience_repair_reviewed.json'


def load_annotations():
    path = annotation_path()
    if not path.is_file():
        raise FileNotFoundError(
            f'Reviewed annotation data missing: {path}. First collect and review '
            'the new trajectories, then write schema_version=1 annotation JSON '
            'with input_sha256, evidence_steps, observed, revision and procedure '
            'per history under annotations. Set TTCL_REPAIR_ANNOTATIONS to use '
            'another reviewed file. No historical labels are substituted.'
        )
    data = json.loads(path.read_text())
    if data.get('schema_version') != 1 or not isinstance(data.get('annotations'), dict):
        raise ValueError(f'Invalid reviewed annotation schema: {path}')
    records = data['annotations']
    for hid, item in records.items():
        if not isinstance(item, dict):
            raise ValueError(f'Invalid reviewed annotation: {hid}')
        digest = item.get('input_sha256')
        steps = item.get('evidence_steps')
        if (not isinstance(digest, str) or len(digest) != 64
                or any(c not in '0123456789abcdef' for c in digest)
                or not isinstance(steps, list)
                or not all(type(step) is int and step >= 1 for step in steps)
                or not all(isinstance(item.get(k), str)
                           for k in ['observed', 'revision', 'procedure'])):
            raise ValueError(f'Invalid reviewed annotation fields: {hid}')
    return records


def annotation_input_sha256(row):
    payload = writer_messages(row['previous'], row['episode'])
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True,
                         separators=(',', ':')).encode()
    return hashlib.sha256(encoded).hexdigest()


def corrected(row):
    item = load_annotations().get(row['id'])
    if item is None or item['input_sha256'] != annotation_input_sha256(row):
        raise ValueError(
            f"History {row['id']} does not match its reviewed annotation input. "
            'New trajectories require a fresh evidence review and annotation content '
            'bindings; reusing an old history ID or source path is insufficient.'
        )
    steps = item['evidence_steps']
    facts, revision, procedure = (item[k] for k in ['observed', 'revision', 'procedure'])
    for step in steps:
        if not 1 <= step <= len(row['episode']['trajectory']):
            raise ValueError('Invalid evidence step')
    if facts == 'KEEP':
        return row['previous'], 'keep', steps
    text = (f'Observed: {facts}\nRevision: {revision}\nProcedure: {procedure}\n'
            'Scope: Similar action mechanics only; object identifiers and locations must be re-observed.')
    if len(text.split()) > 200:
        raise ValueError('Correction exceeds the shared 200-word instruction')
    return text, 'revise', steps
