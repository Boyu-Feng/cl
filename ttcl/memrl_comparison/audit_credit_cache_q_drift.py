"""Audit how native Q broadcasts and content-bound suppression change Q."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from ttcl.icl_mem0_comparison.protocol import read, save, sha

from .audit_credit_cache_online import audit as audit_online
from .audit_credit_train_extension import audit as audit_native


def _text_hash(item: dict) -> str:
    metadata = item['metadata']
    text = ('Task: ' + metadata['task_description'] +
            '\nExperience: ' + metadata['public_abstract'])
    return hashlib.sha256(text.encode()).hexdigest()


def audit(native: Path, online: Path, cache_path: Path) -> dict:
    source_report, online_report = audit_native(native), audit_online(online)
    if (not source_report['complete'] or source_report['missing'] or
            not online_report['complete'] or online_report['missing'] or
            Path(read(online / 'design.json')['source']) != native or
            sha(cache_path) != online_report['cache_sha256']):
        raise ValueError('Source, online chain or cache binding changed')
    cache = read(cache_path)
    source_design = read(native / 'design.json')
    index_by_family = {family:source_design['bootstrap'][family]['index']
                       for family in source_design['families']}
    rows = []
    for entry in cache['entries']:
        family = entry['family']
        boot = index_by_family[family]
        source_dir = (native / 'runs' / 'alfworld' / family /
                      str(source_design['repeat']) / 'memrl')
        online_dir = (online / 'runs' / 'alfworld' / family /
                      str(source_design['repeat']) / 'cache')
        root = read(source_dir / f'episode_{boot:03d}' / 'memory_after.json')
        matches = [mid for mid, item in root['items'].items()
                   if _text_hash(item) == entry['memory_text_sha256']]
        if len(matches) != 1:
            raise ValueError('Cached donor text is not unique in online root')
        mid = matches[0]
        timeline = []
        for index in range(boot, boot + len(source_design['selected_games'][family]) + 1):
            native_memory = read(source_dir / f'episode_{index:03d}' /
                                 'memory_after.json')
            cache_memory = read(online_dir / f'episode_{index:03d}' /
                                'memory_after.json')
            if (mid not in native_memory['items'] or
                    mid not in cache_memory['items'] or
                    _text_hash(native_memory['items'][mid]) !=
                    entry['memory_text_sha256'] or
                    _text_hash(cache_memory['items'][mid]) !=
                    entry['memory_text_sha256']):
                raise ValueError('Experience identity changed along chain')
            native_md = native_memory['items'][mid]['metadata']
            cache_md = cache_memory['items'][mid]['metadata']
            if any(not math.isfinite(float(md['q_value'])) for md in
                   (native_md, cache_md)):
                raise ValueError('Nonfinite Q value')
            item = dict(index=index,
                        native_q=native_md['q_value'],
                        native_visits=native_md['q_visits'],
                        cache_q=cache_md['q_value'],
                        cache_visits=cache_md['q_visits'])
            if index > boot:
                native_retrieval = read(source_dir / f'episode_{index:03d}' /
                                        'retrieval_1.json')
                cache_retrieval = read(online_dir / f'episode_{index:03d}' /
                                       'retrieval_1.json')
                native_row = read(source_dir / f'episode_{index:03d}' / 'row.json')
                cache_row = read(online_dir / f'episode_{index:03d}' / 'row.json')
                item.update(native_first_retrieved=mid in native_retrieval['ids'],
                            cache_first_suppressed=mid in
                            cache_retrieval['suppressed_ids'],
                            native_first_reward=native_row['first_attempt'],
                            cache_first_reward=cache_row['first_attempt'])
            timeline.append(item)
        rows.append(dict(family=family, memory_id=mid,
                         memory_text_sha256=entry['memory_text_sha256'],
                         donor_case=entry['source_case'],
                         donor_seed_deltas=entry['seed_deltas'],
                         timeline=timeline,
                         final_q_difference=timeline[-1]['cache_q'] -
                         timeline[-1]['native_q']))
    return dict(schema='credit_cache_q_drift_audit_v1',
                native_design_sha256=source_report['design_sha256'],
                online_design_sha256=online_report['design_sha256'],
                cache_sha256=sha(cache_path), entries=len(rows), rows=rows,
                caveat='Q change is audited but reward equality on this online chain does not establish a performance gain')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native', type=Path, required=True)
    parser.add_argument('--online', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.native.resolve(), args.online.resolve(),
                   args.cache.resolve())
    save(args.report, report)
    print(json.dumps({key:value for key, value in report.items()
                      if key != 'rows'}, sort_keys=True))


if __name__ == '__main__':
    main()
