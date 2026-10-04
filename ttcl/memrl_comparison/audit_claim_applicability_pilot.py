"""Audit frozen claim-pilot outputs, quote binding and unsupported claims."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

from ttcl.icl_mem0_comparison.protocol import read, sha
from .annotate_claim_applicability_pilot import _parse, prepare


SEARCH_CLAIM = re.compile(r'\b(?:locate|search for|retrieve) (?:the )?([a-z]+)\b', re.I)


def audit(sign_report: Path, output: Path) -> dict:
    design = read(output / 'design.json')
    expected, bindings = prepare(sign_report, output, design['url'])
    if design != expected:
        raise ValueError('Frozen pilot design changed')
    rows = []
    for index, item in enumerate(bindings):
        record = read(output / f'item_{index:02d}.json')
        if (record['index'] != index or
                record['design_sha256'] != sha(output / 'design.json') or
                record['prompt_sha256'] != item['prompt_sha256'] or
                record['input_sha256'] != item['input_sha256'] or
                record['memory_text_sha256'] != item['memory_text_sha256']):
            raise ValueError('Pilot record binding changed')
        try:
            parsed = _parse(record['raw_response'], item['target'],
                            item['peers'], item['observation'])
        except (ValueError, json.JSONDecodeError) as exc:
            if 'features' in record or record.get('parse_error') != str(exc):
                raise ValueError('Stored parse result changed')
            valid = False
        else:
            if record.get('features') != parsed or 'parse_error' in record:
                raise ValueError('Stored features changed')
            valid = True
        try:
            raw = json.loads(record['raw_response'])
        except json.JSONDecodeError:
            raw = {}
        claim = raw.get('claim', '')
        quote = raw.get('observation_quote', '')
        match = SEARCH_CLAIM.search(claim) if isinstance(claim, str) else None
        unsupported_search = bool(
            raw.get('initial_support') == 'supported' and match and
            isinstance(quote, str) and
            not re.search(r'\b' + re.escape(match.group(1)) + r'\b', quote, re.I))
        rows.append(dict(index=index, input_sha256=item['input_sha256'],
                         memory_text_sha256=item['memory_text_sha256'],
                         raw_response_sha256=hashlib.sha256(
                             record['raw_response'].encode()).hexdigest(),
                         quote_valid=valid, initial_support=raw.get('initial_support'),
                         search_object_without_quote_evidence=unsupported_search))
    return dict(schema='alf_claim_applicability_pilot_audit_v1',
                design_sha256=sha(output / 'design.json'),
                script_sha256=sha(Path(__file__)), rows=rows,
                totals=dict(expected=len(bindings), audited=len(rows),
                            quote_valid=sum(row['quote_valid'] for row in rows),
                            search_claims_marked_supported_without_object_evidence=sum(
                                row['search_object_without_quote_evidence']
                                for row in rows)),
                caveat='Posthoc selected sign-flip cases. Literal quote checks cannot certify semantic entailment; unsupported-search count flags a specific false-support pattern only.')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sign-report', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.sign_report.resolve(), args.output.resolve())
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result['totals']))


if __name__ == '__main__':
    main()
