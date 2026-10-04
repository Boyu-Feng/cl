"""Verify fixed-card pair provenance, prompts, seeds, and official records."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from ttcl.icl_mem0_comparison.protocol import read, save, sha
from .evidence_selected_memory import compact, public_rows


def audit(root):
    design = read(root / 'design.json')
    if design['schema'] != 'fixed_raw_card_single_state_pair_v1':
        raise ValueError('Unknown fixed-card pair design')
    if (sha(Path(design['origin']) / 'plan.json') != design['origin_plan_sha256'] or
            sha(root / 'source' / 'probe_fixed_evidence_card.py') != design['runner_sha256']):
        raise ValueError('Frozen plan or runner changed')
    source_root = Path(design['source_root'])
    card_path = source_root / 'evidence_selected_v3' / 'memory' / 'evidence_memory_000000.json'
    trace_path = source_root / 'evidence_selected_v3' / 'episode_001' / 'public_trajectory.json'
    if (sha(card_path) != design['source_card_sha256'] or
            sha(trace_path) != design['source_trace_sha256']):
        raise ValueError('Frozen source changed')
    card = read(card_path)
    source_rows = public_rows(read(trace_path))
    for entry in card['selected']:
        observed = source_rows[entry['step']]
        if entry['action'] != observed['action'] or entry['feedback'] != observed['feedback']:
            raise ValueError('Card is not literal public source evidence')
    expected_lines = [f'  step {x["step"]}: {compact(x["action"], 170)} '
                      f'→ {compact(x["feedback"], 260)}' for x in card['selected']]
    text = ('\nVerbatim public evidence from one past instance; verify current inputs '
            f'[source {card["source_binding"]["initial_query_sha256"][:12]}]:\n' +
            '\n'.join(expected_lines))
    details = []
    for repeat in design['repeats']:
        snapshot = design['source_snapshots'][str(repeat)]
        if sha(Path(snapshot['path'])) != snapshot['sha256']:
            raise ValueError('Native source memory snapshot changed')
        for index in design['indices']:
            stem = root / str(repeat) / f'target_{index:03d}'
            pair = {}
            for arm in design['arms']:
                cell = stem / arm
                row, retrieval = read(cell / 'row.json'), read(cell / 'retrieval.json')
                first = json.loads((cell / 'responses.jsonl').read_text().splitlines()[0])
                if (row['status'] != 'complete' or row.get('memory_update_status') == 'failed' or
                        sha(cell / 'memory_after.json') != row['memory_after_sha256'] or
                        row['initial_query_sha256'] != design['target_bindings'][str(repeat)][str(index)]):
                    raise ValueError('Target identity, score, or snapshot changed')
                pair[arm] = dict(row=row, retrieval=retrieval, first=first)
            base, candidate = (pair[name] for name in design['arms'])
            a, b = base['retrieval'], candidate['retrieval']
            if (b['context'] != a['context'] + text or
                    a['ids'] != b['ids'] or a['candidates'] != b['candidates'] or
                    b['fixed_card_rows_sha256'] != card['rows_sha256'] or
                    base['first']['actual_generation_seed'] != candidate['first']['actual_generation_seed']):
                raise ValueError('Pair differs by more than the fixed card at retrieval')
            details.append(dict(repeat=repeat, canonical_index=index,
                input_sha256=base['row']['initial_query_sha256'],
                without_card_reward=base['row']['reward'],
                with_card_reward=candidate['row']['reward'],
                delta=candidate['row']['reward']-base['row']['reward'],
                actor_seed=base['first']['actual_generation_seed'],
                without_card_first_prompt_sha256=base['row']['first_prompt_sha256'],
                with_card_first_prompt_sha256=candidate['row']['first_prompt_sha256']))
    return dict(schema='fixed_raw_card_pair_audit_v1',
        design_sha256=sha(root / 'design.json'), source_card_sha256=sha(card_path),
        pairs=details, card_source_steps=len(card['selected']),
        caveat='Exact source and retrieval comparison; no independent scorer replay')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.root.resolve())
    save(args.output, result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
