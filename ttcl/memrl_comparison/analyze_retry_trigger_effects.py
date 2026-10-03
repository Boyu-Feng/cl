"""Separate current retry exposure from later online-chain differences."""
from __future__ import annotations

import argparse
from pathlib import Path

from ttcl.icl_mem0_comparison.protocol import read, save, sha

from .run_retry_trigger_train import audit as audit_second_retry
from .run_late_retry_trigger_train import audit as audit_final_retry


def analyze(output: Path, *, require_complete: bool) -> dict:
    output = output.resolve()
    design_schema = read(output / 'design.json')['schema']
    auditors = {
        'retry_trigger_train_online_v1': audit_second_retry,
        'late_retry_trigger_train_online_v1': audit_final_retry,
    }
    if design_schema not in auditors:
        raise ValueError(f'Unexpected retry design: {design_schema}')
    verified = auditors[design_schema](output, require_complete=require_complete)
    details = []
    for row in verified['rows']:
        family, index = row['family'], row['index']
        root = output / family
        native = root / 'vanilla' / f'episode_{index:03d}'
        candidate = root / 'typed_grounded' / f'episode_{index:03d}'
        candidate_row = read(candidate / 'row.json')
        n_retrieval = read(native / 'retrieval_1.json')
        c_retrieval = read(candidate / 'retrieval_1.json')
        current_guidance = candidate_row['guided_choice_calls'] > 0
        if sha(candidate / 'memory_after.json') != candidate_row[
                'memory_after_sha256']:
            raise ValueError('Candidate snapshot changed during effect audit')
        details.append(dict(**row,
                            guided_attempts=candidate_row['guided_attempts'],
                            guided_choice_calls=candidate_row['guided_choice_calls'],
                            current_guidance=current_guidance,
                            first_retrieval_context_equal=n_retrieval[
                                'context_sha256'] == c_retrieval['context_sha256'],
                            first_prompt_equal=read(native / 'row.json')[
                                'first_prompt_sha256'] == candidate_row[
                                'first_prompt_sha256'],
                            possible_online_spillover=(not current_guidance and
                                                       row['delta'] != 0)))
    return dict(schema='retry_trigger_effect_audit_v1',
                design_schema=design_schema,
                parent=str(output), design_sha256=sha(output / 'design.json'),
                completed=verified['completed'], expected=verified['expected'],
                rows=details,
                exposed_pairs=sum(item['current_guidance'] for item in details),
                no_exposure_reward_differences=sum(
                    item['possible_online_spillover'] for item in details),
                note='Descriptive exposure audit; a guided pair also includes subsequent changed writer and memory state, so this does not isolate causal per-attempt reward')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--allow-partial', action='store_true')
    args = parser.parse_args()
    report = analyze(args.output, require_complete=not args.allow_partial)
    save(args.output / ('effects_partial.json' if args.allow_partial else
                        'effects.json'), report)
    print(f"Audited {report['completed']}/{report['expected']} pairs; "
          f"{report['exposed_pairs']} currently guided")


if __name__ == '__main__':
    main()
