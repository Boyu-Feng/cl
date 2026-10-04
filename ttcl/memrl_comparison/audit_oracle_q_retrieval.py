"""Replay later native retrieval with one locally observed Shapley Q value.

This is a fixed-snapshot mechanism audit, not a prospective credit predictor.
The local Shapley sign comes from an earlier task and need not transfer to the
later task. No actor, writer, benchmark episode, or online memory update runs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics

from ttcl.icl_mem0_comparison.protocol import Client, read, sha
from .audit_alf_coalition_credit import audit as audit_coalitions
from .audit_credit_q_opportunity import audit as audit_opportunities
from .memory import Embedder, Memory


def run(outputs: list[Path], workdir: Path) -> dict:
    opportunity = audit_opportunities(outputs)
    if opportunity['totals']['robust_actionable_q_opportunity'] == 0:
        raise ValueError('No downstream ranking opportunities')
    workdir.mkdir(parents=True, exist_ok=True)
    embedder = None
    rows = []
    for source in opportunity['outputs']:
        coalition = audit_coalitions(Path(source['output']))
        design = read(Path(source['output']) / 'design.json')
        origin = Path(design['origin'])
        plan = read(origin / 'plan.json')
        # Retrieval uses the local embedding model; this client only supplies
        # token counting. Its actor endpoint is never contacted here.
        client = Client(plan, int(read(origin / 'design.json')['repeat']))
        if embedder is None:
            embedder = Embedder(plan['embedding'])
        grouped = {}
        for unit in coalition['units']:
            if unit['full_repeat_reward_equal']:
                grouped.setdefault(unit['case'], []).append(unit)
        for case in source['cases']:
            for item in case['memories']:
                if not item['actionable_q_opportunity']:
                    continue
                mid = item['memory_id']
                index = item['memory_index']
                credits = [unit['shapley'][index]
                           for unit in grouped.get(case['case'], [])]
                if (sum(value * item['robust_shapley_sign'] > 0
                        for value in credits) < 2 or
                        any(value * item['robust_shapley_sign'] < 0
                            for value in credits)):
                    raise ValueError('Claimed robust sign disagrees with audited units')
                oracle_q = statistics.fmean(credits)
                family = case['case'].split('/')[1]
                base = origin / 'runs' / 'alfworld' / family / str(client.repeat) / 'memrl'
                relevant = sorted(set(item['candidate_not_selected_at']
                                      if item['robust_shapley_sign'] > 0
                                      else item['selected_at']))
                for later_index in relevant:
                    episode = base / f'episode_{later_index:03d}'
                    snapshot = base / f'episode_{later_index-1:03d}' / 'memory_after.json'
                    source_retrieval = episode / 'retrieval_1.json'
                    reference = read(source_retrieval)
                    memory = Memory(plan, client, workdir / f'case_{len(rows):03d}',
                                    plan['calibration']['alfworld'], embedder=embedder)
                    memory.restore(snapshot)
                    native = memory.retrieve(reference['query'])
                    if (native['ids'] != reference['ids'] or
                            native['context_sha256'] != reference['context_sha256'] or
                            mid not in {candidate['memory_id']
                                        for candidate in native['candidates']}):
                        raise ValueError(f'Native retrieval mismatch: {episode}')
                    native_q = memory.service._q_cache[mid]
                    memory.service._q_cache[mid] = oracle_q
                    changed = memory.retrieve(reference['query'])
                    rows.append(dict(source_output=source['output'],
                                     source_design_sha256=source['source_design_sha256'],
                                     coalition_design_sha256=source['design_sha256'],
                                     prior_case=case['case'],
                                     prior_input_sha256=case['input_sha256'],
                                     memory_id=mid, memory_index=index,
                                     memory_text_sha256=item['memory_text_sha256'],
                                     stable_shapley=credits,
                                     later_index=later_index,
                                     later_input_sha256=read(episode / 'row.json')['input_sha256'],
                                     later_snapshot_sha256=sha(snapshot),
                                     source_retrieval_sha256=sha(source_retrieval),
                                     native_q=native_q, oracle_q=oracle_q,
                                     native_ids=native['ids'], oracle_ids=changed['ids'],
                                     native_context_sha256=native['context_sha256'],
                                     oracle_context_sha256=changed['context_sha256'],
                                     order_changed=native['ids'] != changed['ids'],
                                     membership_changed=set(native['ids']) != set(changed['ids']),
                                     directionally_desired_membership_change=(
                                         mid in changed['ids'] if oracle_q > 0
                                         else mid not in changed['ids'])))
    return dict(schema='alf_oracle_q_retrieval_audit_v1',
                script_sha256=sha(Path(__file__)),
                opportunity_script_sha256=opportunity['script_sha256'],
                rows=rows,
                totals=dict(opportunities=len(rows),
                            order_changed=sum(row['order_changed'] for row in rows),
                            membership_changed=sum(row['membership_changed'] for row in rows),
                            directionally_desired_membership_change=sum(
                                row['directionally_desired_membership_change'] for row in rows)),
                caveat='Earlier-task Shapley is a posthoc local oracle, not later-task truth or a learned prediction; fixed snapshots do not estimate online chain reward')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outputs', type=Path, nargs='+', required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    result = run([p.resolve() for p in args.outputs],
                 args.report.parent / 'retrieval_workspace')
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result['totals']))


if __name__ == '__main__':
    main()
