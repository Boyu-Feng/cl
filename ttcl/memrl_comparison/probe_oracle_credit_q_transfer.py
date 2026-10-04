"""Fixed-snapshot test of a robust local Shapley value as the next-task Q.

This is an oracle mechanism test on an already inspected official-train pair.
It replaces one Q cache entry with the mean stable-coalition Shapley credit,
then compares the resulting retrieved text with native text on the next task.
There is no online writer/Q chain and no prospective policy claim.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics

from ttcl.icl_mem0_comparison.protocol import Client, read, save, sha
from .audit_alf_coalition_credit import audit as audit_coalitions
from .credit_probe import memory_arms, run_alf
from .memory import Embedder, Memory


PRIOR_CASE = 'alfworld/pick_heat_then_place_in_recep/92721/memrl/episode_007'
NEXT_CASE = 'alfworld/pick_heat_then_place_in_recep/92721/memrl/episode_008'
MEMORY_INDEX = 0
REPEATS = (93901, 93902, 93903)


def design_for(origin: Path, prior_probe: Path, output: Path,
               url: str) -> dict:
    prior = audit_coalitions(prior_probe)
    if prior['missing'] or prior['audited'] != prior['expected']:
        raise ValueError('Incomplete source coalition probe')
    prior_design = read(prior_probe / 'design.json')
    if prior_design['origin'] != str(origin):
        raise ValueError('Coalition and native source differ')
    units = [u for u in prior['units'] if u['case'] == PRIOR_CASE and
             u['full_repeat_reward_equal']]
    if len(units) < 2 or any(u['shapley'][MEMORY_INDEX] <= 0 for u in units):
        raise ValueError('Missing robust positive local Shapley credit')
    original_spec, _ = memory_arms(origin, PRIOR_CASE)
    mid = original_spec['ids'][MEMORY_INDEX]
    causal_q = statistics.fmean(u['shapley'][MEMORY_INDEX] for u in units)
    plan = read(origin / 'plan.json')
    plan['url'] = url
    plan['alf']['actor_url'] = url
    next_spec, _ = memory_arms(origin, NEXT_CASE)
    next_path = origin / 'runs' / NEXT_CASE
    snapshot = next_path.parent / 'episode_007' / 'memory_after.json'
    client = Client(plan, 92721)
    memory = Memory(plan, client, output / 'retrieval_memory',
                    plan['calibration']['alfworld'],
                    embedder=Embedder(plan['embedding']))
    memory.restore(snapshot)
    reference = read(next_path / 'retrieval_1.json')
    original = memory.retrieve(reference['query'])
    if original['ids'] != reference['ids'] or \
            original['context_sha256'] != reference['context_sha256'] or \
            mid not in {item['memory_id'] for item in reference['candidates']}:
        raise ValueError('Native next-task retrieval did not reproduce')
    native_q = memory.service._q_cache[mid]
    memory.service._q_cache[mid] = causal_q
    changed = memory.retrieve(reference['query'])
    if changed['ids'] == original['ids'] or mid not in changed['ids']:
        raise ValueError('Causal Q did not change next-task selected memories')
    return dict(schema='alf_oracle_credit_q_transfer_v1',
                origin=str(origin), prior_probe=str(prior_probe),
                prior_probe_design_sha256=sha(prior_probe / 'design.json'),
                prior_probe_analysis_sha256=sha(prior_probe / 'analysis.json'),
                source_design_sha256=sha(origin / 'design.json'),
                source_plan_sha256=sha(origin / 'plan.json'),
                prior_case=PRIOR_CASE, next_case=NEXT_CASE,
                memory_id=mid, memory_index=MEMORY_INDEX,
                memory_text_sha256=original_spec['memory_text_sha256'][mid],
                prior_input_sha256=original_spec['source_input_sha256'],
                next_input_sha256=next_spec['source_input_sha256'],
                next_snapshot_sha256=sha(snapshot),
                next_source_retrieval_sha256=sha(next_path / 'retrieval_1.json'),
                stable_prior_repeats=[u['repeat'] for u in units],
                stable_prior_shapley=[u['shapley'][MEMORY_INDEX] for u in units],
                native_q=native_q, oracle_q=causal_q,
                original_ids=original['ids'], changed_ids=changed['ids'],
                original_context_sha256=original['context_sha256'],
                changed_context_sha256=changed['context_sha256'],
                original_context=original['context'],
                changed_context=changed['context'],
                url=url, repeats=list(REPEATS),
                runner_sha256=sha(Path(__file__)),
                memory_adapter_sha256=sha(Path(__file__).with_name('memory.py')),
                credit_probe_sha256=sha(Path(__file__).with_name('credit_probe.py')),
                budget='Three arms per actor seed: native, oracle_q, native_repeat; at most 50 environment actions each; fixed snapshot and no online updates',
                caveat='Q is set directly to an oracle local Shapley mean, not learned from action-before features or updated through native EMA; this is a development mechanism intervention')


def run(origin: Path, prior_probe: Path, output: Path,
        url: str, prepare_only: bool) -> None:
    design = design_for(origin, prior_probe, output, url)
    output.mkdir(parents=True, exist_ok=True)
    path = output / 'design.json'
    if path.exists():
        if read(path) != design:
            raise ValueError('Frozen oracle-Q transfer design changed')
    else:
        save(path, design)
    if prepare_only:
        print(json.dumps(dict(design_sha256=sha(path),
                              native=design['original_ids'],
                              oracle_q=design['changed_ids'],
                              q=[design['native_q'], design['oracle_q']])))
        return
    plan = read(origin / 'plan.json')
    plan['url'] = url
    plan['alf']['actor_url'] = url
    spec, _ = memory_arms(origin, NEXT_CASE)
    for repeat in REPEATS:
        target = output / f'actor_repeat_{repeat}'
        target.mkdir(parents=True, exist_ok=True)
        summary = target / 'summary.json'
        if summary.exists():
            continue
        contexts = dict(native=design['original_context'],
                        oracle_q=design['changed_context'],
                        native_repeat=design['original_context'])
        result = run_alf(plan, Client(plan, repeat),
                         dict(spec, repeat=repeat), contexts, target)
        save(summary, dict(repeat=repeat, design_sha256=sha(path), replay=result))
        print(json.dumps(dict(repeat=repeat,
                              rewards={name:row['reward'] for name,row in result.items()})),
              flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--origin', type=Path, required=True)
    parser.add_argument('--prior-probe', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--url', default='http://127.0.0.1:18559')
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    run(args.origin.resolve(), args.prior_probe.resolve(), args.output.resolve(),
        args.url, args.prepare_only)


if __name__ == '__main__':
    main()
