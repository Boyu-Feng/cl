"""Single-card on/off CLBench replay from one shared native MemRL snapshot.

Each target task begins independently from the same per-repeat source memory
state. The only prompt difference is one frozen, content-bound public evidence
card. This diagnoses a card, not an entire online-chain policy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil

from ttcl.icl_mem0_comparison.protocol import read, save, sha
from .credit_online_ablation import ControlledActorClient
from .evidence_selected_memory import compact
from .memory import Memory
from .worker import cl_cell


class FixedCardMemory(Memory):
    def __init__(self, plan, client, directory, calibration, card, embedder=None):
        super().__init__(plan, client, directory, calibration, embedder=embedder)
        self.card = card

    def retrieve(self, query):
        result = super().retrieve(query)
        source = self.card['source_binding']['initial_query_sha256']
        lines = [f'  step {x["step"]}: {compact(x["action"], 170)} '
                 f'→ {compact(x["feedback"], 260)}' for x in self.card['selected']]
        text = ('\nVerbatim public evidence from one past instance; verify current inputs '
                f'[source {source[:12]}]:\n' + '\n'.join(lines))
        proposed = result['context'] + text
        tokens = len(self.client.tokenizer.encode(proposed, add_special_tokens=False))
        if tokens > self.plan['memory_tokens']:
            raise ValueError('Frozen card does not fit matched memory budget')
        result['context'] = proposed
        result['tokens'] = tokens
        result['context_sha256'] = hashlib.sha256(proposed.encode()).hexdigest()
        result['fixed_card_source_sha256'] = source
        result['fixed_card_rows_sha256'] = self.card['rows_sha256']
        return result


def run(origin, source_root, output, domain, repeats, indices, url):
    plan = read(origin / 'plan.json')
    if domain not in plan['tasks'] or any(r not in plan['repeats'] for r in repeats):
        raise ValueError('Unknown CLBench domain or repeat')
    if any(i < 1 or i >= int(plan['tasks'][domain] * .2) for i in indices):
        raise ValueError('Targets must be after source and inside calibration prefix')
    card_path = source_root / 'evidence_selected_v3' / 'memory' / 'evidence_memory_000000.json'
    card = read(card_path)
    source_trace = source_root / 'evidence_selected_v3' / 'episode_001' / 'public_trajectory.json'
    if (not card['selected'] or
            card['source_binding']['public_content_sha256'] !=
            hashlib.sha256(json.dumps(read(source_trace), sort_keys=True,
                                      ensure_ascii=False, allow_nan=False).encode()).hexdigest()):
        raise ValueError('Frozen card source trajectory changed')
    target_bindings = {str(repeat): {str(index): read(origin / 'runs' / 'clbench' /
        domain / str(repeat) / 'none' / f'episode_{index+1:03d}' / 'row.json')[
            'initial_query_sha256'] for index in indices} for repeat in repeats}
    if any(value == card['source_binding']['initial_query_sha256']
           for per_repeat in target_bindings.values() for value in per_repeat.values()):
        raise ValueError('A target repeats the source input content; choose distinct targets')
    snapshots = {str(repeat): source_root.parent /
        source_root.name.replace('pilot_v3', f'repeat{repeat}_v3') /
        'memrl' / 'episode_001' / 'memory_after.json' for repeat in repeats}
    # A source run for repeat 303 uses the original pilot directory.
    snapshots[str(repeats[0])] = source_root / 'memrl' / 'episode_001' / 'memory_after.json'
    if any(not p.is_file() for p in snapshots.values()):
        raise ValueError('Missing same-repeat native source memory snapshot')
    design = dict(schema='fixed_raw_card_single_state_pair_v1',
        origin=str(origin), origin_plan_sha256=sha(origin / 'plan.json'),
        source_root=str(source_root), source_card_sha256=sha(card_path),
        source_trace_sha256=sha(source_trace), domain=domain,
        repeats=repeats, indices=indices,
        source_snapshots={k:dict(path=str(v), sha256=sha(v)) for k,v in snapshots.items()},
        target_bindings=target_bindings, url=url.rstrip('/'),
        runner_sha256=sha(Path(__file__)),
        arms=['without_card','with_card'],
        policy='Each cell restores the same frozen native memory snapshot; only one literal card differs; targets are calibration-prefix instances')
    output.mkdir(parents=True, exist_ok=True)
    design_path = output / 'design.json'
    if design_path.is_file() and read(design_path) != design:
        raise ValueError('Frozen single-card design changed')
    save(design_path, design)
    frozen_source = output / 'source'
    frozen_source.mkdir(exist_ok=True)
    for name in ('probe_fixed_evidence_card.py', 'memory.py', 'worker.py',
                 'evidence_selected_memory.py', 'credit_online_ablation.py'):
        original = Path(__file__).with_name(name)
        target = frozen_source / name
        if target.exists() and sha(target) != sha(original):
            raise ValueError('Frozen runner source changed')
        if not target.exists():
            shutil.copy2(original, target)
    os.environ.setdefault('TTCL_BENCH', str(origin / 'source' / 'bench'))
    os.chdir(os.environ['TTCL_BENCH'])
    plan['url'] = url.rstrip('/')
    plan['q_min_threshold'] = plan['q_min_thresholds']['clbench']
    rows = []
    for repeat in repeats:
        client = ControlledActorClient(plan, repeat, plan['alf']['actor_temperature'])
        cal = plan['calibration'][domain]
        native = Memory(plan, client, output / str(repeat) / 'without_card' / 'memory', cal)
        candidate = FixedCardMemory(plan, client, output / str(repeat) / 'with_card' / 'memory',
                                    cal, card, embedder=native.service.embedding_provider)
        for index in indices:
            for arm, memory in [('without_card', native), ('with_card', candidate)]:
                memory.restore(snapshots[str(repeat)])
                target = output / str(repeat) / f'target_{index:03d}' / arm
                if (target / 'row.json').is_file():
                    result = read(target / 'row.json')
                    if sha(target / 'memory_after.json') != result['memory_after_sha256']:
                        raise ValueError('Completed target changed')
                else:
                    result = cl_cell(plan, client, memory, domain, index, arm, target)
                if (result['status'] != 'complete' or
                        result.get('memory_update_status') == 'failed' or
                        result['initial_query_sha256'] != target_bindings[str(repeat)][str(index)]):
                    raise RuntimeError(result.get('error', 'Unscored fixed-card target'))
                retrieval = read(target / 'retrieval.json')
                if arm == 'with_card' and retrieval.get('fixed_card_rows_sha256') != card['rows_sha256']:
                    raise ValueError('Frozen card was not exposed')
                row = dict(repeat=repeat, canonical_index=index, arm=arm,
                    input_sha256=result['initial_query_sha256'],
                    reward=result['reward'], actor_calls=result['actor_calls'],
                    native_context_sha256=read(target / 'retrieval.json')['context_sha256'])
                rows.append(row)
                print(json.dumps(row), flush=True)
    save(output / 'rows.json', rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--origin', type=Path, required=True)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--domain', required=True)
    parser.add_argument('--repeats', type=int, nargs='+', default=[303, 404])
    parser.add_argument('--indices', type=int, nargs='+', default=[1, 2])
    parser.add_argument('--url', default='http://127.0.0.1:18559')
    args = parser.parse_args()
    run(args.origin.resolve(), args.source_root.resolve(), args.output.resolve(),
        args.domain, args.repeats, args.indices, args.url)


if __name__ == '__main__':
    main()
