"""Test source-entity rebinding in one exact-text ALFWorld memory.

Two reviewed train games were selected posthoc after observing opposite
credit signs and source-entity leakage. This is a fixed-snapshot mechanism
test with official reward, not a general policy or online Q/writer update.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

from ttcl.icl_mem0_comparison.protocol import Client, read, save, sha
from .credit_probe import memory_arms, run_alf
from .probe_delayed_memory_activation import design_for as stage_design_for
from .probe_goal_slot_credit import _slots


REPEATS = (94201, 94202, 94203)
ARMS = ('native', 'rebound', 'none', 'native_repeat', 'rebound_repeat')


def _replace_entity(value: str, old: str, new: str) -> tuple[str, int]:
    pattern = re.compile(r'\b' + re.escape(old) + r'(?:\s+\d+)?\b', re.I)
    return pattern.subn(new, value)


def _strip_instance_id(value: str, entity: str) -> tuple[str, int]:
    pattern = re.compile(r'\b' + re.escape(entity) + r'\s+\d+\b', re.I)
    return pattern.subn(entity, value)


def rebind(context: str, current_goal: str) -> tuple[str, dict]:
    if not context.startswith('Task: ') or '\nExperience: ' not in context:
        raise ValueError('Expected source-bound MemRL experience')
    source_task, lesson = context.split('\nExperience: ', 1)
    source_goal = source_task.removeprefix('Task: ')
    source_obj, source_dst, source_op = _slots(source_goal)
    current_obj, current_dst, current_op = _slots(current_goal)
    if source_op != current_op or source_obj == source_dst or \
            current_obj == current_dst:
        raise ValueError('Incompatible task operation or ambiguous binding')
    adapted, obj_count = _replace_entity(lesson, source_obj, current_obj)
    adapted, dst_count = _replace_entity(adapted, source_dst, current_dst)
    if obj_count < 2 or dst_count < 2:
        raise ValueError('Memory lacks repeated source entities to bind')
    # Instance numbers belong to the old game, including when the object
    # name happens to match in both tasks.
    adapted, current_obj_ids = _strip_instance_id(adapted, current_obj)
    adapted, current_dst_ids = _strip_instance_id(adapted, current_dst)
    text = ('Earlier failure lesson with source-game entity names adapted to '
            'the current public goal. Verify every step against current '
            'observations; this does not describe a completed current task.\n'
            'Experience: ' + adapted)
    if source_obj != current_obj and re.search(
            r'\b' + re.escape(source_obj) + r'\b', text, re.I):
        raise ValueError('Unbound source object remains')
    if source_dst != current_dst and re.search(
            r'\b' + re.escape(source_dst) + r'\b', text, re.I):
        raise ValueError('Unbound source destination remains')
    return text, dict(source_goal=source_goal, current_goal=current_goal,
                      source_object=source_obj, current_object=current_obj,
                      source_destination=source_dst,
                      current_destination=current_dst,
                      source_object_replacements=obj_count,
                      source_destination_replacements=dst_count,
                      numbered_current_object_replacements=current_obj_ids,
                      numbered_current_destination_replacements=current_dst_ids)


def design_for(sign_report: Path, output: Path, url: str) -> dict:
    earlier = stage_design_for(sign_report, output, url)
    cases = []
    for source in earlier['cases']:
        origin = Path(source['origin'])
        spec, arms = memory_arms(origin, source['case'])
        if (spec['source_input_sha256'] != source['game_sha256'] or
                spec['memory_text_sha256'][source['memory_id']] !=
                source['memory_text_sha256']):
            raise ValueError('Source text or reviewed game changed')
        native = arms[f"only_{source['memory_index']}"]
        if native != source['memory_context']:
            raise ValueError('Frozen native context changed')
        rebound, binding = rebind(native, source['public_task'])
        cases.append(dict(index=source['index'], origin=source['origin'],
                          source_case=source['case'], game=source['game'],
                          game_sha256=source['game_sha256'],
                          source_plan_sha256=source['source_plan_sha256'],
                          review_sha256=source['review_sha256'],
                          public_task=source['public_task'],
                          memory_id=source['memory_id'],
                          memory_text_sha256=source['memory_text_sha256'],
                          local_shapley_sign=source['local_shapley_sign'],
                          native_context=native,
                          native_context_sha256=hashlib.sha256(
                              native.encode()).hexdigest(),
                          rebound_context=rebound,
                          rebound_context_sha256=hashlib.sha256(
                              rebound.encode()).hexdigest(),
                          entity_binding=binding))
    return dict(schema='alf_source_entity_rebinding_v1',
                sign_report_sha256=sha(sign_report),
                stage_runner_sha256=earlier['runner_sha256'],
                runner_sha256=sha(Path(__file__)),
                credit_probe_sha256=sha(Path(__file__).with_name('credit_probe.py')),
                cases=cases, repeats=list(REPEATS), arms=list(ARMS), url=url,
                budget='Two reviewed official train games; three new actor seeds; five arms each; at most 50 environment actions per arm; fixed single-memory context; no Q/writer updates',
                caveat='Posthoc selected source-entity leak. Rewriting a failed source lesson can alter semantics; effects must be checked against no-memory and both repeat controls.')


def run(sign_report: Path, output: Path,
        url: str, prepare_only: bool) -> None:
    design = design_for(sign_report, output, url)
    output.mkdir(parents=True, exist_ok=True)
    path = output / 'design.json'
    if path.exists():
        if read(path) != design:
            raise ValueError('Frozen rebinding design changed')
    else:
        save(path, design)
    print(json.dumps(dict(design_sha256=sha(path),
                          bindings=[case['entity_binding']
                                    for case in design['cases']])), flush=True)
    if prepare_only:
        return
    for case in design['cases']:
        origin = Path(case['origin'])
        plan = read(origin / 'plan.json')
        plan['url'] = url
        plan['alf']['actor_url'] = url
        spec, _ = memory_arms(origin, case['source_case'])
        contexts = dict(native=case['native_context'],
                        rebound=case['rebound_context'], none='',
                        native_repeat=case['native_context'],
                        rebound_repeat=case['rebound_context'])
        for repeat in REPEATS:
            target = output / f"case_{case['index']}" / f'repeat_{repeat}'
            target.mkdir(parents=True, exist_ok=True)
            summary = target / 'summary.json'
            if summary.exists():
                continue
            result = run_alf(plan, Client(plan, repeat),
                             dict(spec, repeat=repeat), contexts, target)
            save(summary, dict(case=case['index'], repeat=repeat,
                               design_sha256=sha(path), replay=result))
            print(json.dumps(dict(case=case['index'], repeat=repeat,
                                  rewards={name:arm['reward']
                                           for name,arm in result.items()})),
                  flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sign-report', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--url', default='http://127.0.0.1:18559')
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    run(args.sign_report.resolve(), args.output.resolve(), args.url,
        args.prepare_only)


if __name__ == '__main__':
    main()
