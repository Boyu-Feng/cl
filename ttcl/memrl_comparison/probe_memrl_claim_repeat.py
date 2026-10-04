"""Post-hoc same-prompt repeat for MemRL claim-overlay win cases only.

This probes actor sampling noise. Selection follows inspected outcomes and
must never be counted as a new unbiased evaluation set.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ttcl.alfworld_comparison.environment import Actor
from ttcl.experience_evolution.core import seed
from ttcl.icl_mem0_comparison.protocol import read, save, sha
from .probe_memrl_claim_overlay import ClaimOverlayActor, design_for


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    source_design = read(source / 'design.json')
    expected, static = design_for(Path(source_design['origin']),
        Path(source_design['dataset_path']), source_design['url'])
    if source_design != expected:
        raise ValueError('Original overlay design changed')
    selections = []
    for repeat in source_design['repeats']:
        for index, case in enumerate(source_design['cases']):
            stem = source / str(repeat) / f'{index:02d}_{case["input_sha256"][:12]}'
            base, candidate = stem / 'memrl/episode.json', stem / 'memrl_static/episode.json'
            before, after = read(base), read(candidate)
            if before['reward'] == 0 and after['reward'] == 1:
                selections.append(dict(repeat=repeat, index=index,
                    input_sha256=case['input_sha256'], base_sha256=sha(base),
                    static_sha256=sha(candidate)))
    design = dict(schema='alf_memrl_claim_same_prompt_repeat_v1',
        source=str(source), source_design_sha256=sha(source / 'design.json'),
        selection='All inspected static-claim wins over native MemRL; post-hoc mechanism/noise control',
        cases=selections, arms=['memrl_repeat', 'memrl_static_repeat'],
        static_memory=source_design['static_memory'],
        url=source_design['url'], runner_sha256=sha(Path(__file__)),
        budget='One same-seed same-prompt duplicate per original arm for each selected win; max 50 official train steps')
    output.mkdir(parents=True, exist_ok=True)
    frozen = output / 'design.json'
    if frozen.exists():
        if read(frozen) != design:
            raise ValueError('Repeat design changed')
    else:
        save(frozen, design)
    print(json.dumps(dict(design_sha256=sha(frozen), selected=len(selections))), flush=True)
    plan = dict(read(Path(source_design['origin']) / 'plan.json')['alf'],
                actor_url=source_design['url'])
    for selection in selections:
        case = source_design['cases'][selection['index']]
        for arm in design['arms']:
            target = output / str(selection['repeat']) / f'{selection["index"]:02d}_{case["input_sha256"][:12]}' / arm
            target.mkdir(parents=True, exist_ok=True)
            episode_path = target / 'episode.json'
            if episode_path.exists():
                episode = read(episode_path)
                if episode['game'] != case['game'] or episode['memory'] != case['memrl_context']:
                    raise ValueError('Repeat input changed')
            else:
                actor = (Actor(plan) if arm == 'memrl_repeat' else
                         ClaimOverlayActor(plan, case['public_task'], static))
                try:
                    episode = actor.run_many([dict(game=case['game'],
                        memory=case['memrl_context'],
                        seed=seed(selection['repeat'], case['game'], 0, 'actor'), output=target)])[0]
                finally:
                    actor.pool.shutdown(wait=True)
            print(json.dumps(dict(index=selection['index'], repeat=selection['repeat'],
                                  arm=arm, reward=episode['reward'], steps=episode['steps'])),
                  flush=True)


if __name__ == '__main__':
    main()
