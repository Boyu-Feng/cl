"""Paired official-train probe: stage-gated trajectory claims atop frozen MemRL retrieval.

The original per-task MemRL snapshot, retrieval order, Q scores, and context
remain frozen. One arm adds independently sourced claims; one starts empty and
updates claims only after its own completed episodes. No writer or Q training.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from ttcl.alfworld_comparison.environment import Actor, make_env
from ttcl.experience_evolution.core import digest, seed
from ttcl.icl_mem0_comparison.protocol import read, save, sha
from .credit_probe import memory_arms
from .probe_trajectory_experience_actor import relevant_experience, snapshot_memory
from .trajectory_writer_learning import ClaimMemory, teacher_operations


FAMILIES = ('pick_clean_then_place_in_recep', 'pick_cool_then_place_in_recep',
            'pick_heat_then_place_in_recep')
REPEATS = (95101, 95102)
ARMS = ('memrl', 'memrl_static', 'memrl_online')
EXPERIENCE_PREFIX = ('\n\nRelevant experience verified in completed earlier tasks '
                     '(apply only when the stated condition holds):\n')


class ClaimOverlayActor(Actor):
    def __init__(self, plan: dict, task: str, memory: ClaimMemory):
        super().__init__(plan)
        self.task, self.memory = task, memory

    def generate(self, messages: list[dict], random_seed: int) -> dict:
        text, decision = relevant_experience(self.task, messages, self.memory)
        prompt = [dict(message) for message in messages]
        if text:
            prompt[-1]['content'] += EXPERIENCE_PREFIX + text
        result = super().generate(prompt, random_seed)
        result['experience_decision'] = decision
        result['experience_text_sha256'] = digest(text)
        return result


def source_cases(origin: Path) -> list[dict]:
    design = read(origin / 'design.json')
    if design['group_index'] != 7 or len(design['selected_games']) != 6:
        raise ValueError('Expected frozen group8 official-train source')
    plan = read(origin / 'plan.json')
    data_root = Path(plan['alf']['data_root'])
    cases = []
    for family in FAMILIES:
        for offset, game in enumerate(design['selected_games'][family], 1):
            index = design['bootstrap'][family]['index'] + offset
            case = f'alfworld/{family}/{design["repeat"]}/memrl/episode_{index:03d}'
            spec, arms = memory_arms(origin, case)
            path = data_root / spec['original_memrl']['game']
            if (spec['source_input_sha256'] != sha(path) or
                    spec['source_input_sha256'] != game['sha256'] or
                    not spec['original_memrl']['game'].startswith('json_2.1.1/train/')):
                raise ValueError('Original MemRL case input changed')
            env = make_env(path)
            try:
                initial = str(env.reset()['feedback'])
            finally:
                env.close()
            task = initial.split('Your task is to: ', 1)[1].splitlines()[0]
            cases.append(dict(case=case, family=family, game=spec['original_memrl']['game'],
                input_sha256=sha(path), public_task=task,
                initial_observation_sha256=hashlib.sha256(initial.encode()).hexdigest(),
                memrl_context=arms['full'], context_sha256=digest(arms['full']),
                source_row_sha256=spec['original_memrl_row_sha256'],
                retrieval_sha256=spec['retrieval_sha256'],
                snapshot_sha256=spec['snapshot_sha256'], reviewed=True))
    if len(cases) != 9 or len({c['input_sha256'] for c in cases}) != 9:
        raise ValueError('Expected nine distinct preparation games')
    return cases


def initial_claim_memory(dataset: dict, excluded: set[str]) -> tuple[ClaimMemory, list[dict]]:
    memory, source_rows, seen = ClaimMemory(), [], set()
    for row in dataset['rows']['train']:
        source_sha = row['input_sha256']
        if (source_sha in seen or source_sha in excluded or
                'group8' in row['source_episode']):
            continue
        seen.add(source_sha)
        path = Path(row['source_episode'])
        if sha(path) != row['source_episode_sha256']:
            raise ValueError('Claim source episode changed')
        episode = read(path)
        if episode['game'] != row['game']:
            raise ValueError('Claim source game changed')
        result = memory.apply(teacher_operations(memory, episode), episode, source_sha)
        if result['rejected']:
            raise ValueError('Claim source has unsupported evidence')
        source_rows.append(dict(input_sha256=source_sha, episode=str(path),
                                episode_sha256=sha(path)))
    if len(source_rows) < 20 or not memory.entries:
        raise ValueError('Insufficient content-disjoint claim sources')
    return memory, source_rows


def design_for(origin: Path, dataset_path: Path, url: str) -> tuple[dict, ClaimMemory]:
    cases = source_cases(origin)
    dataset = read(dataset_path)
    static, source_rows = initial_claim_memory(dataset, {c['input_sha256'] for c in cases})
    design = dict(schema='alf_memrl_stage_claim_overlay_train_v1',
        origin=str(origin), origin_design_sha256=sha(origin / 'design.json'),
        origin_plan_sha256=sha(origin / 'plan.json'),
        dataset_path=str(dataset_path), dataset_sha256=sha(dataset_path),
        source_rows=source_rows, cases=cases, repeats=list(REPEATS), arms=list(ARMS),
        static_memory=snapshot_memory(static),
        runner_sha256=sha(Path(__file__)), url=url,
        selection='All three official-train cases in each clean, cool, heat family of frozen group8, ordered by source design before overlay results',
        budget='Two actor seeds; original frozen per-case MemRL retrieval and Q context; three arms; max 50 environment actions per episode',
        caveat='Fixed-snapshot MemRL+claims development diagnostic; not a fully evolving MemRL or valid_unseen evaluation')
    return design, static


def run(origin: Path, dataset_path: Path, output: Path, url: str, prepare_only: bool) -> None:
    design, static = design_for(origin, dataset_path, url)
    output.mkdir(parents=True, exist_ok=True)
    frozen = output / 'design.json'
    if frozen.exists():
        if read(frozen) != design:
            raise ValueError('Frozen overlay design changed')
    else:
        save(frozen, design)
    print(json.dumps(dict(design_sha256=sha(frozen), cases=len(design['cases']),
                          static_claims=static.public_state())), flush=True)
    if prepare_only:
        return
    source_plan = read(origin / 'plan.json')
    plan = dict(source_plan['alf'], actor_url=url)
    for repeat in REPEATS:
        online = ClaimMemory()
        for index, case in enumerate(design['cases']):
            for arm in ARMS:
                target = output / str(repeat) / f'{index:02d}_{case["input_sha256"][:12]}' / arm
                target.mkdir(parents=True, exist_ok=True)
                memory = static if arm == 'memrl_static' else online if arm == 'memrl_online' else None
                before = snapshot_memory(memory) if memory is not None else {}
                if (target / 'memory_before.json').exists():
                    if read(target / 'memory_before.json') != before:
                        raise ValueError('Memory before task changed')
                else:
                    save(target / 'memory_before.json', before)
                episode_path = target / 'episode.json'
                if episode_path.exists():
                    episode = read(episode_path)
                    if (episode['game'] != case['game'] or
                            episode['memory'] != case['memrl_context']):
                        raise ValueError('Existing episode has wrong frozen MemRL input')
                else:
                    actor = (Actor(plan) if memory is None else
                             ClaimOverlayActor(plan, case['public_task'], memory))
                    try:
                        episode = actor.run_many([dict(game=case['game'],
                            memory=case['memrl_context'],
                            seed=seed(repeat, case['game'], 0, 'actor'), output=target)])[0]
                    finally:
                        actor.pool.shutdown(wait=True)
                update = None
                if arm == 'memrl_online':
                    operations = teacher_operations(online, episode)
                    result = online.apply(operations, episode, case['input_sha256'])
                    if result['rejected']:
                        raise ValueError('Online update unsupported by source trajectory')
                    update = dict(schema='alf_memrl_claim_update_v1', game=case['game'],
                        input_sha256=case['input_sha256'], source_episode_sha256=sha(episode_path),
                        operations=operations, applied=result['applied'], reviewed=True)
                    path = target / 'reviewed_update.json'
                    if path.exists():
                        if read(path) != update:
                            raise ValueError('Claim update changed')
                    else:
                        save(path, update)
                after = snapshot_memory(memory) if memory is not None else {}
                if (target / 'memory_after.json').exists():
                    if read(target / 'memory_after.json') != after:
                        raise ValueError('Memory after task changed')
                else:
                    save(target / 'memory_after.json', after)
                print(json.dumps(dict(repeat=repeat, index=index, arm=arm,
                    family=case['family'], input_sha256=case['input_sha256'],
                    reward=episode['reward'], steps=episode['steps'],
                    new_claims=len(update['operations']) if update else 0,
                    activated_turns=sum(bool(g.get('experience_decision', {}).get('claim_keys'))
                                        for g in episode['generations']))), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--origin', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--url', default='http://127.0.0.1:18559')
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    run(args.origin.resolve(), args.dataset.resolve(), args.output.resolve(),
        args.url, args.prepare_only)


if __name__ == '__main__':
    main()
