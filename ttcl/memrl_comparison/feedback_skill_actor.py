"""Feedback-grounded ALFWorld skill transfer; no model update or test learning.

The training input is an explicit, content-bound review of official train
episodes. A skill becomes usable only after positive effects in two distinct
games. At evaluation time the table is frozen. The empty-table arm uses the
same state tracker, model, seeds and action budget.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re

from ttcl.alfworld_comparison.environment import Actor, make_env
from ttcl.experience_evolution.environment import clean_command
from ttcl.experience_evolution.core import digest, seed
from ttcl.icl_mem0_comparison.protocol import read, save, sha
from .probe_state_progress_holdout import extended_goal


PREP = re.compile(r'(clean|cool|heat) (\w+ \d+) with (\w+ \d+)', re.I)
TAKE = re.compile(r'take (\w+ \d+) from (\w+ \d+)', re.I)
MOVE = re.compile(r'move (\w+ \d+) to (\w+ \d+)', re.I)


def effect(action: str, observation: str) -> tuple[str, str, str] | None:
    """Return a positive effect only when feedback confirms the exact action."""
    action, observation = action.lower().strip(), observation.lower()
    match = PREP.fullmatch(action)
    if match:
        verb, obj, tool = match.groups()
        if re.search(rf'\byou {verb} the {re.escape(obj)} using the {re.escape(tool)}\.', observation):
            return verb, obj, tool
    match = TAKE.fullmatch(action)
    if match:
        obj, source = match.groups()
        if re.search(rf'\byou pick up the {re.escape(obj)} from the {re.escape(source)}\.', observation):
            return 'take', obj, source
    match = MOVE.fullmatch(action)
    if match:
        obj, destination = match.groups()
        if re.search(rf'\byou move the {re.escape(obj)} to the {re.escape(destination)}\.', observation):
            return 'move', obj, destination
    return None


def build_review(source_roots: list[Path], data_root: Path) -> dict:
    """Extract inspectable targets, bound to input and action/feedback bytes."""
    games: dict[str, dict] = {}
    for root in source_roots:
        for path in sorted(root.glob('**/episode.json')):
            episode = read(path)
            game = episode.get('game', '')
            if not game.startswith('json_2.1.1/train/'):
                continue
            input_path = (data_root / game).resolve()
            if not input_path.is_file() or not input_path.is_relative_to(data_root.resolve()):
                continue
            events = []
            for index, row in enumerate(episode.get('trajectory', [])):
                match = PREP.fullmatch(row['action'].lower().strip())
                if not match:
                    continue
                verb, obj, tool = match.groups()
                result = effect(row['action'], row['observation'])
                if result is None and 'Nothing happens.' not in row['observation']:
                    continue
                events.append(dict(verb=verb, object=obj, tool=tool,
                                   positive=result is not None,
                                   action=row['action'], observation=row['observation'],
                                   evidence_sha256=digest(json.dumps(row, sort_keys=True, ensure_ascii=False)),
                                   source_episode=str(path.resolve()), source_episode_sha256=sha(path),
                                   trajectory_index=index))
            if not events:
                continue
            target = games.setdefault(game, dict(game=game, input_sha256=sha(input_path), events=[]))
            if target['input_sha256'] != sha(input_path):
                raise ValueError('Training input changed during review')
            for event in events:
                if event not in target['events']:
                    target['events'].append(event)
    return dict(schema='alf_feedback_skill_review_v1',
                source_roots=[str(root.resolve()) for root in source_roots],
                source_root_hashes=[digest('\n'.join(sorted(str(p.relative_to(root))+':'+sha(p)
                    for p in root.glob('**/episode.json')))) for root in source_roots],
                targets=sorted(games.values(), key=lambda row: row['game']),
                review_semantics='Exact action and environment feedback; positive effects require matching verb, object ID and tool ID; negatives require Nothing happens; manually inspected before acceptance',
                reviewed=False)


def learn(review: dict, data_root: Path) -> dict:
    if review.get('schema') != 'alf_feedback_skill_review_v1' or review.get('reviewed') is not True:
        raise ValueError('A reviewed annotation manifest is required')
    counts = defaultdict(lambda: defaultdict(set))
    for target in review['targets']:
        game = target['game']
        if not game.startswith('json_2.1.1/train/') or sha(data_root / game) != target['input_sha256']:
            raise ValueError(f'Train input binding changed: {game}')
        for row in target['events']:
            source_path = Path(row['source_episode'])
            if (not any(source_path.is_relative_to(Path(root)) for root in review['source_roots'])
                    or sha(source_path) != row['source_episode_sha256']):
                raise ValueError('Source episode binding changed')
            source = read(source_path)
            if source['game'] != game:
                raise ValueError('Source episode game mismatch')
            original = source['trajectory'][row['trajectory_index']]
            if (digest(json.dumps(original, sort_keys=True, ensure_ascii=False)) != row['evidence_sha256']
                    or original['action'] != row['action'] or original['observation'] != row['observation']):
                raise ValueError('Exact action-feedback binding changed')
            match = PREP.fullmatch(row['action'].lower().strip())
            if not match or tuple(match.groups()) != (row['verb'], row['object'], row['tool']):
                raise ValueError('Action annotation mismatch')
            positive = effect(row['action'], row['observation']) is not None
            if positive != row['positive'] or (not positive and 'Nothing happens.' not in row['observation']):
                raise ValueError('Effect annotation mismatch')
            key = (row['verb'], row['tool'].split()[0])
            counts[key]['positive' if positive else 'negative'].add(target['input_sha256'])
    skills = {}
    for (verb, tool), evidence in sorted(counts.items()):
        pos = evidence['positive']
        if len(pos) < 2:
            continue
        skills.setdefault(verb, []).append(dict(tool=tool, positive_games=len(pos),
            negative_games=len(evidence['negative']), source_input_sha256=sorted(pos)))
    for rows in skills.values():
        rows.sort(key=lambda row: (-row['positive_games'], row['negative_games'], row['tool']))
    return dict(schema='alf_feedback_skill_table_v1', skills=skills,
                review_sha256=digest(json.dumps(review, sort_keys=True, ensure_ascii=False)),
                training_game_count=len(review['targets']))


def public_state(task: str, messages: list[dict]) -> dict:
    try:
        kind, obj, destination = extended_goal(task)
    except ValueError:
        extra = ((r'clean some (\w+) and put it in (\w+)\.', 'clean'),
                 (r'put a cool (\w+) in (\w+)\.', 'cool'))
        for pattern, candidate in extra:
            match = re.fullmatch(pattern, task.lower())
            if match:
                kind, obj, destination = candidate, *match.groups()
                break
        else:
            raise
    visited, moved, prepared, held = [], set(), set(), set()
    current = None
    for index, message in enumerate(messages):
        if message['role'] != 'assistant' or index + 1 >= len(messages):
            continue
        previous_user = messages[index - 1]['content']
        prior_available = previous_user.split('\nAvailable commands:\n', 1)[1].splitlines()
        action = clean_command(message['content'], prior_available).strip().lower()
        feedback = messages[index + 1]['content'].split('\nAvailable commands:\n', 1)[0].lower()
        arrived = re.fullmatch(r'go to (\w+ \d+)', action)
        if arrived and f'you arrive at {arrived.group(1)}.' in feedback:
            current = arrived.group(1)
            visited.append(current)
        event = effect(action, feedback)
        if event:
            verb, item, _ = event
            if verb == 'take':
                held.add(item)
            elif verb == 'move':
                held.discard(item)
                if item.startswith(obj + ' '):
                    moved.add(item)
            elif item.startswith(obj + ' '):
                prepared.add(item)
    return dict(kind=kind, target=obj, destination=destination,
                visited=list(dict.fromkeys(visited)), current=current,
                held=sorted(held), moved=sorted(moved), prepared=sorted(prepared))


def choose_action(state: dict, available: list[str], skill_table: dict) -> tuple[str | None, str]:
    kind, obj, destination = state['kind'], state['target'], state['destination']
    target = rf'{re.escape(obj)} \d+'
    dest = rf'{re.escape(destination)} \d+'
    if kind == 'look':
        lamps = [a for a in available if re.fullmatch(rf'use {dest}', a)]
        if lamps:
            return lamps[0], 'goal_light'
    matching_move = [a for a in available if re.fullmatch(rf'move ({target}) to {dest}', a)]
    for action in matching_move:
        item = MOVE.fullmatch(action).group(1)
        if kind in ('simple', 'two') or item in state['prepared']:
            return action, 'finish_goal'
    if kind in ('clean', 'cool', 'heat'):
        held = [item for item in state['held'] if item.startswith(obj + ' ') and item not in state['prepared']]
        skill = skill_table.get('skills', {}).get(kind, [])
        if held and skill:
            tool_type = skill[0]['tool']
            tools = [a.removeprefix('go to ') for a in available if re.fullmatch(
                rf'go to {re.escape(tool_type)} \d+', a)]
            if state['current'] and re.fullmatch(rf'{re.escape(tool_type)} \d+', state['current']):
                tools.append(state['current'])
            tools.sort(key=lambda tool: (tool != state['current'], tool))
            if tools:
                tool = tools[0]
                if state['current'] != tool:
                    return 'go to ' + tool, 'learned_tool_navigation'
                opening = f'open {tool}'
                if opening in available:
                    return opening, 'learned_tool_open'
                prep = f'{kind} {held[0]} with {tool}'
                if prep in available:
                    return prep, 'learned_tool_effect'
        prep = [a for a in available if re.fullmatch(rf'{kind} {target} with \w+ \d+', a)
                and PREP.fullmatch(a).group(2) in held]
        if prep:
            return prep[0], 'local_prep_attempt'
    takes = [a for a in available if re.fullmatch(rf'take ({target}) from \w+ \d+', a)]
    for action in takes:
        item = TAKE.fullmatch(action).group(1)
        if item not in state['moved'] and item not in state['held']:
            return action, 'take_goal_object'
    return None, 'actor_search'


class FeedbackSkillActor(Actor):
    def __init__(self, plan: dict, task: str, skill_table: dict):
        super().__init__(plan)
        self.task, self.skill_table = task, skill_table

    def generate(self, messages: list[dict], random_seed: int) -> dict:
        user = messages[-1]['content']
        available = user.split('\nAvailable commands:\n', 1)[1].splitlines()
        try:
            state = public_state(self.task, messages)
            command, reason = choose_action(state, available, self.skill_table)
        except ValueError:
            state, command, reason = {}, None, 'unsupported_goal_fallback'
        if command is not None:
            return dict(text=command, finish_reason='public_feedback_skill',
                        usage={'prompt_tokens': 0, 'completion_tokens': 0}, seed=random_seed,
                        prompt_sha256=digest(json.dumps(messages, ensure_ascii=False)),
                        seconds=0.0, public_state=state, skill_reason=reason)
        augmented = [dict(m) for m in messages]
        if state:
            augmented[-1]['content'] += ('\n\nCurrent goal: ' + self.task +
                '\nVisited locations: ' + (', '.join(state['visited'][-20:]) or 'none') +
                '\nChoose a listed command that advances the unfinished goal. '
                'Search a new location when possible; revisit one when carrying the goal object.')
        result = super().generate(augmented, random_seed)
        result['public_state'], result['skill_reason'] = state, reason
        return result


def run(plan_path: Path, review_path: Path, cases_path: Path, output: Path, url: str, prepare_only: bool):
    plan_source, review, cases = read(plan_path), read(review_path), read(cases_path)
    data_root = Path(plan_source['alf']['data_root']).resolve()
    table = learn(review, data_root)
    for case in cases['cases']:
        if (not case.get('reviewed') or not case['game'].startswith('json_2.1.1/train/')
                or sha(data_root / case['game']) != case['input_sha256']):
            raise ValueError('Evaluation case is not content-bound official train')
        if case['input_sha256'] in {t['input_sha256'] for t in review['targets']}:
            raise ValueError('Training source overlaps evaluation case')
        environment = make_env(data_root / case['game'])
        try:
            initial = str(environment.reset()['feedback'])
        finally:
            environment.close()
        if (hashlib.sha256(initial.encode()).hexdigest() != case['initial_observation_sha256']
                or f'Your task is to: {case["public_task"]}' not in initial):
            raise ValueError('Evaluation public instruction changed')
    design = dict(schema='alf_feedback_skill_pilot_v1', cases=cases['cases'],
        arms=['empty_skill', 'learned_skill'], repeats=cases['repeats'],
        plan_path=str(plan_path), plan_sha256=sha(plan_path),
        review_path=str(review_path), review_sha256=sha(review_path),
        cases_path=str(cases_path), cases_sha256=sha(cases_path),
        skill_table=table, runner_sha256=sha(Path(__file__)), url=url,
        budget='Official train evaluation, two arms, same 50-step limit and actor-call limit; frozen skill table; no model training')
    output.mkdir(parents=True, exist_ok=True)
    frozen = output / 'design.json'
    if frozen.exists():
        if read(frozen) != design:
            raise ValueError('Frozen design changed')
    else:
        save(frozen, design)
    print(json.dumps({'design_sha256': sha(frozen), 'skills': table['skills']}), flush=True)
    if prepare_only:
        return
    plan = dict(plan_source['alf'], actor_url=url)
    for case in design['cases']:
        for repeat in design['repeats']:
            for arm in design['arms']:
                target = output / case['family'] / case['input_sha256'][:12] / str(repeat) / arm
                if (target / 'episode.json').exists():
                    if read(target / 'episode.json')['game'] != case['game']:
                        raise ValueError('Existing episode belongs to a different input')
                    continue
                target.mkdir(parents=True, exist_ok=True)
                actor = FeedbackSkillActor(plan, case['public_task'], table if arm == 'learned_skill' else {'skills': {}})
                try:
                    episode = actor.run_many([dict(game=case['game'], memory='',
                        seed=seed(repeat, case['game'], 0, 'actor'), output=target)])[0]
                finally:
                    actor.pool.shutdown(wait=True)
                print(json.dumps({'family': case['family'], 'input_sha256': case['input_sha256'],
                    'repeat': repeat, 'arm': arm,
                    'reward': episode['reward'], 'steps': episode['steps'],
                    'invalid': sum(not row['valid_command'] for row in episode['trajectory']),
                    'learned_actions': sum(g.get('skill_reason', '').startswith('learned_') for g in episode['generations'])}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--reviews', type=Path, required=True)
    parser.add_argument('--cases', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--url', default='http://127.0.0.1:18559')
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    run(args.plan.resolve(), args.reviews.resolve(), args.cases.resolve(), args.output.resolve(), args.url, args.prepare_only)


if __name__ == '__main__':
    main()
