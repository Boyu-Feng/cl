"""Frozen ExpeL rules/examples from the original Delta training environments.

This reuses the 240 already executed actor/baseline episodes on 144 training
instances. It is shared development-task exposure, not identical writer input:
the Delta writer itself did not receive all baseline and final-task trajectories.
No evaluation trajectory is used to build or update this bank.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
from types import SimpleNamespace
from typing import List, Tuple

from ttcl.experience_evolution.core import FAMILIES, read, save, seed
from ttcl.reflexion_expel.upstream import fragments

MAX_RULES = 20
MAX_SUCCESS_BATCH = 8
CRITIC_TOKENS = 768
CRITIC_SEED = 92611


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def _text_sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def _json_sha(value):
    return _text_sha(json.dumps(value, sort_keys=True, ensure_ascii=False))


def _family(game):
    task = PurePosixPath(game).parts[2]
    found = [family for family in FAMILIES if task.startswith(family + '-')]
    if len(found) != 1:
        raise ValueError(f'Unknown ALFWorld task family: {game}')
    return found[0]


def _public_text(episode):
    """Complete actions and public feedback, with no hidden game/score metadata."""
    lines = [episode['initial_observation']]
    for step in episode['trajectory']:
        lines.extend(['> ' + step['action'], step['observation']])
    lines.append('STATUS: ' + ('SUCCESS' if episode['reward'] == 1 else 'FAIL'))
    return '\n'.join(lines)


def _training_sources(source_root):
    """Read only declared train games; bind every episode and game to its hash."""
    source_root = Path(source_root).resolve()
    plan_path, hashes_path = source_root / 'plan.json', source_root / 'data_hashes.json'
    plan, declared_hashes = read(plan_path), read(hashes_path)
    file_hashes = {str(p): _sha(p) for p in [plan_path, hashes_path]}
    data_root = Path(plan['data_root']).resolve()
    batch_size = int(plan['batch_sequences'])
    if batch_size < 1:
        raise ValueError('Invalid original training batch size')
    groups, seen = [], set()
    for ordinal, sequence in enumerate(plan['training']):
        if len(sequence['games']) != 3:
            raise ValueError('Expected the original three-task Delta curriculum')
        batch, position = divmod(ordinal, batch_size)
        prefix = source_root / f'training/delta/batch_{batch:03d}/seq_{position}'
        for task_position, game in enumerate(sequence['games']):
            game_path = PurePosixPath(game)
            if (game_path.is_absolute() or '..' in game_path.parts
                    or len(game_path.parts) < 4
                    or game_path.parts[:2] != ('json_2.1.1', 'train')):
                raise ValueError(f'ExpeL source must be an official train game: {game}')
            if game in seen:
                raise ValueError(f'Duplicate training game: {game}')
            seen.add(game)
            data_file = (data_root / game).resolve()
            if not data_file.is_relative_to(data_root):
                raise ValueError(f'Training game escapes data root: {game}')
            game_hash = _sha(data_file)
            if declared_hashes.get(game) != game_hash:
                raise ValueError(f'Original training game hash mismatch: {game}')
            file_hashes[str(data_file)] = game_hash
            # Stable priority: actual task branch first; then its empty baseline.
            names = [f'task_{task_position}']
            if task_position > 0:
                names.append(f'baseline_{task_position}')
            trials = []
            for name in names:
                path = prefix / name / 'episode.json'
                resolved = path.resolve()
                if not resolved.is_relative_to(source_root):
                    raise ValueError(f'Training episode escapes experiment root: {path}')
                episode = read(path)
                if episode['game'] != game:
                    raise ValueError(f'Training episode game mismatch: {path}')
                if episode.get('status') != 'complete':
                    raise ValueError(f'Incomplete source episode: {path}')
                if episode.get('actor_adapter_enabled') is not False:
                    raise ValueError(f'Source actor was not explicitly frozen: {path}')
                if episode.get('reward') not in (0, 1):
                    raise ValueError(f'Invalid ALFWorld source reward: {path}')
                if episode['steps'] != len(episode['trajectory']):
                    raise ValueError(f'Incomplete source trajectory: {path}')
                episode_hash = _sha(path)
                file_hashes[str(resolved)] = episode_hash
                trials.append({'source': str(resolved), 'source_sha256': episode_hash,
                               'branch': name, 'episode': episode})
            reset_keys = ['seed', 'initial_observation', 'initial_commands_sha256']
            if any(len({_json_sha(t['episode'][key]) for t in trials}) != 1
                   for key in reset_keys):
                raise ValueError(f'Source branches have mismatched resets/seeds: {game}')
            groups.append({'index': len(groups), 'game': game,
                           'family': _family(game), 'trials': trials})
    if not groups:
        raise ValueError('No original Delta training sources found')
    return groups, file_hashes


class _OfficialExpeL:
    """Load only pinned official prompt strings and rule update function bodies."""
    def __init__(self, root):
        root = Path(root) / 'expel'
        self.files = [root / name for name in ['expel.py', 'human.py', 'alfworld.py']]
        self.algorithm = fragments(root / 'expel.py',
            {'parse_rules', 'retrieve_rule_index', 'is_existing_rule', 'update_rules'},
            {'re': re, 'List': List, 'Tuple': Tuple})
        self.prompts = fragments(root / 'human.py',
            {'FORMAT_RULES_OPERATION_TEMPLATE', 'CRITIQUE_SUMMARY_SUFFIX',
             'human_critique_existing_rules_all_success_template',
             'human_critique_existing_rules_template', 'RULE_TEMPLATE'},
            {'HumanMessagePromptTemplate': SimpleNamespace(from_template=lambda text: text)})
        self.instructions = fragments(root / 'alfworld.py',
            {'SYSTEM_CRITIQUE_EXISTING_RULES_INSTRUCTION',
             'SYSTEM_CRITIQUE_ALL_SUCCESS_EXISTING_RULES_INSTRUCTION'})

    def messages(self, rules, successful, failed=None, task=''):
        comparison = failed is not None
        template = ('human_critique_existing_rules_template' if comparison
                    else 'human_critique_existing_rules_all_success_template')
        instruction = ('SYSTEM_CRITIQUE_EXISTING_RULES_INSTRUCTION' if comparison
                       else 'SYSTEM_CRITIQUE_ALL_SUCCESS_EXISTING_RULES_INSTRUCTION')
        system = ('You are an advanced reasoning agent that can add, edit or remove rules '
                  'from your existing rule set, based on forming new critiques of past task '
                  'trajectories. ' + self.instructions[instruction])
        human = self.prompts[template].format(instruction='', task=task,
            success_history=successful, fail_history=failed,
            existing_rules='\n'.join(f'{i}. {text}' for i, (text, _) in enumerate(rules, 1)))
        human += self.prompts['CRITIQUE_SUMMARY_SUFFIX'][
            'full' if len(rules) >= MAX_RULES else 'not_full']
        return [{'role': 'system', 'content': system}, {'role': 'user', 'content': human}]

    def update(self, rules, raw_response):
        operations = self.algorithm['parse_rules'](raw_response)
        rejected = [op for op in operations if op[0].startswith('EDIT ')
                    and int(op[0].split()[1]) < 1]
        accepted = [op for op in operations if op not in rejected]
        updated = self.algorithm['update_rules'](
            copy.deepcopy(rules), copy.deepcopy(accepted), list_full=len(rules) >= MAX_RULES)
        # Official update_rules orders by importance. Enforce the declared hard
        # storage budget only between entire rules, preserving its stable ties.
        return updated[:MAX_RULES], accepted, rejected, updated[MAX_RULES:]


def _generate(client, messages, path, random_seed):
    from .methods import cached_generation
    return cached_generation(client, messages, path, random_seed,
                             model='frozen-actor', tokens=CRITIC_TOKENS)


def _message_tokens(tokenizer, messages):
    if hasattr(tokenizer, 'apply_chat_template'):
        rendered = tokenizer.apply_chat_template(messages, tokenize=False,
                                                add_generation_prompt=True)
    else:
        rendered = '\n'.join(message['content'] for message in messages)
    return len(tokenizer.encode(rendered, add_special_tokens=False))


def build_bank(client, source_root, out, upstream_root, tokenizer=None):
    """Build once from source_root/plan.json; thereafter verify and reuse.

    Critic inference uses the untrained frozen base model. All same-game,
    same-seed mixed outcomes are contrasted first; then one successful episode
    per training game is processed in fixed batches of at most eight.
    """
    out = Path(out)
    tokenizer = tokenizer or client.tokenizer
    groups, source_hashes = _training_sources(source_root)
    upstream = _OfficialExpeL(upstream_root)
    source_hashes.update({str(p.resolve()): _sha(p) for p in upstream.files})
    hashes_path = out / 'source_hashes.json'
    if hashes_path.exists():
        if read(hashes_path) != source_hashes:
            raise ValueError('ExpeL source or upstream hashes changed since bank preparation')
    else:
        save(hashes_path, source_hashes)
    state_path, freeze_path = out / 'state.json', out / 'freeze.json'
    if state_path.exists():
        if not freeze_path.exists():
            raise ValueError('ExpeL bank has state without a completed freeze manifest')
        freeze = read(freeze_path)
        if (freeze['state_sha256'] != _sha(state_path)
                or freeze['source_hashes_sha256'] != _sha(hashes_path)):
            raise ValueError('Frozen ExpeL bank integrity mismatch')
        return read(state_path)
    jobs, successes = [], []
    for group in groups:
        succeeded = [t for t in group['trials'] if t['episode']['reward'] == 1]
        failed = [t for t in group['trials'] if t['episode']['reward'] == 0]
        if not succeeded:
            continue
        chosen = succeeded[0]
        episode = chosen['episode']
        example = {k: group[k] for k in ['index', 'game', 'family']}
        example.update(query=episode['initial_observation'], trajectory=_public_text(episode),
                       source=chosen['source'], source_sha256=chosen['source_sha256'],
                       seed=episode['seed'])
        successes.append(example)
        for failure in failed:
            jobs.append({'kind': 'compare', 'indices': [group['index']],
                         'games': [group['game']], 'success': example['trajectory'],
                         'failure': _public_text(failure['episode']),
                         'task': example['query'],
                         'sources': [chosen['source'], failure['source']]})
    for start in range(0, len(successes), MAX_SUCCESS_BATCH):
        chunk = successes[start:start + MAX_SUCCESS_BATCH]
        jobs.append({'kind': 'all_success', 'indices': [x['index'] for x in chunk],
                     'games': [x['game'] for x in chunk],
                     'success': '\n\n'.join(x['trajectory'] for x in chunk),
                     'failure': None, 'task': '', 'sources': [x['source'] for x in chunk]})
    if not successes:
        raise ValueError('No successful train trajectory available for an ExpeL bank')
    save(out / 'jobs.json', [{'kind': job['kind'], 'indices': job['indices'],
                            'games': job['games'], 'sources': job['sources']} for job in jobs])
    rules, generations = [], []
    context_limit = int(getattr(client, 'context', 32768))
    for index, job in enumerate(jobs):
        messages = upstream.messages(rules, job['success'], job['failure'], job['task'])
        prompt_tokens = _message_tokens(tokenizer, messages)
        if prompt_tokens + CRITIC_TOKENS > context_limit:
            raise ValueError(f'ExpeL critique {index} exceeds context: '
                             f'{prompt_tokens}+{CRITIC_TOKENS}>{context_limit}; no truncation')
        before = copy.deepcopy(rules)
        generation = _generate(client, messages, out / 'generations' / f'{index:03d}.json',
                               seed(CRITIC_SEED, 'expel_alfworld_bank', index))
        generations.append(generation)
        rules, operations, rejected, capped = upstream.update(rules, generation['raw_response'])
        save(out / 'operations' / f'{index:03d}.json', {
            'kind': job['kind'], 'training_indices': job['indices'], 'games': job['games'],
            'sources': job['sources'], 'before': before, 'after': rules,
            'parsed_operations': operations, 'rejected_invalid_indices': rejected,
            'dropped_by_rule_cap': capped, 'max_rules': MAX_RULES,
            'finish_reason': generation.get('finish_reason'), 'prompt_tokens': prompt_tokens})
    state = {'version': 1, 'rules': rules, 'successful_examples': successes,
             'training_games': [group['game'] for group in groups],
             'training_indices': [group['index'] for group in groups],
             'rule_template': upstream.prompts['RULE_TEMPLATE']['alfworld'],
             'source_hashes_sha256': _sha(hashes_path),
             'num_critique_calls': len(jobs),
             'num_compare_calls': sum(job['kind'] == 'compare' for job in jobs),
             'num_success_calls': sum(job['kind'] == 'all_success' for job in jobs),
             'source_episodes': sum(len(group['trials']) for group in groups),
             'source_successes': sum(t['episode']['reward'] == 1
                                     for group in groups for t in group['trials']),
             'test_feedback_used': False, 'frozen_before_evaluation': True,
             'critic_model': 'frozen-actor', 'critic_tokens': CRITIC_TOKENS,
             'critic_seed': CRITIC_SEED, 'max_rules': MAX_RULES,
             'source_scope': 'Original Delta train tasks, actual actor and empty-baseline '
                             'branches; shared development tasks, not identical writer inputs.',
             'development_cost': {
                 'new_environment_episodes': 0,
                 'reused_environment_episodes': sum(len(g['trials']) for g in groups),
                 'critic_calls': len(jobs),
                 'input_tokens': sum(g.get('input_tokens', 0) for g in generations),
                 'output_tokens': sum(g.get('output_tokens', 0) for g in generations),
                 'seconds': sum(g.get('seconds', 0) for g in generations)}}
    state['bank_id'] = _json_sha(state)
    save(state_path, state)
    save(freeze_path, {'state_sha256': _sha(state_path),
                       'source_hashes_sha256': _sha(hashes_path),
                       'test_feedback_used': False})
    return state


def query_context(state, query, family, retriever, tokenizer, budget=2048):
    """Render whole importance-ranked rules, then same-family top-2 examples."""
    if budget < 1:
        raise ValueError('ExpeL context budget must be positive')
    if family not in FAMILIES:
        raise ValueError(f'Unknown retrieval family: {family}')
    token_count = lambda text: len(tokenizer.encode(text, add_special_tokens=False))
    selected_rules, skipped_rules, rule_lines = [], [], []
    template = state['rule_template']
    context = ''
    ordered_rules = sorted(enumerate(state['rules'], 1), key=lambda pair: -pair[1][1])
    for index, (text, importance) in ordered_rules:
        lines = rule_lines + [f'{index}. {text}']
        candidate = template.format(rules='\n'.join(lines))
        item = {'index': index, 'importance': importance}
        if token_count(candidate) > budget:
            skipped_rules.append(dict(item, reason='context_token_budget'))
            continue
        rule_lines = lines
        context = candidate
        selected_rules.append(item)
    documents = [doc for doc in state['successful_examples'] if doc['family'] == family]
    selected, skipped = [], []
    for index, similarity in retriever.rank(query, documents):
        document = documents[index]
        provenance = {'training_index': document['index'], 'game': document['game'],
                      'source': document['source'], 'source_sha256': document['source_sha256']}
        if document['query'] == query:
            skipped.append(dict(provenance, reason='identical_initial_observation'))
            continue
        text = '\n\nSuccessful past household task:\n' + document['trajectory']
        if token_count(context + text) > budget:
            skipped.append(dict(provenance, reason='context_token_budget'))
            continue
        context += text
        selected.append(dict(provenance, cosine=similarity))
        if len(selected) == 2:
            break
    tokens = token_count(context)
    if tokens > budget:
        raise AssertionError('ExpeL renderer exceeded declared context budget')
    return context, {'bank_id': state.get('bank_id'), 'family': family,
                     'budget': budget, 'tokens': tokens,
                     'selected_rules': selected_rules, 'skipped_rules': skipped_rules,
                     'selected': selected, 'skipped': skipped,
                     'same_family_candidates': len(documents),
                     'other_family_examples_excluded': len(state['successful_examples']) - len(documents),
                     'context_sha256': _text_sha(context)}
