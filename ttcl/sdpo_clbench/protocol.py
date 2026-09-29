from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import random

UPSTREAM_COMMIT = '7c457fc1b1f636ae794eb0362ba37d4743b06fbc'
TASKS = {'blind_spectrum_monitoring': 90, 'exploitable_poker': 120,
         'database_exploration': 30, 'cohort_studies': 20}
ARMS = ('frozen', 'sdpo_online')


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
    temp.replace(path)


def append(path, value):
    with Path(path).open('a') as f:
        f.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')


def binding(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    allow_nan=False).encode()).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def seed(*parts):
    return int(binding(parts)[:12], 16) % (2**32)


def selected_calls(events, repeat, index, count=2):
    """Uniform, reward/length-independent action selection, including format failures."""
    return sorted(random.Random(seed('sdpo_actions', repeat, index)).sample(
        range(len(events)), min(count, len(events))))


def teacher_messages(event, public_steps, reward, feedback_chars=12000):
    """Hindsight context contains public evidence and one terminal scalar only.

    Never accepts evaluator metadata, task IDs, labels, hidden simulator state,
    or another/future episode. Continuous rewards are not called success labels.
    """
    if reward is not None and not math.isfinite(reward):
        raise ValueError('Invalid official reward')
    public = [{'step': s['step'], 'feedback': s.get('public_feedback', '')}
              for s in public_steps]
    feedback = json.dumps(public, ensure_ascii=False)
    omitted = max(0, len(feedback) - feedback_chars)
    if omitted:
        # Declared context budget: keep beginning and ending public feedback.
        half = feedback_chars // 2
        feedback = feedback[:half] + '\n[public feedback middle omitted]\n' + feedback[-half:]
    report = {'official_episode_reward': reward, 'higher_reward_is_better': True,
              'reward_scope': 'Whole completed episode; not an action-level correctness label.',
              'format_error': event.get('parse_error'),
              'environment_output': feedback}
    extra = ('\n\nThe following is feedback from your earlier attempt. Use this hindsight '
             'to reconsider the original action; it was not available when you acted.\n'
             + json.dumps(report, ensure_ascii=False)
             + '\nCorrectly solve the original question. Return the required JSON action.')
    messages = copy.deepcopy(event['messages'])
    if messages[-1]['role'] != 'user':
        raise ValueError('Expected an action prompt ending with a user message')
    messages[-1]['content'] += extra
    return messages, {'public_feedback_omitted_chars': omitted,
                      'feedback_binding': binding(report), 'teacher_input_binding': binding(messages)}
