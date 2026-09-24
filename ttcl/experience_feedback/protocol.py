from __future__ import annotations
import json
import math
from ttcl.experience_evolution.core import WRITER_SYSTEM

PUBLIC_FIELDS = ('public_task_brief', 'initial_public_query', 'response_schemas',
                 'steps', 'completed', 'format_failures', 'reward', 'reward_scope',
                 'local_execution_error')


def public_messages(memory, episode):
    # Completed means the official task produced an outcome, NOT that it succeeded.
    # An aborted attempt retains its trajectory and null reward; never invent a zero.
    public = {k: episode[k] for k in PUBLIC_FIELDS}
    reward = public['reward']
    if reward is not None and (not isinstance(reward, (int, float)) or not math.isfinite(reward)):
        raise ValueError('Invalid official reward')
    if public['completed'] and reward is None:
        raise ValueError('Completed task is missing its official reward')
    return [{'role': 'system', 'content': WRITER_SYSTEM},
            {'role': 'user', 'content': json.dumps({'previous_experience': memory,
              'completed_interaction': public}, ensure_ascii=False, allow_nan=False)}]


def training_label(pairs, objective):
    from ttcl.experience_evolution.core import reward_signal
    import statistics
    deltas = [float(yes['reward']) - float(no['reward']) for yes, no in pairs]
    signals = [reward_signal(yes['reward'], no['reward'], objective) for yes, no in pairs]
    return statistics.mean(signals), statistics.mean(deltas)
