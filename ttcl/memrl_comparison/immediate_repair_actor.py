"""Repair only a command outside the current public admissible set.

The first completion is the native actor. If its cleaned command is absent
from the environment's advertised list, a second, guided-choice completion
uses the same observation before any environment step. Both calls are audited
and charged; a valid native command is passed through unchanged.
"""
from __future__ import annotations

import json
from pathlib import Path

from ttcl.alfworld_comparison import environment as alf_environment
from ttcl.experience_evolution.core import digest as content_digest, save, seed
from ttcl.experience_evolution.environment import ACTOR_SYSTEM, clean_command

from .guided_choice_actor import GuidedChoiceActor, advertised_commands
from .memory import digest


class ImmediateRepairActor(GuidedChoiceActor):
    def generate(self, messages, random_seed, *, allow_repair=True):
        commands = advertised_commands(messages)
        original = self.client.complete(
            messages, random_seed, tokens=self.plan['actor_max_tokens'],
            temperature=self.plan['actor_temperature'], top_p=1.)
        raw = original['raw_response']
        proposed = clean_command(raw, commands)
        if proposed in commands or not allow_repair:
            return dict(text=raw, finish_reason=original['finish_reason'],
                        usage=dict(prompt_tokens=original['input_tokens'],
                                   completion_tokens=original['output_tokens']),
                        seed=random_seed, prompt_sha256=digest(messages),
                        rendered_prompt_sha256=original['rendered_prompt_sha256'],
                        seconds=original['seconds'], completion_count=1,
                        repair_used=False, native_command=proposed,
                        repair_skipped_due_to_budget=proposed not in commands)

        repair_seed = seed(random_seed, 'immediate_public_command_repair')
        repair = super().generate(messages, repair_seed)
        if repair['text'] not in commands:
            raise ValueError('Repair returned an unadvertised command')
        return dict(text=repair['text'], finish_reason=repair['finish_reason'],
                    usage=dict(prompt_tokens=original['input_tokens'] +
                               repair['usage']['prompt_tokens'],
                               completion_tokens=original['output_tokens'] +
                               repair['usage']['completion_tokens']),
                    seed=random_seed, prompt_sha256=digest(messages),
                    rendered_prompt_sha256=original['rendered_prompt_sha256'],
                    seconds=original['seconds'] + repair['seconds'],
                    completion_count=2, repair_used=True,
                    native_text=raw, native_command=proposed,
                    native_finish_reason=original['finish_reason'],
                    native_usage=dict(prompt_tokens=original['input_tokens'],
                                      completion_tokens=original['output_tokens']),
                    repair_seed=repair_seed,
                    repair_rendered_prompt_sha256=repair['rendered_prompt_sha256'],
                    repair_guided_choice_sha256=repair['guided_choice_sha256'],
                    repair_guided_choice_count=repair['guided_choice_count'],
                    repair_usage=repair['usage'])

    def run_many(self, jobs):
        # The original protocol allows max_steps actor completions per attempt.
        # A repair consumes one of those calls, not an unbudgeted extra sample.
        if len(jobs) != 1:
            raise ValueError('Immediate repair requires one audited ALFWorld job')
        job = jobs[0]
        output = Path(job['output']) / 'episode.json'
        if output.exists():
            raise FileExistsError(output)
        environment = alf_environment.make_env(Path(self.plan['data_root']) / job['game'])
        try:
            state = environment.reset()
            commands = list(state['admissible_commands'])
            system = ACTOR_SYSTEM + ('\n\nPast experience:\n' + job['memory']
                                    if job['memory'] else '')
            messages = [dict(role='system', content=system)]
            episode = dict(game=job['game'], memory=job['memory'],
                           memory_sha256=content_digest(job['memory']),
                           seed=job['seed'], initial_observation=str(state['feedback']),
                           initial_commands_sha256=content_digest(json.dumps(commands)),
                           trajectory=[], generations=[], reward=0., steps=0,
                           actor_adapter_enabled=True, immediate_repair=True)
            calls = repairs = 0
            for turn in range(self.plan['max_steps']):
                commands = list(state['admissible_commands'])
                messages.append(dict(role='user', content=str(state['feedback']) +
                                     '\nAvailable commands:\n' + '\n'.join(commands)))
                completion = self.generate(messages, seed(job['seed'], turn),
                                           allow_repair=calls + 1 < self.plan['max_steps'])
                episode['generations'].append(completion)
                calls += completion['completion_count']
                repairs += int(completion['repair_used'])
                if calls > self.plan['max_steps']:
                    raise RuntimeError('Actor completion budget exceeded')
                if completion.get('repair_skipped_due_to_budget'):
                    episode['termination'] = 'actor_call_budget_invalid_command'
                    break
                command = clean_command(completion['text'], commands)
                if command not in commands:
                    raise ValueError('Immediate repair left an invalid command')
                state, _, done = environment.step(command)
                messages.append(dict(role='assistant', content=completion['text']))
                episode['trajectory'].append(dict(action=command,
                                                  observation=str(state['feedback']),
                                                  valid_command=True))
                episode['reward'] = float(bool(state['won']))
                episode['steps'] = turn + 1
                if state['won'] or done:
                    episode['termination'] = ('success' if state['won'] else
                                              'budget_or_environment_done')
                    break
                if calls == self.plan['max_steps']:
                    episode['termination'] = 'actor_call_budget'
                    break
            else:
                episode['termination'] = 'budget_or_environment_done'
            episode['status'] = 'complete'
            episode['actor_completion_count'] = calls
            episode['repair_count'] = repairs
            save(output, episode)
            return [episode]
        finally:
            environment.close()
