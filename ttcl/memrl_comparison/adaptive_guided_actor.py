"""Guide only the turn after an unadvertised text command.

The public command list and the actor's own previous output determine the
mode. One completion is made per environment step: native decoding after a
valid action, guided-choice decoding after an invalid one. The rule contains
no benchmark family, object name, reward, or extra model call.
"""
from __future__ import annotations

from pathlib import Path

from ttcl.alfworld_comparison.environment import Actor
from ttcl.experience_evolution.core import save
from ttcl.experience_evolution.environment import clean_command
from .guided_choice_actor import GuidedChoiceActor, advertised_commands
from .memory import digest


class AdaptiveGuidedActor(GuidedChoiceActor):
    def generate(self, messages, random_seed):
        guide = False
        if len(messages) >= 4:
            previous_user, previous_assistant = messages[-3:-1]
            prior_commands = advertised_commands([previous_user])
            proposed = clean_command(previous_assistant['content'], prior_commands)
            guide = proposed not in prior_commands
        if guide:
            return dict(super().generate(messages, random_seed),
                        guided_choice_used=True)
        completion = self.client.complete(
            messages, random_seed, tokens=self.plan['actor_max_tokens'],
            temperature=self.plan['actor_temperature'], top_p=1.)
        return dict(text=completion['raw_response'],
                    finish_reason=completion['finish_reason'],
                    usage=dict(prompt_tokens=completion['input_tokens'],
                               completion_tokens=completion['output_tokens']),
                    seed=random_seed,
                    prompt_sha256=digest(messages),
                    rendered_prompt_sha256=completion['rendered_prompt_sha256'],
                    guided_choice_used=False,
                    seconds=completion['seconds'])

    def run_many(self, jobs):
        results = Actor.run_many(self, jobs)
        for job, episode in zip(jobs, results):
            episode['actor_adapter_enabled'] = True
            episode['adaptive_guidance'] = True
            episode['guided_choice_calls'] = sum(
                generation['guided_choice_used']
                for generation in episode['generations'])
            if (len(episode['generations']) != episode['steps'] or
                    episode['guided_choice_calls'] > episode['steps']):
                raise ValueError('Adaptive actor call count mismatch')
            save(Path(job['output']) / 'episode.json', episode)
        return results
