"""Use public-action constrained decoding only after a failed invalid-heavy retry.

The trigger is computed by the caller from the preceding public trajectory.
Either mode makes exactly one actor completion for each environment step.
"""
from __future__ import annotations

from pathlib import Path

from ttcl.alfworld_comparison.environment import Actor
from ttcl.experience_evolution.core import save

from .guided_choice_actor import GuidedChoiceActor
from .memory import digest


INVALID_RATE_THRESHOLD = 0.4


def should_guide_retry(previous_episode: dict | None) -> bool:
    if previous_episode is None or previous_episode.get('reward'):
        return False
    trajectory = previous_episode.get('trajectory')
    if (not isinstance(trajectory, list) or not trajectory or
            any(type(step.get('valid_command')) is not bool
                for step in trajectory)):
        raise ValueError('Cannot route an unaudited failed trajectory')
    return sum(not step['valid_command'] for step in trajectory) / len(
        trajectory) > INVALID_RATE_THRESHOLD


class RetryTriggeredActor(GuidedChoiceActor):
    guide_retry = False

    def generate(self, messages, random_seed):
        if self.guide_retry:
            return dict(super().generate(messages, random_seed),
                        guided_choice_used=True)
        result = self.client.complete(
            messages, random_seed, tokens=self.plan['actor_max_tokens'],
            temperature=self.plan['actor_temperature'], top_p=1.)
        return dict(text=result['raw_response'],
                    finish_reason=result['finish_reason'],
                    usage=dict(prompt_tokens=result['input_tokens'],
                               completion_tokens=result['output_tokens']),
                    seed=random_seed, prompt_sha256=digest(messages),
                    rendered_prompt_sha256=result['rendered_prompt_sha256'],
                    seconds=result['seconds'], guided_choice_used=False)

    def run_many(self, jobs):
        results = Actor.run_many(self, jobs)
        for job, episode in zip(jobs, results):
            if self.guide_retry and any(not step['valid_command']
                                        for step in episode['trajectory']):
                raise ValueError('Guided retry produced an invalid command')
            if (len(episode['generations']) != episode['steps'] or
                    any(generation['guided_choice_used'] != self.guide_retry
                        for generation in episode['generations'])):
                raise ValueError('Retry actor mode or call count mismatch')
            episode['actor_adapter_enabled'] = self.guide_retry
            episode['guided_choice_decoder'] = self.guide_retry
            save(Path(job['output']) / 'episode.json', episode)
        return results
