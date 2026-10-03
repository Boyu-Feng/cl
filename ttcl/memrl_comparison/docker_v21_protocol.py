"""Keep Docker CLBench task state while updating MemRL after each official instance.

The four-domain worker reconstructs a task for each instance. Sales Prediction
has a persistent /app workspace, so that worker would change the benchmark.
This adapter runs one canonical task sequence per arm and creates a fresh actor
conversation at each instance boundary. Only the MemRL store persists across
instances; the task owns its Docker container and official scoring.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ttcl.icl_mem0_comparison.protocol import read, save, sha, normalize_prompt
from ttcl.icl_mem0_comparison.worker import System, base
from .memory import digest


def input_digest(query) -> str:
    return hashlib.sha256(query.prompt.encode()).hexdigest()


class OnlineDockerSystem(base.ContinualLearningSystem):
    """One actor conversation per instance, one memory bank per ordered run."""

    def __init__(self, plan, client, memory, arm: str, task_name: str,
                 brief: str, output: Path, count: int):
        self.plan, self.client, self.memory = plan, client, memory
        self.arm, self.mode, self.task_name = arm, arm, task_name
        self.brief, self.output, self.count = brief, Path(output), count
        self.active_index = None
        self.inner = None
        self.retrieval = None
        self.task_query = None
        self.initial_prompt_sha256 = None
        self.finalized: set[int] = set()
        self.calls = self.input_tokens = self.output_tokens = self.max_input_tokens = 0
        self._before_writer = None
        self.observation_complete = False
        self.output.mkdir(parents=True, exist_ok=True)

    @property
    def name(self):
        return 'qwen4b_' + self.arm

    def reset(self):
        raise RuntimeError('Docker task and online memory must not reset mid-run')

    def _start(self, query):
        index = query.instance_index
        if index is None or index < 0 or index >= self.count or index in self.finalized:
            raise ValueError('Invalid or repeated canonical Docker instance')
        if self.active_index is not None and self.active_index not in self.finalized:
            raise ValueError('Previous instance has no official outcome')
        target = self.output / f'episode_{index + 1:03d}'
        target.mkdir(parents=True, exist_ok=False)
        self.active_index = index
        self.observation_complete = False
        self.initial_prompt_sha256 = input_digest(query)
        self.task_query = normalize_prompt(query.prompt, self.brief, index, self.count)
        if hasattr(self.memory, 'begin_task'):
            self.memory.begin_task('json_schema', self.initial_prompt_sha256)
        self.retrieval = self.memory.retrieve(self.task_query)
        save(target / 'retrieval.json', self.retrieval)
        inner = System(self.plan, self.client, self.arm, [], None,
                       target, self.brief, index, self.count)
        if self.retrieval['context']:
            inner.messages[0]['content'] += ('\n\nPast experience:\n' +
                                             self.retrieval['context'])
        if hasattr(self.memory, 'decorate_system'):
            self.memory.decorate_system(inner)
        self.inner = inner
        self._before_writer = (self.memory.calls, self.memory.input_tokens,
                               self.memory.output_tokens, self.memory.limit_hits)

    def respond(self, query):
        if self.active_index != query.instance_index:
            self._start(query)
        if self.active_index in self.finalized or self.inner is None:
            raise ValueError('Actor called after completed Docker instance')
        old = (self.inner.calls, self.inner.input_tokens, self.inner.output_tokens)
        try:
            return self.inner.respond(query)
        finally:
            self.calls += self.inner.calls - old[0]
            self.input_tokens += self.inner.input_tokens - old[1]
            self.output_tokens += self.inner.output_tokens - old[2]
            self.max_input_tokens = max(self.max_input_tokens,
                                        self.inner.max_input_tokens)

    def observe(self, observation, next_query=None):
        if self.inner is None:
            raise ValueError('Observation without active Docker instance')
        self.inner.observe(observation, next_query)
        self.observation_complete = base.observation_marks_instance_complete(
            observation)

    def finalize(self, outcome):
        index = outcome.instance_index
        if index in self.finalized:
            return
        if index != self.active_index or self.inner is None or self.retrieval is None:
            raise ValueError('Official outcome does not match active instance')
        if not self.observation_complete:
            raise ValueError('Official outcome arrived before instance completion')
        target = self.output / f'episode_{index + 1:03d}'
        public = self.inner.public_steps
        if not public:
            raise ValueError('No public actions before official Docker outcome')
        trace = json.dumps(public, ensure_ascii=False)
        reward = float(outcome.reward)
        binding = dict(task=self.task_name, canonical_index=index,
                       initial_query_sha256=self.initial_prompt_sha256,
                       public_content_sha256=digest(public),
                       repeat=self.client.repeat)
        update = self.memory.update(self.task_query, trace, reward,
                                    outcome.success, self.retrieval, binding)
        save(target / 'update.json', update)
        after = (self.memory.calls, self.memory.input_tokens,
                 self.memory.output_tokens, self.memory.limit_hits)
        before = self._before_writer
        self.memory.snapshot(target / 'memory_after.json')
        event_path = target / 'responses.jsonl'
        first = json.loads(event_path.open().readline()) if event_path.exists() else {}
        row = dict(benchmark='clbench', task=self.task_name, arm=self.arm,
                   repeat=self.client.repeat, canonical_index=index,
                   instance_id=outcome.instance_id, status='complete',
                   reward=reward, success=outcome.success,
                   initial_query_sha256=self.initial_prompt_sha256,
                   first_prompt_sha256=first.get('rendered_prompt_sha256'),
                   actor_calls=self.inner.calls,
                   actor_input_tokens=self.inner.input_tokens,
                   actor_output_tokens=self.inner.output_tokens,
                   writer_calls=after[0] - before[0],
                   writer_input_tokens=after[1] - before[1],
                   writer_output_tokens=after[2] - before[2],
                   writer_token_limit_hits=after[3] - before[3],
                   memory_after_sha256=sha(target / 'memory_after.json'))
        save(target / 'public_trajectory.json', public)
        save(target / 'row.json', row)
        self.finalized.add(index)

    def get_run_artifacts(self):
        return dict(arm=self.arm, completed_indices=sorted(self.finalized),
                    actor_calls=self.calls)


class OnlineDockerRecorder(base.Recorder):
    """Apply official reward before the runner asks for the next instance."""

    def sync_instance_outcomes(self, outcomes):
        for outcome in outcomes:
            self.system.finalize(outcome)
        super().sync_instance_outcomes(outcomes)
