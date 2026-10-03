"""Use an evidence-bound failure reflection in the common MemRL writer.

The same prompt is used for every failed public trajectory, regardless of
benchmark or task family. Success writes and the memory/Q machinery remain
native. Text retries keep v27's all-attempt retrieval; structured actions
keep its typed projection. This candidate is training-split only until an
independent online evaluation is completed.
"""
from __future__ import annotations

from .grounded_evidence_v27 import TypedGroundedMemory as V27Memory
from .memory import digest


FAILURE_REFLECTION_RULE = """Review this unsuccessful attempt using only the public trace below.
Separate observations from guesses. Identify the earliest assumption or action
that the observations contradict. Give at most four concrete next steps for a
retry, each with a checkable observation or query before acting. Explain what
to avoid repeating. Do not invent an answer, object location, schema field,
measurement, or success claim that the trace did not establish. If the trace
does not identify the cause, state the uncertainty and the next useful check.
Keep the reflection concise and applicable to a similar task.

Current task:
{task}

Public failed trajectory:
{trace}

Evidence-bound retry reflection:"""


def failure_reflection_messages(task: str, trace: str) -> list[dict[str, str]]:
    if not task.strip() or not trace.strip():
        raise ValueError('Failure reflection needs a task and public trace')
    return [{'role': 'user', 'content': FAILURE_REFLECTION_RULE.format(
        task=task, trace=trace)}]


class TypedGroundedMemory(V27Memory):
    def __init__(self, plan, client, directory, calibration, embedder=None):
        super().__init__(plan, client, directory, calibration, embedder=embedder)
        self.signature = digest(dict(base=self.signature, variant='typed_grounded_v28',
                                     failure_writer_rule=FAILURE_REFLECTION_RULE))
        self._failure_write = False
        self._failure_task = ''
        self._failure_trace = ''
        writer = self.service.llm_provider
        previous = writer.generate

        def generate(messages, **kwargs):
            if self._failure_write:
                messages = failure_reflection_messages(self._failure_task,
                                                       self._failure_trace)
            return previous(messages, **kwargs)

        writer.generate = generate

    def update(self, query, public_trace, reward, success, retrieval, binding):
        self._failure_write = success is False
        self._failure_task = query if self._failure_write else ''
        self._failure_trace = public_trace if self._failure_write else ''
        try:
            return super().update(query, public_trace, reward, success,
                                  retrieval, binding)
        finally:
            self._failure_write = False
            self._failure_task = ''
            self._failure_trace = ''
