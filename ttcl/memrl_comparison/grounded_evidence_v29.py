"""Use the native failure writer only as task-local retry context.

First attempts and structured JSON actions retain v27 behavior. After a
failed retryable text attempt, the native MemRL writer still stores its
source-bound memory and receives its normal Q update. For the remaining
attempts on that same task, the actor sees the newly written reflection
instead of cross-task retrieval; the task-local note is cleared at the
next task. No extra model call or retry is added.
"""
from __future__ import annotations

import hashlib

from .grounded_evidence_v27 import TypedGroundedMemory as V27Memory
from .memory import digest


LOCAL_REFLECTION_HEADER = (
    'Reflection from earlier failed attempts on this exact task. '
    'Its explanations are hypotheses; verify them against current observations.\n')


class TypedGroundedMemory(V27Memory):
    def __init__(self, plan, client, directory, calibration, embedder=None):
        super().__init__(plan, client, directory, calibration, embedder=embedder)
        self.local_reflections = []
        self.signature = digest(dict(base=self.signature, variant='typed_grounded_v29',
                                     retry_context='task_local_native_failure_reflections',
                                     header=LOCAL_REFLECTION_HEADER))

    def begin_task(self, interface, binding):
        super().begin_task(interface, binding)
        self.local_reflections = []

    def retrieve(self, query):
        native = super().retrieve(query)
        if not self._text_retry or self._text_attempt == 1 or not self.local_reflections:
            return native
        notes = self.local_reflections[-2:]
        context = LOCAL_REFLECTION_HEADER + '\n\n'.join(
            f'Attempt {entry["attempt"]}: {entry["text"]}' for entry in notes)
        tokens = self._tokens(context)
        if tokens > self.plan['memory_tokens']:
            raise ValueError('Task-local reflection exceeds memory budget')
        self.selected_context = context
        self.selected_public_context = ''
        self.selected_evidence = []
        return dict(native, context=context, ids=[], tokens=tokens,
                    context_sha256=hashlib.sha256(context.encode()).hexdigest(),
                    suppressed_ids=native['ids'],
                    local_source_sha256=[x['public_trace_sha256']
                                         for x in self.local_reflections[-2:]],
                    retry_policy='task_local_failure_reflection',
                    action_interface='retryable_text_command')

    def update(self, query, public_trace, reward, success, retrieval, binding):
        result = super().update(query, public_trace, reward, success,
                                retrieval, binding)
        if self._text_retry and success is False:
            if len(self.abstracts) != 1 or not self.abstracts[0].strip():
                raise ValueError('Missing native failure reflection')
            self.local_reflections.append(dict(
                text=self.abstracts[0],
                attempt=self._text_attempt,
                public_trace_sha256=hashlib.sha256(public_trace.encode()).hexdigest(),
                input_binding=binding))
        return result
