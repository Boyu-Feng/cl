"""Read out independently repeated public tool feedback within native budget.

The same recurrence rule applies to every structured-action task. It cannot
invent feedback or use official rewards: only exact matching action/response
pairs from at least two prior episodes qualify. Retryable text commands keep
native v27 behavior. The v26 typed action operator is unchanged.
"""
from __future__ import annotations

import hashlib

from .grounded_evidence_v15 import _head, _task_focus
from .grounded_evidence_v27 import TypedGroundedMemory as V27Memory
from .grounded_evidence_v13_base import _clip
from .memory import digest
from .recurrent_feedback import MIN_SOURCE_EPISODES, stable_feedback


MAX_READOUTS = 3


class TypedGroundedMemory(V27Memory):
    def __init__(self, plan, client, directory, calibration, embedder=None):
        super().__init__(plan, client, directory, calibration, embedder=embedder)
        self.signature = digest(dict(base=self.signature, variant='typed_grounded_v30',
                                     stable_feedback_sources=MIN_SOURCE_EPISODES,
                                     max_feedback_readouts=MAX_READOUTS))

    def retrieve(self, query):
        native = super().retrieve(query)
        if self._text_retry:
            return native
        groups = stable_feedback(self.public_events)
        if not groups:
            return dict(native, recurrent_feedback=[])
        tokenizer = self.client.tokenizer
        candidates = [(index, _clip(tokenizer, _task_focus(prior), 180))
                      for index, group in enumerate(groups)
                      for prior in group['prior_inputs']]
        vectors = self.service.embedding_provider.embed(
            [_clip(tokenizer, _task_focus(query), 256)] +
            [value for _, value in candidates])
        scores = [float('-inf')] * len(groups)
        for (index, _), vector in zip(candidates, vectors[1:]):
            similarity = sum(a * b for a, b in zip(vectors[0], vector))
            scores[index] = max(scores[index], similarity)
        ranked = sorted(range(len(groups)),
                        key=lambda index: (scores[index], groups[index]['episode_count']),
                        reverse=True)
        context = native['context']
        selected = []
        for index in ranked:
            group = groups[index]
            entry = ('Repeated observed feedback from ' +
                     str(group['episode_count']) + ' prior episodes. '
                     'Check whether it applies to the current task.\n'
                     'Submitted action: ' + _head(tokenizer, group['action'], 100) +
                     '\nObserved response: ' +
                     _head(tokenizer, group['feedback'], 320) +
                     '\nSource events: ' + ', '.join(group['source_sha256'][-2:]))
            proposed = (context + '\n\n' + entry).strip()
            if self._tokens(proposed) > self.plan['memory_tokens']:
                continue
            context = proposed
            selected.append(dict(episode_count=group['episode_count'],
                                 source_sha256=group['source_sha256'],
                                 public_trace_sha256=group['public_trace_sha256'],
                                 similarity=scores[index],
                                 action_sha256=digest(group['action']),
                                 feedback_sha256=hashlib.sha256(
                                     group['feedback'].encode()).hexdigest()))
            if len(selected) >= MAX_READOUTS:
                break
        self.selected_context = context
        # The v15 decorator removes raw single-episode public evidence from
        # tool-call prompts. Recurrent observations are separately qualified,
        # so keep them as part of actor context on those actions.
        self.selected_public_context = ''
        self.selected_evidence = []
        return dict(native, context=context, tokens=self._tokens(context),
                    context_sha256=hashlib.sha256(context.encode()).hexdigest(),
                    recurrent_feedback=selected,
                    readout='exact_cross_episode_action_feedback_recurrence')
