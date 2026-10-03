"""A cautious, source-feedback-only retrieval filter for a MemRL ablation."""
from __future__ import annotations

import hashlib

from .memory import Memory


def filter_retrieval(memory: Memory, retrieval: dict, *, apply: bool) -> dict:
    native_ids = list(retrieval['ids'])
    texts, source_success = {}, {}
    for mid in native_ids:
        metadata = memory.store.get(mid).metadata.model_dump()
        texts[mid] = ('Task: ' + metadata['task_description'] +
                      '\nExperience: ' + metadata['public_abstract'])
        source_success[mid] = metadata.get('success') is True
    native_context = '\n\n'.join(texts[mid] for mid in native_ids)
    if native_context != retrieval['context']:
        raise ValueError('Native retrieval text differs from stored experience')
    selected = [mid for mid in native_ids if source_success[mid]]
    filter_active = apply and len(native_ids) == 3 and 0 < len(selected) < 3
    kept = selected if filter_active else native_ids
    context = '\n\n'.join(texts[mid] for mid in kept)
    return dict(retrieval, ids=kept, context=context,
                tokens=len(memory.client.tokenizer.encode(
                    context, add_special_tokens=False)),
                context_sha256=hashlib.sha256(context.encode()).hexdigest(),
                native_ids=native_ids,
                native_context_sha256=retrieval['context_sha256'],
                suppressed_ids=[mid for mid in native_ids if mid not in kept],
                source_success=source_success,
                success_filter_decision=dict(evaluated=apply,
                                             active=filter_active,
                                             kept_ids=kept))


class SourceSuccessMemory(Memory):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.episode_retrievals = 0

    def begin_episode(self) -> None:
        self.episode_retrievals = 0

    def retrieve(self, query: str) -> dict:
        native = super().retrieve(query)
        result = filter_retrieval(self, native,
                                  apply=self.episode_retrievals == 0)
        self.episode_retrievals += 1
        return result
