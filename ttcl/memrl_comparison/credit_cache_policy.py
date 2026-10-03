"""Apply a frozen content-bound credit cache to native MemRL retrievals."""
from __future__ import annotations

import hashlib

from .causal_credit_cache import decision
from .memory import Memory


def filter_retrieval(memory: Memory, retrieval: dict, family: str,
                     policy: dict, *, apply: bool) -> dict:
    ids = list(retrieval['ids'])
    entries = {}
    text_hashes = {}
    for mid in ids:
        metadata = memory.store.get(mid).metadata.model_dump()
        text = ('Task: ' + metadata['task_description'] +
                '\nExperience: ' + metadata['public_abstract'])
        entries[mid] = text
        text_hashes[mid] = hashlib.sha256(text.encode()).hexdigest()
    native_context = '\n\n'.join(entries[mid] for mid in ids)
    if native_context != retrieval['context']:
        raise ValueError('Native retrieval context changed')
    result = decision(dict(task=family, ids=ids,
                           memory_text_sha256=text_hashes), policy)
    suppressed = ([result['drop_memory_id']]
                  if apply and result['drop_memory_id'] is not None else [])
    kept = [mid for mid in ids if mid not in suppressed]
    context = '\n\n'.join(entries[mid] for mid in kept)
    return dict(retrieval, ids=kept, context=context,
                tokens=len(memory.client.tokenizer.encode(
                    context, add_special_tokens=False)),
                context_sha256=hashlib.sha256(context.encode()).hexdigest(),
                native_ids=ids,
                native_context_sha256=retrieval['context_sha256'],
                suppressed_ids=suppressed,
                credit_cache_decision=dict(evaluated=apply,
                                           drop_position=result['drop_position']
                                           if apply else None,
                                           drop_memory_id=result['drop_memory_id']
                                           if apply else None,
                                           donor_case=result['donor']['source_case']
                                           if apply and result['donor'] else None))


class CacheMemory(Memory):
    def __init__(self, *args, family: str, credit_cache: dict, **kwargs):
        super().__init__(*args, **kwargs)
        self.credit_family = family
        self.credit_cache = credit_cache
        self.episode_retrievals = 0

    def begin_episode(self) -> None:
        self.episode_retrievals = 0

    def retrieve(self, query: str) -> dict:
        native = super().retrieve(query)
        filtered = filter_retrieval(
            self, native, self.credit_family, self.credit_cache,
            apply=self.episode_retrievals == 0)
        self.episode_retrievals += 1
        return filtered
