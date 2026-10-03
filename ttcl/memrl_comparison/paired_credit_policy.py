"""Apply one trained paired-credit rule to native MemRL retrievals.

At most one entry is removed per decision.  The training label estimates the
effect of removing one entry while the others remain; removing several at once
would be a different, unmeasured intervention.
"""
from __future__ import annotations

import hashlib

from .paired_credit_rl import features, predict


def _entry(metadata: dict) -> str:
    return ('Task: ' + metadata['task_description'] +
            '\nExperience: ' + metadata['public_abstract'])


def filter_native_retrieval(memory, retrieval: dict, policy: dict) -> dict:
    ids = list(retrieval['ids'])
    if not ids:
        return dict(retrieval, credit_decisions=[], suppressed_ids=[])
    candidate = {row['memory_id']:row for row in retrieval['candidates']}
    metadata = {mid:memory.store.get(mid).metadata.model_dump() for mid in ids}
    text = {mid:_entry(metadata[mid]) for mid in ids}
    original = '\n\n'.join(text[mid] for mid in ids)
    if original != retrieval['context']:
        raise ValueError('Native context reconstruction changed')
    decisions = []
    for position, mid in enumerate(ids):
        record = dict(metadata=metadata[mid], retrieval=candidate.get(mid, {}),
                      text_characters=len(text[mid]))
        result = predict(policy, features(record, len(ids), position))
        decisions.append(dict(memory_id=mid, position=position,
                              memory_text_sha256=hashlib.sha256(
                                  text[mid].encode()).hexdigest(), **result))
    harmful = [row for row in decisions if row['remove']]
    suppressed = [min(harmful, key=lambda row: row['upper_95'])['memory_id']] if harmful else []
    kept = [mid for mid in ids if mid not in suppressed]
    context = '\n\n'.join(text[mid] for mid in kept)
    return dict(retrieval, ids=kept, context=context,
                tokens=len(memory.client.tokenizer.encode(
                    context, add_special_tokens=False)),
                context_sha256=hashlib.sha256(context.encode()).hexdigest(),
                credit_decisions=decisions, suppressed_ids=suppressed)
