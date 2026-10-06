"""Frozen language-model encoding of a complete public trajectory.

No generated textual rule or ALFWorld-specific field parser is used. The
source is encoded once, before the next task, and the resulting vector is
mapped by the trainable hypernetwork to LoRA factors.
"""

from __future__ import annotations

import torch


def source_text(records: list[dict[str, str]]) -> str:
    if not records:
        raise ValueError("A contextual source needs at least one trajectory step")
    return "Completed prior task trajectory:\n" + "\n".join(
        f"Step {index + 1}\nObservation: {step['observation']}\n"
        f"Action: {step['action']}\nFeedback: {step['feedback']}"
        for index, step in enumerate(records))


def contextual_text_fields(agent, tokenizer, text: str, device: str,
                           max_source_tokens: int = 2048) -> dict:
    ids = tokenizer(text, add_special_tokens=False,
                    return_tensors="pt").input_ids.to(device)
    if ids.shape[1] > max_source_tokens:
        raise ValueError(f"Contextual source exceeds declared token budget: {ids.shape[1]}")
    with torch.no_grad():
        agent.set_source(None)
        output = agent.model(input_ids=ids,
            attention_mask=torch.ones_like(ids),
            output_hidden_states=True, use_cache=False)
        vector = output.hidden_states[-1][:, -1, :].detach().float()
    return {"contextual": vector}


def contextual_source_fields(agent, tokenizer, records, device: str,
                             max_source_tokens: int = 2048) -> dict:
    return contextual_text_fields(agent, tokenizer, source_text(records),
                                  device, max_source_tokens)
