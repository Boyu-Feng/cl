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


def task_context_text(initial_observation: str,
                      current_observation: str) -> str:
    """Keep the public goal and the latest state without parsing commands."""
    text = "Current task observation:\n" + initial_observation
    if current_observation != initial_observation:
        text += "\nLatest feedback:\n" + current_observation
    return text


def contextual_text_fields(agent, tokenizer, text: str, device: str,
                           max_source_tokens: int = 2048,
                           pooling: str = "last") -> dict:
    if pooling not in ("last", "mean", "both"):
        raise ValueError("Unknown contextual token pooling")
    ids = tokenizer(text, add_special_tokens=False,
                    return_tensors="pt").input_ids.to(device)
    if ids.shape[1] > max_source_tokens:
        raise ValueError(f"Contextual source exceeds declared token budget: {ids.shape[1]}")
    with torch.no_grad():
        agent.set_source(None)
        output = agent.model(input_ids=ids,
            attention_mask=torch.ones_like(ids),
            output_hidden_states=True, use_cache=False)
        tokens = output.hidden_states[-1]
        last = tokens[:, -1, :].detach().float()
        if pooling == "last":
            return {"contextual": last}
        mean = tokens.float().mean(dim=1).detach()
        if pooling == "both":
            return {"contextual": last, "pair_contextual": mean}
        return {"contextual": mean}


def contextual_source_fields(agent, tokenizer, records, device: str,
                             max_source_tokens: int = 2048,
                             pooling: str = "last") -> dict:
    return contextual_text_fields(agent, tokenizer, source_text(records),
                                  device, max_source_tokens, pooling)
