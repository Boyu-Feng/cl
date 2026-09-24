"""Auditable memory and reflection updates for the shared ALFWorld actor.

These are method-level adaptations. The actor prompt, commands, and action
budget are owned by the runner and are identical across methods.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import List

from ttcl.experience_evolution.core import digest, read, save, writer_messages
from ttcl.reflexion_expel.upstream import fragments


REFLECTION_HEADER = "Plans from previous failed attempts on this SAME task:"


def public_episode(episode):
    """Allow only completed ALFWorld observations, actions, and public outcomes."""
    if episode.get("status", "complete") != "complete":
        raise ValueError("Memory updates require a completed actor episode")
    initial = episode["initial_observation"]
    trajectory = episode["trajectory"]
    if not isinstance(initial, str) or not isinstance(trajectory, list):
        raise TypeError("Expected an initial observation and an ALFWorld trajectory")
    public_steps = []
    for step in trajectory:
        action, observation = step["action"], step["observation"]
        if not isinstance(action, str) or not isinstance(observation, str):
            raise TypeError("ALFWorld actions and observations must be strings")
        public_steps.append({"action": action, "observation": observation})
    reward = float(episode["reward"])
    if not math.isfinite(reward) or reward not in (0.0, 1.0):
        raise ValueError("Expected the official binary ALFWorld success reward")
    steps = episode["steps"]
    if not isinstance(steps, int) or isinstance(steps, bool) or steps != len(public_steps):
        raise ValueError("Actor step count does not match the completed trajectory")
    return {"initial_observation": initial, "trajectory": public_steps,
            "reward": reward, "steps": steps}


def trajectory_text(episode):
    """Serialize public evidence without game paths, hidden facts, or generations."""
    return json.dumps(public_episode(episode), ensure_ascii=False)


def validate_generation(result, model):
    if not isinstance(result.get("raw_response"), str) or not result["raw_response"].strip():
        raise ValueError("Empty or non-text memory generation")
    if result.get("served_model") != model:
        raise ValueError("Generation model does not match the requested writer")
    for field in ("input_tokens", "output_tokens"):
        value = result[field]
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"Invalid generation cost: {field}")
    if not math.isfinite(result["seconds"]) or result["seconds"] < 0:
        raise ValueError("Invalid generation elapsed time")


def cached_generation(client, messages, path, seed, model="frozen-actor", tokens=768):
    """Generate greedily once; reject resumed requests with changed provenance.

    Returns a generation dict, including ``raw_response``, the original cost
    fields, and ``cache_hit``. Cached costs describe the original physical call,
    not an additional call made while resuming.
    """
    path = Path(path)
    if not isinstance(tokens, int) or isinstance(tokens, bool) or tokens < 1:
        raise ValueError("tokens must be a positive integer")
    request = {"messages": messages, "model": model, "random_seed": int(seed) % 2**32,
               "tokens": tokens, "temperature": 0.0, "top_p": 1.0}
    request_hash = digest(json.dumps(request, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":"), allow_nan=False))
    if path.exists():
        result = read(path)
        if result.get("request") != request or result.get("request_sha256") != request_hash:
            raise ValueError(f"Resume generation request changed: {path}")
        if result.get("messages") != messages:
            raise ValueError(f"Cached generation prompt changed: {path}")
        validate_generation(result, model)
        return dict(result, cache_hit=True)
    result = client.complete(messages, model=model, random_seed=request["random_seed"],
                             tokens=tokens, temperature=0.0, top_p=1.0)
    validate_generation(result, model)
    result = dict(result, request=request, request_sha256=request_hash,
                  messages=messages, cache_hit=False)
    save(path, result)
    return result


def delta_update(client, memory, ep, path, seed, model):
    """Use the unchanged Delta writer instruction on the completed public trace."""
    messages = writer_messages(memory, public_episode(ep))
    return cached_generation(client, messages, path, seed, model=model)


def reflection_prompt(upstream_root, episode, reflections):
    """Load the pinned official ALFWorld reflection builder without its runtime.

    ``upstream_root`` accepts the experiment's frozen upstream directory or the
    checkout's ``current_work`` directory. Only critic examples are loaded;
    the caller's common actor prompt is not modified.
    """
    root = Path(upstream_root)
    candidates = [root / "reflexion/generate_reflections.py",
                  root / "reflexion/alfworld_runs/generate_reflections.py"]
    source = next((p for p in candidates if p.is_file()), None)
    if source is None:
        raise FileNotFoundError(f"Missing pinned Reflexion prompt source in {root}")
    examples = source.with_name("reflexion_few_shot_examples.txt").read_text()
    functions = fragments(source, {"_get_scenario", "_generate_reflection_query"},
                          {"List": List, "FEW_SHOT_EXAMPLES": examples})
    return functions["_generate_reflection_query"](
        "Here is the task:\n" + trajectory_text(episode), list(reflections[-3:]))


def reflexion_update(client, ep, reflections, path, seed, upstream_root):
    """Reflect only after a failed attempt; keep task-local state in the runner."""
    public = public_episode(ep)
    if public["reward"] != 0.0:
        raise ValueError("Reflexion is only defined after an unsuccessful attempt")
    prompt = reflection_prompt(upstream_root, public, reflections)
    return cached_generation(client, [{"role": "user", "content": prompt}], path, seed)


def pack_reflections(reflections, tokenizer, budget=2048):
    """Keep whole recent plans, newest first when selecting within the budget.

    Selected plans are presented chronologically. The official last-three-plan
    limit is retained. A single retained-history plan that cannot fit on its own
    is an error; no plan is silently truncated.
    """
    if not isinstance(budget, int) or isinstance(budget, bool) or budget < 1:
        raise ValueError("Reflection token budget must be a positive integer")
    if any(not isinstance(text, str) or not text.strip() for text in reflections):
        raise ValueError("Reflection plans must be nonempty strings")
    def render(indices):
        if not indices:
            return ""
        return REFLECTION_HEADER + "\n\n" + "\n\n".join(
            f"Trial {index + 1}:\n{reflections[index]}" for index in sorted(indices))
    def count(text):
        return len(tokenizer.encode(text, add_special_tokens=False))
    first = max(0, len(reflections) - 3)
    dropped = [{"index": i, "reason": "last_three_limit"} for i in range(first)]
    selected = []
    for index in range(len(reflections) - 1, first - 1, -1):
        if count(render([index])) > budget:
            raise ValueError(f"Reflection {index} exceeds the {budget}-token memory budget")
        if count(render(selected + [index])) <= budget:
            selected.append(index)
        else:
            dropped.append({"index": index, "reason": "whole_plan_token_budget"})
    selected.sort()
    context = render(selected)
    audit = {"selected_indices": selected, "dropped": sorted(dropped, key=lambda x: x["index"]),
             "tokens": count(context), "budget": budget, "history_limit": 3,
             "context_sha256": digest(context), "truncated": False}
    return context, audit
