"""Independent frozen rollout adding only visible executed-action state."""
from pathlib import Path
from typing import Callable
import hashlib
import json
import torch
from ttcl.trajectory_hyperlora.alfworld_zero_shot_probe import ACTOR_SYSTEM, clean_command, make_env, guarded_commands, generate
from ttcl.trajectory_hyperlora.alfworld_online_from_empty import digest
from ttcl.trajectory_hyperlora.visible_action_state import action_state_text

def run_episode(agent, tokenizer, game: Path, fields: dict,
                *, adapter: bool, device: str, max_steps: int,
                max_new_tokens: int,
                constrain_actions: bool = False,
                fixed_adapter: list[torch.Tensor] | None = None,
                fixed_adapter_selector: Callable[[str], list[torch.Tensor] | None] | None = None,
                memory_text: str | None = None,
                evidence_provider: Callable[[str, str, list[str]], str] | None = None,
                actor_history_turns: int | None = None,
                loop_guard_max: int | None = None) -> dict:
    if adapter and (fixed_adapter is not None or fixed_adapter_selector is not None):
        raise ValueError("Choose trajectory adapter or fixed adapter")
    if fixed_adapter is not None and fixed_adapter_selector is not None:
        raise ValueError("Choose one fixed adapter source")
    if actor_history_turns is not None and actor_history_turns < 1:
        raise ValueError("Actor history window must be positive")
    if loop_guard_max is not None and loop_guard_max < 1:
        raise ValueError("Loop guard threshold must be positive")
    env = make_env(game)
    trajectory = []
    observation_actions: dict[tuple[str, str], int] = {}
    try:
        state = env.reset()
        initial_observation = str(state["feedback"])
        initial_commands = list(state["admissible_commands"])
        with torch.no_grad():
            target_fields = None
            if (adapter and agent.task_conditioned and
                    agent.task_context_scope == "initial"):
                from ttcl.trajectory_hyperlora.contextual_alf_source import (
                    contextual_text_fields, task_context_text,
                )
                target_fields = contextual_text_fields(agent, tokenizer,
                    task_context_text(initial_observation,
                                      initial_observation),
                    device, agent.contextual_source_max_tokens,
                    pooling="both" if agent.task_pair_pooling == "mean"
                        else "last")
            if agent.task_context_scope == "initial" or not adapter:
                agent.set_source(fields if adapter else None,
                                 target_fields=target_fields)
            if fixed_adapter_selector is not None:
                fixed_adapter = fixed_adapter_selector(initial_observation)
            if fixed_adapter is not None:
                if len(fixed_adapter) != len(agent.adapters):
                    raise ValueError("Fixed adapter layer count mismatch")
                for layer, value in zip(agent.adapters, fixed_adapter,
                                        strict=True):
                    expected = (1, layer.base.out_features, layer.rank)
                    if tuple(value.shape) != expected:
                        raise ValueError("Fixed adapter factor shape mismatch")
                    layer.b = value.to(device)
        system = ACTOR_SYSTEM
        if memory_text:
            system += "\n\nPrior attempt record:\n" + memory_text
        messages = [{"role": "system", "content": system}]
        evidence_reads = []
        state_reads = []
        for turn in range(max_steps):
            if (adapter and agent.task_conditioned and
                    agent.task_context_scope == "current"):
                from ttcl.trajectory_hyperlora.contextual_alf_source import (
                    contextual_text_fields, task_context_text,
                )
                current_target = contextual_text_fields(agent, tokenizer,
                    task_context_text(initial_observation,
                                      str(state["feedback"])),
                    device, agent.contextual_source_max_tokens,
                    pooling="both" if agent.task_pair_pooling == "mean"
                        else "last")
                with torch.no_grad():
                    agent.set_source(fields, target_fields=current_target)
            available = list(state["admissible_commands"])
            feedback = str(state["feedback"])
            allowed = guarded_commands(feedback, available,
                                       observation_actions, loop_guard_max)
            messages.append({"role": "user", "content": str(state["feedback"]) +
                             "\nAvailable commands:\n" + "\n".join(available)})
            actor_messages = messages
            if actor_history_turns is not None:
                from ttcl.trajectory_hyperlora.train_alf_next_task import (
                    compact_messages,
                )
                actor_messages = compact_messages(messages, actor_history_turns)
            if evidence_provider is not None:
                hint = evidence_provider(initial_observation, feedback, available)
                if hint:
                    actor_messages = [dict(item) for item in actor_messages]
                    actor_messages[0]["content"] += "\n\nPrior public evidence:\n" + hint
                evidence_reads.append(hashlib.sha256(hint.encode()).hexdigest())
            summary = action_state_text(trajectory)
            actor_messages = [dict(item) for item in actor_messages]
            actor_messages[0]['content'] += '\n\n' + summary
            prompt = tokenizer.apply_chat_template(actor_messages, tokenize=False, add_generation_prompt=True)
            prompt_ids = tokenizer(prompt).input_ids
            if len(prompt_ids) > 4096:
                raise ValueError('State-augmented actor prompt exceeds declared 4096 tokens')
            state_reads.append({'prefix_sha256': digest(trajectory), 'state_sha256': digest(summary),
                                'actor_messages_sha256': digest(actor_messages), 'prompt_tokens': len(prompt_ids)})
            response = generate(agent, tokenizer, actor_messages, device, max_new_tokens,
                                allowed if constrain_actions else None)
            command = clean_command(response, available)
            valid = command in available
            key = (feedback, command)
            observation_actions[key] = observation_actions.get(key, 0) + 1
            state, _, done = env.step(command)
            messages.append({"role": "assistant", "content": response})
            trajectory.append({"turn": turn, "response": response,
                               "command": command, "valid": valid,
                               "observation": str(state["feedback"]),
                               "won": bool(state["won"])})
            if done or state["won"]:
                break
        result = {"status": "complete", "reward": float(bool(state["won"])),
                "steps": len(trajectory), "initial_observation": initial_observation,
                "memory_sha256": hashlib.sha256((memory_text or "").encode()).hexdigest(),
                "initial_commands_sha256": hashlib.sha256(json.dumps(
                    initial_commands).encode()).hexdigest(),
                "invalid_commands": sum(not step["valid"] for step in trajectory),
                "trajectory": trajectory,
                "termination": "success" if state["won"] else "budget_or_environment_done"}
        result["state_reads"] = state_reads
        if evidence_provider is not None:
            result["evidence_read_sha256"] = evidence_reads
        return result
    except Exception as error:
        return {"status": "infrastructure_failure", "error": repr(error),
                "steps": len(trajectory), "trajectory": trajectory}
    finally:
        env.close()
        agent.set_source(None)
