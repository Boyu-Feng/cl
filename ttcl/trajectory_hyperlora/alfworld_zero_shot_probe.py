"""Paired ALFWorld development probe for a synthetic-trained trajectory LoRA.

This is a zero-shot cross-domain diagnostic, not an official ALFWorld score.
The adapter generator is never updated on ALFWorld observations or rewards.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ttcl.experience_evolution.environment import ACTOR_SYSTEM, clean_command, make_env
from ttcl.trajectory_hyperlora.direct_composition_pilot import DirectRelationHyperLoRA
from ttcl.trajectory_hyperlora.relational_router_pilot import tokenize_records


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_records(episode: dict) -> list[dict[str, str]]:
    if episode.get("status") != "complete" or episode.get("reward") != 1:
        raise ValueError("Source must be a completed successful train episode")
    if episode.get("steps") != len(episode.get("trajectory", [])):
        raise ValueError("Source step count mismatch")
    observation = episode["initial_observation"]
    result = []
    for index, step in enumerate(episode["trajectory"]):
        feedback = ("invalid command" if step["valid_command"] is False else
                    "task completed" if index == len(episode["trajectory"]) - 1 else
                    "valid command")
        result.append({"observation": observation, "action": step["action"],
                       "feedback": feedback})
        observation = step["observation"]
    return result


def load_agent(model_path: Path, checkpoint_path: Path, device: str,
               gpu_fraction: float,
               context_mode_override: str | None = None):
    if device.startswith("cuda"):
        torch.cuda.set_per_process_memory_fraction(gpu_fraction, device=device)
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path), torch_dtype=torch.bfloat16, local_files_only=True,
        attn_implementation="sdpa").to(device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint["source_encoder"] not in ("raw", "contextual_lm"):
        raise ValueError("ALFWorld probe needs a supported trajectory encoder checkpoint")
    if (checkpoint["source_encoder"] == "contextual_lm" and
            checkpoint.get("encoder_kind") != "contextual"):
        raise ValueError("Contextual source checkpoint has incompatible encoder kind")
    agent = DirectRelationHyperLoRA(model, rank=checkpoint["rank"],
                                    layers=checkpoint["layers"],
                                    encoder_kind=checkpoint.get(
                                        "encoder_kind", "covariance"),
                                    relation_bottleneck=checkpoint.get(
                                        "relation_bottleneck", "none"),
                                    context_mode=(context_mode_override or
                                        checkpoint.get("context_mode", "none")),
                                    context_strength=checkpoint.get(
                                        "context_strength"),
                                    task_conditioned=checkpoint.get(
                                        "task_conditioned", False)).to(device)
    agent.contextual_source_max_tokens = checkpoint.get(
        "contextual_source_max_tokens", 2048)
    agent.task_pair_pooling = checkpoint.get("task_pair_pooling", "last")
    agent.task_context_scope = checkpoint.get("task_context_scope", "initial")
    agent.model.generation_config.temperature = 1.0
    agent.model.generation_config.top_p = 1.0
    agent.model.generation_config.top_k = 50
    trainable = {name: param for name, param in agent.named_parameters()
                 if param.requires_grad}
    learned_state = dict(checkpoint["trainable_state"])
    learned_state.update(checkpoint.get("relation_state", {}))
    if set(trainable) != set(learned_state):
        raise ValueError("Checkpoint parameter names mismatch")
    for name, param in trainable.items():
        value = learned_state[name]
        if value.shape != param.shape:
            raise ValueError(f"Checkpoint shape mismatch: {name}")
        param.data.copy_(value.to(device))
    agent.eval()
    return agent, tokenizer


def generate(agent, tokenizer, messages: list[dict], device: str,
             max_new_tokens: int,
             admissible_commands: list[str] | None = None) -> str:
    content = tokenizer.apply_chat_template(messages, tokenize=False,
                                            add_generation_prompt=True)
    inputs = tokenizer(content, return_tensors="pt").to(device)
    constraint = None
    if admissible_commands is not None:
        prefix = inputs.input_ids[0].tolist()
        paths = []
        for command in sorted(set(admissible_commands)):
            full = tokenizer(content + command,
                             add_special_tokens=False).input_ids
            if full[:len(prefix)] != prefix:
                raise ValueError("Command tokenization changes the actor prefix")
            continuation = tuple(full[len(prefix):])
            if not continuation or len(continuation) + 1 > max_new_tokens:
                raise ValueError("Admissible command exceeds token budget")
            paths.append(continuation)
        if not paths:
            raise ValueError("Environment supplied no admissible commands")

        def constraint(_batch_index: int, generated_ids: torch.Tensor) -> list[int]:
            suffix = tuple(generated_ids[len(prefix):].tolist())
            allowed = {path[len(suffix)] for path in paths
                       if len(path) > len(suffix) and path[:len(suffix)] == suffix}
            if any(path == suffix for path in paths):
                allowed.add(tokenizer.eos_token_id)
            if not allowed:
                raise ValueError("Generated tokens escaped admissible command trie")
            return sorted(allowed)

    with torch.no_grad():
        output = agent.model.generate(**inputs, do_sample=False,
                                      max_new_tokens=max_new_tokens,
                                      pad_token_id=tokenizer.eos_token_id,
                                      prefix_allowed_tokens_fn=constraint)
    return tokenizer.decode(output[0, inputs.input_ids.shape[1]:],
                            skip_special_tokens=True).strip()


def run_episode(agent, tokenizer, game: Path, fields: dict,
                *, adapter: bool, device: str, max_steps: int,
                max_new_tokens: int,
                constrain_actions: bool = False,
                fixed_adapter: list[torch.Tensor] | None = None,
                memory_text: str | None = None) -> dict:
    if adapter and fixed_adapter is not None:
        raise ValueError("Choose trajectory adapter or fixed adapter")
    env = make_env(game)
    trajectory = []
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
            messages.append({"role": "user", "content": str(state["feedback"]) +
                             "\nAvailable commands:\n" + "\n".join(available)})
            response = generate(agent, tokenizer, messages, device, max_new_tokens,
                                available if constrain_actions else None)
            command = clean_command(response, available)
            valid = command in available
            state, _, done = env.step(command)
            messages.append({"role": "assistant", "content": response})
            trajectory.append({"turn": turn, "response": response,
                               "command": command, "valid": valid,
                               "observation": str(state["feedback"]),
                               "won": bool(state["won"])})
            if done or state["won"]:
                break
        return {"status": "complete", "reward": float(bool(state["won"])),
                "steps": len(trajectory), "initial_observation": initial_observation,
                "memory_sha256": hashlib.sha256((memory_text or "").encode()).hexdigest(),
                "initial_commands_sha256": hashlib.sha256(json.dumps(
                    initial_commands).encode()).hexdigest(),
                "invalid_commands": sum(not step["valid"] for step in trajectory),
                "trajectory": trajectory,
                "termination": "success" if state["won"] else "budget_or_environment_done"}
    except Exception as error:
        return {"status": "infrastructure_failure", "error": repr(error),
                "steps": len(trajectory), "trajectory": trajectory}
    finally:
        env.close()
        agent.set_source(None)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path(
        "current_work/delta-Mem/model/Qwen3-4B-Instruct-2507"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-episode", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("ttcl/data/alfworld_delta"))
    parser.add_argument("--target-game", action="append", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpu-fraction", type=float, default=.40)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.max_steps < 1 or args.max_new_tokens < 1:
        parser.error("Budgets must be positive")
    episode = json.loads(args.source_episode.read_text())
    source_game = episode["game"]
    if "/train/" not in source_game:
        parser.error("Source episode must be an ALFWorld train game")
    records = source_records(episode)
    games = [args.data_root / relative for relative in args.target_game]
    for relative, game in zip(args.target_game, games, strict=True):
        if "/train/" not in relative or relative == source_game:
            parser.error("Targets must be distinct ALFWorld train games")
        if not game.is_file():
            parser.error(f"Target game missing: {game}")
    agent, tokenizer = load_agent(args.model, args.checkpoint,
                                  args.device, args.gpu_fraction)
    fields = tokenize_records(tokenizer, records, args.device)
    report = {"protocol": "Exploratory paired ALFWorld train-game zero-shot probe; synthetic-trained raw hypernetwork; no ALFWorld update; capped budget",
              "source_episode": str(args.source_episode),
              "source_episode_sha256": sha256(args.source_episode),
              "source_game": source_game,
              "checkpoint_sha256": sha256(args.checkpoint),
              "max_steps": args.max_steps,
              "max_new_tokens": args.max_new_tokens,
              "games": []}
    for relative, game in zip(args.target_game, games, strict=True):
        row = {"game": relative, "game_sha256": sha256(game)}
        for arm, enabled in (("base", False), ("generated", True)):
            row[arm] = run_episode(agent, tokenizer, game, fields,
                                   adapter=enabled, device=args.device,
                                   max_steps=args.max_steps,
                                   max_new_tokens=args.max_new_tokens)
        report["games"].append(row)
        print(json.dumps({"game": relative,
                          "base": {k: row["base"].get(k) for k in
                                   ("status", "reward", "invalid_commands")},
                          "generated": {k: row["generated"].get(k) for k in
                                        ("status", "reward", "invalid_commands")}}),
              flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
