"""Online LoRA continual learning from environment trajectories."""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer


class OnlineLoRAMemory:
    """A causal LM whose LoRA adapter is updated after every N trajectories."""

    def __init__(
        self,
        model_path: str,
        output_dir: str,
        *,
        device: str = "cuda:0",
        dtype: str = "bfloat16",
        update_every: int = 8,
        replay_trajectories: int = 32,
        learning_rate: float = 2e-5,
        train_epochs: int = 1,
        max_seq_length: int = 2048,
        max_new_tokens: int = 256,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        target_modules: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj"),
        adapter_path: str | None = None,
    ) -> None:
        if update_every < 1 or replay_trajectories < 1 or train_epochs < 1:
            raise ValueError("update_every, replay_trajectories, and train_epochs must be positive")

        self.device = torch.device(device)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.update_every = update_every
        self.replay_trajectories = replay_trajectories
        self.learning_rate = learning_rate
        self.train_epochs = train_epochs
        self.max_seq_length = max_seq_length
        self.max_new_tokens = max_new_tokens
        self.trajectory_count = 0
        self.update_count = 0
        self._current_trajectory: list[dict[str, Any]] = []
        self._replay: deque[list[dict[str, Any]]] = deque(maxlen=replay_trajectories)

        model_dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }.get(dtype.lower())
        if model_dtype is None:
            raise ValueError("dtype must be bfloat16, float16, or float32")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=model_dtype,
            local_files_only=True,
        ).to(self.device)

        if adapter_path is not None:
            self.model = PeftModel.from_pretrained(model, adapter_path, is_trainable=True)
        else:
            self.model = get_peft_model(
                model,
                LoraConfig(
                    r=lora_r,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                    target_modules=list(target_modules),
                    task_type=TaskType.CAUSAL_LM,
                ),
            )
        self.model.eval()

    def _format_prompt(self, prompt: str, history: list[dict[str, Any]]) -> str:
        previous = []
        for step in history:
            previous.append(f"Task: {step['prompt']}")
            previous.append(f"Agent: {step['response']}")
            if step.get("result") is not None:
                previous.append(f"Environment result: {step['result']}")
        context = "\n".join(previous)
        plain_prompt = f"{context}\nTask: {prompt}" if context else f"Task: {prompt}"
        if not hasattr(self.tokenizer, "apply_chat_template"):
            return f"{plain_prompt}\nAgent:"
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an agent interacting with an environment. "
                    "Follow the task instructions and return only the requested response."
                ),
            },
            {"role": "user", "content": plain_prompt},
        ]
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def respond(self, prompt: str) -> str:
        rendered = self._format_prompt(prompt, self._current_trajectory)
        inputs = self.tokenizer(
            rendered,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_seq_length,
        ).to(self.device)
        self.model.eval()
        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        return self.tokenizer.decode(
            output[0, inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
        ).strip()

    def record_step(self, prompt: str, response: str, result: Any = None) -> None:
        self._current_trajectory.append(
            {"prompt": prompt, "response": response, "result": result}
        )

    def reset(self) -> None:
        """Clear the active trajectory while keeping learned LoRA weights."""
        self._current_trajectory.clear()

    def observe_trajectory(self, trajectory: list[dict[str, Any]]) -> bool:
        """Add one completed trajectory and update LoRA when the window is full."""
        if not trajectory:
            return False
        self._replay.append(list(trajectory))
        self.trajectory_count += 1
        self._current_trajectory.clear()
        if self.trajectory_count % self.update_every != 0:
            return False
        self._update_from_replay()
        return True

    def _training_examples(self) -> list[tuple[str, str]]:
        examples = []
        for trajectory in self._replay:
            history: list[dict[str, Any]] = []
            for step in trajectory:
                prompt = self._format_prompt(str(step["prompt"]), history)
                response = str(step["response"])
                examples.append((prompt, response))
                history.append(step)
        return examples

    def _update_from_replay(self) -> None:
        examples = self._training_examples()
        if not examples:
            return
        self.model.train()
        optimizer = torch.optim.AdamW(
            (parameter for parameter in self.model.parameters() if parameter.requires_grad),
            lr=self.learning_rate,
        )
        for _ in range(self.train_epochs):
            for prompt, response in examples:
                prompt_ids = self.tokenizer(
                    prompt,
                    add_special_tokens=True,
                    truncation=True,
                    max_length=self.max_seq_length,
                )["input_ids"]
                response_ids = self.tokenizer(
                    " " + response + self.tokenizer.eos_token,
                    add_special_tokens=False,
                    truncation=True,
                    max_length=max(1, self.max_seq_length - len(prompt_ids)),
                )["input_ids"]
                input_ids = torch.tensor([prompt_ids + response_ids], device=self.device)
                labels = torch.tensor(
                    [[-100] * len(prompt_ids) + response_ids],
                    device=self.device,
                )
                attention_mask = torch.ones_like(input_ids)
                loss = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                ).loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        self.model.eval()
        self.update_count += 1
        checkpoint = self.output_dir / f"update-{self.update_count}"
        self.model.save_pretrained(checkpoint)
        self.tokenizer.save_pretrained(checkpoint)

    def save(self, path: str | None = None) -> Path:
        destination = Path(path) if path else self.output_dir / "latest"
        self.model.save_pretrained(destination)
        self.tokenizer.save_pretrained(destination)
        return destination


def load_jsonl(path: str | Path) -> list[list[dict[str, Any]]]:
    """Load one completed trajectory per JSONL line."""
    trajectories = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            trajectory = item.get("trajectory", item) if isinstance(item, dict) else item
            if not isinstance(trajectory, list):
                raise ValueError("Each JSONL record must be a trajectory list or contain `trajectory`")
            trajectories.append(trajectory)
    return trajectories
