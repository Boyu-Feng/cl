"""Single-feedback online policy-gradient memory in a LoRA adapter.

Only a scalar reward is accepted by observe(). No self-edit generation,
preference pairs, critic model, or hidden benchmark targets are required.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class RewardBaseline:
    alpha: float = 0.2
    warmup: int = 4
    scale_floor: float = 0.05
    advantage_clip: float = 2.0
    count: int = 0
    mean: float = 0.0
    variance: float = 0.0

    def observe(self, reward: float) -> dict:
        if not math.isfinite(reward):
            raise ValueError("Reward must be finite")
        # Compute the baseline BEFORE incorporating the current reward.
        baseline = self.mean
        scale = max(math.sqrt(max(self.variance, 0.0)), self.scale_floor)
        ready = self.count >= self.warmup
        advantage = (
            max(
                -self.advantage_clip,
                min(self.advantage_clip, (reward - baseline) / scale),
            )
            if ready
            else 0.0
        )
        if self.count == 0:
            self.mean = reward
        else:
            difference = reward - self.mean
            self.mean += self.alpha * difference
            self.variance = (1 - self.alpha) * (
                self.variance + self.alpha * difference * difference
            )
        self.count += 1
        return {
            "reward": reward,
            "baseline": baseline,
            "scale": scale,
            "advantage": advantage,
            "ready": ready,
        }


def clipped_policy_loss(new_logp, old_logp, advantage, clip=0.2):
    """Positive advantage increases sampled-token probability, negative reduces it."""
    ratio = (new_logp - old_logp.detach()).exp()
    return -torch.minimum(
        ratio * advantage, ratio.clamp(1 - clip, 1 + clip) * advantage
    ).mean()


def categorical_kl(logp, reference_logp):
    """Exact vocabulary KL, averaged over sampled response prefixes."""
    return (logp.exp() * (logp - reference_logp.detach())).sum(-1).mean()


@dataclass
class Sample:
    ids: torch.Tensor
    prompt_length: int
    old_logp: torch.Tensor
    text: str
    truncated: bool


class RewardLoRA:
    def __init__(self, args):
        from peft import LoraConfig, get_peft_model
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            GenerationConfig,
            set_seed,
        )

        set_seed(args.seed)
        self.args = args
        self.updates = 0
        self.baseline = RewardBaseline(warmup=args.warmup)
        self.tokenizer = AutoTokenizer.from_pretrained(
            args.model, local_files_only=True
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        base = AutoModelForCausalLM.from_pretrained(
            args.model,
            local_files_only=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        ).to(args.device)
        self.model = get_peft_model(
            base,
            LoraConfig(
                r=16,
                lora_alpha=32,
                lora_dropout=0.0,
                bias="none",
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                task_type="CAUSAL_LM",
            ),
        )
        self.model.gradient_checkpointing_enable()
        self.model.enable_input_require_grads()
        # Keep rollout and training likelihoods consistent (no dropout).
        for module in self.model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0.0
        self.parameters = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            self.parameters, lr=args.learning_rate, weight_decay=0.0
        )
        self.generation = GenerationConfig(
            do_sample=True,
            temperature=1.0,
            top_p=1.0,
            top_k=0,
            repetition_penalty=1.0,
            max_new_tokens=args.max_new_tokens,
            eos_token_id=base.generation_config.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            return_dict_in_generate=True,
            output_scores=True,
        )

    def sample(self, prompt: str) -> Sample:
        rendered = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(
            rendered, add_special_tokens=False, return_tensors="pt"
        ).to(self.args.device)
        prompt_length = inputs.input_ids.shape[1]
        if prompt_length > self.args.max_input_tokens:
            raise ValueError(
                "Prompt exceeds max-input-tokens; reduce history-scans or increase the budget"
            )
        self.model.eval()
        with torch.inference_mode():
            # Pass these as kwargs too: Transformers can replace fields equal to
            # global defaults with the model's sampling defaults when merging configs.
            output = self.model.generate(
                **inputs,
                generation_config=self.generation,
                use_model_defaults=False,
                temperature=1.0,
                top_p=1.0,
                top_k=0,
                repetition_penalty=1.0,
            )
            response = output.sequences[0, prompt_length:]
            # Scores are from the actual sampling distribution, before any update.
            old_logp = torch.stack(
                [
                    F.log_softmax(score[0].float(), dim=-1)[token]
                    for score, token in zip(output.scores, response)
                ]
            )
        eos = self.generation.eos_token_id
        eos = eos if isinstance(eos, list) else [eos]
        return Sample(
            ids=output.sequences.detach().cpu().clone(),
            prompt_length=prompt_length,
            old_logp=old_logp.detach().cpu().clone(),
            text=self.tokenizer.decode(response, skip_special_tokens=True).strip(),
            truncated=int(response[-1]) not in eos,
        )

    def _distribution(self, sample):
        ids = sample.ids.to(self.args.device)
        length = ids.shape[1] - sample.prompt_length
        # Qwen3 supports selecting just the hidden positions needed for the loss.
        logits = (
            self.model(
                input_ids=ids,
                attention_mask=torch.ones_like(ids),
                use_cache=False,
                logits_to_keep=length + 1,
            )
            .logits[0, :-1]
            .float()
        )
        logp = F.log_softmax(logits, dim=-1)
        tokens = ids[0, sample.prompt_length :]
        selected = logp.gather(-1, tokens[:, None]).squeeze(-1)
        return logp, selected

    def observe(self, sample: Sample, reward: float, *, update: bool) -> dict:
        stats = self.baseline.observe(float(reward))
        stats.update(
            {
                "accepted": False,
                "update_count": self.updates,
                "response_tokens": sample.ids.shape[1] - sample.prompt_length,
            }
        )
        if not update or not stats["ready"] or abs(stats["advantage"]) < 1e-6:
            stats["reason"] = (
                "disabled_or_terminal" if not update else "warmup_or_zero_advantage"
            )
            return stats
        snapshot = [p.detach().clone() for p in self.parameters]
        optimizer_snapshot = copy.deepcopy(self.optimizer.state_dict())
        old_selected = sample.old_logp.to(self.args.device)
        self.model.eval()
        with torch.no_grad():
            old_distribution, _ = self._distribution(sample)
            with self.model.disable_adapter():
                reference, _ = self._distribution(sample)
        losses = []
        try:
            self.model.train()
            for _ in range(self.args.update_steps):
                self.optimizer.zero_grad(set_to_none=True)
                distribution, selected = self._distribution(sample)
                policy = clipped_policy_loss(selected, old_selected, stats["advantage"])
                regularizer = categorical_kl(distribution, reference)
                loss = policy + self.args.kl_beta * regularizer
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite policy loss")
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(
                    self.parameters, 0.5, error_if_nonfinite=True
                )
                self.optimizer.step()
                losses.append(float(loss.detach()))
                del loss, policy, regularizer, distribution, selected
            self.model.eval()
            with torch.no_grad():
                after, selected = self._distribution(sample)
                update_kl = float(categorical_kl(old_distribution, after))
                reference_kl = float(categorical_kl(after, reference))
                probability_change = float((selected - old_selected).mean())
            stats.update(
                {
                    "update_kl": update_kl,
                    "reference_kl": reference_kl,
                    "mean_sampled_logprob_change": probability_change,
                    "grad_norm_before_clip": float(norm),
                    "losses": losses,
                }
            )
            if not math.isfinite(update_kl) or update_kl > self.args.max_update_kl:
                raise FloatingPointError("Update exceeded the KL limit")
        except (FloatingPointError, RuntimeError) as exc:
            with torch.no_grad():
                for parameter, saved in zip(self.parameters, snapshot):
                    parameter.copy_(saved)
            self.optimizer.load_state_dict(optimizer_snapshot)
            self.optimizer.zero_grad(set_to_none=True)
            self.model.eval()
            # Numerical/KL rejection is recoverable; other runtime errors must surface.
            if isinstance(exc, FloatingPointError):
                stats["reason"] = "kl_or_numerical_rejection"
                return stats
            raise
        self.optimizer.zero_grad(set_to_none=True)
        self.updates += 1
        stats.update(
            {"accepted": True, "reason": "updated", "update_count": self.updates}
        )
        return stats

    def save(self, path):
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
