"""RAMP: reward-guided advantage memory and same-context preferences.

No teacher, reward model, generated memory text, or hidden task labels are used.
Loss and training dependencies are imported where they are used.
"""

from __future__ import annotations

import copy
import math
import random
from collections import deque
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class RewardConfig:
    learning_rate: float = 2e-5
    epochs: int = 1
    max_seq_length: int = 8192
    capacity: int = 64
    replay_size: int = 8
    warmup: int = 2
    baseline_decay: float = 0.9
    scale_floor: float = 0.1
    advantage_clip: float = 2.0
    reward_min: float = 0.0
    reward_max: float = 1.0
    beta: float = 1.0
    point_weight: float = 1.0
    pair_weight: float = 1.0
    anchor_weight: float = 0.05
    trust_weight: float = 0.1
    success_threshold: float = 0.7
    min_pair_gap: float = 0.05
    max_pairs: int = 16
    max_grad_norm: float = 1.0
    max_token_drift: float = 0.2
    seed: int = 42
    advantage_mode: str = "historical"
    max_backtracks: int = 0
    backtrack_factor: float = 0.5

    def __post_init__(self):
        if self.advantage_mode not in ("historical", "group"):
            raise ValueError("advantage_mode must be historical or group")
        if type(self.max_backtracks) is not int or self.max_backtracks < 0:
            raise ValueError("max_backtracks must be a nonnegative integer")
        if not math.isfinite(self.backtrack_factor) or not 0 < self.backtrack_factor < 1:
            raise ValueError("backtrack_factor must be finite and in (0, 1)")
        for name in ("epochs", "max_seq_length", "capacity", "warmup", "max_pairs"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.max_seq_length < 2 or self.replay_size < 0:
            raise ValueError("max_seq_length must be >= 2 and replay_size >= 0")
        for name in (
            "learning_rate",
            "scale_floor",
            "advantage_clip",
            "beta",
            "max_grad_norm",
            "max_token_drift",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("point_weight", "pair_weight", "anchor_weight", "trust_weight"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if not 0 <= self.baseline_decay < 1:
            raise ValueError("baseline_decay must be in [0, 1)")
        if not (
            math.isfinite(self.reward_min)
            and math.isfinite(self.reward_max)
            and self.reward_min < self.reward_max
        ):
            raise ValueError("reward_min must be finite and below reward_max")
        if not 0 <= self.success_threshold <= 1 or not 0 < self.min_pair_gap <= 1:
            raise ValueError(
                "success_threshold and min_pair_gap use normalized rewards"
            )


@dataclass(frozen=True)
class Experience:
    uid: int
    prompt: str  # Fully rendered context, including any system/history messages.
    response: str
    reward: float  # Normalized into [0, 1].
    task_id: str
    valid: bool
    advantage: float
    baseline: float | None
    scale: float
    response_ids: tuple[int, ...] | None = None
    group_id: int | None = None
    group_size: int = 1


class RewardBuffer:
    def __init__(self, config: RewardConfig):
        self.config = config
        self.replay: deque[Experience] = deque(maxlen=config.capacity)
        self.pending: list[Experience] = []
        self.stats: dict[str, dict] = {}
        self.count = 0
        self.rng = random.Random(config.seed)

    def _validate(self, prompt, response, reward, task_id, valid, response_ids):
        c = self.config
        if not isinstance(prompt, str) or not prompt or not isinstance(response, str):
            raise ValueError("prompt must be nonempty text and response must be text")
        if not isinstance(task_id, str) or not task_id or not isinstance(valid, bool):
            raise ValueError("task_id must be nonempty text and valid must be boolean")
        if response_ids is not None and (
            not isinstance(response_ids, (list, tuple))
            or any(type(token) is not int or token < 0 for token in response_ids)
        ):
            raise ValueError("response_ids must be a list of nonnegative integer token IDs")
        reward = float(reward)
        if not math.isfinite(reward) or not c.reward_min <= reward <= c.reward_max:
            raise ValueError("reward is non-finite or outside the configured range")
        return (reward - c.reward_min) / (c.reward_max - c.reward_min)

    def _update_stats(self, task_id, reward):
        stats = self.stats.setdefault(task_id, {"count": 0, "mean": 0.0, "variance": 0.0})
        if stats["count"] == 0:
            stats["mean"] = reward
        else:
            delta = reward - stats["mean"]
            decay = self.config.baseline_decay
            stats["mean"] += (1 - decay) * delta
            stats["variance"] = decay * (stats["variance"] + (1 - decay) * delta * delta)
        stats["count"] += 1

    def observe(
        self,
        prompt,
        response,
        reward,
        *,
        task_id="default",
        valid=True,
        response_ids=None,
    ):
        c = self.config
        if c.advantage_mode == "group":
            return self.observe_group(prompt, [{"response": response, "reward": reward,
                                               "valid": valid, "response_ids": response_ids}],
                                      task_id=task_id)[0]
        reward = self._validate(prompt, response, reward, task_id, valid, response_ids)
        stats = self.stats.setdefault(
            task_id, {"count": 0, "mean": 0.0, "variance": 0.0}
        )
        baseline = stats["mean"] if stats["count"] else None
        scale = max(math.sqrt(max(0.0, stats["variance"])), c.scale_floor)
        advantage = 0.0
        if stats["count"] >= c.warmup:
            advantage = max(
                -c.advantage_clip, min(c.advantage_clip, (reward - baseline) / scale)
            )
        if not valid:
            advantage = min(
                0.0, advantage
            )  # Invalid output is never a positive target.
        self.count += 1
        item = Experience(
            self.count,
            prompt,
            response,
            reward,
            task_id,
            valid,
            advantage,
            baseline,
            scale,
            tuple(response_ids) if response_ids is not None else None,
        )
        self.pending.append(item)
        # Causal baseline: update statistics only AFTER assigning this advantage.
        self._update_stats(task_id, reward)
        return item

    def observe_group(self, prompt, candidates, *, task_id="default"):
        """Record feedback for candidates generated from one identical current context.

        In group mode, center rewards within this group only. Neither preceding
        questions nor later rewards can change these advantages. A singleton or
        tied group has zero point signal. This spends one reward per candidate;
        it must not be presented as single-feedback online evaluation.
        """
        if not isinstance(candidates, (list, tuple)) or not candidates:
            raise ValueError("candidates must be a nonempty list or tuple")
        rewards = []
        for candidate in candidates:
            if not isinstance(candidate, dict) or not {"response", "reward"} <= candidate.keys():
                raise ValueError("each candidate requires response and reward")
            if candidate.get("prompt", prompt) != prompt or candidate.get("task_id", task_id) != task_id:
                raise ValueError("all candidates must share the identical prompt and task_id")
            rewards.append(self._validate(prompt, candidate["response"], candidate["reward"],
                                          task_id, candidate.get("valid", True),
                                          candidate.get("response_ids")))
        # Validate the entire group before mutating any state.
        if self.config.advantage_mode == "historical":
            return [self.observe(prompt, x["response"], x["reward"], task_id=task_id,
                                 valid=x.get("valid", True), response_ids=x.get("response_ids"))
                    for x in candidates]
        c = self.config
        baseline = math.fsum(rewards) / len(rewards)
        scale = max(math.sqrt(math.fsum((r - baseline) ** 2 for r in rewards) / len(rewards)),
                    c.scale_floor)
        tied = max(rewards) == min(rewards)
        group_id = self.count + 1
        items = []
        for candidate, reward in zip(candidates, rewards):
            advantage = 0.0 if tied else max(-c.advantage_clip, min(c.advantage_clip,
                                                                  (reward - baseline) / scale))
            valid = candidate.get("valid", True)
            if not valid:
                advantage = min(0.0, advantage)
            self.count += 1
            ids = candidate.get("response_ids")
            item = Experience(self.count, prompt, candidate["response"], reward, task_id,
                              valid, advantage, baseline, scale,
                              tuple(ids) if ids is not None else None, group_id, len(candidates))
            items.append(item)
            self._update_stats(task_id, reward)  # Audit only; never used for group advantages.
        self.pending.extend(items)
        return items

    def select(self):
        # Fresh data is never evicted by the bounded replay capacity.
        old = list(self.replay)
        successes = [
            x for x in old if x.valid and x.reward >= self.config.success_threshold
        ]
        n_success = min(len(successes), (self.config.replay_size + 1) // 2)
        chosen = self.rng.sample(successes, n_success)
        used = {x.uid for x in chosen}
        others = [x for x in old if x.uid not in used]
        chosen += self.rng.sample(
            others, min(len(others), self.config.replay_size - len(chosen))
        )
        return list(self.pending) + chosen

    def commit(self):
        self.replay.extend(self.pending)
        self.pending.clear()


def preference_pairs(examples, config):
    """Never infer a chosen/rejected pair across different rendered contexts."""
    pairs = []
    for i, left in enumerate(examples):
        for j in range(i + 1, len(examples)):
            right = examples[j]
            if (left.task_id, left.prompt) != (right.task_id, right.prompt):
                continue
            if left.response == right.response:
                continue
            winner, loser = (i, j) if left.reward > right.reward else (j, i)
            gap = examples[winner].reward - examples[loser].reward
            if examples[winner].valid and gap >= config.min_pair_gap:
                pairs.append((winner, loser, gap))
    return sorted(pairs, key=lambda p: (-p[2], p[0], p[1]))[: config.max_pairs]


def point_loss(delta, advantage, beta):
    """Gradient at delta=0 is -advantage/2; weight is NOT normalized away."""
    import torch.nn.functional as F

    sign = 1.0 if advantage >= 0 else -1.0
    return abs(advantage) * F.softplus(-sign * beta * delta) / beta


def pair_loss(winner_delta, loser_delta, gap, beta):
    import torch.nn.functional as F

    return gap * F.softplus(-beta * (winner_delta - loser_delta)) / beta


class RewardLearner:
    """Update only the model's already-trainable parameters (normally LoRA).

    Reference token log probabilities are cached before each update; no second LM
    is needed. This is a discriminative reward surrogate, not unbiased PPO/GRPO.
    Optional drift backtracking reuses the exact examples and reference without
    new feedback. An accepted reduced learning rate persists; complete failure
    restores the original weights and Adam state, including the learning rate.
    """

    def __init__(self, model, tokenizer, config: RewardConfig):
        import torch

        self.model, self.tokenizer, self.config = model, tokenizer, config
        self.buffer = RewardBuffer(config)
        self.parameters = [p for p in model.parameters() if p.requires_grad]
        if not self.parameters:
            raise ValueError("Model has no trainable parameters")
        self.optimizer = torch.optim.AdamW(
            self.parameters, lr=config.learning_rate, weight_decay=0.0
        )
        self.updates = 0
        self.attempts = 0
        self.optimizer_steps = 0
        self.retained_optimizer_steps = 0

    def observe(self, *args, **kwargs):
        return self.buffer.observe(*args, **kwargs)

    def observe_group(self, *args, **kwargs):
        return self.buffer.observe_group(*args, **kwargs)

    def encode(self, item):
        prompt = self.tokenizer.encode(item.prompt, add_special_tokens=False)
        if item.response_ids is not None:
            response = list(item.response_ids)
        else:
            response = self.tokenizer.encode(item.response, add_special_tokens=False)
            if response and self.tokenizer.eos_token_id is not None:
                response.append(self.tokenizer.eos_token_id)
        if not prompt or not response:
            return None, "empty_tokens"
        if len(prompt) + len(response) > self.config.max_seq_length:
            return None, "overlength"  # Never assign a full-answer reward to a prefix.
        return (prompt + response, len(prompt)), None

    def token_logps(self, encoded):
        import torch
        import torch.nn.functional as F

        ids, start = encoded
        batch = torch.tensor([ids], dtype=torch.long, device=self.parameters[0].device)
        logits = (
            self.model(
                input_ids=batch, attention_mask=torch.ones_like(batch), use_cache=False
            )
            .logits[0, start - 1 : -1]
            .float()
        )
        # Cross entropy avoids explicitly materializing another vocabulary-sized tensor.
        return -F.cross_entropy(logits, batch[0, start:], reduction="none")

    def update(self):
        import torch

        c = self.config
        if not self.buffer.pending:
            return {"accepted": False, "reason": "no_new_data", "update": self.updates}
        self.attempts += 1
        selected = self.buffer.select()
        examples, encoded, skipped = [], [], []
        for item in selected:
            tokens, reason = self.encode(item)
            if reason:
                skipped.append({"uid": item.uid, "reason": reason})
            else:
                examples.append(item)
                encoded.append(tokens)
        pairs = preference_pairs(examples, c)

        def positive(x):
            return (x.valid and x.reward > 0 and x.reward >= c.success_threshold
                    and (c.advantage_mode != "group" or x.advantage > 0))

        active = any(
            c.point_weight * abs(x.advantage) > 0
            or (c.anchor_weight > 0 and positive(x))
            for x in examples
        )
        active = active or (c.pair_weight > 0 and bool(pairs))
        report = {
            "attempt": self.attempts,
            "update": self.updates,
            "accepted": False,
            "fresh_count": len(self.buffer.pending),
            "example_count": len(examples),
            "pair_count": len(pairs),
            "point_count": sum(c.point_weight > 0 and x.advantage != 0 for x in examples),
            "anchor_count": sum(c.anchor_weight > 0 and positive(x) for x in examples),
            "advantage_mode": c.advantage_mode,
            "optimizer_steps": 0,
            "retained_optimizer_steps": 0,
            "backtrack_count": 0,
            "backtrack_history": [],
            "skipped": skipped,
            "positive_advantages": sum(x.advantage > 0 for x in examples),
            "negative_advantages": sum(x.advantage < 0 for x in examples),
            "zero_advantages": sum(x.advantage == 0 for x in examples),
            "source_uids": [x.uid for x in examples],
        }
        if not active:
            self.buffer.commit()
            return {**report, "reason": "no_reward_signal"}
        # eval() disables dropout but still permits autograd.
        self.model.eval()
        with torch.no_grad():
            reference = [self.token_logps(x).detach().cpu() for x in encoded]
        saved_parameters = [p.detach().cpu().clone() for p in self.parameters]
        saved_optimizer = copy.deepcopy(self.optimizer.state_dict())
        initial_lrs = [group["lr"] for group in self.optimizer.param_groups]
        report["learning_rate_before"] = initial_lrs[0]

        def restore():
            # load_state_dict can share tensor storage with its argument. A fresh
            # deepcopy prevents retries from mutating the saved Adam moments.
            with torch.no_grad():
                for parameter, saved in zip(self.parameters, saved_parameters):
                    parameter.copy_(saved.to(parameter.device))
            self.optimizer.load_state_dict(copy.deepcopy(saved_optimizer))

        for trial in range(c.max_backtracks + 1):
            report["backtrack_count"] = trial
            if trial:
                # The failed attempt already restored weights and Adam moments.
                for group, initial_lr in zip(self.optimizer.param_groups, initial_lrs):
                    group["lr"] = initial_lr * c.backtrack_factor ** trial
            trial_lrs = [group["lr"] for group in self.optimizer.param_groups]
            trial_steps = 0
            losses, grad_norms = [], []
            branch_losses = {name: [] for name in ("point", "pair", "anchor", "trust")}
            proposed = None
            report.pop("max_token_rms_drift", None)
            try:
                for _ in range(c.epochs):
                    self.optimizer.zero_grad(set_to_none=True)
                    total = 0.0
                    branches = {name: 0.0 for name in branch_losses}
                    for item, tokens, ref in zip(examples, encoded, reference):
                        logps = self.token_logps(tokens)
                        change = logps - ref.to(logps.device)
                        point = c.point_weight * point_loss(
                            change.mean(), item.advantage, c.beta
                        )
                        trust = c.trust_weight * change.square().mean()
                        anchor = -c.anchor_weight * item.reward * logps.mean() if positive(item) else logps.new_zeros(())
                        loss = point + trust + anchor
                        for name, value in (("point", point), ("trust", trust), ("anchor", anchor)):
                            branches[name] += float(value.detach()) / len(examples)
                        loss = loss / len(examples)
                        if not torch.isfinite(loss):
                            raise FloatingPointError("nonfinite_loss")
                        loss.backward()
                        total += float(loss.detach())
                    if c.pair_weight:
                        for winner, loser, gap in pairs:
                            lp_w = self.token_logps(encoded[winner]).mean()
                            lp_l = self.token_logps(encoded[loser]).mean()
                            loss = (
                                c.pair_weight
                                * pair_loss(
                                    lp_w - reference[winner].mean().to(lp_w.device),
                                    lp_l - reference[loser].mean().to(lp_l.device),
                                    gap,
                                    c.beta,
                                )
                                / len(pairs)
                            )
                            if not torch.isfinite(loss):
                                raise FloatingPointError("nonfinite_loss")
                            loss.backward()
                            total += float(loss.detach())
                            branches["pair"] += float(loss.detach())
                    norm = torch.nn.utils.clip_grad_norm_(self.parameters, c.max_grad_norm)
                    if not torch.isfinite(norm):
                        raise FloatingPointError("nonfinite_gradient")
                    self.optimizer.step()
                    self.optimizer_steps += 1
                    report["optimizer_steps"] += 1
                    trial_steps += 1
                    losses.append(total)
                    grad_norms.append(float(norm))
                    for name in branch_losses:
                        branch_losses[name].append(branches[name])
                with torch.no_grad():
                    proposed = [self.token_logps(x).cpu() for x in encoded]
                    drifts = [float((lp - ref).square().mean().sqrt())
                              for lp, ref in zip(proposed, reference)]
                if not all(math.isfinite(x) for x in drifts):
                    raise FloatingPointError("nonfinite_drift")
                report["max_token_rms_drift"] = max(drifts)
                if max(drifts) > c.max_token_drift:
                    raise FloatingPointError("drift_limit")
            except BaseException as exc:
                # Also undo partial epochs and unexpected execution failures.
                restore()
                if not isinstance(exc, FloatingPointError):
                    raise
                report["reason"] = str(exc)
            else:
                self.updates += 1
                self.retained_optimizer_steps += trial_steps
                report["retained_optimizer_steps"] = trial_steps
                report.update(accepted=True, update=self.updates, reason="accepted")
            finally:
                self.optimizer.zero_grad(set_to_none=True)
                self.model.eval()
            report["backtrack_history"].append({
                "trial": trial, "learning_rate": trial_lrs[0],
                "learning_rates": trial_lrs, "optimizer_steps": trial_steps,
                "max_token_rms_drift": report.get("max_token_rms_drift"),
                "reason": report["reason"], "accepted": report["accepted"],
            })
            if report["accepted"] or report["reason"] != "drift_limit":
                break
        report["learning_rate_after"] = self.optimizer.param_groups[0]["lr"]
        self.buffer.commit()
        report.update(losses=losses, grad_norms=grad_norms, branch_losses=branch_losses,
                      trust_active_steps=sum(value > 0 for value in branch_losses["trust"]))
        # A snapshot-relative quadratic trust term has zero gradient at the
        # first full-batch step. With epochs=1 only the post-step drift guard
        # constrains the move; log this honestly rather than calling it KL.
        if proposed is not None and all(torch.isfinite(lp).all() for lp in proposed):
            before = [float(lp.mean()) for lp in reference]
            proposal = [float(lp.mean()) for lp in proposed]
            after = proposal if report["accepted"] else before
            deltas = [new - old for new, old in zip(after, before)]
            report["logprob_diagnostics"] = [
                {"uid": item.uid, "advantage": item.advantage,
                 "mean_logprob_before": old, "mean_logprob_proposed": attempted,
                 "mean_logprob_after": new, "delta": delta}
                for item, old, attempted, new, delta in zip(examples, before, proposal, after, deltas)
            ]
            signed = [math.copysign(1.0, item.advantage) * delta
                      for item, delta in zip(examples, deltas) if item.advantage != 0]
            report["mean_signed_logprob_delta"] = math.fsum(signed) / len(signed) if signed else None
            report["fraction_advantages_followed"] = sum(x > 0 for x in signed) / len(signed) if signed else None
            pair_gains = [deltas[winner] - deltas[loser] for winner, loser, _ in pairs]
            report["mean_pair_margin_gain"] = math.fsum(pair_gains) / len(pair_gains) if pair_gains else None
        return report

    def state_dict(self):
        """Audit state only; excludes weights, optimizer moments and RNG state."""
        return {
            "config": asdict(self.config),
            "updates": self.updates,
            "attempts": self.attempts,
            "optimizer_steps": self.optimizer_steps,
            "retained_optimizer_steps": self.retained_optimizer_steps,
            "optimizer_learning_rates": [group["lr"] for group in self.optimizer.param_groups],
            "observations": self.buffer.count,
            "baseline_stats": self.buffer.stats,
            "replay": [asdict(x) for x in self.buffer.replay],
            "pending": [asdict(x) for x in self.buffer.pending],
        }
