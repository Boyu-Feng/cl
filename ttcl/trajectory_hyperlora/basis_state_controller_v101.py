"""Frozen-generator, persistent-coefficient LoRA experience state.

The generated basis matrices and their source hypernetwork are immutable.  At
test time only the small coefficient state is updated from reward-confirmed
actions; failure-only observations decay the write gate without inventing an
action label.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import torch
from torch.nn import functional as F


@dataclass
class BasisState:
    coefficients: torch.Tensor
    gate_logit: torch.Tensor


class BasisStateController:
    def __init__(self, bases: Sequence[Sequence[torch.Tensor]], *, lr: float = 1e-2,
                 decay: float = 0.02, anchor_strength: float = 0.0) -> None:
        if not bases or not all(len(x) == len(bases[0]) for x in bases):
            raise ValueError("basis bank must contain equally shaped adapter lists")
        self.bases = tuple(tuple(b.detach().clone() for b in row) for row in bases)
        self.k = len(self.bases)
        device = self.bases[0][0].device
        self.state = BasisState(
            coefficients=torch.zeros(self.k, device=device),
            gate_logit=torch.tensor(0.0, device=device),
        )
        self.lr = lr
        self.decay = decay
        if anchor_strength < 0:
            raise ValueError("anchor_strength must be nonnegative")
        self.anchor_strength = anchor_strength
        self.anchor = tuple(x.detach().clone() for x in self.matrices())

    @property
    def gate(self) -> torch.Tensor:
        return torch.sigmoid(self.state.gate_logit)

    def append_basis(self, basis: Sequence[torch.Tensor], *, initial_logit: float = 0.5) -> None:
        """Add an immutable basis generated from a newly confirmed trajectory."""
        if len(basis) != len(self.bases[0]) and self.bases:
            raise ValueError("basis layer count changed")
        if self.bases and any(tuple(x.shape) != tuple(y.shape)
                              for x, y in zip(basis, self.bases[0], strict=True)):
            raise ValueError("basis shape changed")
        if not torch.isfinite(torch.tensor(initial_logit)):
            raise ValueError("initial basis logit must be finite")
        self.bases = (*self.bases,
                      tuple(x.detach().clone() for x in basis))
        self.k = len(self.bases)
        self.state = BasisState(torch.cat((self.state.coefficients,
                                           torch.tensor([initial_logit], device=self.state.coefficients.device))),
                                self.state.gate_logit)

    def append_residual_basis(self, residual: Sequence[torch.Tensor], *,
                              max_fraction: float = 0.25,
                              initial_logit: float = 0.0,
                              orthogonalize: bool = False) -> float:
        """Append a bounded trajectory-derived residual direction.

        ``residual`` is expected to be the signed direction computed from an
        observed trajectory (for example, failure-loss ascent).  It is
        normalized against the current bank's effective Frobenius norm before
        being stored, so one unusually long trajectory cannot dominate the
        persistent state.  The bases remain immutable after insertion.
        """
        if not 0 < max_fraction <= 1:
            raise ValueError("max_fraction must be in (0, 1]")
        if len(residual) != len(self.bases[0]):
            raise ValueError("residual layer count changed")
        prepared = [x.detach().clone() for x in residual]
        if orthogonalize:
            # Project the concatenated residual away from every existing
            # basis direction. This preserves only novel failure evidence.
            flat = torch.cat([x.float().flatten() for x in prepared])
            for row in self.bases:
                direction = torch.cat([x.float().flatten() for x in row])
                denom = direction.square().sum()
                if denom > 0:
                    flat = flat - (flat @ direction / denom) * direction
            offset = 0
            prepared = []
            for x in residual:
                n = x.numel()
                prepared.append(flat[offset:offset + n].reshape_as(x).to(x.dtype))
                offset += n
        total = torch.sqrt(sum(x.detach().float().square().sum() for x in prepared))
        reference = torch.sqrt(sum(x.detach().float().square().sum()
                                   for x in self.matrices()))
        if not torch.isfinite(total) or total == 0:
            raise ValueError("residual must be finite and nonzero")
        scale = max_fraction * reference / total if reference > 0 else max_fraction
        bounded = [x.detach().clone() * scale for x in prepared]
        self.append_basis(bounded, initial_logit=initial_logit)
        return float(scale)

    def matrices(self) -> list[torch.Tensor]:
        weights = torch.softmax(self.state.coefficients, dim=0) * self.gate
        return [sum(weights[i] * self.bases[i][j] for i in range(self.k))
                for j in range(len(self.bases[0]))]

    def update_success(self, basis_logits: torch.Tensor, action: int) -> float:
        """Legacy logit-mixture surrogate, NOT the loss of the mounted B mixture."""
        if basis_logits.ndim != 2 or basis_logits.shape[0] != self.k:
            raise ValueError("basis_logits must have shape [num_bases, num_actions]")
        if not 0 <= action < basis_logits.shape[1]:
            raise ValueError("invalid action logits")
        # A small differentiable inner step through the coefficient state.  The
        # base model/hypernetwork/bases never receive gradients or mutations.
        c = self.state.coefficients.detach().clone().requires_grad_(True)
        g = self.state.gate_logit.detach().clone().requires_grad_(True)
        weights = torch.softmax(c, 0) * torch.sigmoid(g)
        # Legacy diagnostic only: neural-network outputs are nonlinear in B.
        logits = (weights[:, None] * basis_logits).sum(0)
        loss = F.cross_entropy(logits.unsqueeze(0), torch.tensor([action], device=logits.device))
        grad_c, grad_g = torch.autograd.grad(loss, (c, g), allow_unused=True)
        if grad_c is not None:
            self.state.coefficients = (c - self.lr * grad_c).detach()
        if grad_g is not None:
            self.state.gate_logit = (g - self.lr * grad_g).detach().clamp(-6, 6)
        return float(loss.detach())

    def update_mounted(self, loss_fn: Callable[[list[torch.Tensor]], torch.Tensor]) -> dict:
        """Differentiate the actual actor loss through the mounted B mixture.

        Caller freezes the actor and supplies a loss using only completed past
        feedback. Nothing is committed if the loss or gradients are invalid.
        This is imitation credit when loss_fn is successful-action CE; it is
        not by itself an estimator of future task reward improvement.
        """
        c = self.state.coefficients.detach().clone().requires_grad_(True)
        g = self.state.gate_logit.detach().clone().requires_grad_(True)
        weights = c.softmax(0) * g.sigmoid()
        matrices = [sum(weights[i] * self.bases[i][j] for i in range(self.k))
                    for j in range(len(self.bases[0]))]
        before = [b.detach().clone() for b in matrices]
        loss = loss_fn(matrices)
        if self.anchor_strength:
            anchor_penalty = sum((x - a).square().mean()
                                 for x, a in zip(matrices, self.anchor, strict=True))
            loss = loss + self.anchor_strength * anchor_penalty
        if loss.ndim != 0 or not torch.isfinite(loss):
            raise ValueError("mounted actor loss must be a finite scalar")
        dc, dg = torch.autograd.grad(loss, (c, g))
        if not torch.isfinite(dc).all() or not torch.isfinite(dg):
            raise ValueError("nonfinite coefficient gradient")
        self.state = BasisState((c - self.lr * dc).detach(),
                                (g - self.lr * dg).detach().clamp(-6, 6))
        delta = sum(float((new - old).square().sum())
                    for new, old in zip(self.matrices(), before, strict=True)) ** .5
        return {"loss": float(loss.detach()), "matrix_delta_norm": delta,
                "anchor_strength": self.anchor_strength,
                "coefficient_grad_norm": float(dc.norm()),
                "gate_gradient": float(dg), "matrix_changed": delta > 0}

    def update_failure_mounted(self, loss_fn: Callable[[list[torch.Tensor]], torch.Tensor]) -> dict:
        """Use observed failed actions as negative evidence via bounded ascent."""
        c = self.state.coefficients.detach().clone().requires_grad_(True)
        g = self.state.gate_logit.detach().clone().requires_grad_(True)
        weights = c.softmax(0) * g.sigmoid()
        matrices = [sum(weights[i] * self.bases[i][j] for i in range(self.k))
                    for j in range(len(self.bases[0]))]
        loss = loss_fn(matrices)
        if loss.ndim != 0 or not torch.isfinite(loss):
            raise ValueError("failed-action loss must be finite")
        dc, dg = torch.autograd.grad(loss, (c, g))
        if not torch.isfinite(dc).all() or not torch.isfinite(dg):
            raise ValueError("nonfinite failure gradient")
        self.state = BasisState((c + self.lr * dc).detach(),
                                (g - self.decay - self.lr * dg).detach().clamp(-6, 6))
        return {"loss": float(loss.detach()), "coefficient_grad_norm": float(dc.norm()),
                "gate": float(self.gate), "negative_update": True}

    def update_failure(self) -> None:
        """Record failure without fabricating a corrective action direction."""
        self.state.gate_logit = (self.state.gate_logit - self.decay).detach().clamp(-6, 6)


class FeatureBasisRouter:
    """Small online router from a task key to immutable basis coefficients.

    The actor, generator, and bases stay frozen.  Only this low-dimensional
    key-to-mixture map is updated from the observed mounted-B loss, making the
    persistent experience explicit in the router rather than in a growing
    adapter bank.
    """

    def __init__(self, bases: Sequence[Sequence[torch.Tensor]], key_dim: int, *,
                 lr: float = 1e-2, decay: float = 0.02) -> None:
        if key_dim < 1:
            raise ValueError("key_dim must be positive")
        self.bases = tuple(tuple(x.detach().clone() for x in row) for row in bases)
        self.k = len(self.bases)
        device = self.bases[0][0].device
        self.router = torch.nn.Parameter(torch.zeros(self.k, key_dim, device=device))
        self.bias = torch.nn.Parameter(torch.zeros(self.k, device=device))
        self.gate_logit = torch.nn.Parameter(torch.tensor(0., device=device))
        self.lr, self.decay = lr, decay

    @staticmethod
    def _key(key: torch.Tensor) -> torch.Tensor:
        key = key.detach().float().flatten()
        if key.numel() == 0 or not torch.isfinite(key).all() or key.norm() == 0:
            raise ValueError("router key must be a nonzero finite vector")
        return torch.nn.functional.normalize(key, dim=0)

    def _matrices(self, key: torch.Tensor) -> list[torch.Tensor]:
        key = self._key(key)
        weights = torch.softmax(self.router @ key + self.bias, dim=0) * self.gate_logit.sigmoid()
        return [sum(weights[i] * self.bases[i][j] for i in range(self.k))
                for j in range(len(self.bases[0]))]

    def matrices(self, key: torch.Tensor) -> list[torch.Tensor]:
        return [x.detach() for x in self._matrices(key)]

    def update_mounted(self, key: torch.Tensor,
                       loss_fn: Callable[[list[torch.Tensor]], torch.Tensor], *,
                       failure: bool = False) -> dict:
        loss = loss_fn(self._matrices(key))
        if loss.ndim != 0 or not torch.isfinite(loss):
            raise ValueError("router loss must be a finite scalar")
        grads = torch.autograd.grad(loss, (self.router, self.bias, self.gate_logit))
        if not all(torch.isfinite(g).all() for g in grads):
            raise ValueError("nonfinite router gradient")
        direction = 1.0 if failure else -1.0
        with torch.no_grad():
            self.router.add_(direction * self.lr * grads[0])
            self.bias.add_(direction * self.lr * grads[1])
            self.gate_logit.add_(direction * self.lr * grads[2])
            if failure:
                self.gate_logit.sub_(self.decay)
            self.gate_logit.clamp_(-6, 6)
        return {"loss": float(loss.detach()), "failure": failure,
                "router_grad_norm": float(grads[0].norm()),
                "gate": float(self.gate_logit.sigmoid())}


@dataclass
class PrototypeEntry:
    key: torch.Tensor
    basis: tuple[torch.Tensor, ...]
    count: int = 1


class SuccessPrototypeMemory:
    """Task-keyed bank of immutable B matrices from confirmed successes."""

    def __init__(self, fallback: Sequence[torch.Tensor], *, capacity: int = 16,
                 threshold: float = 0.25) -> None:
        if capacity < 1 or not -1 <= threshold <= 1:
            raise ValueError("invalid prototype memory settings")
        self.fallback = tuple(x.detach().clone() for x in fallback)
        self.capacity, self.threshold = capacity, threshold
        self.entries: list[PrototypeEntry] = []

    @staticmethod
    def _key(key: torch.Tensor) -> torch.Tensor:
        key = key.detach().float().flatten()
        if key.numel() == 0 or not torch.isfinite(key).all() or key.norm() == 0:
            raise ValueError("prototype key must be finite and nonzero")
        return torch.nn.functional.normalize(key, dim=0).cpu()

    def read(self, key: torch.Tensor) -> tuple[list[torch.Tensor], dict]:
        if not self.entries:
            return [x.detach() for x in self.fallback], {"entries": 0, "selected": None}
        key = self._key(key)
        sims = torch.tensor([torch.dot(key, e.key) for e in self.entries])
        selected = int(sims.argmax())
        if float(sims[selected]) < self.threshold:
            return [x.detach() for x in self.fallback], {
                "entries": len(self.entries), "selected": None,
                "best_cosine": float(sims[selected])}
        return [x.detach() for x in self.entries[selected].basis], {
            "entries": len(self.entries), "selected": selected,
            "best_cosine": float(sims[selected])}

    def add(self, key: torch.Tensor, basis: Sequence[torch.Tensor]) -> dict:
        key = self._key(key)
        item = PrototypeEntry(key, tuple(x.detach().clone() for x in basis))
        self.entries.append(item)
        if len(self.entries) > self.capacity:
            self.entries.pop(0)
        return {"entries": len(self.entries)}


class AnchoredDeltaController:
    """Frozen initial B plus bounded residuals from confirmed trajectories."""

    def __init__(self, anchor: Sequence[torch.Tensor], *, lr: float = 1e-2,
                 max_fraction: float = 0.25) -> None:
        self.anchor = tuple(x.detach().clone() for x in anchor)
        self.deltas: list[tuple[torch.Tensor, ...]] = []
        self.coefficients = torch.empty(0, device=self.anchor[0].device)
        self.lr, self.max_fraction = lr, max_fraction

    def matrices(self) -> list[torch.Tensor]:
        if not self.deltas:
            return [x.detach() for x in self.anchor]
        return [self.anchor[j] + sum(self.coefficients[i] * self.deltas[i][j]
                                     for i in range(len(self.deltas)))
                for j in range(len(self.anchor))]

    def add_success_delta(self, basis: Sequence[torch.Tensor], *,
                          initial_weight: float = 0.0) -> dict:
        if len(basis) != len(self.anchor):
            raise ValueError("delta layer count changed")
        raw = [x.detach() - a for x, a in zip(basis, self.anchor, strict=True)]
        norm = torch.sqrt(sum(x.float().square().sum() for x in raw))
        anchor_norm = torch.sqrt(sum(x.float().square().sum() for x in self.anchor))
        if not torch.isfinite(norm) or norm == 0:
            return {"added": False, "reason": "zero_delta"}
        if not 0 <= initial_weight <= 1:
            raise ValueError("initial_weight must be in [0, 1]")
        scale = self.max_fraction * anchor_norm / norm if anchor_norm > 0 else self.max_fraction
        self.deltas.append(tuple(x * scale for x in raw))
        self.coefficients = torch.cat((self.coefficients,
                                       torch.tensor([initial_weight], device=self.coefficients.device)))
        return {"added": True, "delta_norm": float((norm * scale).detach()),
                "count": len(self.deltas)}

    def update_mounted(self, loss_fn: Callable[[list[torch.Tensor]], torch.Tensor], *,
                       failure: bool = False) -> dict:
        if not self.deltas:
            return {"updated": False, "reason": "empty"}
        c = self.coefficients.detach().clone().requires_grad_(True)
        matrices = [self.anchor[j] + sum(c[i] * self.deltas[i][j]
                                         for i in range(len(self.deltas)))
                    for j in range(len(self.anchor))]
        loss = loss_fn(matrices)
        if loss.ndim != 0 or not torch.isfinite(loss):
            raise ValueError("delta loss must be finite")
        grad = torch.autograd.grad(loss, c)[0]
        direction = 1.0 if failure else -1.0
        self.coefficients = (c + direction * self.lr * grad).detach().clamp(-1, 1)
        return {"updated": True, "loss": float(loss.detach()),
                "gradient_norm": float(grad.norm()), "failure": failure}


class SignedBasisController:
    """Bounded signed composition with the frozen mixture as its start point."""

    def __init__(self, bases: Sequence[Sequence[torch.Tensor]], *, lr: float = 1e-2,
                 bound: float = 1.0, tied: bool = False,
                 preserve_l1: bool = False, positive: bool = False) -> None:
        if not bases or bound <= 0:
            raise ValueError("invalid signed basis settings")
        self.bases = tuple(tuple(x.detach().clone() for x in row) for row in bases)
        self.lr, self.bound = lr, bound
        self.tied = tied
        self.preserve_l1 = preserve_l1
        self.positive = positive
        device = self.bases[0][0].device
        self.coefficients = torch.full((len(self.bases),), .5 / len(self.bases), device=device)

    def matrices(self) -> list[torch.Tensor]:
        return [sum(self.coefficients[i] * self.bases[i][j]
                    for i in range(len(self.bases)))
                for j in range(len(self.bases[0]))]

    def update_mounted(self, loss_fn: Callable[[list[torch.Tensor]], torch.Tensor], *,
                       failure: bool = False) -> dict:
        c = self.coefficients.detach().clone().requires_grad_(True)
        matrices = [sum(c[i] * self.bases[i][j] for i in range(len(self.bases)))
                    for j in range(len(self.bases[0]))]
        loss = loss_fn(matrices)
        if loss.ndim != 0 or not torch.isfinite(loss):
            raise ValueError("signed composition loss must be finite")
        grad = torch.autograd.grad(loss, c)[0]
        if not torch.isfinite(grad).all():
            raise ValueError("nonfinite signed composition gradient")
        if self.tied:
            # Project the update onto the all-ones coefficient direction.
            # Mean (not sum) matches the common-mode step of the full router.
            grad = grad.mean().expand_as(grad)
        direction = 1.0 if failure else -1.0
        updated = (c + direction * self.lr * grad).detach().clamp(
            0 if self.positive else -self.bound, self.bound)
        if self.preserve_l1:
            magnitude = updated.abs().sum()
            if magnitude > 0:
                updated = updated * (.5 / magnitude)
        self.coefficients = updated
        return {"updated": True, "loss": float(loss.detach()),
                "gradient_norm": float(grad.norm()), "failure": failure,
                "coefficient_min": float(self.coefficients.min()),
                "coefficient_max": float(self.coefficients.max()),
                "coefficients": self.coefficients.cpu().tolist(),
                "tied": self.tied, "preserve_l1": self.preserve_l1,
                "positive": self.positive}


@dataclass
class PersistentBasisEntry:
    key: torch.Tensor
    controller: BasisStateController
    count: int = 1


class PersistentBasisMemory:
    """Bounded key-addressed persistent state over one frozen basis bank.

    The basis matrices are shared and immutable.  Each memory slot owns only
    its coefficient/gate state, so one task's successful trajectory cannot
    overwrite another task's state.  Query reads interpolate slot states by
    cosine attention; writes update the nearest slot or create a new one.
    """

    def __init__(self, bases: Sequence[Sequence[torch.Tensor]], *, capacity: int = 16,
                 temperature: float = 0.05, slot_similarity: float = 0.9,
                 lr: float = 1e-2,
                 decay: float = 0.02) -> None:
        if capacity < 1 or temperature <= 0 or not 0 <= slot_similarity <= 1:
            raise ValueError("invalid persistent basis memory settings")
        self.bases = tuple(tuple(x.detach().clone() for x in row) for row in bases)
        self.capacity, self.temperature = capacity, temperature
        self.slot_similarity, self.lr, self.decay = slot_similarity, lr, decay
        self.entries: list[PersistentBasisEntry] = []

    @staticmethod
    def _key(key: torch.Tensor) -> torch.Tensor:
        key = key.detach().float().flatten().cpu()
        if key.numel() == 0 or not torch.isfinite(key).all() or key.norm() == 0:
            raise ValueError("memory key must be a nonzero finite vector")
        return F.normalize(key, dim=0)

    def _new(self, key: torch.Tensor) -> PersistentBasisEntry:
        return PersistentBasisEntry(key, BasisStateController(
            self.bases, lr=self.lr, decay=self.decay))

    def _nearest(self, key: torch.Tensor) -> tuple[int, torch.Tensor]:
        sims = torch.stack([torch.dot(key, e.key) for e in self.entries])
        return int(sims.argmax()), sims

    def _slot_for_write(self, key: torch.Tensor) -> tuple[int, PersistentBasisEntry]:
        """Select or allocate a key-addressed state slot for an update."""
        key = self._key(key)
        if not self.entries:
            self.entries.append(self._new(key))
            return 0, self.entries[0]
        index, sims = self._nearest(key)
        if len(self.entries) < self.capacity and float(sims[index]) < self.slot_similarity:
            self.entries.append(self._new(key))
            index = len(self.entries) - 1
        return index, self.entries[index]

    def write_success(self, key: torch.Tensor, basis_logits: torch.Tensor,
                      action: int) -> dict:
        index, entry = self._slot_for_write(key)
        loss = entry.controller.update_success(basis_logits, action)
        entry.count += 1
        return {"slot": index, "loss": loss, "entries": len(self.entries)}

    def write_mounted_success(self, key: torch.Tensor,
                              loss_fn: Callable[[list[torch.Tensor]], torch.Tensor]) -> dict:
        """Apply actual mounted-B success credit to the key's persistent slot."""
        index, entry = self._slot_for_write(key)
        audit = entry.controller.update_mounted(loss_fn)
        entry.count += 1
        return {"slot": index, "entries": len(self.entries), **audit}

    def write_mounted_failure(self, key: torch.Tensor,
                              loss_fn: Callable[[list[torch.Tensor]], torch.Tensor]) -> dict:
        """Apply observed failed-action negative evidence to a key's slot."""
        index, entry = self._slot_for_write(key)
        audit = entry.controller.update_failure_mounted(loss_fn)
        return {"slot": index, "entries": len(self.entries), **audit}

    def write_failure(self, key: torch.Tensor) -> dict:
        if not self.entries:
            return {"written": False, "reason": "empty"}
        index, _ = self._nearest(self._key(key))
        self.entries[index].controller.update_failure()
        return {"written": True, "slot": index, "entries": len(self.entries)}

    def read(self, key: torch.Tensor) -> tuple[list[torch.Tensor] | None, dict]:
        if not self.entries:
            return None, {"entries": 0, "weights": []}
        key = self._key(key)
        sims = torch.stack([torch.dot(key, e.key) for e in self.entries])
        weights = torch.softmax(sims / self.temperature, dim=0)
        matrices = [sum(weights[i] * self.entries[i].controller.matrices()[j]
                        for i in range(len(self.entries)))
                    for j in range(len(self.bases[0]))]
        return matrices, {"entries": len(self.entries), "weights": weights.tolist(),
                          "selected": int(weights.argmax()), "cosine": sims.tolist()}
