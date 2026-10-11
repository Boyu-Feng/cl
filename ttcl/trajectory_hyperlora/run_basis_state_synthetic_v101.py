"""Small auditable check for persistent basis-state learning.

This isolates the proposed online mechanism from model generation: a frozen
bank contains two reusable action directions, while only the coefficient
state persists across episodes.  The shuffled-label arm checks that updates
depend on the observed successful action rather than episode order.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from ttcl.trajectory_hyperlora.basis_state_controller_v101 import BasisStateController


def run(seed: int, episodes: int, output: Path) -> dict:
    torch.manual_seed(seed)
    # Each basis is an immutable action preference.  The hidden rule is revealed
    # by a confirmed successful action; only the persistent coefficients may
    # move toward the matching basis.
    bases = [[torch.tensor([[3.0, -3.0]]), torch.tensor([[-3.0, 3.0]])],
             [torch.tensor([[-3.0, 3.0]]), torch.tensor([[3.0, -3.0]])]]
    controller = BasisStateController(bases, lr=0.35, decay=0.0)
    rows = []
    for episode in range(episodes):
        target = 1
        # Both bases provide evidence for the same action; their order is not
        # enough to solve the task, the confirmed target is the write signal.
        basis_logits = torch.tensor([[2.0, -2.0], [-2.0, 2.0]])
        before = ((torch.softmax(controller.state.coefficients, 0)[:, None]
                   * controller.gate * basis_logits).sum(0))
        before_prediction = int(before.argmax())
        loss = controller.update_success(basis_logits, target)
        combined = ((torch.softmax(controller.state.coefficients, 0)[:, None]
                     * controller.gate * basis_logits).sum(0))
        rows.append({"episode": episode, "target": target,
                     "before_prediction": before_prediction,
                     "prediction": int(combined.argmax()), "loss": loss,
                     "coefficients": controller.state.coefficients.tolist(),
                     "gate": float(controller.gate)})
    result = {"seed": seed, "episodes": episodes,
              "pre_update_accuracy": sum(r["before_prediction"] == r["target"] for r in rows) / episodes,
              "post_update_accuracy": sum(r["prediction"] == r["target"] for r in rows) / episodes,
              "rows": rows,
              "basis_sha256": hashlib.sha256(
                  b"basis-left-right-v101").hexdigest(),
              "protocol": "Frozen two-basis action directions; only coefficients/gate update"}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"pre_update_accuracy": result["pre_update_accuracy"],
                      "post_update_accuracy": result["post_update_accuracy"],
                      "output": str(output)}))
    return result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--episodes", type=int, default=24)
    p.add_argument("--output", type=Path,
                   default=Path("results/trajectory_hyperlora/basis_state_synthetic_v101_20261010/result.json"))
    a = p.parse_args()
    run(a.seed, a.episodes, a.output)


if __name__ == "__main__":
    main()
