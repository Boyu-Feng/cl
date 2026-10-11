"""Real XLand smoke test for persistent basis-state updates.

Uses one held-out ruleset from the existing v94 plan.  The hypernetwork and
two frozen basis matrices are never updated; only the key-addressed state is
written after a reward-confirmed transition.  This is an interface pilot,
not a full benchmark.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ttcl.trajectory_hyperlora import run_continual_xland_v94 as v94
from ttcl.trajectory_hyperlora.basis_state_controller_v101 import PersistentBasisMemory
from ttcl.trajectory_hyperlora.joint_domains_v93 import write, filehash


ROOT = v94.ROOT
SOURCE = ROOT / "results/trajectory_hyperlora/continual_xland_v94_20261010"
OUT = ROOT / "results/trajectory_hyperlora/basis_state_xland_smoke_v101b_20261010"


def run(device: str) -> dict:
    raise RuntimeError("Invalidated: zero untrained bases and no direct LoRA update; see v101 audit correction")
    if (OUT / "result.json").exists():
        raise RuntimeError("refusing overwrite")
    OUT.mkdir(parents=True, exist_ok=True)
    runner = v94.Runner(device)
    try:
        plan = v94.read(SOURCE / "plan.json")
        task = plan["test"][0]
        collection = v94.read(SOURCE / "collection.json")
        source = collection["test"][0][0]
        # Generate one frozen basis from a completed source trajectory and keep
        # the zero update as an explicit null basis.
        with torch.no_grad():
            source_fields = runner.fields(source)
            runner.agent.set_source({"contextual": source_fields["last"]})
            generated = [a.b.detach().clone() for a in runner.agent.adapters]
            runner.agent.set_source(None)
        zero = runner.zero()
        memory = PersistentBasisMemory([zero, generated], capacity=1, lr=.2, decay=.02)
        rows = []
        direct = runner.zero()
        for episode in range(4):
            key = torch.tensor([1.0], device=runner.device)
            weighted, read_audit = memory.read(key)
            weighted = weighted or runner.zero()
            # Same environment seed for all arms; seed 3940000+97*1009+2 is
            # retained from the prior audited on-policy pilot and is known to
            # contain one positive transition.
            seed = 3940000 + int(task["ruleset_id"]) * 1009 + episode
            base_ep, _ = runner.rollout(task, seed, runner.zero())
            direct_ep, _ = runner.rollout(task, seed, direct)
            state_ep, _ = runner.rollout(task, seed, weighted)
            positive = next((x for x in state_ep["trajectory"] if float(x["reward"]) > 0), None)
            update = {"written": False, "reason": "no_reward"}
            if positive is not None:
                basis_logits = torch.stack([
                    runner.logits(positive["observation"], b) for b in (zero, generated)
                ])
                update = memory.write_success(key, basis_logits, int(positive["action"]))
            else:
                memory.write_failure(key)
            rows.append({"episode": episode + 1,
                         "base_return": base_ep["return"],
                         "direct_lora_return": direct_ep["return"],
                         "persistent_basis_return": state_ep["return"],
                         "base_positive": base_ep["positive"],
                         "direct_lora_positive": direct_ep["positive"],
                         "persistent_basis_positive": state_ep["positive"],
                         "read": read_audit, "update": update})
        result = {"protocol": "one held-out XLand ruleset, four episodes per arm; official replay",
                  "task_id": task["task_id"], "ruleset_id": task["ruleset_id"],
                  "source_sha256": source["content_sha256"],
                  "plan_sha256": filehash(SOURCE / "plan.json"), "rows": rows}
        result["summary"] = {k: sum(r[k] > 0 for r in rows) for k in
                             ("base_return", "direct_lora_return", "persistent_basis_return")}
        write(OUT / "result.json", result)
        return result
    finally:
        runner.worker.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda:2")
    print(json.dumps(run(p.parse_args().device), indent=2))
