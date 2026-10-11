"""Four-ruleset XLand pilot for the v101 persistent basis state."""
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
OUT = ROOT / "results/trajectory_hyperlora/basis_state_xland_pilot_v101c_20261010"


def run(device: str, task_count: int = 4, episodes: int = 2) -> dict:
    raise RuntimeError("Invalidated: zero untrained bases and no direct LoRA update; see v101 audit correction")
    if (OUT / "result.json").exists():
        raise RuntimeError("refusing overwrite")
    OUT.mkdir(parents=True, exist_ok=True)
    runner = v94.Runner(device)
    try:
        plan = v94.read(SOURCE / "plan.json")
        collection = v94.read(SOURCE / "collection.json")
        rows = []
        for ti, task in enumerate(plan["test"][:task_count]):
            source = collection["test"][ti][0]
            with torch.no_grad():
                sf = runner.fields(source)
                runner.agent.set_source({"contextual": sf["last"]})
                generated = [a.b.detach().clone() for a in runner.agent.adapters]
                runner.agent.set_source(None)
            zero = runner.zero()
            memory = PersistentBasisMemory([zero, generated], capacity=2, lr=.2, decay=.02)
            direct = runner.zero()
            for episode in range(episodes):
                key = sf["mean"].detach()
                weighted, read_audit = memory.read(key)
                weighted = weighted or runner.zero()
                seed = 3940000 + int(task["ruleset_id"]) * 1009 + episode
                base_ep, _ = runner.rollout(task, seed, runner.zero())
                direct_ep, _ = runner.rollout(task, seed, direct)
                state_ep, _ = runner.rollout(task, seed, weighted)
                positive = next((x for x in state_ep["trajectory"]
                                 if float(x["reward"]) > 0), None)
                if positive is not None:
                    logits = torch.stack([
                        runner.logits(positive["observation"], b)
                        for b in (zero, generated)])
                    update = memory.write_success(key, logits,
                                                  int(positive["action"]))
                else:
                    update = memory.write_failure(key)
                rows.append({"task_id": task["task_id"],
                             "ruleset_id": task["ruleset_id"],
                             "episode": episode + 1,
                             "base_return": base_ep["return"],
                             "direct_lora_return": direct_ep["return"],
                             "persistent_basis_return": state_ep["return"],
                             "read": read_audit, "update": update})
        result = {"protocol": "four held-out XLand rulesets, two shared-seed episodes per arm; official replay",
                  "plan_sha256": filehash(SOURCE / "plan.json"), "rows": rows,
                  "summary": {k: sum(r[k] > 0 for r in rows) for k in
                               ("base_return", "direct_lora_return", "persistent_basis_return")}}
        write(OUT / "result.json", result)
        return result
    finally:
        runner.worker.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--device", default="cuda:2")
    print(json.dumps(run(p.parse_args().device), indent=2))
