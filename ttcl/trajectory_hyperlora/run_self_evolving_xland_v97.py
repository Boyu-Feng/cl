"""XLand self-evolving hypernetwork pilot.

The updater starts from a random state and is never fitted on the XLand train
split.  During evaluation, a real positive reward supplies the action label
for an online hypernetwork update; failures only produce a trajectory B update.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
from torch.nn import functional as F

from ttcl.trajectory_hyperlora import run_continual_xland_v94 as v94
from ttcl.trajectory_hyperlora.joint_domains_v93 import read, write, filehash
from ttcl.trajectory_hyperlora.self_evolving_update_v97 import SelfEvolvingUpdater

SOURCE = v94.ROOT / "results/trajectory_hyperlora/continual_xland_v94b_20261010"
OUT = v94.ROOT / "results/trajectory_hyperlora/continual_xland_v97_self_evolving_20261010"


class Runner(v94.Runner):
    def mount(self, bb):
        for adapter, b in zip(self.agent.adapters, bb, strict=True):
            adapter.b = b * 1.0

    def train(self, collection):
        # Deliberately skip all offline training.  The checkpoint is a manifest
        # consumed by v94's evaluator, not a fitted updater state.
        torch.manual_seed(970)
        self.rule = type(self.rule)(self.agent.model.config.hidden_size, self.shapes).to(self.device)
        self.rule.train()
        self.evolver = SelfEvolvingUpdater(self.rule, self.aa, self.norms, lr=1e-4)
        self.rule_optimizer = self.evolver.optimizer
        self._seen_successes = 0
        write(OUT / "zero_pretrain_audit.json", {
            "passed": True,
            "train_split_used": False,
            "updater_checkpoint_loaded": False,
            "random_seed": 970,
            "update_signal": "real positive reward action only",
        })
        write(OUT / "trained.pt", {"pretraining": "none", "random_seed": 970})
        return self.zero()

    def hyper_update(self, bb, ep):
        before = v94.tensor_hash(bb)
        with torch.no_grad():
            candidate, gate = self.rule(self.fields(ep), bb, self.aa, self.norms)
        candidate = [x.detach() for x in candidate]
        positives = [row for row in ep["trajectory"] if float(row.get("reward", 0.0)) > 0]
        updated = False
        loss_value = None
        # v94's shared arm replays a frozen collection (seeds 1.94m/2.94m).
        # It is a comparison stream, not this arm's online experience.  Only
        # genuine on-policy seeds (3.94m+) may update the hypernetwork.
        online_own_episode = int(ep.get("seed", 0)) >= 3940000
        if positives and online_own_episode:
            row = positives[0]
            self.rule_optimizer.zero_grad(set_to_none=True)
            out, _ = self.rule(self.fields(ep), bb, self.aa, self.norms)
            loss = F.cross_entropy(
                self.logits(row["observation"], out).unsqueeze(0),
                torch.tensor([int(row["action"])], device=self.device),
            )
            if torch.isfinite(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.rule.parameters(), 1.0,
                                               error_if_nonfinite=True)
                self.rule_optimizer.step()
                updated = True
                loss_value = float(loss.detach())
                self._seen_successes += 1
        return candidate, dict(
            updates=1, source_hash=ep["content_sha256"], before=before,
            after=v94.tensor_hash(candidate), gate=float(gate.detach()),
            self_updated=updated, self_loss=loss_value,
            success_updates=self._seen_successes,
            online_own_episode=online_own_episode,
        )


def main(device: str):
    if (OUT / "plan.json").exists() or (OUT / "evaluation.json").exists():
        raise RuntimeError(f"refusing to overwrite {OUT}")
    source_plan = read(SOURCE / "plan.json")
    collection = read(SOURCE / "collection.json")
    assert collection["complete"]
    OUT.mkdir(parents=True, exist_ok=True)
    plan = dict(source_plan)
    plan["seed"] = 970
    plan["pretraining"] = "none: updater randomized and learned only from own positive rewards"
    plan["source_collection_sha256"] = filehash(SOURCE / "collection.json")
    write(OUT / "plan.json", plan)
    write(OUT / "collection.json", collection)
    original_out = v94.OUT
    v94.OUT = OUT
    runner = None
    try:
        torch.manual_seed(970)
        torch.cuda.manual_seed_all(970)
        runner = Runner(device)
        prior = runner.train(collection)
        runner.evaluate(collection, prior)
    finally:
        if runner is not None:
            runner.worker.close()
        v94.OUT = original_out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda:0")
    main(p.parse_args().device)
