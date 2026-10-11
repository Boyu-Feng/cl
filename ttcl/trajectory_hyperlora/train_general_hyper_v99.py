"""Train one domain-blind frozen trajectory-to-LoRA hypernetwork (v99).

The prepared corpus contains no domain metadata in encoder inputs.  A matched
direct-LoRA arm is trained with the same rows, order, optimizer and budget.
Both states are frozen before any transfer evaluation.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ttcl.trajectory_hyperlora.direct_composition_pilot import DirectRelationHyperLoRA
from ttcl.trajectory_hyperlora.joint_domains_v93 import MODEL, digest, filehash, read, write


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "results/trajectory_hyperlora/general_corpus_v99_20261010"
OUT = ROOT / "results/trajectory_hyperlora/general_hyper_v99_20261010"


def main(device: str):
    raise RuntimeError("v99 invalidated: use a new reviewed corpus excluding target datasets and a fresh output directory")
    if (OUT / "complete.json").exists():
        raise RuntimeError(f"refusing to overwrite {OUT}")
    data = read(DATA_DIR / "dataset.json")
    manifest = read(DATA_DIR / "manifest.json")
    assert digest(data) == manifest["dataset_sha256"]
    OUT.mkdir(parents=True, exist_ok=True)
    write(OUT / "protocol.json", {
        "protocol": "general_hyper_v99",
        "corpus_manifest_sha256": filehash(DATA_DIR / "manifest.json"),
        "dataset_sha256": manifest["dataset_sha256"],
        "frozen_before_transfer": True,
        "updates_per_arm": len(data["schedule"]),
        "encoder_fields": manifest["encoder_fields"],
        "forbidden_encoder_fields": manifest["forbidden_encoder_fields"],
        "seed": 9902,
    })
    torch.manual_seed(9902)
    torch.cuda.manual_seed_all(9902)
    tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, attn_implementation="sdpa", local_files_only=True
    ).to(device)
    agent = DirectRelationHyperLoRA(model, rank=4, layers=2, encoder_kind="contextual", task_conditioned=True).to(device)
    agent.eval()
    hyper_params = {n: p for n, p in agent.named_parameters() if p.requires_grad}
    ha = [a.a for a in agent.adapters]
    sa = [torch.nn.Parameter(a.detach().clone()) for a in ha]
    sb = [torch.nn.Parameter(torch.zeros(1, a.base.out_features, a.rank, device=device)) for a in agent.adapters]
    norms = [float(a.base.weight.float().norm()) for a in agent.adapters]
    shapes = [(a.base.out_features, 4) for a in agent.adapters]
    # Encode source/query once.  The cache is based only on the permitted text.
    cache = {}
    def enc(text):
        key = digest(text)
        if key not in cache:
            ids = tok(text, add_special_tokens=False).input_ids
            used = ids if len(ids) <= 2048 else ids[:1024] + ids[-1024:]
            agent.set_source(None)
            with torch.no_grad():
                h = model(input_ids=torch.tensor([used], device=device), output_hidden_states=True, use_cache=False).hidden_states[-1]
            cache[key] = {"contextual": h[:, -1].float().detach(), "pair_contextual": h.float().mean(1).detach()}
        return key
    prepared = []
    for row in data["rows"]:
        ids = tok.apply_chat_template(row["messages"], tokenize=True, add_generation_prompt=True)
        answer = tok(row["answer"] + tok.eos_token, add_special_tokens=False).input_ids
        prepared.append({"prefix": ids, "answer": answer, "source": enc(row["source"]), "query": enc(row["query"])})
    torch.save({"fields": cache, "prepared": prepared}, OUT / "cache.pt")
    opt_h = torch.optim.AdamW(list(hyper_params.values()), lr=1e-4, weight_decay=0)
    opt_l = torch.optim.AdamW(sa + sb, lr=1e-4, weight_decay=0)
    history = []
    started = time.time()
    for step, idx in enumerate(data["schedule"], 1):
        row = prepared[idx]
        ids = torch.tensor([row["prefix"] + row["answer"]], device=device)
        labels = torch.tensor([[-100] * len(row["prefix"]) + row["answer"]], device=device)
        losses = {}
        for arm in ("hyper", "lora"):
            if arm == "hyper":
                for a, p in zip(agent.adapters, ha, strict=True):
                    a.a = p
                agent.set_source({k: v.to(device) for k, v in cache[row["source"]].items()},
                                 target_fields={k: v.to(device) for k, v in cache[row["query"]].items()})
                params = list(hyper_params.values()); opt = opt_h
            else:
                for a, p, b in zip(agent.adapters, sa, sb, strict=True):
                    a.a = p
                    a.b = b * 1.0
                params = sa + sb; opt = opt_l
            opt.zero_grad(set_to_none=True)
            loss = model(input_ids=ids, labels=labels, use_cache=False).loss
            assert torch.isfinite(loss), (step, arm)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
            opt.step()
            losses[arm] = {"loss": float(loss.detach()), "grad_norm": float(norm)}
        history.append({"step": step, "row_index": idx, **losses})
        if step == 1 or step % 10 == 0:
            print(json.dumps(history[-1]), flush=True)
            write(OUT / "status.json", {"stage": "training", "updates": step, "total": len(data["schedule"]), "elapsed": time.time() - started})
        if step % 100 == 0:
            torch.save({"step": step, "hyper": {n: p.detach().cpu() for n, p in hyper_params.items()},
                        "static_a": [p.detach().cpu() for p in sa], "static_b": [p.detach().cpu() for p in sb]}, OUT / f"checkpoint_{step:04d}.pt")
            write(OUT / "training.json", history)
    torch.save({"step": len(history), "hyper": {n: p.detach().cpu() for n, p in hyper_params.items()},
                "static_a": [p.detach().cpu() for p in sa], "static_b": [p.detach().cpu() for p in sb],
                "corpus_manifest_sha256": filehash(DATA_DIR / "manifest.json")}, OUT / "trained.pt")
    write(OUT / "training.json", history)
    write(OUT / "complete.json", {"training_complete": True, "updates_per_arm": len(history),
                                   "checkpoint_sha256": filehash(OUT / "trained.pt"),
                                   "corpus_manifest_sha256": filehash(DATA_DIR / "manifest.json"),
                                   "frozen_before_transfer": True})
    write(OUT / "status.json", {"stage": "training_complete", "updates": len(history), "frozen": True})


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--device", default="cuda:3")
    main(p.parse_args().device)
