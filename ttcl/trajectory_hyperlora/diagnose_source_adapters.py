"""Measure whether reviewed ALFWorld sources produce distinguishable LoRA updates."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import statistics

import torch

from ttcl.trajectory_hyperlora.alfworld_zero_shot_probe import load_agent
from ttcl.trajectory_hyperlora.prepare_alf_next_task import (
    partition, validate_review,
)
from ttcl.trajectory_hyperlora.relational_router_pilot import tokenize_records


def effective_gram(adapters: list[list[torch.Tensor]],
                   factors: list[torch.Tensor]) -> torch.Tensor:
    """Inner products of B @ A without materializing each full LoRA matrix."""
    count = len(adapters)
    kernel = torch.zeros((count, count), dtype=torch.float64)
    for layer, a in enumerate(factors):
        b = torch.stack([item[layer].double() for item in adapters])
        gram = a.double() @ a.double().T
        kernel += torch.einsum("ior,rs,jos->ij", b, gram, b)
    return kernel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path(
        "current_work/delta-Mem/model/Qwen3-4B-Instruct-2507"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_next_task_candidates_20261005.json"))
    parser.add_argument("--review", type=Path, default=Path(
        "data/annotations/trajectory_hyperlora_alf_20261005_reviewed_v2.json"))
    parser.add_argument("--wrong-source-report", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpu-fraction", type=float, default=.40)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output exists; do not overwrite a frozen diagnostic")
    approved = validate_review(json.loads(args.candidates.read_text()),
                               json.loads(args.review.read_text()))
    sources = {}
    for row in approved:
        if partition(row["source_game"]) != "train":
            raise ValueError("Reviewed source is outside the training partition")
        prior = sources.get(row["source_game"])
        if prior and (prior["source_episode_sha256"] != row["source_episode_sha256"]
                      or prior["source_records"] != row["source_records"]):
            raise ValueError("Reviewed source game has conflicting content")
        sources[row["source_game"]] = row
    if len(sources) < 2:
        raise ValueError("Need at least two reviewed train sources")
    wrong_report = json.loads(args.wrong_source_report.read_text())
    checkpoint_sha = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    if wrong_report["checkpoint_sha256"] != checkpoint_sha:
        raise ValueError("Wrong-source report uses a different checkpoint")
    agent, tokenizer = load_agent(args.model, args.checkpoint,
                                  args.device, args.gpu_fraction)
    games = sorted(sources)
    adapters = []
    latents = []
    for game in games:
        fields = tokenize_records(tokenizer, sources[game]["source_records"],
                                  args.device)
        with torch.no_grad():
            latents.append(agent.encode(fields).detach().cpu().squeeze(0).double())
            agent.set_source(fields)
            adapters.append([layer.b.detach().cpu().squeeze(0).clone()
                             for layer in agent.adapters])
            agent.set_source(None)
    factors = [layer.a.detach().cpu() for layer in agent.adapters]
    latent_tensor = torch.stack(latents)
    latent_mean = latent_tensor.mean(0)
    latent_norms = latent_tensor.norm(dim=1).clamp_min(1e-12)
    latent_cosine = (latent_tensor @ latent_tensor.T) / (
        latent_norms[:, None] * latent_norms[None, :])
    latent_centered = latent_tensor - latent_mean
    latent_summary = {
        "mean_pair_cosine": statistics.mean(
            float(latent_cosine[i, j]) for i in range(len(games))
            for j in range(i + 1, len(games))),
        "centered_rms_over_mean_norm": float(
            latent_centered.square().sum(1).mean().sqrt() /
            latent_mean.norm().clamp_min(1e-12))}
    kernel = effective_gram(adapters, factors)
    norms = kernel.diag().clamp_min(0).sqrt()
    cosine = kernel / (norms[:, None] * norms[None, :]).clamp_min(1e-12)
    distance = (kernel.diag()[:, None] + kernel.diag()[None, :] -
                2 * kernel).clamp_min(0).sqrt()
    index = {game: i for i, game in enumerate(games)}
    by_family = defaultdict(list)
    for game in games:
        by_family[sources[game]["family"]].append(index[game])
    family_summary = {}
    for family, members in sorted(by_family.items()):
        pairs = [(i, j) for pos, i in enumerate(members)
                 for j in members[pos + 1:]]
        family_summary[family] = {
            "sources": len(members),
            "mean_effective_norm": statistics.mean(float(norms[i]) for i in members),
            "mean_within_cosine": statistics.mean(float(cosine[i, j])
                                                   for i, j in pairs) if pairs else None,
            "mean_within_distance": statistics.mean(float(distance[i, j])
                                                     for i, j in pairs) if pairs else None}
    all_pairs = [(i, j) for i in range(len(games)) for j in range(i + 1, len(games))]
    cross_pairs = [(i, j) for i, j in all_pairs
                   if sources[games[i]]["family"] != sources[games[j]]["family"]]
    eigenvalues = torch.linalg.eigvalsh(kernel).clamp_min(0)
    centered = kernel - kernel.mean(0, keepdim=True) - \
        kernel.mean(1, keepdim=True) + kernel.mean()
    centered_eigenvalues = torch.linalg.eigvalsh(centered).clamp_min(0)
    geometry = {
        "mean_all_pair_cosine": statistics.mean(float(cosine[i, j])
                                                for i, j in all_pairs),
        "min_all_pair_cosine": min(float(cosine[i, j]) for i, j in all_pairs),
        "mean_cross_family_cosine": statistics.mean(float(cosine[i, j])
                                                    for i, j in cross_pairs),
        "uncentered_top_energy_fraction": float(
            eigenvalues[-1] / eigenvalues.sum().clamp_min(1e-12)),
        "centered_top_energy_fraction": float(
            centered_eigenvalues[-1] /
            centered_eigenvalues.sum().clamp_min(1e-12))}
    pair_rows = []
    for row in wrong_report["games"]:
        correct, wrong = row["correct_source_game"], row["wrong_source_game"]
        if correct not in index or wrong not in index:
            raise ValueError("Probe source absent from reviewed train sources")
        i, j = index[correct], index[wrong]
        pair_rows.append({"game": row["game"],
                          "correct_source_game": correct,
                          "wrong_source_game": wrong,
                          "wrong_source_reward": row["wrong_source"]["reward"],
                          "effective_cosine": float(cosine[i, j]),
                          "effective_distance": float(distance[i, j]),
                          "distance_over_correct_norm": float(
                              distance[i, j] / norms[i].clamp_min(1e-12))})
    result = {"protocol": "Read-only geometry of effective B@A updates from reviewed train trajectories; no retraining or test-game labels",
              "checkpoint_sha256": checkpoint_sha,
              "review_sha256": hashlib.sha256(args.review.read_bytes()).hexdigest(),
              "wrong_source_report_sha256": hashlib.sha256(
                  args.wrong_source_report.read_bytes()).hexdigest(),
              "source_count": len(games),
              "source_latent_geometry": latent_summary,
              "global_geometry": geometry,
              "family_summary": family_summary,
              "wrong_source_pairs": pair_rows,
              "mean_pair_cosine": statistics.mean(
                  row["effective_cosine"] for row in pair_rows)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"source_count": result["source_count"],
                      "source_latent_geometry": latent_summary,
                      "global_geometry": geometry,
                      "family_summary": family_summary,
                      "wrong_source_pairs": [{"game": row["game"],
                                              "wrong_source_reward": row["wrong_source_reward"],
                                              "effective_cosine": row["effective_cosine"],
                                              "distance_over_correct_norm":
                                                  row["distance_over_correct_norm"]}
                                             for row in pair_rows]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
