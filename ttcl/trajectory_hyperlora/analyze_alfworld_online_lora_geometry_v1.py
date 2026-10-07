"""Decompose online trajectory LoRA matrices without changing the frozen run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from ttcl.trajectory_hyperlora.alfworld_online_context_vector_mean_v1 import (
    update_fields, vector_hash,
)
from ttcl.trajectory_hyperlora.alfworld_online_from_empty import (
    digest, file_hash, records_from_episode,
)
from ttcl.trajectory_hyperlora.alfworld_online_persistent_context_lora_v1 import (
    bounded_source_text,
)
from ttcl.trajectory_hyperlora.alfworld_zero_shot_probe import load_agent
from ttcl.trajectory_hyperlora.contextual_alf_source import (
    contextual_text_fields, task_context_text,
)


def flat(tensors):
    return torch.cat([x.detach().cpu().float().reshape(-1) for x in tensors])


def cosine(x, y):
    return float(F.cosine_similarity(x.reshape(1, -1),
                                     y.reshape(1, -1)).item())


def source_and_pair(agent, fields, target):
    source_latent = agent.contextual_latent(fields["contextual"].float())
    source_vector = fields.get("pair_contextual", fields["contextual"])
    target_vector = target.get("pair_contextual", target["contextual"])
    source = F.layer_norm(source_vector.float(), (source_vector.shape[-1],))
    tgt = F.layer_norm(target_vector.float(), (target_vector.shape[-1],))
    pair = torch.cat((source * tgt, (source - tgt).abs()), dim=-1)
    pair_latent = agent.task_pair_latent(pair)
    return source_latent, pair_latent


def factors(agent):
    return [layer.b.detach().cpu().float().squeeze(0).clone()
            for layer in agent.adapters]


def matrix_stats(agent, current, source_part, pair_part, bias):
    rows = []
    for index, (layer, b, s, p, c) in enumerate(zip(
            agent.adapters, current, source_part, pair_part, bias, strict=True)):
        a = layer.a.detach().cpu().float()
        reduced = b @ torch.linalg.qr(a.T, mode="reduced").R.T / layer.rank
        singular = torch.linalg.svdvals(reduced)
        reconstruction = (s + p + c - b).norm() / b.norm().clamp_min(1e-12)
        rows.append({"layer": index,
            "b_frobenius": float(b.norm()),
            "source_component_frobenius": float(s.norm()),
            "pair_component_frobenius": float(p.norm()),
            "bias_component_frobenius": float(c.norm()),
            "source_pair_cosine": cosine(s.flatten(), p.flatten()),
            "component_relative_reconstruction_error": float(reconstruction),
            "delta_w_frobenius": float(reduced.norm()),
            "base_w_frobenius": float(layer.base.weight.detach().float().norm()),
            "delta_to_base_frobenius_ratio": float(
                reduced.norm() / layer.base.weight.detach().float().norm()),
            "delta_w_singular_values": [float(x) for x in singular]})
    return rows


def analyze(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    run = json.loads(args.report.read_text())
    if (run["checkpoint_sha256"] != file_hash(args.checkpoint) or
            run["failures"] or len(run["games"]) != 36 or
            run["context_tokens"] != args.context_tokens):
        raise ValueError("Changed or incomplete frozen online run")
    agent, tokenizer = load_agent(args.model, args.checkpoint,
                                  args.device, args.gpu_fraction)
    if (agent.encoder_kind != "contextual" or
            agent.task_context_scope != "current" or
            agent.task_pair_pooling != "mean"):
        raise ValueError("Expected current-context contextual hypernetwork")
    fields = None
    before_last_update = None
    updates = 0
    prior = []
    rows = []
    for row in run["games"]:
        if (row["prior_episode_count"] != len(prior) or
                row["prior_episodes_sha256"] != digest(prior) or
                row["source_vector_sha256_before"] != vector_hash(fields)):
            raise ValueError("Changed online source state")
        target = contextual_text_fields(agent, tokenizer,
            task_context_text(row["online"]["initial_observation"],
                              row["online"]["initial_observation"]),
            args.device, args.context_tokens, pooling="both")
        entry = {"game": row["game"], "family": row["family"],
                 "prior_success_count": updates,
                 "base_reward": row["base"]["reward"],
                 "online_reward": row["online"]["reward"]}
        if fields is not None:
            with torch.no_grad():
                agent.set_source(fields, target_fields=target)
                current = factors(agent)
                source_latent, pair_latent = source_and_pair(agent, fields,
                                                            target)
                source_part = [F.linear(source_latent, head.weight, None)
                    .reshape(head.out_features // layer.rank, layer.rank)
                    .detach().cpu().float()
                    for head, layer in zip(agent.b_heads, agent.adapters,
                                            strict=True)]
                pair_part = [F.linear(pair_latent, head.weight, None)
                    .reshape(head.out_features // layer.rank, layer.rank)
                    .detach().cpu().float()
                    for head, layer in zip(agent.b_heads, agent.adapters,
                                            strict=True)]
                bias = [head.bias.detach().cpu().float().reshape(
                    head.out_features // layer.rank, layer.rank)
                    for head, layer in zip(agent.b_heads, agent.adapters,
                                            strict=True)]
                entry["matrix"] = matrix_stats(agent, current, source_part,
                                                pair_part, bias)
                if before_last_update is not None:
                    agent.set_source(before_last_update, target_fields=target)
                    prior_b = factors(agent)
                    x, y = flat(current), flat(prior_b)
                    entry["last_update_same_target_relative_b_change"] = float(
                        (x - y).norm() / y.norm().clamp_min(1e-12))
                    entry["last_update_same_target_b_cosine"] = cosine(x, y)
                agent.set_source(None)
        episode = row["online"]
        records = records_from_episode(episode)
        if row["new_records_sha256"] != digest(records):
            raise ValueError("Own trajectory records changed")
        prior.append({"game": row["game"], "reward": episode["reward"],
                      "records": records})
        if episode["reward"]:
            text, original, retained = bounded_source_text(
                tokenizer, records, args.context_tokens)
            if (row["source_tokens_original"] != original or
                    row["source_tokens_retained"] != retained):
                raise ValueError("Changed truncation of own trajectory")
            fresh = contextual_text_fields(agent, tokenizer, text,
                args.device, args.context_tokens, pooling="both")
            before_last_update = fields
            fields = update_fields(fields, fresh, updates)
            updates += 1
        if (row["memory_updated"] != bool(episode["reward"]) or
                row["update_count_after"] != updates or
                row["source_vector_sha256_after"] != vector_hash(fields)):
            raise ValueError("Online vector update does not reproduce")
        rows.append(entry)
        print(json.dumps({"n": len(rows), "updates": updates}), flush=True)
    matrix = [m for row in rows for m in row.get("matrix", [])]
    changed = [row for row in rows if
               "last_update_same_target_relative_b_change" in row]
    summary = {"games": len(rows), "matrix_observations": len(matrix),
        "updates": updates,
        "mean_delta_to_base_frobenius_ratio": sum(
            x["delta_to_base_frobenius_ratio"] for x in matrix) / len(matrix),
        "mean_source_component_frobenius": sum(
            x["source_component_frobenius"] for x in matrix) / len(matrix),
        "mean_pair_component_frobenius": sum(
            x["pair_component_frobenius"] for x in matrix) / len(matrix),
        "mean_bias_component_frobenius": sum(
            x["bias_component_frobenius"] for x in matrix) / len(matrix),
        "max_component_relative_reconstruction_error": max(
            x["component_relative_reconstruction_error"] for x in matrix),
        "mean_last_update_same_target_relative_b_change": sum(
            x["last_update_same_target_relative_b_change"] for x in changed
            ) / len(changed),
        "min_last_update_same_target_b_cosine": min(
            x["last_update_same_target_b_cosine"] for x in changed)}
    result = {"protocol": "Read-only exact online source replay and low-rank matrix decomposition on frozen official 36-game run; current task initial feedback; no model training or environment interaction",
        "report_sha256": file_hash(args.report),
        "checkpoint_sha256": file_hash(args.checkpoint),
        "summary": summary, "games": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path(
        "current_work/delta-Mem/model/Qwen3-4B-Instruct-2507"))
    parser.add_argument("--checkpoint", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_sibling_simple40_onpolicy600_20261006.pt"))
    parser.add_argument("--report", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_online_reward_gate_valid_unseen_36_20261007.json"))
    parser.add_argument("--output", type=Path, default=Path(
        "results/trajectory_hyperlora/alf_online_lora_geometry_valid_unseen36_20261007.json"))
    parser.add_argument("--context-tokens", type=int, default=2048)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpu-fraction", type=float, default=.65)
    analyze(parser.parse_args())


if __name__ == "__main__":
    main()
