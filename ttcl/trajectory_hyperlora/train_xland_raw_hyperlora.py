"""Raw trajectory -> generated LoRA -> future action, without rule features.

This small frozen-policy pilot tests the learning objective before a costly LLM
run. The only parsing is the public XLand observation/action/reward schema.
Rule kinds, tile transitions, target actions, and object matches are never
features. The target query is passed to the policy without source episodes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


def checked_query(query: dict) -> tuple[dict, int]:
    content = query["model_input"]
    digest = hashlib.sha256(json.dumps(content, sort_keys=True,
        separators=(",", ":")).encode()).hexdigest()
    if not query["reviewed_target"] or digest != query["input_sha256"]:
        raise ValueError("Unreviewed target or changed model input")
    return content, int(query["target_action"])


def state_tensor(state: dict) -> list[int]:
    return [value for row in state["observation"] for tile in row
            for value in tile] + list(state["pocket"])


def encode_query(content: dict) -> tuple[list[list[int]], list[list[int]],
                                         list[list[int]], list[int],
                                         list[float], list[float], list[int],
                                         list[int], list[int], list[int]]:
    before, after, goals, actions, rewards, dones = [], [], [], [], [], []
    episodes, step_positions = [], []
    for episode_index, episode in enumerate(content["source_episodes"]):
        for step_index, step in enumerate(episode["steps"]):
            before.append(state_tensor(step["state"]))
            after.append(state_tensor(step["next_state"]))
            goals.append(episode["goal"])
            actions.append(step["action"])
            rewards.append(float(step["reward"]))
            dones.append(float(step["done"]))
            episodes.append(episode_index)
            step_positions.append(step_index)
    if not before or len(before) > 16:
        raise ValueError("Expected 1..16 public trajectory steps")
    return (before, after, goals, actions, rewards, dones,
            state_tensor(content["target_initial_state"]), content["goal"],
            episodes, step_positions)


def make_split(items: list[dict], device: str) -> dict[str, torch.Tensor]:
    rows, labels, opposite = [], [], []
    for item in items:
        if set(item["arms"]) == {"hold_near", "near_hold"}:
            variants, queries_per_arm = ("hold_near", "near_hold"), 2
        elif set(item["arms"]) == {"hold", "near", "tile_near"}:
            variants, queries_per_arm = ("hold", "near", "tile_near"), 1
        else:
            raise ValueError("Unsupported audited trajectory group")
        start = len(rows)
        for arm_index, variant in enumerate(variants):
            for query_index in range(queries_per_arm):
                query = item["arms"][variant]["queries"][query_index]
                content, target = checked_query(query)
                rows.append(encode_query(content))
                labels.append(target)
                opposite.append(start + ((arm_index + 1) % len(variants)) *
                                queries_per_arm + query_index)
    n, length = len(rows), max(len(row[0]) for row in rows)
    data = {
        "before": torch.zeros(n, length, 52, dtype=torch.long),
        "after": torch.zeros(n, length, 52, dtype=torch.long),
        "goals": torch.zeros(n, length, 2, dtype=torch.long),
        "actions": torch.zeros(n, length, dtype=torch.long),
        "rewards": torch.zeros(n, length),
        "dones": torch.zeros(n, length),
        "episode_index": torch.zeros(n, length, dtype=torch.long),
        "step_index": torch.zeros(n, length, dtype=torch.long),
        "mask": torch.zeros(n, length, dtype=torch.bool),
        "query_state": torch.zeros(n, 52, dtype=torch.long),
        "query_goal": torch.zeros(n, 2, dtype=torch.long),
        "target": torch.tensor(labels, dtype=torch.long),
        "opposite": torch.tensor(opposite, dtype=torch.long),
    }
    for i, row in enumerate(rows):
        before, after, goals, actions, rewards, dones, state, goal, \
            episodes, step_positions = row
        m = len(before)
        for key, values in (("before", before), ("after", after),
                            ("goals", goals), ("actions", actions),
                            ("rewards", rewards), ("dones", dones),
                            ("episode_index", episodes),
                            ("step_index", step_positions)):
            data[key][i, :m] = torch.tensor(values)
        data["mask"][i, :m] = True
        data["query_state"][i] = torch.tensor(state)
        data["query_goal"][i] = torch.tensor(goal)
    if data["target"].min() < 0 or data["target"].max() > 5:
        raise ValueError("Action outside public XLand action space")
    if data["before"].max() > 63 or data["after"].max() > 63:
        raise ValueError("Tile value outside pilot embedding table")
    data["group_size"] = len(variants) * queries_per_arm
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in data.items()}


class RawHyperLoRA(nn.Module):
    def __init__(self, width: int = 64, rank: int = 8,
                 order_invariant_source: bool = False) -> None:
        super().__init__()
        self.rank = rank
        self.order_invariant_source = order_invariant_source
        # Shared, generic value encoder. No named object/action/rule slots.
        self.value = nn.Embedding(64, 16)
        self.state = nn.Sequential(nn.Linear(52 * 16, width), nn.GELU(),
                                   nn.Linear(width, width))
        self.goal = nn.Sequential(nn.Linear(2 * 16, width), nn.GELU(),
                                  nn.Linear(width, width))
        self.action = nn.Embedding(6, width)
        self.event = nn.Sequential(nn.Linear(5 * width + 2, width), nn.GELU(),
                                   nn.Linear(width, width))
        layer = nn.TransformerEncoderLayer(width, 4, 2 * width,
                                           dropout=0, batch_first=True,
                                           norm_first=True)
        self.trajectory = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        self.cls = nn.Parameter(torch.zeros(1, 1, width))
        self.position = nn.Embedding(17, width)
        self.hyper = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width),
                                   nn.Tanh(), nn.Linear(width, 6 * rank))
        nn.init.normal_(self.hyper[-1].weight, std=.02)
        nn.init.zeros_(self.hyper[-1].bias)
        # The policy trunk is fixed. A shared LoRA A and trajectory-generated B
        # modify only its final action projection.
        self.policy = nn.Sequential(nn.Linear(2 * width, width), nn.GELU())
        self.base_head = nn.Linear(width, 6)
        for parameter in self.policy.parameters():
            parameter.requires_grad_(False)
        for parameter in self.base_head.parameters():
            parameter.requires_grad_(False)
        self.lora_a = nn.Linear(width, rank, bias=False)
        nn.init.normal_(self.lora_a.weight, std=.02)

    def state_vector(self, values: torch.Tensor) -> torch.Tensor:
        return self.state(self.value(values).flatten(-2))

    def goal_vector(self, values: torch.Tensor) -> torch.Tensor:
        return self.goal(self.value(values).flatten(-2))

    def source_latent(self, data: dict, source: str = "correct") -> torch.Tensor:
        """Encode public source trajectories without seeing target labels."""
        before = self.state_vector(data["before"])
        after = self.state_vector(data["after"])
        goal = self.goal_vector(data["goals"])
        action = self.action(data["actions"])
        event = self.event(torch.cat((before, after, goal, action,
                     after - before, data["rewards"].unsqueeze(-1),
                     data["dones"].unsqueeze(-1)), dim=-1))
        batch, length, _ = event.shape
        if self.order_invariant_source:
            # Each episode is an ordered sequence. Independent episodes are a
            # set: no episode-index embedding or fixed probe slot is exposed.
            episode_latents, active_episodes = [], []
            for index in range(int(data["episode_index"].max()) + 1):
                present = data["mask"] & (data["episode_index"] == index)
                events = event + self.position(data["step_index"] + 1)
                sequence = torch.cat((self.cls.expand(batch, -1, -1) +
                                      self.position.weight[0], events), dim=1)
                padding = torch.cat((torch.zeros(batch, 1, dtype=torch.bool,
                    device=event.device), ~present), dim=1)
                episode_latents.append(self.trajectory(
                    sequence, src_key_padding_mask=padding)[:, 0])
                active_episodes.append(present.any(dim=1))
            active = torch.stack(active_episodes, dim=1).float()
            latent = (torch.stack(episode_latents, dim=1) *
                      active.unsqueeze(-1)).sum(dim=1) / active.sum(
                          dim=1, keepdim=True).clamp_min(1)
        else:
            sequence = torch.cat((self.cls.expand(batch, -1, -1), event), dim=1)
            sequence = sequence + self.position(torch.arange(length + 1,
                                                           device=event.device))
            padding = torch.cat((torch.zeros(batch, 1, dtype=torch.bool,
                                           device=event.device), ~data["mask"]), dim=1)
            latent = self.trajectory(sequence, src_key_padding_mask=padding)[:, 0]
        if source == "wrong":
            latent = latent[data["opposite"]]
        elif source != "correct":
            raise ValueError(source)
        return latent

    def compile_adapter(self, data: dict, source: str = "correct") -> torch.Tensor:
        """Compile source episodes once; output is a mountable LoRA B factor."""
        latent = self.source_latent(data, source)
        return self.hyper(latent).reshape(len(latent), 6, self.rank)

    def policy_logits(self, query_state: torch.Tensor,
                      query_goal: torch.Tensor,
                      adapter_b: torch.Tensor | None) -> torch.Tensor:
        """The actor sees only a target query and the generated parameters."""
        query = torch.cat((self.state_vector(query_state),
                           self.goal_vector(query_goal)), dim=-1)
        hidden = self.policy(query)
        logits = self.base_head(hidden)
        if adapter_b is not None:
            logits = logits + 2.0 * torch.einsum(
                "bar,br->ba", adapter_b, self.lora_a(hidden)) / self.rank
        return logits

    def forward(self, data: dict, source: str = "correct") -> torch.Tensor:
        adapter_b = None if source == "none" else self.compile_adapter(data, source)
        return self.policy_logits(data["query_state"], data["query_goal"],
                                  adapter_b)


def score(model: RawHyperLoRA, data: dict) -> dict:
    model.eval()
    with torch.no_grad():
        logits = {kind: model(data, kind) for kind in ("correct", "wrong", "none")}
        predictions = {key: value.argmax(-1) for key, value in logits.items()}
        targets = data["target"]
        grouped = predictions["correct"].reshape(-1, data["group_size"])
        result = {"n": len(targets),
                  **{key: int((prediction == targets).sum())
                     for key, prediction in predictions.items()},
                  "swap_changes": int((predictions["correct"] !=
                                       predictions["wrong"]).sum())}
        if data["group_size"] == 4:
            result.update(within_history_distinct=int(
                (grouped[:, 0] != grouped[:, 1]).sum() +
                (grouped[:, 2] != grouped[:, 3]).sum()),
                within_history_count=2 * len(grouped))
        else:
            result.update(all_actions_distinct=int(sum(
                len(set(row.tolist())) == data["group_size"] for row in grouped)),
                item_count=len(grouped))
        return result


def run(args: argparse.Namespace) -> dict:
    if args.output.exists() or args.checkpoint.exists():
        raise FileExistsError("Fresh output paths required")
    raw = args.annotations.read_bytes()
    annotations = json.loads(raw)
    torch.manual_seed(args.seed)
    split = {name: make_split(annotations["split"][name], args.device)
             for name in ("train", "dev", "test")}
    model = RawHyperLoRA(args.width, args.rank,
                         args.order_invariant_source).to(args.device)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                  lr=args.lr, weight_decay=args.weight_decay)
    best_dev, best_state, best_step = -1, None, 0
    history = []
    for step in range(1, args.steps + 1):
        model.train()
        train = split["train"]
        logits = model(train)
        loss = F.cross_entropy(logits, train["target"])
        if args.pair_weight:
            # The same target public query has the opposite action under the
            # matched source. This supervises source-dependent action changes
            # without naming the hidden rule or engineering a state delta.
            competing = train["target"][train["opposite"]]
            advantage = logits.gather(1, train["target"][:, None]).squeeze(1) - \
                logits.gather(1, competing[:, None]).squeeze(1)
            loss = loss + args.pair_weight * F.softplus(
                args.pair_margin - advantage).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % args.eval_every == 0 or step == args.steps:
            dev_score = score(model, split["dev"])["correct"]
            history.append({"step": step, "loss": float(loss.detach()),
                            "train": score(model, train)["correct"],
                            "dev": dev_score})
            if dev_score > best_dev:
                best_dev, best_step = dev_score, step
                best_state = {key: value.detach().cpu().clone()
                              for key, value in model.state_dict().items()}
    assert best_state is not None
    model.load_state_dict(best_state)
    result = {"protocol": "Raw public XLand state/action/reward trajectory -> transformer hypernetwork -> generated LoRA B on frozen small policy. No rule or tile-delta feature. Future-action CE, train-only fitting, dev-selected checkpoint, rule-content-disjoint test; mechanisms may recur across splits.",
              "annotations_sha256": hashlib.sha256(raw).hexdigest(),
              "seed": args.seed, "steps": args.steps, "best_step": best_step,
              "width": args.width, "rank": args.rank, "lr": args.lr,
              "pair_weight": args.pair_weight, "pair_margin": args.pair_margin,
              "order_invariant_source": args.order_invariant_source,
              "history": history,
              **{name: score(model, split[name])
                 for name in ("train", "dev", "test")}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    torch.save({"state_dict": best_state, "annotations_sha256":
                result["annotations_sha256"], "seed": args.seed}, args.checkpoint)
    print(json.dumps({"seed": args.seed, "best_step": best_step,
                      **{name: result[name] for name in ("train", "dev", "test")}}),
          flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, default=Path(
        "data/annotations/xland_crossed_reviewed_v1_20261005.json"))
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--lr", type=float, default=.001)
    parser.add_argument("--weight-decay", type=float, default=0)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--pair-weight", type=float, default=0)
    parser.add_argument("--pair-margin", type=float, default=1)
    parser.add_argument("--order-invariant-source", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    if (args.steps < 1 or args.eval_every < 1 or args.lr <= 0 or
            args.pair_weight < 0):
        parser.error("Invalid training budget")
    run(args)


if __name__ == "__main__":
    main()
