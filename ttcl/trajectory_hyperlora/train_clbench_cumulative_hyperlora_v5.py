"""Continue v4 cumulative-source training on the same reviewed train targets."""

from __future__ import annotations

import argparse
from pathlib import Path

from ttcl.trajectory_hyperlora.clbench_online_parameter_memory_v2 import ROOT
from ttcl.trajectory_hyperlora.train_clbench_cumulative_hyperlora_v4 import checked
from ttcl.trajectory_hyperlora.train_clbench_hyperlora_v1 import train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path,
        default=ROOT / "data/annotations/clbench_cumulative_hyperlora_v4_candidates_20261008.json")
    parser.add_argument("--review", type=Path,
        default=ROOT / "data/annotations/clbench_cumulative_hyperlora_v4_reviewed_20261008.json")
    parser.add_argument("--checkpoint", type=Path,
        default=ROOT / "results/trajectory_hyperlora/clbench_cumulative_hyperlora_v4_20261008.pt")
    parser.add_argument("--checkpoint-out", type=Path,
        default=ROOT / "results/trajectory_hyperlora/clbench_cumulative_hyperlora_v5_20261008.pt")
    parser.add_argument("--output", type=Path,
        default=ROOT / "results/trajectory_hyperlora/clbench_cumulative_hyperlora_v5_train_20261008.json")
    parser.add_argument("--model", type=Path,
        default=ROOT / "current_work/delta-Mem/model/Qwen3-4B-Instruct-2507")
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--gpu-fraction", type=float, default=.85)
    parser.add_argument("--source-tokens", type=int, default=8192)
    parser.add_argument("--context-limit", type=int, default=16384)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--log-every", type=int, default=20)
    args = parser.parse_args()
    train(args, include_thinking=True, reviewed_data=checked(args))


if __name__ == "__main__":
    main()
