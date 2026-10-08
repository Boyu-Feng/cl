"""Schema-conforming CLBench hyper-LoRA distillation from official trajectories.

Unlike v1, poker labels retain its required ``thinking`` JSON field. Source
memory still removes that field so executed moves dominate evidence encoding.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ttcl.trajectory_hyperlora.clbench_online_parameter_memory_v2 import ROOT
from ttcl.trajectory_hyperlora.train_clbench_hyperlora_v1 import prepare, train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "train"))
    parser.add_argument("--candidates", type=Path, default=ROOT / "data/annotations/clbench_hyperlora_v2_candidates_20261008.json")
    parser.add_argument("--review", type=Path, default=ROOT / "data/annotations/clbench_hyperlora_v2_reviewed_20261008.json")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "results/trajectory_hyperlora/alf_sibling_simple40_onpolicy600_20261006.pt")
    parser.add_argument("--checkpoint-out", type=Path, default=ROOT / "results/trajectory_hyperlora/clbench_hyperlora_v2_20261008.pt")
    parser.add_argument("--output", type=Path, default=ROOT / "results/trajectory_hyperlora/clbench_hyperlora_v2_train_20261008.json")
    parser.add_argument("--model", type=Path, default=ROOT / "current_work/delta-Mem/model/Qwen3-4B-Instruct-2507")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpu-fraction", type=float, default=.85)
    parser.add_argument("--source-tokens", type=int, default=2048)
    parser.add_argument("--context-limit", type=int, default=16384)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=10)
    args = parser.parse_args()
    (prepare if args.command == "prepare" else train)(args, include_thinking=True)


if __name__ == "__main__":
    main()
