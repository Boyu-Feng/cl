"""Execute the two unchanged SDPO loss functions from the pinned official source.

The distributed veRL launcher is replaced by a single-device LoRA runner. Only
the masked-sum utility is supplied locally; algorithm function ASTs are intact.
The full source and Apache-2.0 license live in current_work/SDPO_reference/.
"""
from __future__ import annotations
import ast
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional


def load_loss():
    import torch
    import torch.nn.functional as F
    path = Path(__file__).resolve().parents[2] / 'current_work/SDPO_reference/verl/trainer/ppo/core_algos.py'
    tree = ast.parse(path.read_text())
    wanted = {'agg_loss', 'compute_self_distillation_loss'}
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in wanted]
    if {n.name for n in nodes} != wanted:
        raise ValueError('Missing official SDPO loss functions')
    namespace = {'torch': torch, 'F': F, 'Any': Any, 'Optional': Optional,
                 'verl_F': SimpleNamespace(masked_sum=lambda x, mask: (x * mask).sum())}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['compute_self_distillation_loss']
