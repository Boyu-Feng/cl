"""Deploy the frozen v17 gate for an eligible one-to-two LoRA write.

The gate was trained only on old one-source memory. It refuses later
prefixes until separately reviewed full-prefix labels support them.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from ttcl.trajectory_hyperlora.alfworld_online_from_empty import file_hash
from ttcl.trajectory_hyperlora.centered_feedback_residual_v3 import (
    normalized_source, normalized_target,
)
from ttcl.trajectory_hyperlora.train_alf_incremental_kernel_gate_v17 import (
    kernel,
)


class FrozenOneToTwoGate:
    def __init__(self, model_path: Path, *, pool_review: Path,
                 features_path: Path):
        state = torch.load(model_path, map_location='cpu', weights_only=True)
        if (state['pool_review_sha256'] != file_hash(pool_review) or
                state['features_sha256'] != file_hash(features_path) or
                state['source_embeddings'].shape != (6, 10240) or
                state['task_embeddings'].shape != (78, 2560) or
                state['source_center'].shape != (10240,) or
                state['task_center'].shape != (2560,) or
                state['train_task_ids'].shape != (360,) or
                state['train_source_ids'].shape != (360,) or
                state['train_labels'].shape != (360,)):
            raise ValueError('Changed frozen v17 gate lineage or dimensions')
        self.selected = tuple(state['selected'])
        if self.selected[0] not in ('always_update', 'always_freeze',
                                    'product', 'task', 'source', 'sum'):
            raise ValueError('Unknown frozen update decision')
        self.source = state['source_embeddings'].double().numpy()
        self.task = state['task_embeddings'].double().numpy()
        self.source_center = state['source_center'].float()
        self.task_center = state['task_center'].float()
        self.task_ids = state['train_task_ids'].long().numpy()
        self.source_ids = state['train_source_ids'].long().numpy()
        self.labels = state['train_labels'].double().numpy()
        if (np.any(self.task_ids < 0) or np.any(self.task_ids >= 78) or
                np.any(self.source_ids < 0) or
                np.any(self.source_ids >= 6)):
            raise ValueError('Changed gate training pair indexing')
        self.center = float(self.labels.mean())
        self.weights = None
        if self.selected[0] not in ('always_update', 'always_freeze'):
            matrix = kernel(self.source, self.task,
                            self.task_ids, self.source_ids, self.selected)
            ridge = self.selected[3]
            self.weights = np.linalg.solve(matrix +
                ridge*np.eye(len(self.labels)), self.labels-self.center)

    def predict(self, old_global: torch.Tensor,
                old_delta: torch.Tensor,
                new_global: torch.Tensor,
                new_delta: torch.Tensor,
                target_vector: torch.Tensor, *, old_count: int) -> float:
        if old_count != 1:
            raise ValueError('Frozen v17 gate is valid only for a 1-to-2 write')
        if self.selected[0] == 'always_update':
            return 1.
        if self.selected[0] == 'always_freeze':
            return -1.
        old = normalized_source(old_global.cpu(), old_delta.cpu())
        new = normalized_source((old_global+new_global).cpu()/2,
                                (old_delta+new_delta).cpu()/2)
        source_raw = torch.cat((F.normalize(old, dim=0),
                                F.normalize(new-old, dim=0)))
        source = F.normalize(source_raw-self.source_center, dim=0)
        task_raw = normalized_target(target_vector.cpu())
        task = F.normalize(task_raw-self.task_center, dim=0)
        source = source.double().numpy()
        task = task.double().numpy()
        mode, task_temperature, source_temperature, _ = self.selected
        task_similarity = np.clip(self.task @ task, -1., 1.)
        source_similarity = np.clip(self.source @ source, -1., 1.)
        task_kernel = np.exp(-(1-task_similarity)/task_temperature)[
            self.task_ids]
        source_kernel = np.exp(-(1-source_similarity)/source_temperature)[
            self.source_ids]
        if mode == 'product':
            query = task_kernel*source_kernel
        elif mode == 'task':
            query = task_kernel
        elif mode == 'source':
            query = source_kernel
        elif mode == 'sum':
            query = (task_kernel+source_kernel)/2
        else:
            raise ValueError('Unknown frozen update kernel')
        return float(self.center+query @ self.weights)

    def should_write(self, *args, **kwargs) -> bool:
        return self.predict(*args, **kwargs) > 0.
