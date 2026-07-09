"""Loss-function factory."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn


def get_class_weights(
    class_counts: Sequence[int],
    smoothing: float = 1.0,
) -> torch.Tensor:
    """Compute inverse-frequency class weights.

    Args:
        class_counts: Number of samples per class (in label-index order).
        smoothing: Laplace-style smoothing to avoid division-by-zero.

    Returns:
        A 1-D float tensor of length ``len(class_counts)``.
    """
    counts = np.asarray(class_counts, dtype=np.float64) + smoothing
    weights = counts.sum() / (len(counts) * counts)
    # Normalise to mean 1.0 for training stability.
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def create_loss(
    class_weights: Optional[Sequence[float]] = None,
    label_smoothing: float = 0.0,
) -> nn.Module:
    """Build a :class:`~torch.nn.CrossEntropyLoss` with optional weighting.

    Args:
        class_weights: Optional per-class weights. ``None`` produces the
            un-weighted loss.
        label_smoothing: Forwarded to ``CrossEntropyLoss`` (default 0).

    Returns:
        The configured loss module.
    """
    weight: Optional[torch.Tensor] = None
    if class_weights is not None:
        weight = torch.as_tensor(class_weights, dtype=torch.float32)
    return nn.CrossEntropyLoss(weight=weight, label_smoothing=label_smoothing)