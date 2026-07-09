"""Helpers for inspecting & manipulating the DINOv3 + LoRA model."""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Parameter counts
# ---------------------------------------------------------------------------
def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Return total and trainable parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


def format_param_count(model: nn.Module) -> str:
    """Pretty-print parameter counts."""
    counts = count_parameters(model)
    total = counts["total"]
    trainable = counts["trainable"]
    pct = (trainable / max(total, 1)) * 100
    return (
        f"Total params: {total:,}  |  "
        f"Trainable: {trainable:,}  ({pct:.2f}%)"
    )


# ---------------------------------------------------------------------------
# FLOPs estimation (best-effort)
# ---------------------------------------------------------------------------
def estimate_macs(model: nn.Module, input_size: Tuple[int, int, int, int] = (1, 3, 224, 224)) -> Optional[float]:
    """Estimate MACs using the ``thop`` library if available.

    Returns ``None`` (and logs a warning) if ``thop`` is not installed or
    the model cannot be profiled.
    """
    try:
        from thop import profile
    except ImportError:
        return None
    model.eval()
    dummy = torch.zeros(*input_size)
    try:
        macs, _ = profile(model, inputs=(dummy,), verbose=False)
        return float(macs)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Freezing utilities
# ---------------------------------------------------------------------------
def freeze_backbone(model: nn.Module, backbone_attr: str = "backbone") -> int:
    """Freeze every parameter in the backbone. Returns the number frozen."""
    backbone = getattr(model, backbone_attr, None)
    if backbone is None:
        raise AttributeError(f"Model has no attribute '{backbone_attr}'.")
    n = 0
    for p in backbone.parameters():
        if p.requires_grad:
            p.requires_grad = False
            n += p.numel()
    return n


def unfreeze_backbone(model: nn.Module, backbone_attr: str = "backbone") -> int:
    """Unfreeze every parameter in the backbone. Returns the number unfrozen."""
    backbone = getattr(model, backbone_attr, None)
    if backbone is None:
        raise AttributeError(f"Model has no attribute '{backbone_attr}'.")
    n = 0
    for p in backbone.parameters():
        if not p.requires_grad:
            p.requires_grad = True
            n += p.numel()
    return n


def freeze_all_but_head_and_lora(model: nn.Module, head_attr: str = "head",
                                 backbone_attr: str = "backbone") -> int:
    """Freeze everything except the classification head and LoRA adapters."""
    n = freeze_backbone(model, backbone_attr=backbone_attr)
    # Ensure head remains trainable
    head = getattr(model, head_attr, None)
    if head is not None:
        for p in head.parameters():
            p.requires_grad = True
    return n


def list_trainable_named_parameters(model: nn.Module, top: Optional[int] = 30
                                    ) -> List[Tuple[str, int]]:
    """Return a list of (qualified_name, num_params) for trainable layers."""
    items: List[Tuple[str, int]] = []
    for name, p in model.named_parameters():
        if p.requires_grad:
            items.append((name, p.numel()))
    items.sort(key=lambda x: x[1], reverse=True)
    return items[:top] if top is not None else items