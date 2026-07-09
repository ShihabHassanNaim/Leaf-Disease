"""DINOv3 ViT + LoRA classification model."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoConfig, AutoImageProcessor, AutoModel

from utils.helpers import PathLike, ensure_dir, get_logger, save_json

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------
@dataclass
class DinoV3LoRAConfig:
    """Container for model-related configuration.

    Attributes:
        backbone_name: HuggingFace model id (e.g. ``facebook/dinov3-vits16-pretrain-lvd1689m``).
        num_classes: Output dimension of the classifier head.
        cache_dir: Optional HF cache directory.
        lora_enabled: If ``True`` apply LoRA via PEFT.
        lora_r: LoRA rank.
        lora_alpha: LoRA scaling factor.
        lora_dropout: LoRA dropout.
        lora_bias: ``"none"`` | ``"all"`` | ``"lora_only"``.
        lora_target_modules: Module-name patterns to wrap with LoRA.
        freeze_backbone: Freeze all non-LoRA backbone parameters.
    """

    backbone_name: str
    num_classes: int = 6
    cache_dir: Optional[str] = None
    lora_enabled: bool = True
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_bias: str = "none"
    lora_target_modules: Tuple[str, ...] = ("qkv", "proj")
    freeze_backbone: bool = True


# ---------------------------------------------------------------------------
# Head definition
# ---------------------------------------------------------------------------
class ClassifierHead(nn.Module):
    """Single linear layer mapping CLS hidden state → class logits."""

    def __init__(self, hidden_size: int, num_classes: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, cls_feature: torch.Tensor) -> torch.Tensor:
        return self.fc(self.dropout(cls_feature))


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------
class DinoV3LoRAClassifier(nn.Module):
    """DINOv3 backbone + optional LoRA + linear classification head.

    Forward signature::

        forward(pixel_values: Tensor) -> Dict[str, Tensor]

    Returns a dictionary with ``logits`` and the optional ``hidden_state``
    used for Grad-CAM hooks.
    """

    def __init__(self, config: DinoV3LoRAConfig) -> None:
        super().__init__()
        self.config = config
        logger.info("Loading DINOv3 backbone '%s' from HuggingFace…",
                    config.backbone_name)

        kwargs: Dict[str, Any] = {}
        if config.cache_dir:
            ensure_dir(config.cache_dir)
            os.environ.setdefault("HF_HOME", str(Path(config.cache_dir).resolve()))
            kwargs["cache_dir"] = str(Path(config.cache_dir).resolve())

        backbone_config = AutoConfig.from_pretrained(config.backbone_name, **kwargs)
        self.backbone = AutoModel.from_pretrained(
            config.backbone_name, config=backbone_config, **kwargs,
        )

        # Detect hidden size and image size from the config
        hidden_size = getattr(backbone_config, "hidden_size", None)
        if hidden_size is None:
            raise AttributeError("Backbone config has no `hidden_size`.")
        self.hidden_size = int(hidden_size)

        self.image_size = int(getattr(backbone_config, "image_size", 224))
        self.patch_size = int(getattr(backbone_config, "patch_size", 16))

        # Apply LoRA if requested
        if config.lora_enabled:
            lora_cfg = LoraConfig(
                r=config.lora_r,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                bias=config.lora_bias,
                target_modules=list(config.lora_target_modules),
                # Only LoRA layers will be trainable; the rest stay frozen.
                modules_to_save=None,
            )
            self.backbone = get_peft_model(self.backbone, lora_cfg)
            logger.info("Applied LoRA (r=%d, alpha=%d, dropout=%.2f, targets=%s) to backbone.",
                        config.lora_r, config.lora_alpha, config.lora_dropout,
                        list(config.lora_target_modules))
        else:
            logger.info("LoRA disabled — full backbone fine-tuning.")

        # Classification head (always trainable)
        self.head = ClassifierHead(self.hidden_size, config.num_classes)

        # Freeze backbone (LoRA adapters stay trainable thanks to PEFT)
        if config.freeze_backbone:
            for name, p in self.backbone.named_parameters():
                if "lora_" in name.lower():
                    p.requires_grad = True
                else:
                    p.requires_grad = False

        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in self.parameters())
        logger.info(
            "Model ready — hidden_size=%d, image_size=%d, patch_size=%d.  "
            "Trainable params: %s / %s (%.2f%%)",
            self.hidden_size, self.image_size, self.patch_size,
            f"{n_trainable:,}", f"{n_total:,}",
            100.0 * n_trainable / max(n_total, 1),
        )

    # -----------------------------------------------------------------
    def forward(self, pixel_values: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Run the backbone (forward + optional LoRA) and the classification head.

        Uses the **CLS token** (first position) of the last hidden state as
        the global image descriptor.  For DINOv3 the CLS token is index 0.
        """
        outputs = self.backbone(pixel_values=pixel_values)
        # ``last_hidden_state`` shape: (B, 1 + N_patches, hidden_size)
        last_hidden = outputs.last_hidden_state
        cls_feature = last_hidden[:, 0]                # (B, hidden_size)
        logits = self.head(cls_feature)
        return {
            "logits": logits,
            "hidden_state": last_hidden,
            "cls_feature": cls_feature,
        }

    # -----------------------------------------------------------------
    def merge_and_unload_lora(self) -> nn.Module:
        """Merge LoRA weights into the backbone and return the underlying base model."""
        if not isinstance(self.backbone, PeftModel):
            logger.warning("Backbone is not a PeftModel — nothing to merge.")
            return self.backbone
        merged = self.backbone.merge_and_unload()
        logger.info("LoRA adapters merged and unloaded into base backbone.")
        return merged

    # -----------------------------------------------------------------
    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device


# ---------------------------------------------------------------------------
# Builder helper
# ---------------------------------------------------------------------------
def build_model(
    backbone_name: str,
    num_classes: int,
    cache_dir: Optional[str] = None,
    lora_config: Optional[Dict[str, Any]] = None,
    freeze_backbone: bool = True,
) -> DinoV3LoRAClassifier:
    """Factory that creates a :class:`DinoV3LoRAClassifier` from primitives."""
    cfg_kwargs: Dict[str, Any] = dict(
        backbone_name=backbone_name,
        num_classes=num_classes,
        cache_dir=cache_dir,
        freeze_backbone=freeze_backbone,
    )
    if lora_config is not None:
        cfg_kwargs.update(
            lora_enabled=lora_config.get("enabled", True),
            lora_r=lora_config.get("r", 8),
            lora_alpha=lora_config.get("alpha", 16),
            lora_dropout=lora_config.get("dropout", 0.05),
            lora_bias=lora_config.get("bias", "none"),
            lora_target_modules=tuple(lora_config.get(
                "target_modules", ("qkv", "proj")
            )),
        )
    cfg = DinoV3LoRAConfig(**cfg_kwargs)
    return DinoV3LoRAClassifier(cfg)


# ---------------------------------------------------------------------------
# Checkpoint I/O
# ---------------------------------------------------------------------------
def save_checkpoint(
    model: DinoV3LoRAClassifier,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    epoch: int,
    metrics: Dict[str, float],
    path: PathLike,
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    """Persist training state to disk."""
    out = Path(path)
    ensure_dir(out.parent)
    payload: Dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
        "metrics": metrics,
        "config": {
            "backbone_name": model.config.backbone_name,
            "num_classes": model.config.num_classes,
            "lora_enabled": model.config.lora_enabled,
            "lora_r": model.config.lora_r,
            "lora_alpha": model.config.lora_alpha,
            "lora_dropout": model.config.lora_dropout,
            "lora_bias": model.config.lora_bias,
            "lora_target_modules": list(model.config.lora_target_modules),
            "freeze_backbone": model.config.freeze_backbone,
            "hidden_size": model.hidden_size,
            "image_size": model.image_size,
            "patch_size": model.patch_size,
        },
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler_state_dict"] = scheduler.state_dict()
    if extra:
        payload["extra"] = extra
    torch.save(payload, out)
    logger.info("Saved checkpoint → %s (epoch=%d)", out, epoch)
    return out


def load_checkpoint(
    path: PathLike,
    model: DinoV3LoRAClassifier,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    strict: bool = True,
    map_location: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Load a checkpoint into *model* (and optionally optimizer/scheduler)."""
    payload = torch.load(str(path), map_location=map_location or model.device)
    model.load_state_dict(payload["model_state_dict"], strict=strict)
    if optimizer is not None and "optimizer_state_dict" in payload:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in payload:
        scheduler.load_state_dict(payload["scheduler_state_dict"])
    logger.info("Loaded checkpoint %s (epoch=%s, metrics=%s)",
                path, payload.get("epoch"), payload.get("metrics"))
    return payload


def export_class_mapping(class_names: List[str], out_dir: PathLike) -> Path:
    """Persist class-name → index mapping for downstream inference."""
    out = ensure_dir(out_dir) / "classes.json"
    save_json({"class_to_idx": {c: i for i, c in enumerate(class_names)}}, out)
    return out