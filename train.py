"""Main training script for DINOv2/DINOv3 + LoRA chilli leaf disease classification.

Supports CPU-only training (mixed precision is auto-disabled on CPU).
"""

from __future__ import annotations

import argparse
import json
import os
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from tqdm import tqdm
import yaml

# Project imports
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.dataset import create_dataloaders
from models.dinov3_lora import (
    DinoV3LoRAClassifier,
    build_model,
    load_checkpoint,
    save_checkpoint,
    export_class_mapping,
)
from models.model_utils import count_parameters, format_param_count
from utils.helpers import (
    ensure_dir,
    format_seconds,
    get_device,
    get_logger,
    gpu_memory_summary,
    resolve_path_str,
    set_seed,
)
from utils.metrics import (
    compute_classification_metrics,
    compute_confusion_matrix,
    plot_confusion_matrix,
)
from utils.losses import create_loss, get_class_weights
from utils.early_stopping import EarlyStopping


# AMP is CUDA-only; gracefully fall back to plain fp32 on CPU.
try:
    from torch.cuda.amp import autocast, GradScaler  # type: ignore
    _CUDA_AVAILABLE_FOR_AMP = True
except Exception:  # pragma: no cover
    autocast = None  # type: ignore
    GradScaler = None  # type: ignore
    _CUDA_AVAILABLE_FOR_AMP = False


logger = get_logger(__name__, log_file=PROJECT_ROOT / "logs" / "training.log")


# ---------------------------------------------------------------------------
# Optimizer / scheduler builders
# ---------------------------------------------------------------------------
def build_optimizer(model: nn.Module, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    """Build AdamW with no weight decay on bias / norm / LoRA-A parameters."""
    decay_params, no_decay_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if (
            param.ndim <= 1
            or name.endswith(".bias")
            or "norm" in name.lower()
            or "lora_A" in name
        ):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(param_groups, lr=lr)


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: Dict[str, Any],
    steps_per_epoch: int,
):
    """Build a learning-rate scheduler.

    Currently supports:
      - ``"cosine"`` (with optional warm-up).
    """
    scheduler_cfg = cfg.get("scheduler", {"type": "cosine"})
    training_cfg = cfg["training"]
    total_epochs = int(training_cfg["epochs"])
    warmup_epochs = int(training_cfg.get("warmup_epochs", 0))

    if scheduler_cfg.get("type", "cosine") == "cosine":
        eta_min = float(scheduler_cfg.get("eta_min", 1e-6))
        t_max = max(1, int(scheduler_cfg.get("t_max", total_epochs)) * steps_per_epoch)
        warmup_iters = max(0, warmup_epochs * steps_per_epoch)
        warmup_lr_start = float(training_cfg["learning_rate"]) * 0.01

        def lr_lambda(step: int) -> float:
            if warmup_iters > 0 and step < warmup_iters:
                # Linear warm-up from 1 % to 100 % of base LR.
                frac = (step + 1) / max(1, warmup_iters)
                return warmup_lr_start + (1.0 - warmup_lr_start) * frac
            progress = (step - warmup_iters) / max(1, t_max - warmup_iters)
            progress = min(max(progress, 0.0), 1.0)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            # Scale between eta_min/lr and 1.0 (eta_min is treated relative to base lr).
            return cosine * (1.0 - eta_min) + eta_min

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    # Fallback: no scheduling
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _step: 1.0)


# ---------------------------------------------------------------------------
# Train / validation loops
# ---------------------------------------------------------------------------
def _autocast_ctx(enabled: bool, device_type: str):
    """Return an autocast context manager that works on both CPU and CUDA."""
    if not enabled or autocast is None:
        # No-op context manager on CPU.
        from contextlib import nullcontext
        return nullcontext()
    return autocast(device_type=device_type)


def train_one_epoch(
    model: DinoV3LoRAClassifier,
    loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[GradScaler],
    device: torch.device,
    scheduler,
    grad_clip: float,
    use_amp: bool,
    accum_steps: int,
    epoch: int,
    save_every_steps: int = 0,
    best_metric: float = -1.0,
    ckpt_dir=None,
    history=None,
    config_path: str = "",
) -> Tuple[float, float]:
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    optimizer.zero_grad(set_to_none=True)
    pbar = tqdm(loader, desc=f"Epoch {epoch + 1} [train]", ncols=110)

    device_type = "cuda" if device.type == "cuda" else "cpu"
    accum_steps = max(1, int(accum_steps))
    step_count = 0

    for step, (images, labels) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with _autocast_ctx(use_amp, device_type):
            outputs = model(images)
            logits = outputs["logits"] if isinstance(outputs, dict) else outputs
            loss = criterion(logits, labels) / accum_steps

        if use_amp and scaler is not None and device.type == "cuda":
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (step + 1) % accum_steps == 0:
            if grad_clip and grad_clip > 0:
                if use_amp and scaler is not None and device.type == "cuda":
                    scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            if use_amp and scaler is not None and device.type == "cuda":
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()

        total_loss += loss.item() * accum_steps * labels.size(0)
        preds = logits.argmax(dim=-1)
        correct += int((preds == labels).sum().item())
        total += labels.size(0)

        if step_count % 10 == 0:
            current_lr = optimizer.param_groups[0]["lr"]
            pbar.set_postfix(loss=f"{loss.item() * accum_steps:.4f}",
                             acc=f"{correct / max(1, total):.4f}",
                             lr=f"{current_lr:.2e}")
            # ---- Mid-epoch checkpoint (every save_every_steps) ----
            _ses = save_every_steps
            if _ses > 0 and step_count > 0 and (step_count % _ses == 0):
                try:
                    save_checkpoint(
                        model,
                        optimizer,
                        scheduler,
                        epoch,
                        {"best_metric": float(best_metric), "step": int(step_count)},
                        str(ckpt_dir / "running_model.pt"),
                        extra={"history": history,
                               "config_path": config_path,
                               "step_count": int(step_count)},
                    )
                    logger.info("  ~ mid-epoch checkpoint at step %d (running_model.pt)", step_count)
                except Exception as _e:
                    logger.warning("Mid-epoch checkpoint failed: %s", _e)
        step_count += 1

    avg_loss = total_loss / max(1, total)
    accuracy = correct / max(1, total)
    return avg_loss, accuracy


@torch.no_grad()
def validate_one_epoch(
    model: DinoV3LoRAClassifier,
    loader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool,
    epoch: int,
) -> Tuple[float, float, list, list]:
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    y_true: list = []
    y_pred: list = []
    pbar = tqdm(loader, desc=f"Epoch {epoch + 1} [val]  ", ncols=110)
    device_type = "cuda" if device.type == "cuda" else "cpu"

    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with _autocast_ctx(use_amp, device_type):
            outputs = model(images)
            logits = outputs["logits"] if isinstance(outputs, dict) else outputs
            loss = criterion(logits, labels)
        total_loss += float(loss.item()) * labels.size(0)
        preds = logits.argmax(dim=-1)
        correct += int((preds == labels).sum().item())
        total += labels.size(0)
        y_true.extend(labels.cpu().tolist())
        y_pred.extend(preds.cpu().tolist())
        pbar.set_postfix(loss=f"{loss.item():.4f}",
                         acc=f"{correct / max(1, total):.4f}")

    return total_loss / max(1, total), correct / max(1, total), y_true, y_pred


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train DINOv2/DINOv3 + LoRA chilli-leaf classifier.")
    p.add_argument("--config", type=str, default=str(PROJECT_ROOT / "configs" / "config.yaml"))
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--output-dir", type=str, default=None,
                   help="Override outputs root for this run.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as fh:
        config: Dict[str, Any] = yaml.safe_load(fh)

    # Apply CLI overrides
    if args.epochs is not None:
        config["training"]["epochs"] = int(args.epochs)
    if args.batch_size is not None:
        config["training"]["batch_size"] = int(args.batch_size)
    if args.resume is not None:
        config["training"]["resume_from_checkpoint"] = args.resume

    set_seed(config["seed"],
             deterministic=bool(config.get("deterministic", True)),
             benchmark=bool(config.get("benchmark", False)))

    device = get_device(prefer_cuda=True)
    use_amp = bool(config["training"].get("mixed_precision", False)) and device.type == "cuda"
    if config["training"].get("mixed_precision", False) and not use_amp:
        logger.warning("Mixed precision requested but CUDA is unavailable; running in fp32.")

    # ------------- Data -------------
    dataset_dir = resolve_path_str(PROJECT_ROOT, config["paths"]["dataset_dir"])
    image_size = int(config["dataset"]["image_size"])
    batch_size = int(config["training"]["batch_size"])

    train_loader, val_loader, test_loader, class_names = create_dataloaders(
        root=dataset_dir,
        image_size=image_size,
        batch_size=batch_size,
        num_workers=int(config["training"]["num_workers"]),
        pin_memory=bool(config["training"]["pin_memory"]) and device.type == "cuda",
        train_ratio=float(config["dataset"]["train_ratio"]),
        val_ratio=float(config["dataset"]["val_ratio"]),
        test_ratio=float(config["dataset"]["test_ratio"]),
        seed=int(config["seed"]),
        class_names=config.get("classes"),
    )
    num_classes = len(class_names)
    logger.info("Found %d classes: %s", num_classes, class_names)

    # ------------- Model -------------
    model_cfg = config["model"]
    model = build_model(
        backbone_name=model_cfg["backbone_name"],
        num_classes=num_classes,
        cache_dir=model_cfg.get("cache_dir"),
        lora_config=config.get("lora"),
        freeze_backbone=bool(model_cfg.get("freeze_backbone", True)),
    )
    model.to(device)
    logger.info(format_param_count(model))

    # Optional resume
    start_epoch = 0
    best_metric = -math.inf
    if config["training"].get("resume_from_checkpoint"):
        payload = load_checkpoint(
            config["training"]["resume_from_checkpoint"], model, map_location=device,
        )
        start_epoch = int(payload.get("epoch", -1)) + 1
        best_metric = float(payload.get("best_metric", best_metric))
        logger.info("Resumed from epoch %d (best_metric=%.4f)", start_epoch - 1, best_metric)

    # ------------- Loss / optimiser / scheduler -------------
    weights = None
    if config["training"].get("weighted_loss", False):
        train_ds = train_loader.dataset
        if hasattr(train_ds, "get_class_counts"):
            counts = train_ds.get_class_counts()
            weights = get_class_weights(counts).tolist()
            logger.info("Using weighted loss: %s", weights)
    criterion = create_loss(class_weights=weights)

    optimizer = build_optimizer(
        model,
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )

    steps_per_epoch = max(1, math.ceil(len(train_loader) /
                                       max(1, int(config["training"]["gradient_accumulation_steps"]))))
    scheduler = build_scheduler(optimizer, config, steps_per_epoch=steps_per_epoch)

    scaler = None
    if use_amp and GradScaler is not None:
        scaler = GradScaler(enabled=True)

    
    # ------------- Mid-epoch checkpointing (every N steps) -------------
    save_every_steps = int(os.environ.get("SAVE_EVERY_STEPS", "200"))# ------------- Early stopping -------------
    es_cfg = config["training"].get("early_stopping", {}) or {}
    early_stopping = None
    if es_cfg.get("enabled", False):
        early_stopping = EarlyStopping(
            patience=int(es_cfg.get("patience", 3)),
            min_delta=float(es_cfg.get("min_delta", 1e-4)),
            mode="max",
        )
        early_stopping.best_value = best_metric if best_metric > -math.inf else None

    # ------------- Output dirs -------------
    run_root = Path(args.output_dir) if args.output_dir else PROJECT_ROOT
    ckpt_dir = ensure_dir(run_root / config["paths"]["checkpoints_dir"])
    metrics_dir = ensure_dir(run_root / config["paths"]["metrics_dir"])
    cm_dir = ensure_dir(run_root / config["paths"]["confusion_matrix_dir"])

    # ------------- Training loop -------------
    history: Dict[str, list] = {
        "epoch": [],
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "lr": [],
    }
    total_epochs = int(config["training"]["epochs"])
    grad_clip = float(config["training"].get("grad_clip_max_norm", 0.0))
    accum_steps = max(1, int(config["training"].get("gradient_accumulation_steps", 1)))
    save_every = bool(config["training"].get("save_every_epoch", False))

    logger.info("Starting training: epochs=%d, batch_size=%d, use_amp=%s, device=%s",
                total_epochs, batch_size, use_amp, device)
    logger.info("GPU/CPU memory before training: %s", gpu_memory_summary(device))

    train_start = time.time()
    for epoch in range(start_epoch, total_epochs):
        epoch_start = time.time()

        train_loss, train_acc = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            scheduler=scheduler,
            grad_clip=grad_clip,
            use_amp=use_amp,
            accum_steps=accum_steps,
            epoch=epoch,
            save_every_steps=save_every_steps,
            best_metric=best_metric,
            ckpt_dir=ckpt_dir,
            history=history,
            config_path=args.config,
        )
        val_loss, val_acc, _, _ = validate_one_epoch(
            model=model, loader=val_loader, criterion=criterion,
            device=device, use_amp=use_amp, epoch=epoch,
        )

        current_lr = optimizer.param_groups[0]["lr"]
        epoch_time = time.time() - epoch_start
        logger.info(
            "Epoch %d/%d | train_loss=%.4f train_acc=%.4f | val_loss=%.4f val_acc=%.4f | "
            "lr=%.2e | time=%s",
            epoch + 1, total_epochs, train_loss, train_acc, val_loss, val_acc,
            current_lr, format_seconds(epoch_time),
        )

        history["epoch"].append(epoch + 1)
        history["train_loss"].append(float(train_loss))
        history["train_acc"].append(float(train_acc))
        history["val_loss"].append(float(val_loss))
        history["val_acc"].append(float(val_acc))
        history["lr"].append(float(current_lr))

        # Track best metric
        improved = val_acc > best_metric
        if improved:
            best_metric = val_acc

        # Save last & (optionally) per-epoch checkpoints
        save_checkpoint(
            model,
            optimizer,
            scheduler,
            epoch,
            {"val_acc": float(val_acc), "val_loss": float(val_loss), "best_metric": float(best_metric)},
            str(ckpt_dir / "last_model.pt"),
            extra={"history": history, "config_path": args.config,
                   "class_to_idx": {c: i for i, c in enumerate(class_names)}},
        )
        if save_every:
            save_checkpoint(
                model, optimizer, scheduler, epoch,
                {"val_acc": float(val_acc), "val_loss": float(val_loss), "best_metric": float(best_metric)},
                str(ckpt_dir / f"epoch_{epoch + 1:02d}.pt"),
                extra={"history": history, "config_path": args.config,
                       "class_to_idx": {c: i for i, c in enumerate(class_names)}},
            )

        # Best checkpoint
        if improved:
            save_checkpoint(
                model, optimizer, scheduler, epoch,
                {"val_acc": float(val_acc), "val_loss": float(val_loss), "best_metric": float(best_metric)},
                str(ckpt_dir / "best_model.pt"),
                extra={"history": history, "config_path": args.config,
                       "class_to_idx": {c: i for i, c in enumerate(class_names)}},
            )
            logger.info("  ↳ New best (val_acc=%.4f) — saved best_model.pt", best_metric)

        # Save history JSON after every epoch
        with open(metrics_dir / "training_history.json", "w", encoding="utf-8") as fh:
            json.dump(history, fh, indent=2)

        # Early stopping
        if early_stopping is not None:
            early_stopping.step(val_acc)
            if early_stopping.should_stop:
                logger.info("Early stopping triggered at epoch %d", epoch + 1)
                break

    total_time = time.time() - train_start
    logger.info("Training finished in %s. Best val_acc=%.4f",
                format_seconds(total_time), best_metric)

    # ------------- Final confusion matrix on val split -------------
    cm_loader = val_loader
    _, _, y_true, y_pred = validate_one_epoch(
        model=model, loader=cm_loader, criterion=criterion,
        device=device, use_amp=use_amp, epoch=total_epochs - 1,
    )
    cm = compute_confusion_matrix(y_true, y_pred, class_names=class_names)
    plot_confusion_matrix(cm, class_names, cm_dir / "final_val_cm.png",
                          title="Final Validation Confusion Matrix")
    metrics = compute_classification_metrics(y_true, y_pred, class_names=class_names)
    with open(metrics_dir / "final_val_metrics.json", "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)

    # Export class mapping for inference.py
    export_class_mapping(class_names, ckpt_dir)

    logger.info("Artefacts written to %s", ckpt_dir)


if __name__ == "__main__":
    main()

