"""Evaluate a trained DINOv3 + LoRA checkpoint on the test split."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import torch
import yaml
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.dataset import create_dataloaders, ChilliLeafDataset, build_eval_transforms
from models.dinov3_lora import DinoV3LoRAClassifier, build_model, load_checkpoint
from models.model_utils import count_parameters, format_param_count
from utils.helpers import (
    ensure_dir,
    format_seconds,
    get_device,
    get_logger,
    gpu_memory_summary,
    humanize_int,
    load_json,
    resolve_path_str,
    save_json,
    set_seed,
)
from utils.metrics import (
    classification_report_text,
    compute_classification_metrics,
    compute_confusion_matrix,
    save_metrics_bundle,
)
from utils.losses import create_loss

logger = get_logger(__name__, log_file=PROJECT_ROOT / "logs" / "evaluate.log")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate DINOv3 + LoRA on test set.")
    parser.add_argument("--config", type=str, default=str(PROJECT_ROOT / "configs" / "config.yaml"))
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to the .pt checkpoint to load.")
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"])
    parser.add_argument("--batch-size", type=int, default=None)
    return parser.parse_args()


@torch.no_grad()
def run_inference(model: DinoV3LoRAClassifier, loader, device: torch.device,
                  num_classes: int) -> tuple[list, list, list]:
    """Run inference over *loader*; return (y_true, y_pred, y_prob_all).

    y_prob_all is a list of numpy arrays — one vector per sample.
    """
    model.eval()
    y_true: List[int] = []
    y_pred: List[int] = []
    y_prob: List[List[float]] = []
    pbar = tqdm(loader, desc="Inference", ncols=100)
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        outputs = model(images)
        logits = outputs["logits"] if isinstance(outputs, dict) else outputs
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        preds = probs.argmax(axis=-1)
        y_true.extend(labels.tolist())
        y_pred.extend(int(p) for p in preds)
        y_prob.extend(p.tolist() for p in probs)
    return y_true, y_pred, y_prob


def main() -> None:
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as fh:
        config: Dict[str, Any] = yaml.safe_load(fh)

    set_seed(config["seed"], deterministic=config.get("deterministic", True),
             benchmark=config.get("benchmark", False))

    dataset_dir = resolve_path_str(PROJECT_ROOT, config["paths"]["dataset_dir"])
    image_size = int(config["dataset"]["image_size"])
    batch_size = int(args.batch_size or config["evaluation"]["batch_size"])

    _, val_loader, test_loader, class_names = create_dataloaders(
        root=dataset_dir,
        image_size=image_size,
        batch_size=batch_size,
        num_workers=int(config["evaluation"]["num_workers"]),
        pin_memory=True,
        train_ratio=float(config["dataset"]["train_ratio"]),
        val_ratio=float(config["dataset"]["val_ratio"]),
        test_ratio=float(config["dataset"]["test_ratio"]),
        seed=int(config["seed"]),
        class_names=config.get("classes"),
    )
    loader = test_loader if args.split == "test" else val_loader

    # Build model then load weights (so config matches checkpoint exactly)
    model_cfg = config["model"]
    model = build_model(
        backbone_name=model_cfg["backbone_name"],
        num_classes=int(config["num_classes"]),
        cache_dir=model_cfg.get("cache_dir"),
        lora_config=config.get("lora"),
        freeze_backbone=bool(model_cfg.get("freeze_backbone", True)),
    )
    device = get_device(prefer_cuda=True)
    model.to(device)
    payload = load_checkpoint(args.checkpoint, model, map_location=device)
    logger.info("Loaded checkpoint %s (epoch=%s)",
                args.checkpoint, payload.get("epoch"))
    logger.info(format_param_count(model))

    t0 = time.time()
    y_true, y_pred, y_prob = run_inference(model, loader, device, num_classes=int(config["num_classes"]))
    elapsed = time.time() - t0
    logger.info("Inference completed in %s on %d samples",
                format_seconds(elapsed), len(y_true))

    metrics = compute_classification_metrics(y_true, y_pred, class_names=class_names)
    cm = compute_confusion_matrix(y_true, y_pred, class_names=class_names)
    report_text = classification_report_text(y_true, y_pred, class_names=class_names)

    metrics_dir = ensure_dir(resolve_path_str(PROJECT_ROOT, config["paths"]["metrics_dir"]))
    cm_dir = ensure_dir(resolve_path_str(PROJECT_ROOT, config["paths"]["confusion_matrix_dir"]))

    paths = save_metrics_bundle(
        metrics, cm, class_names, report_text, metrics_dir,
        prefix=f"{args.split}_evaluation",
    )
    # Also save confusion matrixes under the dedicated folder
    from utils.metrics import plot_confusion_matrix
    plot_confusion_matrix(cm, class_names, cm_dir / f"{args.split}_cm_raw.png",
                          title=f"Confusion Matrix — {args.split} (Counts)")
    cm_norm = cm.astype("float32") / np_max_axis(cm, axis=1)
    plot_confusion_matrix(cm_norm, class_names, cm_dir / f"{args.split}_cm_normalized.png",
                          title=f"Confusion Matrix — {args.split} (Normalized)", normalize=True)

    metrics.update({
        "split": args.split,
        "checkpoint": str(args.checkpoint),
        "num_samples": len(y_true),
        "elapsed_seconds": elapsed,
        "param_counts": count_parameters(model),
        "gpu_memory": gpu_memory_summary(device),
        "y_pred_distribution": y_pred_distribution(y_pred, class_names),
    })
    save_json(metrics, metrics_dir / f"{args.split}_metrics_full.json")

    logger.info("=== %s metrics ===", args.split.upper())
    logger.info("Accuracy : %.4f", metrics["accuracy"])
    logger.info("Macro F1 : %.4f", metrics["macro_f1"])
    logger.info("Weighted F1: %.4f", metrics["weighted_f1"])
    logger.info("Macro Precision: %.4f", metrics["macro_precision"])
    logger.info("Macro Recall   : %.4f", metrics["macro_recall"])
    logger.info("Artefacts:\n%s", "\n".join(f"  {k}: {v}" for k, v in paths.items()))


def np_max_axis(cm, axis):
    import numpy as np
    return np.maximum(cm.sum(axis=axis, keepdims=True), 1)


def y_pred_distribution(preds, class_names):
    import collections
    counts = collections.Counter(preds)
    return {class_names[i]: counts.get(i, 0) for i in range(len(class_names))}


if __name__ == "__main__":
    main()