"""Standalone Grad-CAM visualisation for DINOv3 + LoRA chilli leaf classifier.

For each test image, saves a side-by-side PNG with the original image, the
heatmap, and the overlay.  Files are split into ``correct/`` and ``incorrect/``
sub-folders under the configured Grad-CAM output directory.

Usage::

    python gradcam.py --checkpoint checkpoints/best_model.pt \\
                      --max-per-class 4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.dataset import (
    ChilliLeafDataset,
    build_eval_transforms,
    create_dataloaders,
)
from models.dinov3_lora import build_model, load_checkpoint
from utils.gradcam import (
    ViTGradCAM,
    load_image_rgb,
    overlay_heatmap,
    save_heatmap_visual,
)
from utils.helpers import (
    ensure_dir,
    get_device,
    get_logger,
    resolve_path_str,
    set_seed,
)

logger = get_logger(__name__, log_file=PROJECT_ROOT / "logs" / "gradcam.log")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate Grad-CAM heatmaps.")
    p.add_argument("--config", type=str, default=str(PROJECT_ROOT / "configs" / "config.yaml"))
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--max-per-class", type=int, default=4,
                   help="Maximum number of samples to visualise per class.")
    p.add_argument("--image-size", type=int, default=None)
    p.add_argument("--split", type=str, default="test", choices=["val", "test"])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)

    set_seed(config["seed"], deterministic=config.get("deterministic", True),
             benchmark=config.get("benchmark", False))

    dataset_dir = resolve_path_str(PROJECT_ROOT, config["paths"]["dataset_dir"])
    output_dir = ensure_dir(resolve_path_str(PROJECT_ROOT, config["paths"]["gradcam_dir"]))
    correct_dir = ensure_dir(output_dir / "correct")
    incorrect_dir = ensure_dir(output_dir / "incorrect")

    # DataLoaders + class names
    image_size = int(args.image_size or config["model"].get("image_size") or config["dataset"]["image_size"])
    _, _, test_loader, class_names = create_dataloaders(
        root=dataset_dir,
        image_size=image_size,
        batch_size=int(config["evaluation"]["batch_size"]),
        num_workers=int(config["evaluation"]["num_workers"]),
        pin_memory=True,
        train_ratio=float(config["dataset"]["train_ratio"]),
        val_ratio=float(config["dataset"]["val_ratio"]),
        test_ratio=float(config["dataset"]["test_ratio"]),
        seed=int(config["seed"]),
        class_names=config.get("classes"),
    )
    loader = test_loader if args.split == "test" else test_loader  # 'val' possible but defaults to test

    # Build model + load checkpoint
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
    logger.info("Loaded checkpoint %s (epoch=%s)", args.checkpoint, payload.get("epoch"))

    # Per-class selection state
    counter_per_class: Dict[int, Dict[str, int]] = {
        i: {"correct": 0, "incorrect": 0} for i in range(len(class_names))
    }

    cam = ViTGradCAM(model)

    def remaining_budget(cls_idx: int) -> Tuple[int, int]:
        s = counter_per_class[cls_idx]
        return args.max_per_class - s["correct"], args.max_per_class - s["incorrect"]

    total_saved = 0
    with torch.no_grad():
        for images, labels in iter(loader):
            for i in range(images.size(0)):
                cls_idx = int(labels[i])
                image_tensor = images[i:i+1].to(device)

                # Decide whether to save (correct or incorrect)
                with torch.no_grad():
                    logits = model(image_tensor)["logits"]
                    pred_idx = int(logits.argmax(dim=-1).item())
                is_correct = pred_idx == cls_idx
                budget_correct, budget_incorrect = remaining_budget(cls_idx)
                if is_correct and budget_correct <= 0:
                    continue
                if not is_correct and budget_incorrect <= 0:
                    continue

                # Compute Grad-CAM heatmap (manual gradient)
                heatmap = cam.generate(image_tensor, target_class=pred_idx)

                # Undo normalisation for visualisation
                inv_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
                inv_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
                img_unnorm = (image_tensor[0].cpu() * inv_std[0].cpu() + inv_mean[0].cpu())
                img_np = (img_unnorm.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype("uint8")

                tag = "correct" if is_correct else "incorrect"
                sub_dir = correct_dir if is_correct else incorrect_dir
                fname = f"{class_names[cls_idx]}_pred-{class_names[pred_idx]}_{counter_per_class[cls_idx][tag]:02d}.png"
                save_path = sub_dir / fname
                title = (
                    f"True: {class_names[cls_idx]}  |  Pred: {class_names[pred_idx]}  |  "
                    f"{'CORRECT' if is_correct else 'INCORRECT'}"
                )
                save_heatmap_visual(img_np, heatmap, save_path, title=title)
                counter_per_class[cls_idx][tag] += 1
                total_saved += 1

                if all(
                    counter_per_class[c]["correct"] >= args.max_per_class and
                    counter_per_class[c]["incorrect"] >= args.max_per_class
                    for c in counter_per_class
                ):
                    break
            else:
                continue
            break

    cam.remove_hooks()
    logger.info("Saved %d Grad-CAM visualisations to %s", total_saved, output_dir)
    for i, name in enumerate(class_names):
        s = counter_per_class[i]
        logger.info("  %s — correct=%d, incorrect=%d",
                    name, s["correct"], s["incorrect"])


if __name__ == "__main__":
    main()