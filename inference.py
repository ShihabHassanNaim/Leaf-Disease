"""Single-image inference for DINOv3 + LoRA chilli leaf disease classifier.

Usage::

    python inference.py --checkpoint checkpoints/best_model.pt \\
                        --image path/to/leaf.jpg \\
                        --output outputs/predictions/leaf.json

Prints the predicted class, confidence, and the full probability vector.
Saves a JSON file containing all of the above when ``--output`` is given.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import torch
import yaml
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.dataset import build_eval_transforms
from models.dinov3_lora import build_model, load_checkpoint
from utils.helpers import (
    get_device,
    get_logger,
    load_json,
    resolve_path_str,
    save_json,
)

logger = get_logger(__name__, log_file=PROJECT_ROOT / "logs" / "inference.log")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run inference on a single image.")
    p.add_argument("--config", type=str, default=str(PROJECT_ROOT / "configs" / "config.yaml"))
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--classes-json", type=str,
                   default=str(PROJECT_ROOT / "configs" / "classes.json"))
    p.add_argument("--image", type=str, required=True, help="Path to the input image.")
    p.add_argument("--output", type=str, default=None,
                   help="Optional path for the JSON output.")
    p.add_argument("--top-k", type=int, default=3)
    return p.parse_args()


def load_image(path: str, image_size: int = 224) -> torch.Tensor:
    img = Image.open(path).convert("RGB").resize(
        (image_size, image_size), Image.BILINEAR
    )
    return build_eval_transforms(image_size)(img).unsqueeze(0)


def main() -> None:
    args = parse_args()

    with open(args.config, "r", encoding="utf-8") as fh:
        config: Dict[str, Any] = yaml.safe_load(fh)

    # Load class names (preferred) — falls back to config if missing.
    try:
        classes_payload = load_json(args.classes_json)
        class_names = list(classes_payload["class_to_idx"].keys())
    except FileNotFoundError:
        class_names = list(config.get("classes", []))
        if not class_names:
            raise RuntimeError(
                "Class names could not be loaded — provide classes.json "
                "or populate `classes` in config.yaml."
            )

    # Build model
    model_cfg = config["model"]
    model = build_model(
        backbone_name=model_cfg["backbone_name"],
        num_classes=len(class_names),
        cache_dir=model_cfg.get("cache_dir"),
        lora_config=config.get("lora"),
        freeze_backbone=bool(model_cfg.get("freeze_backbone", True)),
    )
    device = get_device(prefer_cuda=True)
    model.to(device)

    payload = load_checkpoint(args.checkpoint, model, map_location=device)
    model.eval()

    # Predict
    image_size = int(model.image_size)
    tensor = load_image(args.image, image_size=image_size).to(device)
    with torch.no_grad():
        outputs = model(tensor)
        logits = outputs["logits"] if isinstance(outputs, dict) else outputs
        probs = torch.softmax(logits, dim=-1).cpu().numpy().squeeze()

    predicted_idx = int(probs.argmax())
    predicted_class = class_names[predicted_idx]
    confidence = float(probs[predicted_idx]) * 100.0
    per_class = {
        class_names[i]: float(probs[i]) * 100.0 for i in range(len(class_names))
    }
    top_k_idx = probs.argsort()[::-1][: args.top_k]
    top_k = [
        {"class": class_names[int(i)], "confidence_pct": float(probs[int(i)]) * 100.0}
        for i in top_k_idx
    ]

    result: Dict[str, Any] = {
        "image_path": str(Path(args.image).resolve()),
        "predicted_class": predicted_class,
        "predicted_index": predicted_idx,
        "confidence_pct": confidence,
        "probabilities_pct": per_class,
        "top_k": top_k,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_epoch": payload.get("epoch"),
        "image_size": image_size,
    }

    # Console output
    print("=" * 60)
    print(f"Image        : {result['image_path']}")
    print(f"Predicted    : {predicted_class}")
    print(f"Confidence   : {confidence:.2f}%")
    print("Probabilities:")
    for name, p in per_class.items():
        bar = "#" * int(round(p / 2))
        print(f"  {name:<25s} {p:6.2f}%  {bar}")
    print(f"Top-{args.top_k}:")
    for entry in top_k:
        print(f"  {entry['class']:<25s} {entry['confidence_pct']:6.2f}%")
    print("=" * 60)

    if args.output:
        save_json(result, args.output)
        print(f"\nSaved prediction -> {args.output}")
    return result


if __name__ == "__main__":
    main()