"""Evaluation metrics: accuracy, precision, recall, F1, confusion matrix."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from .helpers import PathLike, ensure_dir, save_json

# ---------------------------------------------------------------------------
# Core computations
# ---------------------------------------------------------------------------
def _to_numpy(x: Union[torch.Tensor, np.ndarray, Sequence]) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def compute_accuracy(y_true: Union[torch.Tensor, np.ndarray, Sequence],
                     y_pred: Union[torch.Tensor, np.ndarray, Sequence]) -> float:
    """Top-1 accuracy in ``[0, 1]``."""
    yt = _to_numpy(y_true).astype(int)
    yp = _to_numpy(y_pred).astype(int)
    return float(accuracy_score(yt, yp))


def compute_classification_metrics(
    y_true: Union[torch.Tensor, np.ndarray, Sequence],
    y_pred: Union[torch.Tensor, np.ndarray, Sequence],
    class_names: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Compute a full battery of classification metrics.

    Returns:
        Dictionary with accuracy, macro/weighted precision/recall/F1, and
        per-class breakdown.
    """
    yt = _to_numpy(y_true).astype(int)
    yp = _to_numpy(y_pred).astype(int)

    acc = float(accuracy_score(yt, yp))
    precisions, recalls, f1s, supports = precision_recall_fscore_support(
        yt, yp, labels=np.arange(len(class_names)) if class_names else None,
        zero_division=0,
    )
    macro_f1 = float(f1_score(yt, yp, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(yt, yp, average="weighted", zero_division=0))
    macro_precision = float(np.mean(precisions))
    macro_recall = float(np.mean(recalls))

    per_class: Dict[str, Dict[str, float]] = {}
    if class_names is None:
        class_names = [f"class_{i}" for i in range(len(precisions))]
    for i, name in enumerate(class_names):
        per_class[name] = {
            "precision": float(precisions[i]),
            "recall": float(recalls[i]),
            "f1": float(f1s[i]),
            "support": int(supports[i]),
        }

    return {
        "accuracy": acc,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "per_class": per_class,
    }


def compute_confusion_matrix(
    y_true: Union[torch.Tensor, np.ndarray, Sequence],
    y_pred: Union[torch.Tensor, np.ndarray, Sequence],
    class_names: Optional[Sequence[str]] = None,
    normalize: Optional[str] = None,
) -> np.ndarray:
    """Scikit-learn's ``confusion_matrix`` wrapper with optional normalization.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        class_names: Optional list of class names for ordering.
        normalize: ``"true"`` | ``"pred"`` | ``"all"`` | ``None``.
    """
    yt = _to_numpy(y_true).astype(int)
    yp = _to_numpy(y_pred).astype(int)
    labels = list(range(len(class_names))) if class_names else None
    cm = confusion_matrix(yt, yp, labels=labels, normalize=normalize)
    return cm


def classification_report_text(
    y_true: Union[torch.Tensor, np.ndarray, Sequence],
    y_pred: Union[torch.Tensor, np.ndarray, Sequence],
    class_names: Optional[Sequence[str]] = None,
) -> str:
    """Return a pretty-printed scikit-learn classification report."""
    yt = _to_numpy(y_true).astype(int)
    yp = _to_numpy(y_pred).astype(int)
    return classification_report(
        yt, yp,
        target_names=list(class_names) if class_names else None,
        digits=4, zero_division=0,
    )


# ---------------------------------------------------------------------------
# Plotting & persistence
# ---------------------------------------------------------------------------
def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: Sequence[str],
    save_path: PathLike,
    title: str = "Confusion Matrix",
    normalize: bool = False,
    figsize: tuple = (8, 6),
) -> Path:
    """Save a heat-map of the confusion matrix."""
    out = ensure_dir(Path(save_path).parent) / Path(save_path).name
    fig, ax = plt.subplots(figsize=figsize)
    try:
        import seaborn as sns  # optional; gracefully falls back to plain matplotlib
        sns.heatmap(
            cm,
            annot=True,
            fmt=".2f" if normalize else "d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
            cbar=True,
            square=True,
            ax=ax,
        )
    except Exception:
        im = ax.imshow(cm, cmap="Blues", interpolation="nearest")
        ax.set_xticks(np.arange(len(class_names)))
        ax.set_yticks(np.arange(len(class_names)))
        ax.set_xticklabels(class_names)
        ax.set_yticklabels(class_names)
        ax.figure.colorbar(im, ax=ax)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                v = cm[i, j]
                ax.text(j, i, f"{v:.2f}" if normalize else f"{int(v)}",
                        ha="center", va="center",
                        color="white" if v > cm.max() / 2 else "black")

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


def save_metrics_bundle(
    metrics: Dict[str, Any],
    cm: np.ndarray,
    class_names: Sequence[str],
    report_text: str,
    out_dir: PathLike,
    prefix: str = "eval",
) -> Dict[str, str]:
    """Persist metrics JSON, classification report JSON, and confusion matrix.

    Returns:
        Mapping of artefact name → file path.
    """
    out = ensure_dir(out_dir)
    paths = {
        "metrics_json": str(save_json(metrics, out / f"{prefix}_metrics.json")),
        "report_json": str(save_json(
            {"classification_report": report_text, "per_class": metrics["per_class"]},
            out / f"{prefix}_report.json",
        )),
        "report_txt": str(_save_text(report_text, out / f"{prefix}_report.txt")),
        "confusion_matrix_raw": str(plot_confusion_matrix(
            cm, class_names, out / f"{prefix}_cm_raw.png",
            title="Confusion Matrix (Counts)",
        )),
        "confusion_matrix_norm": str(plot_confusion_matrix(
            cm.astype(np.float32) / np.maximum(cm.sum(axis=1, keepdims=True), 1),
            class_names, out / f"{prefix}_cm_normalized.png",
            title="Confusion Matrix (Normalized)",
            normalize=True,
        )),
    }
    return paths


def _save_text(text: str, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path