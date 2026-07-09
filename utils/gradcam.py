"""Grad-CAM adapted for DINOv3 (Vision Transformer).

For ViTs the standard "last convolutional feature map" is unavailable.
We instead use the **last transformer block's output** projected back to
patch space (excluding the CLS token).  Gradients of the target class
score w.r.t. each patch token are pooled to weight the patch
activations, producing a heat-map of shape ``(1+H/p, 1+W/p)``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from .helpers import PathLike, ensure_dir

# ---------------------------------------------------------------------------
# Hook-based Grad-CAM engine
# ---------------------------------------------------------------------------
class ViTGradCAM:
    """Compute Grad-CAM heatmaps for a DINOv3 + LoRA classifier.

    The classifier is expected to expose ``model.backbone`` (a HuggingFace
    ViTModel) and ``model.head`` (a linear classifier).  Hooks are
    attached to the **last transformer block's output** so that
    backward-gradients of the target logit can be propagated.
    """

    def __init__(self, model: nn.Module, target_layer: Optional[nn.Module] = None) -> None:
        """Args:
            model: The end-to-end classifier wrapping a ``backbone`` and a
                ``head``.  Both must produce / consume the CLS token.
            target_layer: Optional override; defaults to the last layer of
                ``model.backbone.encoder.layer``.
        """
        self.model = model
        self.model.eval()
        self.target_layer = target_layer or self._auto_find_target_layer()
        self._activations: Optional[torch.Tensor] = None
        self._gradients: Optional[torch.Tensor] = None
        self._handles: List[torch.utils.hooks.RemovableHandle] = []
        self._register_hooks()

    # -----------------------------------------------------------------
    def _auto_find_target_layer(self) -> nn.Module:
        """Discover the last transformer block of the backbone."""
        backbone = getattr(self.model, "backbone", None)
        if backbone is None:
            raise AttributeError(
                "Model must expose a `backbone` attribute for Grad-CAM."
            )
        # HuggingFace AutoModel: backbone.encoder.layer[-1]
        encoder = getattr(backbone, "encoder", None)
        if encoder is not None and hasattr(encoder, "layer"):
            return encoder.layer[-1]
        # Fallback — last submodule
        return list(backbone.modules())[-1]

    def _register_hooks(self) -> None:
        def fwd_hook(_module, _inputs, output):
            # HF block output is a tuple; first element is hidden states.
            if isinstance(output, tuple):
                output = output[0]
            self._activations = output

        def bwd_hook(_module, grad_input, grad_output):
            # grad_output is a tuple; first element is gradient w.r.t. output.
            if isinstance(grad_output, tuple):
                grad_output = grad_output[0]
            self._gradients = grad_output

        self._handles.append(self.target_layer.register_forward_hook(fwd_hook))
        self._handles.append(self.target_layer.register_full_backward_hook(bwd_hook))

    # -----------------------------------------------------------------
    def remove_hooks(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def __enter__(self) -> "ViTGradCAM":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.remove_hooks()

    # -----------------------------------------------------------------
    @torch.enable_grad()
    def generate(
        self,
        pixel_values: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> np.ndarray:
        """Run a forward+backward pass and return a heatmap as a numpy array.

        Args:
            pixel_values: Input batch of shape ``(B, 3, H, W)``.  ``B=1``
                is recommended.
            target_class: Class index to explain.  ``None`` ⇒ predicted
                class.

        Returns:
            Float32 array of shape ``(H, W)`` in ``[0, 1]``.
        """
        was_training = self.model.training
        self.model.eval()
        self.model.zero_grad()

        pixel_values = pixel_values.clone().detach().requires_grad_(False)
        outputs = self.model(pixel_values)
        logits = outputs["logits"] if isinstance(outputs, dict) else outputs
        if target_class is None:
            target_class = int(logits.argmax(dim=-1).item())

        score = logits[0, target_class]
        score.backward(retain_graph=False)

        activations = self._activations  # (B, T, D)
        gradients = self._gradients      # (B, T, D)
        if activations is None or gradients is None:
            raise RuntimeError("Hooks did not capture activations/gradients.")

        # Pool gradients over feature dimension → weights per token.
        weights = gradients[0].mean(dim=-1)            # (T,)
        cam_tokens = (weights[:, None] * activations[0]).sum(dim=-1)  # (T,)

        # Drop CLS token (index 0) — keep patch tokens.
        patch_tokens = cam_tokens[1:]
        num_patches = patch_tokens.shape[0]
        grid = int(round(np.sqrt(num_patches)))
        if grid * grid != num_patches:
            raise RuntimeError(
                f"Token count {num_patches} is not a perfect square; cannot reshape to grid."
            )
        heatmap = patch_tokens.reshape(grid, grid).detach().cpu().numpy()

        # ReLU & normalize to [0, 1]
        heatmap = np.maximum(heatmap, 0.0)
        heatmap = heatmap / (heatmap.max() + 1e-8)

        # Resize to image size
        image_h, image_w = pixel_values.shape[-2:]
        heatmap = cv2.resize(heatmap, (image_w, image_h), interpolation=cv2.INTER_CUBIC)
        heatmap = np.clip(heatmap, 0.0, 1.0)

        # Restore mode
        if was_training:
            self.model.train()
        return heatmap.astype(np.float32)


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------
def overlay_heatmap(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.45,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """Overlay a heatmap on an RGB image (uint8 in, uint8 out)."""
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    if image.shape[:2] != heatmap.shape[:2]:
        heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    colored = cv2.applyColorMap(np.uint8(255 * heatmap), colormap)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(image, 1.0 - alpha, colored, alpha, 0)
    return overlay


def save_heatmap_visual(
    image: np.ndarray,
    heatmap: np.ndarray,
    save_path: PathLike,
    title: Optional[str] = None,
) -> Path:
    """Save a side-by-side visualisation (image | heatmap | overlay)."""
    ensure_dir(Path(save_path).parent)
    overlay = overlay_heatmap(image, heatmap)
    fig, axes = plt_subplots(1, 3, titles=("Original", "Heatmap", "Overlay"))
    axes[0].imshow(image)
    axes[1].imshow(heatmap, cmap="jet")
    axes[2].imshow(overlay)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt_close(fig)
    return Path(save_path)


def plt_subplots(nrows: int, ncols: int, titles: Sequence[str]):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))
    # Flatten axes to a 1D iterable so both 1-row and N-row layouts work.
    flat_axes = np.atleast_1d(axes).ravel()
    for ax, t in zip(flat_axes, titles):
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(t)
    return fig, flat_axes


def plt_close(fig) -> None:
    import matplotlib.pyplot as plt
    plt.close(fig)


def load_image_rgb(path: PathLike, image_size: int = 224) -> Tuple[np.ndarray, Image.Image]:
    """Read an image from disk and return both a numpy uint8 array and PIL image.

    The returned numpy array is resized to ``image_size × image_size``.
    """
    pil = Image.open(str(path)).convert("RGB")
    pil = pil.resize((image_size, image_size), Image.BILINEAR)
    return np.asarray(pil), pil