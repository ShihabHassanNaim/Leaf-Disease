"""Dataset loading for Chilli Leaf Disease Classification."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms as T

from utils.helpers import PathLike, get_logger, set_seed

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------
def build_train_transforms(image_size: int = 224) -> T.Compose:
    """Augmentation pipeline suitable for natural-image classification."""
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.2),
        T.RandomRotation(degrees=15),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def build_eval_transforms(image_size: int = 224) -> T.Compose:
    """Deterministic eval/preprocessing transforms (ImageNet stats)."""
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------
class ChilliLeafDataset(Dataset):
    """In-memory image-folder dataset.

    Args:
        root: Path containing one sub-directory per class.
        class_names: Optional explicit ordering of class names.  When
            ``None`` the directories are sorted alphabetically.
        recursive: If ``True`` (default) sub-folders are scanned via
            :meth:`Path.rglob`, which handles the
            ``Class/Class/*.jpg`` layout shipped with the dataset.
        valid_extensions: Lower- or upper-case file suffixes to keep.
        transform: Callable applied to each PIL image.
    """

    def __init__(
        self,
        root: PathLike,
        class_names: Optional[Sequence[str]] = None,
        recursive: bool = True,
        valid_extensions: Sequence[str] = (
            ".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG",
        ),
        transform: Optional[Callable] = None,
    ) -> None:
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(f"Dataset root not found: {self.root}")

        if class_names is None:
            class_names = sorted(
                d.name for d in self.root.iterdir() if d.is_dir()
            )
        self.class_names: List[str] = list(class_names)
        self.class_to_idx: Dict[str, int] = {
            c: i for i, c in enumerate(self.class_names)
        }
        self.transform = transform

        samples: List[Tuple[Path, int]] = []
        for cls in self.class_names:
            cls_dir = self.root / cls
            if not cls_dir.exists():
                logger.warning("Class directory missing: %s", cls_dir)
                continue
            iterator = cls_dir.rglob("*") if recursive else cls_dir.iterdir()
            for p in iterator:
                if not p.is_file():
                    continue
                if p.suffix not in valid_extensions:
                    continue
                samples.append((p, self.class_to_idx[cls]))
        if not samples:
            raise RuntimeError(
                f"No images found under {self.root} with extensions {valid_extensions}."
            )
        self.samples = samples
        logger.info(
            "Loaded %d images across %d classes from %s",
            len(self.samples), len(self.class_names), self.root,
        )

    # -----------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        path, label = self.samples[index]
        try:
            with Image.open(path) as img:
                img = img.convert("RGB")
        except Exception as exc:  # pragma: no cover - I/O robustness
            raise RuntimeError(f"Failed to load image {path}: {exc}") from exc
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    # -----------------------------------------------------------------
    def class_counts(self) -> Dict[str, int]:
        """Return a mapping of class name → sample count."""
        counts: Dict[str, int] = {c: 0 for c in self.class_names}
        for _, idx in self.samples:
            counts[self.class_names[idx]] += 1
        return counts


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------
def stratified_split(
    dataset: ChilliLeafDataset,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[ChilliLeafDataset, ChilliLeafDataset, ChilliLeafDataset]:
    """Split a dataset into train/val/test subsets using PyTorch ``random_split``.

    Uses a fixed generator seeded with *seed* so the split is reproducible.
    Approximate ratios (the actual sizes may differ by ±1 sample per split).
    """
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    n = len(dataset)
    n_train = int(round(train_ratio * n))
    n_val = int(round(val_ratio * n))
    n_test = n - n_train - n_val
    generator = torch.Generator().manual_seed(seed)
    splits = random_split(dataset, [n_train, n_val, n_test], generator=generator)
    logger.info(
        "Split dataset (seed=%d) → train=%d  val=%d  test=%d (total=%d)",
        seed, n_train, n_val, n_test, n,
    )
    return splits[0], splits[1], splits[2]


def split_indices(
    n: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[List[int], List[int], List[int]]:
    """Helper that returns the raw indices used by :func:`stratified_split`.

    Useful for the notebook, where you'd like to inspect the splits.
    """
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")
    generator = torch.Generator().manual_seed(seed)
    indices = list(range(n))
    random.Random(seed).shuffle(indices)
    n_train = int(round(train_ratio * n))
    n_val = int(round(val_ratio * n))
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]
    return train_idx, val_idx, test_idx


# ---------------------------------------------------------------------------
# DataLoaders
# ---------------------------------------------------------------------------
def create_dataloaders(
    root: PathLike,
    image_size: int = 224,
    batch_size: int = 16,
    num_workers: int = 4,
    pin_memory: bool = True,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    class_names: Optional[Sequence[str]] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    """End-to-end: build transforms, splits, and DataLoaders.

    Returns:
        ``(train_loader, val_loader, test_loader, class_names)``.
    """
    set_seed(seed)

    train_tf = build_train_transforms(image_size)
    eval_tf = build_eval_transforms(image_size)

    # Train uses augmentation; val/test must not — so we wrap each subset
    # in a tiny view that swaps the transform.
    base_train = ChilliLeafDataset(
        root=root, class_names=class_names,
        transform=train_tf,
    )
    class_names_out = base_train.class_names

    # We must use the SAME indices as random_split to keep train/val/test
    # boundaries aligned with the unsplit dataset's transform.  Trick:
    # attach *no* transform to the base dataset, and apply transforms
    # dynamically in __getitem__ via a thin wrapper.
    base_eval = ChilliLeafDataset(
        root=root, class_names=class_names_out,
        transform=None,
    )

    n = len(base_eval)
    n_train = int(round(train_ratio * n))
    n_val = int(round(val_ratio * n))
    n_test = n - n_train - n_val
    generator = torch.Generator().manual_seed(seed)
    train_sub, val_sub, test_sub = random_split(
        base_eval, [n_train, n_val, n_test], generator=generator,
    )

    train_ds = _TransformSubset(train_sub, train_tf)
    val_ds = _TransformSubset(val_sub, eval_tf)
    test_ds = _TransformSubset(test_sub, eval_tf)

    common = dict(num_workers=num_workers, pin_memory=pin_memory)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=False, **common,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, drop_last=False, **common,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, drop_last=False, **common,
    )
    return train_loader, val_loader, test_loader, class_names_out


class _TransformSubset(Dataset):
    """Wrap a :class:`torch.utils.data.Subset` to inject a transform per split."""

    def __init__(self, subset, transform: Optional[Callable]) -> None:
        self.subset = subset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.subset)

    def __getitem__(self, index: int):
        path, label = self.subset.dataset.samples[self.subset.indices[index]]
        with Image.open(path) as img:
            img = img.convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label