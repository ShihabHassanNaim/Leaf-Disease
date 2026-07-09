"""Helper utilities: seeding, JSON I/O, formatting, GPU stats."""

from __future__ import annotations

import json
import logging
import os
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

import numpy as np
import torch

PathLike = Union[str, os.PathLike]

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def get_logger(
    name: str,
    log_file: Optional[PathLike] = None,
    level: Union[str, int] = "INFO",
    console: bool = True,
) -> logging.Logger:
    """Create or return a configured logger.

    Args:
        name: Logger name (typically ``__name__``).
        log_file: Optional path to a log file. Parent directories are created
            automatically.
        level: Log level string or numeric constant.
        console: Whether to also stream to ``stdout``.

    Returns:
        A configured ``logging.Logger`` instance.
    """
    logger = logging.getLogger(name)
    if getattr(logger, "_puku_configured", False):
        return logger

    logger.setLevel(level)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if console:
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        logger.addHandler(sh)

    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    logger.propagate = False
    logger._puku_configured = True  # type: ignore[attr-defined]
    return logger


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------
def set_seed(seed: int, deterministic: bool = True, benchmark: bool = False) -> None:
    """Seed every relevant RNG for reproducible runs.

    Args:
        seed: The integer seed value.
        deterministic: If ``True`` sets PyTorch deterministic algorithms.
        benchmark: If ``True`` enables cuDNN benchmark (off by default for
            reproducibility on small GPUs).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = benchmark
    os.environ["PYTHONHASHSEED"] = str(seed)


def resolve_path(root: Path, value: Union[str, os.PathLike]) -> Path:
    """Resolve a (possibly relative) path against *root*.

    Accepts strings, ``os.PathLike`` objects, and existing :class:`Path`
    instances.  Absolute paths are returned unchanged; relative paths are
    expanded as ``root / value`` and made absolute.
    """
    p = Path(os.path.expanduser(str(value)))
    if not p.is_absolute():
        p = (root / p).resolve()
    return p


def resolve_path_str(root: Path, value: str) -> Path:
    """Resolve a (possibly relative) path string against *root*."""
    return resolve_path(root, value)


# ---------------------------------------------------------------------------
# Filesystem & JSON
# ---------------------------------------------------------------------------
def ensure_dir(path: PathLike) -> Path:
    """Create *path* (including parents) and return it as :class:`Path`."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(data: Any, path: PathLike, indent: int = 4) -> Path:
    """Persist *data* to JSON, creating parent directories if needed."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=indent, ensure_ascii=False, default=_json_default)
    return out


def load_json(path: PathLike) -> Any:
    """Load JSON from *path*."""
    with Path(path).open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _json_default(obj: Any) -> Any:
    """Best-effort JSON encoder for numpy / torch scalars."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


# ---------------------------------------------------------------------------
# GPU utilities
# ---------------------------------------------------------------------------
def get_device(prefer_cuda: bool = True) -> torch.device:
    """Return the best available :class:`torch.device`."""
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def gpu_memory_summary(device: Optional[torch.device] = None) -> Dict[str, float]:
    """Return a snapshot of CUDA memory in MB.

    Returns an empty dict on CPU devices.
    """
    if not torch.cuda.is_available():
        return {}
    dev = device or torch.device("cuda")
    free, total = torch.cuda.mem_get_info(dev)
    allocated = torch.cuda.memory_allocated(dev)
    reserved = torch.cuda.memory_reserved(dev)
    peak = torch.cuda.max_memory_allocated(dev)
    mb = 1024 ** 2
    return {
        "free_mb": free / mb,
        "total_mb": total / mb,
        "allocated_mb": allocated / mb,
        "reserved_mb": reserved / mb,
        "peak_mb": peak / mb,
    }


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """Count total and trainable parameters in *model*."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------
def format_seconds(seconds: float) -> str:
    """Format *seconds* as ``Hh Mm Ss``."""
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m {s:02d}s"
    if m:
        return f"{m}m {s:02d}s"
    return f"{s}s"


def humanize_int(n: int) -> str:
    """Insert thousands separators (e.g. 21_500_000 → ``21,500,000``)."""
    return f"{n:,}"


def to_list(x: Iterable) -> List:
    """Materialize an iterable to a list, returning the input unchanged if
    it is already a list/tuple."""
    if isinstance(x, (list, tuple)):
        return list(x)
    return list(x)