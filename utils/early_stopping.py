"""Early-stopping utility."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class EarlyStopping:
    """Track a metric and trigger when it stops improving.

    Args:
        patience: Number of epochs to wait after the last improvement.
        min_delta: Minimum change to qualify as an improvement.
        mode: ``"min"`` for loss-like metrics, ``"max"`` for accuracy-like
            metrics.
    """

    patience: int = 5
    min_delta: float = 1e-4
    mode: str = "max"

    best_value: Optional[float] = None
    counter: int = 0
    should_stop: bool = False

    def __post_init__(self) -> None:
        if self.mode not in {"min", "max"}:
            raise ValueError(f"mode must be 'min' or 'max', got {self.mode!r}")
        self.best_value = None
        self.counter = 0
        self.should_stop = False

    def _is_improvement(self, current: float) -> bool:
        if self.best_value is None:
            return True
        if self.mode == "min":
            return current < self.best_value - self.min_delta
        return current > self.best_value + self.min_delta

    def step(self, value: float) -> bool:
        """Update internal state with a new metric value.

        Returns:
            ``True`` when the current value is the new best (i.e. caller
            should checkpoint), otherwise ``False``.
        """
        if self._is_improvement(value):
            self.best_value = value
            self.counter = 0
            return True
        self.counter += 1
        if self.counter >= self.patience:
            self.should_stop = True
        return False

    def reset(self) -> None:
        """Reset the tracker (useful between training/evaluation phases)."""
        self.best_value = None
        self.counter = 0
        self.should_stop = False

    def state_dict(self) -> dict:
        return {
            "best_value": self.best_value,
            "counter": self.counter,
            "should_stop": self.should_stop,
            "patience": self.patience,
            "min_delta": self.min_delta,
            "mode": self.mode,
        }

    def load_state_dict(self, state: dict) -> None:
        self.best_value = state.get("best_value")
        self.counter = state.get("counter", 0)
        self.should_stop = state.get("should_stop", False)
        self.patience = state.get("patience", self.patience)
        self.min_delta = state.get("min_delta", self.min_delta)
        self.mode = state.get("mode", self.mode)